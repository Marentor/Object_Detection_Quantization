import time

import numpy as np
import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import json
import pickle
import pandas as pd

mod_path = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"

ds = tfds.load('coco/2017', split='validation', shuffle_files=True)

with open('instances_val2017.json') as f:
   dic = json.load(f)
   

   
df=pd.DataFrame.from_dict(dic["annotations"])
df=df[["image_id","category_id"]]

detector = hub.load(mod_path)


inference_time = []
precisions = []
start_time = time.time()
benchm ={}

#saved_model_dir= 'models/original_model'
saved_model_dir='saved_model'
#converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter = tf.lite.TFLiteConverter.from_keras_model(detector )
converter.optimizations = [tf.lite.Optimize.DEFAULT]

### for float 16 this is commented in ###
#converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
interpreter.resize_tensor_input(0, [1, 320, 320, 3])

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

end_time = time.time()
benchm["init_time"]= end_time - start_time

images=[]
for count, example in enumerate(ds):
    if count % 30==0:
        print(count)
        
    image=example["image"]
    #image=tf.dtypes.cast(image, tf.int8)
    images.append(image)


    input_tensor=tf.image.resize(images[count],[320,320])
    input_tensor = tf.dtypes.cast(input_tensor, tf.uint8)
    input_tensor = tf.expand_dims(input_tensor, 0)
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.allocate_tensors()
    #signature_defs = interpreter.get_signature_list()
    #print(signature_defs)


    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    # # run the inference
    
    
    
    start_time = time.time()
    interpreter.invoke()
    boxes=interpreter.get_tensor(output_details[4]['index'])
    boxes = np.squeeze(boxes)
    classes=np.squeeze(interpreter.get_tensor(output_details[5]['index']))
    scores=np.squeeze(interpreter.get_tensor(output_details[6]['index']))
    end_time = time.time()
    
    inference_time.append(end_time - start_time)
    
    current_id= int(example["image/id"])
    scores = [x for x in scores if x >0.5]
    classes = set(classes[:len(scores)])
    classes = {int(x) for x in classes}
    real_classes= set(df.loc[df["image_id"]==current_id,"category_id"])
    
    if len(classes)!=0:
        precisions.append(len(classes.intersection(real_classes))/len(classes))


    if count == 1000:
        break
    
    
benchm["precisions"]=precisions
benchm["inference_times"]=inference_time

with open("temp", "wb") as f:
    f.write(tflite_quant_model)

benchm["size_Model"]= os.stat("temp").st_size / float(2**20)

with open('int8.pkl', 'wb') as f:
    pickle.dump(benchm, f)
