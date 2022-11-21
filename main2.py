import time

import numpy as np
import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

mod_path = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"

ds = tfds.load('coco/2017', split='test', shuffle_files=True)

detector = hub.load(mod_path)

images=[]
for count, image in enumerate(ds):
    image=image["image"]
    #image=tf.dtypes.cast(image, tf.int8)
    images.append(image)
    if count >20:
        break

#saved_model_dir= 'models/original_model'
saved_model_dir='saved_model'
#converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter = tf.lite.TFLiteConverter.from_keras_model(detector )
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]
tflite_quant_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
interpreter.resize_tensor_input(0, [1, 320, 320, 3])

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_tensor=tf.image.resize(images[0],[320,320])
input_tensor = tf.dtypes.cast(input_tensor, tf.uint8)
input_tensor = tf.expand_dims(input_tensor, 0)
input_index = interpreter.get_input_details()[0]["index"]
interpreter.allocate_tensors()
#signature_defs = interpreter.get_signature_list()
#print(signature_defs)


interpreter.set_tensor(input_details[0]['index'], input_tensor)
# # run the inference
interpreter.invoke()

boxes=interpreter.get_tensor(output_details[4]['index'])
boxes = np.squeeze(boxes)
classes=np.squeeze(interpreter.get_tensor(output_details[5]['index']))
scores=np.squeeze(interpreter.get_tensor(output_details[6]['index']))
