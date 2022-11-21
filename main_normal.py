import time

import numpy as np
import os

import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import json

import pickle

mod_path = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"

ds = tfds.load('coco/2017', split='validation', shuffle_files=True)

with open('instances_val2017.json') as f:
   dic = json.load(f)
   

   
df=pd.DataFrame.from_dict(dic["annotations"])
df=df[["image_id","category_id"]]

start_time = time.time()

detector = hub.load(mod_path)

end_time = time.time()

benchm ={}
benchm["init_time"]= end_time - start_time


inference_time = []
precisions = []
for count, example in enumerate(ds):
    if count % 30==0:
        print(count)
    
    
    image = example["image"]
    image = tf.expand_dims(image, 0)
    start_time = time.time()
    result= detector(image)
    end_time = time.time()
    result = {key: value.numpy() for key, value in result.items()}
    scores = result["detection_scores"][0]
    boxes = result["detection_boxes"][0]
    classes = result["detection_classes"][0]
    num_detections = result['num_detections']
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

benchm["size_Model"]= os.stat("ssd_mobilenet_v2_2.tar").st_size / float(2**20)

with open('normal.pkl', 'wb') as f:
    pickle.dump(benchm, f)






