import time

import numpy as np
import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

mod_path = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"

ds = tfds.load('coco/2017', split='test', shuffle_files=True)

detector = hub.load(mod_path)




def save_module(url, save_path):
    module = hub.KerasLayer(url)
    model = tf.keras.Sequential(module)
    tf.saved_model.save(model, save_path)
  
saved_loc= 'saved_model'
#save_module(mod_path, saved_loc)




##lines for quantization
#converter = tf.lite.TFLiteConverter.from_concrete_functions([detector.signatures["serving_default"]])


def representative_data_gen():
  for input_value in ds.batch(1).take(10):
    yield [input_value]
    

#converter = tf.lite.TFLiteConverter.from_saved_model(saved_loc)
converter = tf.lite.TFLiteConverter.from_keras_model(detector)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()


#detector = tflite_quant_model 

inference = []
for count, example in enumerate(ds):
    if count % 10==9:
        print(count)
        break
    
    
    image = example["image"]
    image = tf.expand_dims(image, 0)
    start_time = time.time()
    result= detector(image)
    end_time = time.time()
    result = {key: value.numpy() for key, value in result.items()}
    scores = result["detection_scores"]
    boxes = result["detection_boxes"]
    classes = result["detection_classes"]
    num_detections = result['num_detections']
    inference.append(end_time - start_time)
    print(scores)
    print(classes)
    
