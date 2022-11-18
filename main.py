import time

import numpy as np
import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

ds = tfds.load('coco/2017', split='test', shuffle_files=True)

detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

##lines for quantization
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8  # or tf.uint8
# converter.inference_output_type = tf.int8  # or tf.uint8
# tflite_quant_model = converter.convert()

inference = []
for count, example in enumerate(ds):
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
    print(scores + '\n' + classes + '\n')
    if count == 10: break
