import time

import numpy as np
import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

mod_path = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"

ds = tfds.load('coco/2017', split='test', shuffle_files=True)

detector = hub.load(mod_path)


if False:
    
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



def save_module(url, save_path):
    module = hub.KerasLayer(url)
    model = tf.keras.Sequential(module)
    tf.saved_model.save(model, save_path)
  
saved_loc= 'saved_model'
#save_module(mod_path, saved_loc)

images=[]
for count, example in enumerate(ds):
    images.append(example["image"])
    if count >20:
        break

##lines for quantization
def representative_dataset():
  for data in tf.data.Dataset.from_tensor_slices(images).batch(1).take(10):
    yield []

def representative_dataset_as():
  for data in ds.batch(1).take(10):
    yield {
      "image": data["image"],#  tf.dtypes.cast(data.image, tf.int8),
      #"bias": data["bias"], # tf.dtypes.cast(data.bias, tf.int8),
    }
    
'''def representative_dataset_test():
  for data in ds.batch(1).take(100):
    yield [data]'''

#converter = tf.lite.TFLiteConverter.from_saved_model(saved_loc)
converter = tf.lite.TFLiteConverter.from_keras_model(detector, )

converter.representative_dataset = representative_dataset
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.int8  # or tf.uint8
#converter.inference_output_type = tf.int8  # or tf.uint8

#converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                       tf.lite.OpsSet.TFLITE_BUILTINS]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Ensure that if any ops can't be quantized, the converter throws an error
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
#converter.inference_input_type = tf.uint8
#converter.inference_output_type = tf.uint8

'''converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
#converter.signatures = detector.signatures['serving_default']
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.experimental_new_converter = True
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8'''
tflite_quant_model = converter.convert()


interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
input_details  = interpreter.get_input_details()[0]
print('input: ', input_details['dtype'])
output_details = interpreter.get_output_details()[0]
print('output: ', output_details['dtype'])

interpreter.allocate_tensors()

predictions = np.zeros((10,), dtype=int)
for count, example in enumerate(ds):
    if count % 10==9:
        print(count)
        break
        

    image = example["image"]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details ['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      #test_image = test_image / input_scale + input_zero_point
    
    image = tf.expand_dims(image, 0)
    interpreter.set_tensor(input_details["index"], image)
    interpreter.invoke()
    output = interpreter.get_tensor(image)[0]
    
    predictions[count] = output.argmax()

    
