import json
import pickle
import pandas as pd

from matplotlib import pyplot as plt

with open('int8.pkl', 'rb') as f:
    int8 = pickle.load(f)
    
with open('normal.pkl', 'rb') as f:
    float32 = pickle.load(f)
    
with open('float16.pkl', 'rb') as f:
    float16 = pickle.load(f)
    
    
plt.title('Inference times of int8 Quantization')
plt.xlabel('Seconds')
    
plt.hist(int8["inference_times"], 100)



plt.savefig('inf_times_int8.png')

plt.show()
####

plt.title('Inference times of float16 Quantization')
plt.xlabel('Seconds')
    
plt.hist(float16["inference_times"], 100)


plt.savefig('inf_times_float16.png')

plt.show()

####

plt.title('Inference times of float32 Quantization')
plt.xlabel('Seconds')
    
plt.hist(float32["inference_times"], 100)


plt.savefig('inf_times_float32.png')

plt.show()

####

print(sum(float32["precisions"])/len(float32["precisions"]))
print(sum(float32["inference_times"])/len(float32["inference_times"]))
print(float32["size_Model"])
print(float32["init_time"])

print("-------------")

print(sum(float16["precisions"])/len(float16["precisions"]))
print(sum(float16["inference_times"])/len(float16["inference_times"]))
print(float16["size_Model"])
print(float16["init_time"])

print("-------------")

print(sum(int8["precisions"])/len(int8["precisions"]))
print(sum(int8["inference_times"])/len(int8["inference_times"]))
print(int8["size_Model"])
print(int8["init_time"])

print("-------------")

