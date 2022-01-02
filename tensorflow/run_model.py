import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50
import tensorflow.contrib.slim as slim

tf.keras.backend.set_image_data_format('channels_last')

# Create input from image
img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
img_arr = image.img_to_array(img_sgl)
img_arr2 = np.expand_dims(img_arr, axis=0)
img_arr3 = resnet50.preprocess_input(img_arr2)

# Load model
COMPILED_MODEL_DIR = './ssd_mobilenet_v3_large_coco_2020_01_14/neuron_compiled'
predictor_inferentia = tf.contrib.predictor.from_saved_model(COMPILED_MODEL_DIR)

input=  tf.random.uniform(
    [4,224,224,3], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None
)

# Run inference
model_feed_dict={'input': input}
infa_rslts = predictor_inferentia(model_feed_dict);

# Display results
print(resnet50.decode_predictions(infa_rslts["output"], top=5)[0])


# Sample output will look like below:
#[('n02123045', 'tabby', 0.68817204), ('n02127052', 'lynx', 0.12701613), ('n02123159', 'tiger_cat', 0.08736559), ('n02124075', 'Egyptian_cat', 0.063844085), ('n02128757', 'snow_leopard', 0.009240591)]