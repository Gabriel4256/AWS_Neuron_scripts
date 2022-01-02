import os
import time
import shutil
import tensorflow as tf
import tensorflow.neuron as tfn
import tensorflow.compat.v1.keras as keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Create a workspace
WORKSPACE = './ssd_mobilenet_v3_large_coco_2020_01_14'
os.makedirs(WORKSPACE, exist_ok=True)

# Prepare export directory (old one removed)
model_dir = os.path.join(WORKSPACE, 'saved_model')
compiled_model_dir = os.path.join(WORKSPACE, 'neuron_compiled')
# shutil.rmtree(model_dir, ignore_errors=True)
# shutil.rmtree(compiled_model_dir, ignore_errors=True)

# Instantiate Keras ResNet50 model
keras.backend.set_learning_phase(0)
keras.backend.set_image_data_format('channels_last')

# model = tf.saved_model.load_v2(export_dir=model_dir, tags=None)
model = tf.compat.v2.saved_model.load(model_dir)
#  ResNet50(weights='imagenet')
# print(model.summary())

# Export SavedModel
tf.saved_model.simple_save(
    session            = keras.backend.get_session(),
    export_dir         = "./tmp",
    inputs             = {'input': tf.convert_to_tensor([1,224,224,3])},
    outputs            = {'output': tf.convert_to_tensor([1])}
)
# Compile using Neuron
tfn.saved_model.compile("./tmp", compiled_model_dir)
