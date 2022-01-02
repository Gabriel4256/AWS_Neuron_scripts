import tensorflow as tf
import tensorflow.neuron as tfn

MODEL_NAME = 'efficientNetB0'

# model = tf.keras.applications.MobileNetV3Large(
#     input_shape=None, alpha=1.0, minimalistic=False, include_top=True,
#     weights='imagenet', input_tensor=None, classes=1000, pooling=None,
#     dropout_rate=0.2, classifier_activation='softmax',
# )

model = tf.keras.applications.efficientnet.EfficientNetB0(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000,
    classifier_activation='softmax'
)

tf.keras.models.save_model(
    model, './' + MODEL_NAME
)

example_inputs = tf.random.uniform(
    [1,224,224,3], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None
)



model_neuron = tfn.trace(model, example_inputs)
model_neuron.save('./' + MODEL_NAME + '_neuron')

print(model_neuron(example_inputs))
print(model(example_inputs))
# model_loaded = tf.saved_model.load('./model_dir')
# predict_func = model_loaded['serving_default']


# model_loaded_neuron = tfn.trace(predict_func, example_inputs2)
# model_loaded_neuron.save('./model_loaded_neuron_dir')