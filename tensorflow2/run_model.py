import tensorflow as tf

model = tf.keras.models.load_model('./efficientNetB0_neuron')

example_inputs = tf.random.uniform(
    [1,224,224,3], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None
)

output = model(example_inputs)
print(output)