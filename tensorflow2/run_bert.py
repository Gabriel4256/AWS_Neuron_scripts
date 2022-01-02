from transformers import pipeline
import tensorflow as tf
import tensorflow.neuron as tfn

class TFBertForSequenceClassificationDictIO(tf.keras.Model):
    def __init__(self, model_wrapped):
        super().__init__()
        self.model_wrapped = model_wrapped
        self.aws_neuron_function = model_wrapped.aws_neuron_function
    def call(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        logits = self.model_wrapped([input_ids, attention_mask])
        return [logits]

class TFBertForSequenceClassificationFlatIO(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def call(self, inputs):
        input_ids, attention_mask = inputs
        output = self.model({'input_ids': input_ids, 'attention_mask': attention_mask})
        return output['logits']


reloaded_model = tf.keras.models.load_model('./distilbert_b128')
# rewrapped_model = TFBertForSequenceClassificationDictIO(reloaded_model)

from datasets import load_dataset
dataset = load_dataset('amazon_polarity')

string_inputs = dataset['test'][:128]['content']

# model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
# neuron_pipe = pipeline('sentiment-analysis', model=model_name, framework='tf')

# neuron_pipe.model = rewrapped_model
# neuron_pipe.model.config = pipe.model.config

# example_inputs = neuron_pipe.tokenizer(string_inputs)

print(reloaded_model(string_inputs))

#now you can reinsert our reloaded model back into our pipeline