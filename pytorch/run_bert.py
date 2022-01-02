import tensorflow  # to workaround a protobuf version conflict issue
import torch
import torch.neuron
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import transformers
import os
import warnings

# Load TorchScript back
model_neuron = torch.jit.load('bert_neuron.pt')
# Verify the TorchScript works on both example inputs
paraphrase_classification_logits_neuron = model_neuron(*example_inputs_paraphrase)
not_paraphrase_classification_logits_neuron = model_neuron(*example_inputs_not_paraphrase)
classes = ['not paraphrase', 'paraphrase']
paraphrase_prediction = paraphrase_classification_logits_neuron[0][0].argmax().item()
not_paraphrase_prediction = not_paraphrase_classification_logits_neuron[0][0].argmax().item()
print('BERT says that "{}" and "{}" are {}'.format(sequence_0, sequence_2, classes[paraphrase_prediction]))
print('BERT says that "{}" and "{}" are {}'.format(sequence_0, sequence_1, classes[not_paraphrase_prediction]))

def get_input_with_padding(batch, batch_size, max_length):
    ## Reformulate the batch into three batch tensors - default batch size batches the outer dimension
    encoded = batch['encoded']
    inputs = torch.squeeze(encoded['input_ids'], 1)
    attention = torch.squeeze(encoded['attention_mask'], 1)
    token_type = torch.squeeze(encoded['token_type_ids'], 1)
    quality = list(map(int, batch['quality']))

    if inputs.size()[0] != batch_size:
        print("Input size = {} - padding".format(inputs.size()))
        remainder = batch_size - inputs.size()[0]
        zeros = torch.zeros( [remainder, max_length], dtype=torch.long )
        inputs = torch.cat( [inputs, zeros] )
        attention = torch.cat( [attention, zeros] )
        token_type = torch.cat( [token_type, zeros] )

    assert(inputs.size()[0] == batch_size and inputs.size()[1] == max_length)
    assert(attention.size()[0] == batch_size and attention.size()[1] == max_length)
    assert(token_type.size()[0] == batch_size and token_type.size()[1] == max_length)

    return (inputs, attention, token_type), quality

def count(output, quality):
    assert output.size(0) >= len(quality)
    correct_count = 0
    count = len(quality)

    batch_predictions = [ row.argmax().item() for row in output ]

    for a, b in zip(batch_predictions, quality):
        if int(a)==int(b):
            correct_count += 1

    return correct_count, count