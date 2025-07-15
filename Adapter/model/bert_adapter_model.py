import torch
from transformers import BertModel, BertTokenizer
from Adapter.model.adapter import Adapter


def build_bert_with_adapter(config):
    model_name = config["model_name"]
    hidden_size = config["hidden_size"]
    adapter_size = config["adapter_size"]
    layer_idx = config["layer_idx"]

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    adapter = Adapter(hidden_size=hidden_size, adapter_size=adapter_size)

    layer_module = model.encoder.layer[layer_idx].output
    original_forward = layer_module.forward

    def new_forward(self, hidden_states, input_tensor, **kwargs):
        hidden_states = original_forward(hidden_states, input_tensor, **kwargs)
        hidden_states[0] = adapter(hidden_states[0])
        return hidden_states

    layer_module.forward = new_forward.__get__(layer_module, type(layer_module))

    for param in model.parameters():
        param.requires_grad = False

    for param in adapter.parameters():
        param.requires_grad = True

    return tokenizer, model, adapter
