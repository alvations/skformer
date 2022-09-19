import gc

import numpy as np

import torch
from torch import nn
from torch.nn.utils import prune

from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
	"""Mean Pooling - Take attention mask into account for correct averaging."""
	token_embeddings = model_output[0]  #First element of model_output contains all token embeddings.
	input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
	return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def prune_model_l1_unstructured(model, layer_type, proportion):
    for module in model.modules():
        if isinstance(module, layer_type):
            prune.l1_unstructured(module, 'weight', proportion)
            prune.remove(module, 'weight')
    return model


class Vectorizer:
	def __init__(self, model_name, to_prune=True):
		# Load tokenizer.
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		# Load model with some clean_up processes.
		_model = AutoModel.from_pretrained(model_name)
		self.model = prune_model_l1_unstructured(_model, nn.Linear, 0.1) if to_prune else _model
		del _model
		gc.collect()

	def transform(self, sentences):
	    # Tokenize sentences
	    encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

	    # Compute token embeddings
	    with torch.no_grad():
	        model_output = self.model(**encoded_input)

	    # Perform mean pooling on the batch.
	    return mean_pooling(model_output, encoded_input['attention_mask']).cpu().detach().numpy()



