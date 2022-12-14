import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer



def fetch_graph_embedding(hidden_size):
    return torch.rand(hidden_size)

class FusionModel(nn.Module):

    def __init__(self, model_name: str, device):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, return_dict=True, output_hidden_states=True, output_attentions=True)
        self.hidden_size = 768
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(self.hidden_size, 1)
        self.device = device

    def forward(self, input_ids, attention_mask):
        batch_size, n_choices, seq_length = input_ids.size()[0], input_ids.size()[1], input_ids.size()[-1]

        input_ids = input_ids.view(-1, seq_length) # Transform (batch, n_choices, seq_length) to (batch*n_choices, seq_length)
        attention_mask = attention_mask.view(-1, seq_length)# Transform (batch, n_choices, seq_length) to (batch*n_choices, seq_length)
        pooled_output = self.encoder(input_ids, attention_mask)[1] # Get the pooled output from the encoder (batch_size, hidden_size)
        
        pooled_output = self.dropout(pooled_output) #
        logits = self.head(pooled_output)
        reshaped_logits = logits.view(-1, n_choices)
        return reshaped_logits
