import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def fetch_graph_embedding(hidden_size):
    return torch.rand(hidden_size)

class FusionModel(nn.Module):

    def __init__(self, model_name: str, use_graph=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, return_dict=True, output_hidden_states=True, output_attentions=True)
        self.hidden_size = self.encoder.config.to_dict()['hidden_size']
        self.dropout = nn.Dropout()
        self.head = nn.Linear(hidden_size, 1)
        self.use_graph = use_graph

        if self.use_graph:
            import stanza
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.ner_tagger = stanza.Pipeline(lang='en', processors='tokenize,ner')

    def forward(self, input_ids, attention_mask):
        batch_size, n_choices, seq_length = input_ids.size()[0], input_ids.size()[1], input_ids.size()[-1]

        if self.use_graph:
            graph_embeddings = []
            for batch in input_ids:
                local_embedding = []
                for choice in batch:
                    text = self.tokenizer.decode(choice, skip_special_tokens=True)
                    doc = self.ner_tagger(text)
                    entities = [ent.text for ent in doc.ents]
                    if entities:
                        local_embedding.append(fetch_graph_embedding(self.hidden_size)) # add graph rep of all mentioned entities
                    else:
                        local_embedding.append(torch.ones(self.hidden_size))
                    break #only consider first one.
                if local_embedding:
                    neighbourhood = None
                    for i, emb in enumerate(local_embedding):
                        if i == 0:
                            neighbourhood = emb
                        else:
                            neighbourhood = torch.matmul(neighbourhood, emb)
                    graph_embeddings.append(neighbourhood)
            
            graph_embeddings = torch.stack(graph_embeddings) # (batch_size, hidden_size)


        input_ids = input_ids.view(-1, seq_length) # Transform (batch, n_choices, seq_length) to (batch*n_choices, seq_length)
        attention_mask = attention_mask.view(-1, seq_length)# Transform (batch, n_choices, seq_length) to (batch*n_choices, seq_length)
        pooled_output = self.encoder(input_ids, attention_mask)[1] # Get the pooled output from the encoder (batch_size, hidden_size)
        pooled_output = self.dropout(pooled_output) #
        logits = self.head(pooled_output)
        reshaped_logits = logits.view(-1, n_choices)
        return reshaped_logits