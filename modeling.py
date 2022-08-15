import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from concept_net import PathRetriever
#import stanza
from stop import eng_stop_words
from sentence_transformers import SentenceTransformer, util
import json


def fetch_graph_embedding(hidden_size):
    return torch.rand(hidden_size)

class FusionModel(nn.Module):

    def __init__(self, model_name: str, device, use_graph=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, return_dict=True, output_hidden_states=True, output_attentions=True)
        self.hidden_size = 768
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(self.hidden_size, 1)
        self.use_graph = use_graph
        self.device = device
        self.scorer = SentenceTransformer('all-MiniLM-L6-v2')

        if self.use_graph:
            self.PG = PathRetriever("./data/conceptnet/")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            #self.ner_tagger = stanza.Pipeline(lang='en', processors='tokenize,ner')

    def forward(self, input_ids, attention_mask):
        batch_size, n_choices, seq_length = input_ids.size()[0], input_ids.size()[1], input_ids.size()[-1]
        best_paths = []
        best_paths_attention_masks = []

        if self.use_graph:
            for batch in input_ids:
                answers_tmp = []
                question = None
                for block in batch:
                    text = self.tokenizer.decode(block, skip_special_tokens=False)
                    text = text.replace('[PAD]', '')
                    q, a, e = text.split('[SEP]')
                    question = q
                    answers_tmp.append(a)
                q_words = []
                answer_words = []
                for q in question.split(" "):
                    if self.PG.is_entity(q) and q not in eng_stop_words:
                        q_words.append(q)
                for a in answers_tmp:
                    a_tmp = []
                    for a_word in a.split(" "):
                        if self.PG.is_entity(a_word): a_tmp.append(a_word)
                    answer_words.append(a_tmp)

                paths = []

                top_score = 0.0
                best_path = ""

                for q in q_words:
                    for a in answer_words:
                        for a_local in a:
                            path =  self.PG.get_path(q, a_local)
                            if path != -1: paths.append(path)
                
                for path in paths:
                    path_emb = self.scorer.encode(path, convert_to_tensor=True, show_progress_bar=False)
                    question_emb = self.scorer.encode(question, convert_to_tensor=True, show_progress_bar=False)
                    score = util.cos_sim(path_emb, question_emb)
                    if score > top_score:
                        top_score = score
                        best_path = path
                tokenized_path = self.tokenizer(best_path, padding="max_length", max_length=129)
                ids = tokenized_path["input_ids"]
                ats = tokenized_path["attention_mask"]
                best_paths.append(ids[1:]) #Ignore CLS
                best_paths_attention_masks.append(ats[1:]) #Ignore CLS

        

        tmp_i = [[x] * n_choices for x in best_paths]
        tmp_a = [[x] * n_choices for x in best_paths_attention_masks]

        path_representations_i = torch.LongTensor(tmp_i).to(self.device)
        path_representations_a = torch.LongTensor(tmp_a).to(self.device)

        input_ids = torch.cat((input_ids,path_representations_i), dim=-1)
        attention_mask = torch.cat((attention_mask,path_representations_a), dim=-1)

        seq_length  = seq_length*2

        input_ids = input_ids.view(-1, seq_length) # Transform (batch, n_choices, seq_length) to (batch*n_choices, seq_length)
        attention_mask = attention_mask.view(-1, seq_length)# Transform (batch, n_choices, seq_length) to (batch*n_choices, seq_length)
        pooled_output = self.encoder(input_ids, attention_mask)[1] # Get the pooled output from the encoder (batch_size, hidden_size)
        
        pooled_output = self.dropout(pooled_output) #
        logits = self.head(pooled_output)
        reshaped_logits = logits.view(-1, n_choices)
        return reshaped_logits
