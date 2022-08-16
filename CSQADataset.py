from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
from datasets import load_dataset, disable_caching
from concept_net import PathRetriever
#import stanza
from stop import eng_stop_words
from sentence_transformers import SentenceTransformer, util
import json


class CSQADataset(Dataset):

    def __init__(self, dataset, split, MODEL, use_graph):
        options = ["A", "B", "C", "D", "E"]
        self.use_graph = use_graph
        if self.use_graph == True:
            print("Graph mode activated...")
            self.scorer = SentenceTransformer('all-MiniLM-L6-v2')
            self.PG = PathRetriever("./data/conceptnet/")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.dataset = load_dataset(dataset, split=split)
        self.num_choices = 4 if dataset == "openbookqa" else 5
        self.question_key = "question_stem" if dataset == "openbookqa" else "question"
        use_batched_processing = True if self.use_graph == False else False
        disable_caching()
        self.tokenized_dataset = self.dataset.map(self.preprocess_function, batched=use_batched_processing)

        
        self.label_indexer = {v:k for k,v in enumerate(options)}
        self.label_inverter = {k:v for k,v in enumerate(options)}
        self.labels = [self.label_indexer[x] for x in self.dataset["answerKey"]]

    def __len__(self):
        return len(self.dataset)


    def find_best_path(self, question, options):
        '''
            Finds paths between concepts indentified in the quesition and the answer candidates.
            Each path is scored against the original QA context using cosine similarity using SentenceTransformer.
        '''
        answers_tmp = options
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
        return best_path

    def preprocess_function(self, examples):
        question_key = self.question_key
        n_choices = self.num_choices

        if self.use_graph == True:
            question = examples[question_key]
            options = examples["choices"]["text"]
            best_path = self.find_best_path(question, options)
            questions = []
            q = [examples[question_key] + " " + best_path for i in range(n_choices)]
            questions.append(q)
            choices = []
            choices.append(examples["choices"]["text"])

        else:  
            if isinstance(examples[question_key], str):
                questions = []
                q = [examples[question_key] for i in range(n_choices)]
                questions.append(q)
                choices = []
                choices.append(examples["choices"]["text"])
            
            else:
                questions = [[context]*n_choices for context in list(examples[question_key])]
                choices = []
                for element in examples["choices"]:
                    choices.append(element["text"])
        
        q_flat = sum(questions, [])
        c_flat = sum(choices, [])

        tokenized_examples = self.tokenizer(q_flat, c_flat, truncation=True, padding="max_length", max_length=128)
        unflatten =  {k: [v[i : i + n_choices] for i in range(0, len(v), n_choices)] for k, v in tokenized_examples.items()}

        return unflatten

    def __getitem__(self, idx):
        input_ids = torch.LongTensor(self.tokenized_dataset[idx]["input_ids"]).squeeze(0)
        attention_masks = torch.BoolTensor(self.tokenized_dataset[idx]["attention_mask"]).squeeze(0)
        y = self.labels[idx]
        return idx, input_ids, attention_masks, y