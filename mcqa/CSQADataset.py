from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
from datasets import load_dataset


class CSQADataset(Dataset):

    def __init__(self, dataset, split, MODEL):
        options = ["A", "B", "C", "D", "E"]
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)

        self.dataset = load_dataset(dataset, split=split)
        self.num_choices = 4 if dataset == "openbookqa" else 5
        self.question_key = "question_stem" if dataset == "openbookqa" else "question"
        self.tokenized_dataset = self.dataset.map(self.preprocess_function, batched=True)
        
        self.label_indexer = {v:k for k,v in enumerate(options)}
        self.label_inverter = {k:v for k,v in enumerate(options)}
        self.labels = [self.label_indexer[x] for x in self.dataset["answerKey"]]
        
        #self.my_pykeen_model = torch.load('trained_model.pkl')


    
    def __len__(self):
        return len(self.dataset)


    def preprocess_function(self, examples):
        question_key = self.question_key
        n_choices = self.num_choices
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

        '''
        for ele in unflatten["input_ids"]:
            for qa_context in ele:
                dec = tokenizer.decode(qa_context)
                print(dec)
        '''
        return unflatten

    def __getitem__(self, idx):
        input_ids = torch.LongTensor(self.tokenized_dataset[idx]["input_ids"]).squeeze(0)
        attention_masks = torch.BoolTensor(self.tokenized_dataset[idx]["attention_mask"]).squeeze(0)
        y = self.labels[idx]
        return idx, input_ids, attention_masks, y