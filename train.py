import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from datasets import load_dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("scibert_scivocab_uncased")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def preprocess_function(examples):
    questions = []
    options = []

    if isinstance(examples["id"], list):
        questions = [[context] * 4 for context in examples["question"]]
        for i, element in enumerate(examples["choices"]):
            try:
                for key, value in element.items():
                    if key == "text":
                        options.append(value)
            except Exception as e:
                print(f"Failed at idx: {i}")
                print(element)
    else:
        x = examples["choices"]["text"]
        n = len(x)
        q = []
        for i in range(4):
            q.append(examples["question"])
        questions.append(q)
        if n == 3:
            x.append(x[0])
        if n == 5:
            x = x[0:4]
        options.append(x)
  
    
    
    #Flatten for tokenization
    flat_q = sum(questions, [])
    flat_a = sum(options, [])

   
    #Tokenize [[CLS] Question [SEP] Choice]
    tokenized_examples = tokenizer(flat_q, flat_a, padding="max_length", max_length=128, truncation=True)

    #Unflatten 
    unflatten =  {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    
    '''Decode and print
    for ele in unflatten["input_ids"]:
        for qa_context in ele:
            dec = tokenizer.decode(qa_context)
            print(dec)
    '''

    return unflatten

def label_cleaner(dirty_labels):
    d = {
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D",
        "E": "A",
        "1": "A",
        "2": "B",
        "3": "C",
        "4": "D",
        "5": "A"
    }

    return [d[k] for k in dirty_labels]

def find_q(ids:str):
    print(f"Searching for {ids}")
    dataset = load_dataset("ai2_arc", "ARC-Easy")
    bingo = None
    for i, ele in enumerate(dataset["test"]):
        if ele["id"] == ids:
            print(f"FOUND at: {i}")
            bingo = i
            break
    print(dataset["test"][bingo])

class ArcDataset(Dataset):
    def __init__(self, split):
        dataset = load_dataset("ai2_arc", "ARC-Easy")
        
        self.X = dataset[split].map(preprocess_function, batched=False)
        self.input_ids = self.X["input_ids"]
        self.attention_masks = self.X["attention_mask"]
        self.labels = label_cleaner(list(dataset[split]["answerKey"]))
        self.label_indexer = {"A":0, "B":1, "C":2, "D":3}
        self.label_inverter = {0:"A", 1:"B" , 2:"C", 3:"D"}

        for t in list(self.label_indexer.keys()):
            assert t == self.label_inverter[self.label_indexer[t]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.LongTensor(self.input_ids[idx]).squeeze(0), torch.BoolTensor(self.attention_masks[idx]).squeeze(0), self.label_indexer[self.labels[idx]])


def train():
    BATCH_SIZE = 32
    EPOCHS = 3
    train_dataset = ArcDataset("train")
    val_dataset = ArcDataset("validation")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AutoModelForMultipleChoice.from_pretrained("scibert_scivocab_uncased", num_labels=4).to(device)
    optimiser = AdamW(model.parameters(), lr=3e-5)
    criterion = CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for i, (input_ids, attention_masks, y) in enumerate(tqdm(train_loader)):
            
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            optimiser.zero_grad()
            y = torch.LongTensor(y)
            y = y.to(device)
            out = model(input_ids=input_ids, attention_mask=attention_masks, return_dict=True).logits
            loss = criterion(out, y)
            train_loss += loss.item()
            loss.backward()
            optimiser.step()
            

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            n = 0
            for i, (input_ids, attention_masks, y) in enumerate(tqdm(val_loader)):
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                y = torch.LongTensor(y)
                y = y.to(device)
                out = model(input_ids=input_ids, attention_mask=attention_masks, return_dict=True).logits
                y_hat = (torch.argmax(out, dim=1))
                correct += (y_hat == y).float().sum()
                loss = criterion(out, y)
                val_loss += loss.item()
                n += 1*BATCH_SIZE

            accuracy = correct / n

        print(f"Epoch {epoch}, avg. train loss: {train_loss/len(train_loader)} avg. val loss: {val_loss / len(val_loader)}. Val. accuracy: {accuracy}")
        
if __name__ == "__main__":
    train()
