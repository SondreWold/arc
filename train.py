import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from datasets import load_dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
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
        for i in range(n):
            q.append(examples["question"])
        questions.append(q)
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
        "E": "E",
        "1": "A",
        "2": "B",
        "3": "C",
        "4": "D",
        "5": "E"
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
        self.label_indexer = {"A":0, "B":1, "C":2, "D":3, "E":4}
        self.label_inverter = {0:"A", 1:"B" , 2:"C", 3:"D", 4:"E"}

        for t in list(self.label_indexer.keys()):
            assert t == self.label_inverter[self.label_indexer[t]]

        for i, element in enumerate(self.X["choices"]):
            if len(element["label"]) != 4:
                del self.input_ids[i]
                del self.attention_masks[i]
                del self.labels[i]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.LongTensor(self.input_ids[idx]).squeeze(0), torch.BoolTensor(self.attention_masks[idx]).squeeze(0), self.label_indexer[self.labels[idx]]


def train():
    BATCH_SIZE = 16
    EPOCHS = 1
    train_dataset = ArcDataset("test")
    val_dataset = ArcDataset("validation")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AutoModelForMultipleChoice.from_pretrained("distilbert-base-uncased", num_labels=4).to(device)
    optimiser = AdamW(model.parameters(), lr=0.005)
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
            for i, (input_ids, attention_masks, y) in enumerate(tqdm(val_loader)):
                y = torch.LongTensor(y)
                out = model(input_ids=input_ids, attention_mask=attention_masks, return_dict=True).logits
                loss = criterion(out, y)
                val_loss += loss.item()

        print(f"Epoch {epoch}, avg. train loss: {train_loss/len(train_loader)} avg. val loss: {val_loss / len(val_loader)}")
        

if __name__ == "__main__":
    train()
