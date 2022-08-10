import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoModelForMultipleChoice, AutoTokenizer, set_seed, AutoModel
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import logging
import argparse
from tqdm import tqdm
import os
import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FusionModel(nn.Module):

    def __init__(self, model_name, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, return_dict=True, output_hidden_states=True, output_attentions=True)
        self.dropout = nn.Dropout()
        self.head = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        #get the CLS representation from transformer. torch.FloatTensor of shape (batch_size, hidden_size))
        batch_size, n_choices, seq_length = input_ids.size()[0], input_ids.size()[1], input_ids.size()[-1]
        input_ids = input_ids.view(-1, seq_length)
        attention_mask = attention_mask.view(-1, seq_length)
        pooled_output = self.encoder(input_ids, attention_mask)[1] #expects (batch_size, max_seq, dim) (16, 128, 768)?
        pooled_output = self.dropout(pooled_output)
        logits = self.head(pooled_output)
        reshaped_logits = logits.view(-1, n_choices)
        return reshaped_logits



def parse_args():
    parser = argparse.ArgumentParser(
        description="Train on the CommonsenseQA dataset")

    parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="The batch size to use during training.",
    )

    parser.add_argument(
    "--model",
    type=str,
    default="bert-base-uncased",
    help="The pretrained model to use",
    )

    parser.add_argument(
    "--epochs",
    type=int,
    default=3,
    help="The number of epochs.",
    )

    parser.add_argument(
    "--debug",
    type=bool,
    default=False,
    help="The number of epochs.",
    )

    parser.add_argument(
    "--lr",
    type=float,
    default=3e-5,
    help="The learning rate).",
    )

    parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="The rng seed",
    )

    args = parser.parse_args()

    return args


class CSQADataset(Dataset):

    def __init__(self, split, MODEL):
        options = ["A", "B", "C", "D", "E"]
        self.dataset = load_dataset("commonsense_qa", split=split)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.label_indexer = {v:k for k,v in enumerate(options)}
        self.label_inverter = {k:v for k,v in enumerate(options)}
        self.tokenized_dataset = self.dataset.map(self.preprocess_function, batched=True)
        self.labels = [self.label_indexer[x] for x in self.dataset["answerKey"]]
    
    def __len__(self):
        return len(self.dataset)


    def preprocess_function(self, examples):
        if isinstance(examples["question"], str):
            questions = []
            q = [examples["question"] for i in range(5)]
            questions.append(q)
            choices = []
            choices.append(examples["choices"]["text"])
        
        else:
            questions = [[context]*5 for context in list(examples["question"])]
            choices = []
            for element in examples["choices"]:
                choices.append(element["text"])
        
        q_flat = sum(questions, [])
        c_flat = sum(choices, [])

        tokenized_examples = self.tokenizer(q_flat, c_flat, truncation=True, padding="max_length", max_length=128)
        unflatten =  {k: [v[i : i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}

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

def main(args):
    EPOCHS = args.epochs
    LR = args.lr
    BATCH_SIZE = args.batch_size
    MODEL_NAME = args.model
    checker = AutoTokenizer.from_pretrained(MODEL_NAME)
    logging.info(f"Initializing model with id \"{MODEL_NAME}\" and hyperparameters epochs={EPOCHS}, batch size={BATCH_SIZE}, lr={LR}")

    config = {
        "learning_rate": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE
    }

    if args.debug == False:
        wandb.init(project="csqa_test", entity="sondrewo", config=config)

    train_dataset = CSQADataset("train", MODEL_NAME)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = CSQADataset("validation", MODEL_NAME)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    #model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, num_labels=5).to(device)
    model = FusionModel(MODEL_NAME, 5).to(device)
    criterion = CrossEntropyLoss()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.001,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR)

    for epoch in range(EPOCHS):
        logging.info(f"Staring training at epoch {epoch}")
        train_loss = 0.0
        model.train()
        for i, (idx, input_ids, attention_masks, y) in enumerate(tqdm(train_loader)):
            y = torch.LongTensor(y)
            optimizer.zero_grad()
            input_ids, attention_masks, y = input_ids.to(device), attention_masks.to(device), y.to(device)
            y_hat = model(input_ids, attention_masks)
            loss = criterion(y_hat, y)
            train_loss += loss.item()
            wandb.log({"batch_loss_train": loss})
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            n = 0
            for i, (idx, input_ids, attention_masks, y) in enumerate(tqdm(val_loader)):
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                y = torch.LongTensor(y)
                y = y.to(device)
                out = model(input_ids=input_ids, attention_mask=attention_masks)
                y_hat = nn.Softmax(out)
                y_hat = (torch.argmax(out, dim=1))
                correct += (y_hat == y).float().sum()
                loss = criterion(out, y)
                val_loss += loss.item()
                n += 1*BATCH_SIZE

            accuracy = correct / n

        t_l = train_loss / len(train_loader)
        v_l = val_loss / len(val_loader)
        print(f"Epoch {epoch}, avg. train loss: {t_l} avg. val loss: {v_l}. Val. accuracy: {accuracy}")
        wandb.log({"train_loss_epoch": t_l})
        wandb.log({"val_loss": v_l})
        wandb.log({"accuracy": accuracy})



if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    main(args)
    
    if args.seed is not None:
        set_seed(args.seed)