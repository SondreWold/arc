import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, set_seed
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import logging
import argparse
from tqdm import tqdm
import os
import wandb
from modeling import FusionModel
from CSQADataset import CSQADataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = True #Fox only (RTX3090 and A100)

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
    "--weight_decay",
    type=int,
    default=1e-4,
    help="The batch size to use during training.",
    )

    parser.add_argument(
    "--model",
    type=str,
    default="bert-base-uncased",
    help="The pretrained model to use",
    )

    parser.add_argument(
    "--dataset",
    type=str,
    default="commonsense_qa",
    help="The dataset to use",
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

def main(args):
    EPOCHS = args.epochs
    LR = args.lr
    BATCH_SIZE = args.batch_size
    MODEL_NAME = args.model
    DATASET_NAME = args.dataset
    assert DATASET_NAME in ["openbookqa", "commonsense_qa"]

    logging.info(f"Initializing model with id \"{MODEL_NAME}\" for dataset {DATASET_NAME} and hyperparameters epochs={EPOCHS}, batch size={BATCH_SIZE}, lr={LR}")

    config = {
        "learning_rate": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "weight_decay": args.weight_decay,
        "dataset": DATASET_NAME,
        "model_name": MODEL_NAME

    }

    if args.debug == False:
        wandb.init(project="multiple_choice", entity="sondrewo", config=config)

    train_dataset = CSQADataset(DATASET_NAME, "train", MODEL_NAME)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = CSQADataset(DATASET_NAME, "validation", MODEL_NAME)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = FusionModel(MODEL_NAME).to(device)
    criterion = CrossEntropyLoss()

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(grouped_parameters, lr=LR)

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
            loss.backward()
            optimizer.step()
            if args.debug == True:
                break

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
                if args.debug == True:
                    break

            accuracy = correct / n

        t_l = train_loss / len(train_loader)
        v_l = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch}, avg. train loss: {t_l} avg. val loss: {v_l}. Val. accuracy: {accuracy}")
        if args.debug == False:
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