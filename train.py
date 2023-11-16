import argparse
import torch
import json
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaModel, LlamaConfig
from datasets import load_dataset

from utils import *

def tokenize_function(example, tokenizer):
    return tokenizer(example["sentence1"], example["sentence2"], padding="max_length", truncation=True, return_tensors='pt')

def collate_batch(batch):
    return {
        'input_ids': torch.tensor([item['input_ids'] for item in batch]),
        'attention_mask': torch.tensor([item['attention_mask'] for item in batch]),
        'labels': torch.tensor([item['input_ids'] for item in batch])
    }

def main():
    parser = argparse.ArgumentParser(description='Train a Custom Llama model.')
    parser.add_argument('--config_path', type=str, default='configs/config.json', help='Path to config file.')
    parser.add_argument('--model_name', type=str, default='custom1.pth', help='Model name to save.')
    parser.add_argument('--model_dir', type=str, default='models', help='Path to model dir.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id.')
    args = parser.parse_args()
    
    printt("\n\n\n\nLoading model...\n----------------------------------\n\n\n\n", 'y')

    with open(args.config_path) as f:
        config = json.load(f)

    configuration = LlamaConfig(**config)
    llama2_model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")
    llama2_model.config = configuration
    llama2_model.model = LlamaModel(configuration)
    llama2_model.vocab_size = configuration.vocab_size
    llama2_model.lm_head = nn.Linear(configuration.hidden_size, configuration.vocab_size, bias=False)
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    
    
    printt("\n\n\n\nLoading dataset...\n----------------------------------\n\n\n\n", 'y')

    raw_datasets = load_dataset("glue", "mrpc")
    tokenized_datasets = raw_datasets.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    train_dataset = tokenized_datasets["train"]
    valid_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate_batch)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate_batch)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(llama2_model.parameters(), lr=0.001)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    llama2_model.to(device)
    
    
    
    printt("\n\n\n\nTraining/Validation in progress...\n----------------------------------\n\n\n\n", 'y')

    for epoch in range(args.epochs):
        llama2_model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['input_ids'].to(device)  # In causal LM, labels are usually the input_ids
            outputs = llama2_model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        printt(f"Epoch {epoch} --- Average training loss: {total_loss / len(train_dataloader)}", 'b')

        llama2_model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for batch in valid_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['input_ids'].to(device)
                outputs = llama2_model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_eval_loss += loss.item()
        printt(f"Epoch {epoch} --- Validation loss: {total_eval_loss / len(valid_dataloader)}", 'b')



    printt("\n\n\n\nTesting in progress...\n----------------------------------\n\n\n\n", 'y')
    llama2_model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['input_ids'].to(device)
            outputs = llama2_model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_test_loss += loss.item()
    printt(f"Test loss: {total_test_loss / len(test_dataloader)}", 'b')
    
    
    printt("\n\n\n\nSaving model...\n----------------------------------\n\n\n\n", 'r')

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    torch.save(llama2_model.state_dict(), f"{args.model_dir}/{args.model_name}")
    printt(f"Model saved at {args.model_dir}/{args.model_name}", 'g')

if __name__ == "__main__":
    main()
