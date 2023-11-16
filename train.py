import argparse
import torch
import json
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaModel, LlamaConfig
from datasets import load_dataset

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
    parser.add_argument('--model_path', type=str, default='models/custom1', help='Path to save model.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs.')
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    configuration = LlamaConfig(**config)
    llama2_model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")
    llama2_model.config = configuration
    llama2_model.model = LlamaModel(configuration)
    llama2_model.vocab_size = configuration.vocab_size
    llama2_model.lm_head = nn.Linear(configuration.hidden_size, configuration.vocab_size, bias=False)
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llama2_model.to(device)

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
        print(f"Average training loss: {total_loss / len(train_dataloader)}")

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
        print(f"Validation loss: {total_eval_loss / len(valid_dataloader)}")



    # eval on test set
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
    print(f"Test loss: {total_test_loss / len(test_dataloader)}")


    torch.save(llama2_model.state_dict(), args.model_path)

if __name__ == "__main__":
    main()
