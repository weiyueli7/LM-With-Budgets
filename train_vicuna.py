import argparse
import os
import pandas as pd
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

# Constants
MAX_LENGTH = 1024

# Function for Preprocessing
def preprocess(RAW_DATA, TOKENIZER, MAX_LENGTH=1024):
    return TOKENIZER(RAW_DATA, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")

# Custom Dataset Class
class SupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer):
        super(SupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess(self.data[i], self.tokenizer)
        ret = {
            "input_ids": ret["input_ids"][0],
            "labels": ret["input_ids"][0],
            "attention_mask": ret["attention_mask"][0],
        }
        self.cached_data_dict[i] = ret
        return ret

# Argument Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='Training arguments for fine-tuning Llama model')
    parser.add_argument('--config', type=str, default="config.json", help='Config file')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--save_steps', type=int, default=1200, help='Save steps')
    parser.add_argument('--version', type=int, default=1, help='Version for checkpoint directory')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--logging_steps', type=int, default=1, help='Logging steps')
    parser.add_argument('--save_total_limit', type=int, default=10, help='Total number of saved checkpoints')
    parser.add_argument('--evaluation_strategy', type=str, default="no", choices=["no", "steps", "epoch"], help='Evaluation strategy')
    parser.add_argument('--save_strategy', type=str, default="steps", choices=["no", "steps", "epoch"], help='Save strategy')
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], help='Learning rate scheduler type')
    return parser.parse_args()

# Main Function
def main():
    args = parse_arguments()

    # Data loading
    dir_path = "../SlimPajama-6B/data"
    data_path = os.listdir(dir_path)
    train_raw_data, valid_raw_data, test_raw_data = [], [], []

    for path in tqdm(data_path):
        data = pd.read_parquet(os.path.join(dir_path, path))['text'].to_list()
        if path.startswith('test'):
            test_raw_data += data
        elif path.startswith('train'):
            train_raw_data += data
        elif path.startswith('valid'):
            valid_raw_data += data

    # Tokenizer and Datasets
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_fast=False)
    train_data_loader = SupervisedDataset(train_raw_data, tokenizer)
    valid_data_loader = SupervisedDataset(valid_raw_data, tokenizer)
    test_data_loader = SupervisedDataset(test_raw_data, tokenizer)

    # Model Configuration
    with open(f"configs/{args.config}", 'r') as file:
        config_dict = json.load(file)
    config = LlamaConfig.from_dict(config_dict)
    model = LlamaForCausalLM(config)

    # Training Arguments
    training_args = TrainingArguments(
        bf16=True, 
        output_dir=f'./checkpoints_{args.version}',
        num_train_epochs=args.epochs,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        fsdp_transformer_layer_cls_to_wrap='LlamaDecoderLayer',
        tf32=True,
        gradient_checkpointing=True,
    )

    # Trainer
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_data_loader, 
        eval_dataset=valid_data_loader
    )

    # Training and Evaluation
    trainer.train()
    trainer.evaluate(eval_dataset=test_data_loader)

if __name__ == '__main__':
    main()
