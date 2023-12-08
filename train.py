import argparse
import os
import pandas as pd
import json
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

from data import *


# Argument Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='Training arguments for fine-tuning Llama model')
    parser.add_argument('--wandb_run_name', type=str, default='my_run', help='Wandb run name')
    parser.add_argument('--data_dir_path', type=str, default="../SlimPajama-6B/data", help='Directory path for the data')
    parser.add_argument('--data_pt', type=int, default=566406, help='Number of Data Points for training')
    parser.add_argument('--config', type=str, default="config.json", help='Config file')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=8e-4, help='Learning rate')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--save_steps', type=int, default=1200, help='Save steps')
    parser.add_argument('--version', type=str, default="medium", help='Version for checkpoint directory: small, medium, large')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--logging_steps', type=int, default=1, help='Logging steps')
    parser.add_argument('--save_total_limit', type=int, default=10, help='Total number of saved checkpoints')
    parser.add_argument('--evaluation_strategy', type=str, default="no", choices=["no", "steps", "epoch"], help='Evaluation strategy')
    parser.add_argument('--save_strategy', type=str, default="steps", choices=["no", "steps", "epoch"], help='Save strategy')
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], help='Learning rate scheduler type')
    parser.add_argument('--cuda_visible_devices', type=str, default="0,1,2,3,4,5,6,7", help='CUDA visible devices')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    return parser.parse_args()

# Main Function
def main():
    args = parse_arguments()

    # Data loading
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    MAX_LENGTH = args.max_length

    # Data loading
    dir_path = args.data_dir_path
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
    train_data_loader = SupervisedDataset(train_raw_data[:args.data_pt], tokenizer, MAX_LENGTH)
    valid_data_loader = SupervisedDataset(valid_raw_data, tokenizer, MAX_LENGTH)
    test_data_loader = SupervisedDataset(test_raw_data, tokenizer, MAX_LENGTH)

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
        report_to="wandb",
        run_name=args.wandb_run_name
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
