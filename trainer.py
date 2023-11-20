import json
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
import wandb
from transformers import LlamaForCausalLM, LlamaConfig 

# def tokenize(tokenizer, MAX_LENGTH=1024, SEED=4, dataset_path="cerebras/SlimPajama-627B"):
#     raw_datasets = load_dataset(dataset_path, split=['train', 'validation', 'test'], streaming=True)
#     tokenized_datasets = {split: dataset.shuffle(seed=SEED).map(
#                             lambda x: tokenizer(
#                                 x["text"],
#                                 truncation=True,
#                                 padding="max_length",
#                                 max_length=MAX_LENGTH,
#                                 return_tensors="pt",
#                             ), batched=True) 
#                           for split, dataset in zip(['train', 'validation', 'test'], raw_datasets)}
#     return tokenized_datasets

def tokenize(MAX_LENGTH=1024, OPTIMAL_NUM_TOKENS=64_000_000, SEED=4, dataset_path="cerebras/SlimPajama-627B", model_name="lmsys/vicuna-7b-v1.5"):
    # Load only testing data
    raw_datasets = load_dataset(dataset_path, streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    shuffled_datasets = raw_datasets.shuffle(seed=SEED)

    tokenized_datasets = {}
    for key in raw_datasets.keys():
        tokenized_datasets[key] = shuffled_datasets[key].map(
            lambda x: tokenizer(
                x["text"],
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
        )

    return tokenized_datasets

def main():
    parser = argparse.ArgumentParser(description='Train a Custom Llama model.')
    parser.add_argument('--config_path', type=str, default='configs/config.json', help='Path to config file.')
    parser.add_argument('--model_name', type=str, default='custom1.pth', help='Model name to save.')
    parser.add_argument('--model_dir', type=str, default='models', help='Path to model dir.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id.')
    parser.add_argument('--tokenizer_model', type=str, default='lmsys/vicuna-7b-v1.5', help='Tokenizer model name.')
    args = parser.parse_args()

    print("\n\n\n\nLoading model...\n----------------------------------\n\n\n\n")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)

    with open(args.config_path, 'r') as file:
        config_dict = json.load(file)
    config = LlamaConfig.from_dict(config_dict)
    llama2_model = LlamaForCausalLM(config)
    
    print("\n\n\n\nLoading dataset...\n----------------------------------\n\n\n\n")

    tokenized_datasets = tokenize(tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    print("\n\n\n\nTraining...\n----------------------------------\n\n\n\n")

    trainer = Trainer(
        model=llama2_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    wandb.init(project="llama_training_project")

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Validation Results: {eval_results}")

    trainer.save_model(args.model_dir + '/' + args.model_name)

    wandb.finish()

if __name__ == "__main__":
    main()
