from datasets import load_dataset
from transformers import AutoTokenizer


def tokenize(MAX_LENGTH=1024, OPTIMAL_NUM_TOKENS=64_000_000, SEED=4, dataset_path="cerebras/SlimPajama-627B"):
    raw_datasets = load_dataset(dataset_path, streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
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


