from datasets import load_dataset
from transformers import AutoTokenizer
import transformers

# def tokenize(MAX_LENGTH=1024, OPTIMAL_NUM_TOKENS=64_000_000, SEED=4, dataset_path="cerebras/SlimPajama-627B", model_name="lmsys/vicuna-7b-v1.5"):
#     raw_datasets = load_dataset(dataset_path, streaming=True)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     shuffled_datasets = raw_datasets.shuffle(seed=SEED)

#     tokenized_datasets = {}
#     for key in raw_datasets.keys():
#         tokenized_datasets[key] = shuffled_datasets[key].map(
#             lambda x: tokenizer(
#                 x["text"],
#                 truncation=True,
#                 padding="max_length",
#                 max_length=MAX_LENGTH,
#                 return_tensors="pt",
#             )
#         )

#     return tokenized_datasets
def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )
        
        
