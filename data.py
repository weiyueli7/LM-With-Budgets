from datasets import load_dataset
from transformers import AutoTokenizer
import transformers
import pandas as pd
from torch.utils.data import Dataset


def preprocess(RAW_DATA, TOKENIZER, MAX_LENGTH=1024):
    dataset =  TOKENIZER(
                           RAW_DATA,
                           truncation=True,
                           padding="max_length",
                           max_length=MAX_LENGTH,
                           return_tensors="pt",
    )
    return dataset


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        sources = [example for example in raw_data]
        data_dict = preprocess(RAW_DATA=sources, TOKENIZER=tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["input_ids"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.input_ids[i],
            attention_mask=self.attention_mask[i],
        )
        
