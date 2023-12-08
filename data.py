from datasets import load_dataset
from transformers import AutoTokenizer
import transformers
import pandas as pd
from torch.utils.data import Dataset


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

        ret = preprocess(self.data[i], self.tokenizerd)
        ret = {
            "input_ids": ret["input_ids"][0],
            "labels": ret["input_ids"][0],
            "attention_mask": ret["attention_mask"][0],
        }
        self.cached_data_dict[i] = ret
        return ret
        
