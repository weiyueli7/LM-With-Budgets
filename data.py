from datasets import load_dataset
import transformers
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import torch

# def preprocess(RAW_DATA, TOKENIZER, MAX_LENGTH=1024):
#     dataset =  TOKENIZER(
#                            RAW_DATA,
#                            truncation=True,
#                            padding="max_length",
#                            max_length=MAX_LENGTH,
#                            return_tensors="pt",
#     )
#     return dataset



# def get_slimpajama_dataset(dataset_path):
#     print('Loading dataset...')
#     train_dataset = []
#     validation_dataset = []
#     test_dataset = []
#     for file in tqdm(os.listdir(dataset_path)):
#         cur_text = pd.read_parquet(os.path.join(dataset_path, file))['text'].to_list()
#         if file.startswith('train'):
#             train_dataset.extend(cur_text)
#         elif file.startswith('valid'):
#             validation_dataset.extend(cur_text)
#         elif file.startswith('test'):
#             test_dataset.extend(cur_text)
#     return train_dataset, validation_dataset, test_dataset

# def get_tokenizer(model_name):
#     print('Loading tokenizer...')
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     return tokenizer


# class SlimPajamaDataset(Dataset):
#     """Dataset for SlimPajama."""

#     def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
#         super(SlimPajamaDataset, self).__init__()

#         sources = [example for example in raw_data]
#         data_dict = preprocess(RAW_DATA=sources, TOKENIZER=tokenizer)

#         self.input_ids = data_dict["input_ids"]
#         self.labels = data_dict["input_ids"]
#         self.attention_mask = data_dict["attention_mask"]

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, i):
#         return dict(
#             input_ids=self.input_ids[i],
#             labels=self.input_ids[i],
#             attention_mask=self.attention_mask[i],
#         )
        
def tokenize_function(examples, tokenizer, max_length=1024):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

class SlimPajamaDataset(Dataset):
    """Dataset for SlimPajama with lazy loading."""

    def __init__(self, file_paths, tokenizer):
        self.file_paths = file_paths
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = pd.read_parquet(file_path)['text'].tolist()
        tokenized_data = tokenize_function({"text": data}, self.tokenizer)
        return {key: torch.tensor(val) for key, val in tokenized_data.items()}

def get_file_paths(dataset_path, prefix):
    return [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.startswith(prefix)]