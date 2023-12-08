from torch.utils.data import Dataset


# Function for Preprocessing
def preprocess(RAW_DATA, TOKENIZER, MAX_LENGTH=1024):
    """
    Preprocess the raw data
    :param RAW_DATA: Raw data
    :param TOKENIZER: Tokenizer
    :param MAX_LENGTH: Maximum length of the input
    :return: Preprocessed data
    """
    return TOKENIZER(
        RAW_DATA, 
        truncation=True, 
        padding="max_length", 
        max_length=MAX_LENGTH, 
        return_tensors="pt"
        )

# Custom Dataset Class
class SupervisedDataset(Dataset):
    """
    Custom Dataset class for supervised learning
    :param raw_data: Raw data
    :param tokenizer: Tokenizer
    :return: None
    """
    def __init__(self, raw_data, tokenizer, max_length=1024):
        super(SupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data = raw_data
        self.max_length = max_length
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        """
        Get the item at index i using lazy loading technique
        """
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess(self.data[i], self.tokenizer, self.max_length)
        ret = {
            "input_ids": ret["input_ids"][0],
            "labels": ret["input_ids"][0],
            "attention_mask": ret["attention_mask"][0],
        }
        self.cached_data_dict[i] = ret
        return ret
