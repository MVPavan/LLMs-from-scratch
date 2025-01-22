from time import time_ns
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class LlamaDLPreTokenized(Dataset):
    def __init__(self, data_file, max_len:int=512, stride:int=1):
        self.load_data(data_file)
        self.max_len = max_len
        self.stride = stride
        if self.max_len <= 0:
            raise ValueError("max_len must be greater than 0.")
        if self.stride <= 0:
            raise ValueError("stride must be greater than 0.")
        if len(self.data) < self.max_len:
            raise ValueError("Data length must be greater than max_len.")
        print("Length of dataset: ", self.__len__()/1e6)
        
    
    def load_data(self, data_file):
        st = time_ns()
        self.data = torch.tensor(pd.read_parquet(data_file, engine="pyarrow").values)
        print("Time to load data: ", (time_ns()-st)/1e9)

    def __len__(self):
        return (len(self.data) - self.max_len - 1) // self.stride

    def __getitem__(self, idx):
        idx = idx*self.stride
        input_ids = self.data[idx:idx+self.max_len]
        target_ids = self.data[idx+1:idx+self.max_len+1]
        return input_ids, target_ids


train_dataset = LlamaDLPreTokenized(
    "/media/data_2/datasets/large_models_data/gutenberg_data/train_tokenized.parquet",
    max_len=512,
    stride=256
)
val_dataset = LlamaDLPreTokenized(
    "/media/data_2/datasets/large_models_data/gutenberg_data/val_tokenized.parquet",
    max_len=512,
    stride=256
)
# test_dataset = LlamaDLPreTokenized(
#     "/media/data_2/datasets/large_models_data/gutenberg_data/test_tokenized.parquet",
#     max_len=512,
#     stride=256
# )