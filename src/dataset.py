# src/dataset.py

import torch
from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_tensor = torch.tensor(self.src_data[idx], dtype=torch.long)
        tgt_tensor = torch.tensor(self.tgt_data[idx], dtype=torch.long)
        return src_tensor, tgt_tensor
