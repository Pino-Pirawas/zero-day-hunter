import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class NetworkTrafficDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(data: np.ndarray,
                   batch_size: int = 256,
                   shuffle: bool = True) -> DataLoader:
    dataset = NetworkTrafficDataset(data)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)
