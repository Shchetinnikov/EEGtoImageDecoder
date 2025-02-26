import os
import numpy as np
from torch.utils.data import Dataset

class TUABDataset(Dataset):
    def __init__(self, path, normal='/normal/01_tcp_ar', abnormal='/abnormal/01_tcp_ar'):
        self.normal_path = path + normal
        self.abnormal_path = path + abnormal
        self.file_paths = []
        self.labels = []

        for file_name in os.listdir(self.normal_path):
            file_path = self.normal_path + "/" + file_name
            self.file_paths.append(file_path)
            self.labels.append(0)

        for file_name in os.listdir(self.abnormal_path):
            file_path = self.abnormal_path + "/" + file_name
            self.file_paths.append(file_path)
            self.labels.append(1)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self.file_paths) + idx
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        return (np.load(file_path, allow_pickle=True), label)