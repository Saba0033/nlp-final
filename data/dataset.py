import json

from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, path):
        self.rows = []
        with open(path) as f:
            for line in f:
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return row["query"], row["document"]
