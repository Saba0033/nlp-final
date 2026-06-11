import argparse
import yaml
from torch.utils.data import DataLoader

from data.dataset import PairDataset
from model.model import TextEncoder, infonce_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_ds = PairDataset(cfg["data"]["train_path"])
    loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)

    model = TextEncoder(cfg["model"]["backbone"], cfg["model"]["embedding_dim"])
    optimizer = None

    for epoch in range(cfg["training"]["epochs"]):
        for queries, docs in loader:
            pass

    raise NotImplementedError


if __name__ == "__main__":
    main()
