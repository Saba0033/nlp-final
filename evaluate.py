import argparse
import json
import yaml

from baseline.bm25 import BM25Retriever
from search.metrics import mrr, recall_at_k


def load_test(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def eval_bm25(test_rows, k_values):
    raise NotImplementedError


def eval_neural(test_rows, k_values):
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["baseline", "neural"], required=True)
    args = parser.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    test_rows = load_test(cfg["data"]["test_path"])
    k_values = cfg["evaluation"]["k_values"]

    if args.method == "baseline":
        eval_bm25(test_rows, k_values)
    else:
        eval_neural(test_rows, k_values)


if __name__ == "__main__":
    main()
