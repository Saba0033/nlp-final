import json
import random
import re
from pathlib import Path

import yaml


def load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def clean_text(text):
    # no pretrained tokenizer, so we lowercase ourselves + fix extra spaces
    return re.sub(r"\s+", " ", text).strip().lower()


def n_words(text):
    return len(text.split())


def good_pair(query, document, min_words, max_words):
    if not query or not document:
        return False
    if n_words(document) < min_words or n_words(document) > max_words:
        return False
    return True


def cut_to_max(document, max_words):
    words = document.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return document


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_by_groups(groups, seed):
    # split by article so same text doesnt end up in train and test
    random.seed(seed)
    random.shuffle(groups)
    n = len(groups)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = [p for g in groups[:n_train] for p in g]
    val = [p for g in groups[n_train:n_train + n_val] for p in g]
    test = [p for g in groups[n_train + n_val:] for p in g]
    return train, val, test


def save_splits(out_dir, train, val, test):
    out_dir = Path(out_dir)
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        for r in rows:
            r["split"] = name
        write_jsonl(out_dir / f"{name}.jsonl", rows)
        print(f"  {out_dir.name}/{name}: {len(rows)} pairs")
