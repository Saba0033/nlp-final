from datasets import load_dataset

from preprocess_utils import (
    clean_text,
    cut_to_max,
    good_pair,
    load_cfg,
    save_splits,
    split_by_groups,
)


def build_squad(cfg):
    min_words = cfg["data"]["min_words"]
    max_words = cfg["data"]["max_words"]

    print("downlaoding squad...")
    ds = load_dataset("rajpurkar/squad", split="train")

    # question -> context paragaph. answer span not needed for us
    by_title = {}
    for ex in ds:
        query = clean_text(ex["question"])
        document = cut_to_max(clean_text(ex["context"]), max_words)
        if not good_pair(query, document, min_words, max_words):
            continue
        title = ex["title"]
        by_title.setdefault(title, []).append({
            "query": query,
            "document": document,
            "doc_id": ex["id"],
            "source": "squad",
        })
    return list(by_title.values())


def main():
    cfg = load_cfg()
    groups = build_squad(cfg)
    save_splits(cfg["data"]["squad_dir"], *split_by_groups(groups, cfg["data"]["seed"]))
    print("squad done")


if __name__ == "__main__":
    main()
