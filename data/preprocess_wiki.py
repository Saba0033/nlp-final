from datasets import load_dataset

from preprocess_utils import (
    clean_text,
    cut_to_max,
    good_pair,
    load_cfg,
    save_splits,
    split_by_groups,
)


def build_wiki(cfg):
    min_words = cfg["data"]["min_words"]
    max_words = cfg["data"]["max_words"]
    n_articles = cfg["data"]["wiki_articles"]

    print("downlaoding wikipedia")
    stream = load_dataset(
        "wikimedia/wikipedia", "20231101.simple", split="train", streaming=True
    )

    groups = []
    for i, article in enumerate(stream):
        if i >= n_articles:
            break
        title = clean_text(article["title"])
        paras = [clean_text(p) for p in article["text"].split("\n") if p.strip()]
        pairs = []
        for j, para in enumerate(paras):
            para = cut_to_max(para, max_words)
            if not good_pair(title, para, min_words, max_words):
                continue
            pairs.append({
                "query": title,
                "document": para,
                "doc_id": f"wiki-{i}-{j}",
                "source": "wiki",
            })
        if pairs:
            groups.append(pairs)
    return groups


def main():
    cfg = load_cfg()
    groups = build_wiki(cfg)
    save_splits(cfg["data"]["wiki_dir"], *split_by_groups(groups, cfg["data"]["seed"]))
    print("wiki done")


if __name__ == "__main__":
    main()
