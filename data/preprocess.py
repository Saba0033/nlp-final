import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path("data/processed")
JM_BOOK = Path("data/raw/jm_book.txt")
SEED = 42
N_ARTICLES = 3000
MIN_WORDS = 40
MAX_WORDS = 300


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def wiki_to_pairs(articles):
    by_article = []
    for i, article in enumerate(articles):
        title = article["title"]
        paragraphs = [p.strip() for p in article["text"].split("\n") if p.strip()]
        pairs = []
        for j, para in enumerate(paragraphs):
            words = para.split()
            if len(words) < MIN_WORDS:
                continue
            if len(words) > MAX_WORDS:
                para = " ".join(words[:MAX_WORDS])
            pairs.append({
                "query": title,
                "document": para,
                "doc_id": f"wiki-{i}-{j}",
            })
        if pairs:
            by_article.append(pairs)
    return by_article


def split_articles(by_article):
    random.seed(SEED)
    random.shuffle(by_article)
    n = len(by_article)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train = [p for group in by_article[:n_train] for p in group]
    val = [p for group in by_article[n_train : n_train + n_val] for p in group]
    test = [p for group in by_article[n_train + n_val :] for p in group]
    return train, val, test


def process_wiki():
    print("downlaoding wikipedia...")
    stream = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
    articles = []
    for i, article in enumerate(stream):
        articles.append(article)
        if i + 1 >= N_ARTICLES:
            break
    by_article = wiki_to_pairs(articles)
    train, val, test = split_articles(by_article)

    for name, rows in [("train", train), ("val", val), ("test", test)]:
        for row in rows:
            row["split"] = name
        write_jsonl(OUT / f"{name}.jsonl", rows)
        print(f"{name}: {len(rows)} paurs")


def process_jm():
    if not JM_BOOK.exists():
        print(f"no {JM_BOOK} - add book text for demo corpus")
        return

    words = JM_BOOK.read_text().split()
    chunks = []
    for i in range(0, len(words), 250):
        chunk = " ".join(words[i : i + 250]).strip()
        if chunk:
            chunks.append(chunk)

    corpus = [{"document": c, "doc_id": f"jm-{i}"} for i, c in enumerate(chunks)]
    write_jsonl(OUT / "corpus.jsonl", corpus)
    print(f"corpus: {len(corpus)} jm chunks")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    process_wiki()
    process_jm()


if __name__ == "__main__":
    main()
