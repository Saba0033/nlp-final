import json
import statistics
from pathlib import Path

import yaml


def load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def n_words(text):
    return len(text.split())


def split_stats(rows):
    if not rows:
        return {"count": 0}

    q_len = [n_words(r["query"]) for r in rows]
    d_len = [n_words(r["document"]) for r in rows]
    uniq_q = len(set(r["query"] for r in rows))
    uniq_d = len(set(r["document"] for r in rows))

    # if same query/doc repeates a lot, infonce gets false negatives in batch
    dup_query_rate = round(1 - uniq_q / len(rows), 2)
    dup_doc_rate = round(1 - uniq_d / len(rows), 2)

    return {
        "count": len(rows),
        "unique_queries": uniq_q,
        "unique_documents": uniq_d,
        "dup_query_rate": dup_query_rate,
        "dup_doc_rate": dup_doc_rate,
        "query_words_avg": round(statistics.mean(q_len), 1),
        "query_words_min_max": [min(q_len), max(q_len)],
        "doc_words_avg": round(statistics.mean(d_len), 1),
        "doc_words_min_max": [min(d_len), max(d_len)],
    }


def leakge(train_rows, test_rows):
    train_docs = set(r["document"] for r in train_rows)
    test_docs = set(r["document"] for r in test_rows)
    return len(train_docs & test_docs)


def analyze_dataset(data_dir):
    data_dir = Path(data_dir)
    out = {"name": data_dir.name, "splits": {}}
    loaded = {}
    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}.jsonl"
        if path.exists():
            loaded[split] = read_jsonl(path)
            out["splits"][split] = split_stats(loaded[split])
    if "train" in loaded and "test" in loaded:
        out["train_test_doc_overlap"] = leakge(loaded["train"], loaded["test"])
    out["_train_rows"] = loaded.get("train", [])
    return out


def print_report(rep):
    print(f"\n{rep['name']}")
    for split, s in rep["splits"].items():
        print(f"  {split}: {s}")
    if "train_test_doc_overlap" in rep:
        print(f"  train/test overlap: {rep['train_test_doc_overlap']}")
    for r in rep["_train_rows"][:2]:
        print("  ex query:", r["query"][:80])
        print("  ex doc  :", r["document"][:100], "...")


def main():
    cfg = load_cfg()
    reports = {}
    for key, d in [("wikipedia", cfg["data"]["wiki_dir"]),
                   ("squad", cfg["data"]["squad_dir"])]:
        rep = analyze_dataset(d)
        print_report(rep)
        rep.pop("_train_rows")
        reports[key] = rep

    out = Path("data/processed/stats.json")
    with open(out, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
