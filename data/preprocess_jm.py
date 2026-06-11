from pathlib import Path

from preprocess_utils import clean_text, load_cfg, write_jsonl

JM_BOOK = Path("data/raw/jm_book.txt")


def build_jm(out_path):
    if not JM_BOOK.exists():
        print(f"no {JM_BOOK} yet")
        return
    words = JM_BOOK.read_text().split()
    rows = []
    for i in range(0, len(words), 250):
        chunk = clean_text(" ".join(words[i:i + 250]))
        if chunk:
            rows.append({"document": chunk, "doc_id": f"jm-{len(rows)}"})
    write_jsonl(Path(out_path), rows)
    print(f"corpus: {len(rows)} jm chunks")


def main():
    cfg = load_cfg()
    build_jm(cfg["data"]["corpus_path"])
    print("jm done")


if __name__ == "__main__":
    main()
