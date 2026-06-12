import re
from pathlib import Path

from preprocess_utils import clean_text, load_cfg, write_jsonl

JM_BOOK = Path("data/raw/jm_book.txt")
CHUNK_WORDS = 250

# pdftotext leaves junk: page headers, chapter banners, page numbers, etc.
# drop lines that match these before chunking.
HEADER = re.compile(r"speech and language processing.*jurafsky", re.I)
CHAPTER_BANNER = re.compile(r"^\s*c\s*hapter", re.I)
COPYRIGHT = re.compile(r"copyright|all\s+rights|draft of|^all$", re.I)


def looks_like_junk(line):
    s = line.strip()
    if not s:
        return True
    if HEADER.search(s) or CHAPTER_BANNER.search(s) or COPYRIGHT.search(s):
        return True
    # page numbers / single tokens on their own line
    if len(s.split()) <= 2 and not s.endswith("."):
        return True
    # mostly non-latin (chinese/greek examples) -> drop
    latin = sum(c.isascii() and c.isalpha() for c in s)
    if latin < len(s) * 0.5:
        return True
    return False


def read_clean_text():
    lines = JM_BOOK.read_text(errors="ignore").splitlines()
    kept = [ln for ln in lines if not looks_like_junk(ln)]
    return " ".join(kept)


def build_jm(out_path):
    if not JM_BOOK.exists():
        print(f"no {JM_BOOK} yet")
        return
    words = read_clean_text().split()
    rows = []
    for i in range(0, len(words), CHUNK_WORDS):
        chunk = clean_text(" ".join(words[i:i + CHUNK_WORDS]))
        # skip tiny tail chunks, they make bad search results
        if chunk and len(chunk.split()) >= 50:
            rows.append({"document": chunk, "doc_id": f"jm-{len(rows)}"})
    write_jsonl(Path(out_path), rows)
    print(f"corpus: {len(rows)} jm chunks (~{CHUNK_WORDS} words each)")


def main():
    cfg = load_cfg()
    build_jm(cfg["data"]["corpus_path"])
    print("jm done")


if __name__ == "__main__":
    main()
