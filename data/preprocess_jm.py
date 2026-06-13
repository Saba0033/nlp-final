import re
from pathlib import Path

from preprocess_utils import clean_text, load_cfg, write_jsonl

JM_BOOK = Path("data/raw/jm_book.txt")
MIN_WORDS = 200
MAX_WORDS = 300

# pdf leaves garbage lines, fitler them out
HEADER = re.compile(r"speech and language processing.*jurafsky", re.I)
CHAPTER_BANNER = re.compile(r"^\s*c\s*hapter", re.I)
COPYRIGHT = re.compile(r"copyright|all\s+rights|draft of|^all$", re.I)
SECTION = re.compile(r"^\d+(\.\d+)+\.?\s*$")  # 2.1, 2.3.1 etc
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
YEAR = re.compile(r"^(19|20)\d\d[.,;:]?$")  # 2017, 1999. etc
NUMBER = re.compile(r"^\d+[.,;:]?$")


def looks_like_junk(line):
    s = line.strip()
    if not s:
        return True
    if SECTION.match(s):
        return False  # dont drop section numbers
    if HEADER.search(s) or CHAPTER_BANNER.search(s) or COPYRIGHT.search(s):
        return True
    # page nums and random short lines
    if len(s.split()) <= 2 and not s.endswith("."):
        return True
    # chinese/greek examples in book
    latin = sum(c.isascii() and c.isalpha() for c in s)
    if latin < len(s) * 0.5:
        return True
    return False


def read_sections():
    # split book into sections by 2.1, 2.2 ... headres
    lines = JM_BOOK.read_text(errors="ignore").splitlines()
    sections = []
    buf = []
    for line in lines:
        s = line.strip()
        if SECTION.match(s):
            if buf:
                sections.append(" ".join(buf))
            buf = []
            continue
        if looks_like_junk(line):
            continue
        buf.append(s)
    if buf:
        sections.append(" ".join(buf))
    return sections


def split_sentences(text):
    # split on . ? ! but keep sentence together
    out = []
    for bit in SENT_SPLIT.split(text.strip()):
        bit = bit.strip()
        if bit:
            out.append(bit)
    return out


def is_reference_junk(chunk):
    # whole book has a big index at the end + bibliography after every chapter.
    # those turn into chunks full of years ("2017.") and bare page numbers, which
    # are useless to search. drop them so the demo corpus is only real content.
    toks = chunk.split()
    years = sum(1 for w in toks if YEAR.match(w))
    nums = sum(1 for w in toks if NUMBER.match(w))
    return years >= 6 or nums / len(toks) > 0.12


def merge_sentences(sentences):
    # glue sentences until we hit 200-300 words
    chunks = []
    buf = []
    n = 0
    for sent in sentences:
        w = len(sent.split())
        if w > MAX_WORDS:
            # weird long line, just chop it (shouldnt hapen much)
            if buf:
                chunks.append(" ".join(buf))
                buf, n = [], 0
            words = sent.split()
            for i in range(0, len(words), MAX_WORDS):
                piece = " ".join(words[i:i + MAX_WORDS])
                if len(piece.split()) >= MIN_WORDS:
                    chunks.append(piece)
            continue
        if n + w > MAX_WORDS and n >= MIN_WORDS:
            chunks.append(" ".join(buf))
            buf, n = [], 0
        buf.append(sent)
        n += w
    if n >= MIN_WORDS:
        chunks.append(" ".join(buf))
    return chunks


def build_jm(out_path):
    if not JM_BOOK.exists():
        print(f"no {JM_BOOK} yet")
        return
    rows = []
    for section in read_sections():
        sents = split_sentences(section)
        for chunk in merge_sentences(sents):
            chunk = clean_text(chunk)
            wc = len(chunk.split())
            if MIN_WORDS <= wc <= MAX_WORDS and not is_reference_junk(chunk):
                rows.append({"document": chunk, "doc_id": f"jm-{len(rows)}"})
    write_jsonl(Path(out_path), rows)
    print(f"corpus: {len(rows)} chunks")


def main():
    cfg = load_cfg()
    build_jm(cfg["data"]["corpus_path"])
    print("jm done")


if __name__ == "__main__":
    main()
