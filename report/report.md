# Neural Search Engine - Data Report

## 0. მოკლე overview

პროექტი: neural search engine - user query → encoder → embedding → vector search → top-k chunks.

ჩვენ contrastive learning-ს InfoNCE loss-ით ვაკეთებთ (sentence-transformers library-ს არ ვიყენებთ).
encoder-ს **scratch-იდან** ვწვრთნით - pretrained მოდელი (BERT და ა.შ.) არ გამოგვიყენებია.
BM25 baseline-თან შევადარებთ.


---

## 1. Pipeline - რა რიგით ხდება ყველაფერი

data დამუშავება 3 ეტაპადაა. quality analysis **preprocess-ის შემდეგ** ხდება, training-ის **წინ**.

```
[1] preprocess          →  jsonl files (train/val/test)
[2] analyze (quality)   →  stats.json + console report
[3] training/eval       →  PairDataset reads jsonl
```

**რატომ analyze preprocess-ის შემდეგ:**
- quality metrics (dup rate, length, leakage) მხოლოდ **processed** pairs-ზე აქვს აზრი
- raw SQuAD/Wikipedia-ზე სხვა რიცხვები გამოვა - filtering/split-მდე leakage-ს ვერ ვნახავ
- თუ preprocess-ს ხელახლა გავუშვებ (მაგ. min_words შევცვალეთ), analyze-იც ხელახლა უნდა გაეშვას

**რომელი script როდის:**

| ეტაპი | Script | Output |
|-------|--------|--------|
| 1a | `python data/preprocess_squad.py` | `data/processed/squad/*.jsonl` |
| 1b | `python data/preprocess_wiki.py` | `data/processed/wiki/*.jsonl` |
| 1c | `python data/preprocess_jm.py` | `data/processed/corpus.jsonl` (demo) |
| 2 | `python data/analyze.py` | `data/processed/stats.json` |
| 3 | training (`train.py`) | model checkpoints |

analyze **ავტომატურად არ იძახება** preprocess-იდან - ცალკე step-ია, რომ preprocess-ის rebuild-ის დროს analyze-ის rerun optional იყოს. practice-ში ყოველთვის ორივე ერთად ვუშვებთ.

---

## 2. რა data ავირჩიეთ

### 2.1 რა dataset-ები ავირჩიეთ და რატომ

**SQuAD v1.1** (`rajpurkar/squad`)
- assignment-ის "question-answer" ტიპს ემთხვევა
- question = natural language query (~10 word avg)
- context paragraph = searchable chunk
- v1.1 აირჩიეთ v2-ის ნაცვლად, რადგან v2-ში unanswerable questions-ია - 45k+ row skip-დებოდა და retrieval positive pair-ისთვის ნაკლებად სასარგებლოა

**Wikipedia Simple English** (`wikimedia/wikipedia/20231101.simple`)
- assignment-ის "title-main text" მაგალითს პირდაპირ ემთხვევა
- ამაზეც ვწვრთნით - ცალკე მოდელი, შემდეგ SQuAD მოდელს ვადარებთ (იხ. 6.6)
- query = title, ანუ ვნახავთ რამდენად განსხვავდება title-query retrieval question-query-ისგან

**J&M book text** (demo corpus)
- assignment-ის demo ამ წიგნზე უნდა იყოს (Speech and Language Processing, SLP3 draft)
- training-ში **არ** მონაწილეობს - მხოლოდ search/demo-ს corpus-ია
- SLP3-ის თავისუფალი draft chapters (PDF) გადმოვწერეთ, `pdftotext`-ით text-ად ვაქციეთ
- დეტალები იხ. section 3.3

### 2.2 რა არ გავითვალისწინეთ / რატომ

- **SQuAD v2** - unanswerable questions, ჩვენს positive-pair setup-ს არ ემთხვევა
- **Full English Wikipedia** - ძალიან დიდი, simple english საკმარისი comparison-ისთვის
- **MS MARCO** - კარგი retrieval dataset-ია, SQuad 1.1 ვარჩიეთ
- **Triplet format** - InfoNCE-ს წყვილები სჭირდება, hard negative data-ში არ ვინახავთ

---

## 3. Raw data → our format

### 3.1 SQuAD

**Raw format** (HuggingFace, ~87,599 rows train split):

| column | meaning |
|--------|---------|
| `id` | unique example id |
| `title` | wikipedia article title |
| `context` | one paragraph from article |
| `question` | question about that paragraph |
| `answers` | `{text: [...], answer_start: [...]}` |

Raw task = reading comprehension (find exact answer span in paragraph).
Our task = retrieval (find the right paragraph given question).

**Transformation:**

```
question  →  query
context   →  document
id        →  doc_id
(+ source, split added at save time)
answers, title dropped (title used only for grouping/split)
```

**Example output row:**

```json
{
  "query": "what three factors do scientists believe are the cause of sexual orientation?",
  "document": "scientists do not know the exact cause of sexual orientation...",
  "doc_id": "570f8a7d5ab6b81900390f03",
  "source": "squad",
  "split": "train"
}
```

(text lowercased - იხ. section 4, scratch vocab-ის გამო)

**Filtering-ის შემდეგ რიცხვები (train split):**

| metric | value |
|--------|-------|
| pairs | 68,536 |
| unique queries | 68,322 |
| unique documents | 14,605 |

raw train ~87,599 row იყო, სამივე split-ში სულ ~85,571 დარჩა - ანუ ~2k გაიფილტრა (too short/long/empty/dup).

ერთ paragraph-ზე საშუალოდ ~4-5 question მოდის - ეს ჩანს dup_doc_rate-ში (ქვემოთ).

### 3.2 Wikipedia

**Raw format** (streamed, 15,000 articles):

| column | meaning |
|--------|---------|
| `id` | article id |
| `url` | wikipedia url |
| `title` | article title |
| `text` | full article, paragraphs separated by `\n` |

**Transformation:**

```
title              →  query (same for all paragraphs in article)
each paragraph     →  separate document row
doc_id             →  wiki-{article_idx}-{para_idx}
```

**Filtering-ის შემდეგ (train split):**

| metric | value |
|--------|-------|
| pairs | 38,070 |
| unique queries | 8,870 |
| unique documents | 38,020 |
| articles used | 15,000 (config limit) |

### 3.3 J&M demo corpus

ეს არ არის training data. assignment ითხოვს demo-ს Jurafsky & Martin-ის წიგნზე
(*Speech and Language Processing*), ამიტომ ცალკე searchable corpus გვჭირდება.

**საიდან:** SLP3 draft chapters თავისუფლად ხელმისაწვდომია PDF-ად (stanford.edu).
3 თავი გადმოვწერეთ (Words and Tokens, N-grams, Vector Semantics) და `pdftotext`-ით ტექსტად ვაქციეთ.

**raw -> chunks (`preprocess_jm.py`):**
- pdftotext-ის junk-ის გასუფთავება: page header-ები, "C HAPTER" banner-ები, copyright/draft ხაზები,
  page numbers, non-latin მაგალითები (chinese/greek)
- ტექსტი ~250 სიტყვიან chunk-ებად დაყოფა (assignment ითხოვს 200-300 word chunks)
- ძალიან პატარა tail chunk (<50 word) ამოვრიცხეთ

**output:** `data/processed/corpus.jsonl`, **155 chunk**, თითო ~250 word.
format: `{"document": "...", "doc_id": "jm-0"}` - query/source/split არ აქვს, რადგან ეს index-ია, არა pair.

eval/demo-ზე query-ს ამ corpus-ში ვეძებთ (top-k chunks). training corpus (squad/wiki) აქ არ ერევა.

---

## 4. Cleaning და filtering

მნიშვნელოვანი: **pretrained მოდელს არ ვიყენებთ** (BERT და ა.შ.). მოდელს scratch-იდან ვწვრთნით,
ანუ tokenizer/vocab თვითონ უნდა ავაგოთ. ამიტომ cleaning უფრო მეტს აკეთებს ვიდრე pretrained-ის შემთხვევაში.

### 4.1 რა გავაკეთეთ

- ზედმეტი space-ების მოშორება - ტექსტი სუფთა რჩება
- ცარიელი query ან document გამოვრიცხეთ
- **lowercasing** - რადგან საკუთარი vocab გვაქვს, "The" და "the" ერთი token უნდა იყოს (pretrained რომ გვქონდეს, tokenizer თვითონ გააკეთებდა)
- ძალიან მოკლე პარაგრაფები (<40 სიტყვა) ამოვრიცხეთ - პატარა chunk-ს retrieval-ში ცუდად ეძებს
- ძალიან გრძელი დოკუმენტები (>300 სიტყვა) მოვჭერით - memory-ს ვზოგავთ, encoder-ის max_length-ში ეტევა
- იგივე (query, document) წყვილი ერთხელ რჩება - duplicate rows training-ს არაფერს არ უმატებს

### 4.2 რა არ გავაკეთეთ

- stemming/lemmatization - მნიშვნელობა შეიძლება დაიკარგოს, vocab-ს თვითონ ვაშენებთ სიტყვებიდან
- stopword removal - "the", "is" და ა.შ. retrieval-ში context-ის ნაწილია
- feature selection - encoder representation-ს end-to-end სწავლობს

min/max word config-შია (`config.yaml`: 40/300). რამე შევცვლეთ -> preprocess + analyze ხელახლა.

### 4.3 vocabulary (scratch-ის გამო)

pretrained tokenizer არ გვაქვს, ამიტომ vocab training text-იდან აიგება (Alex-ის training კოდში):
- ყველა სიტყვა train split-იდან, frequency-ით დალაგებული
- top N სიტყვა vocab-ში (`config.yaml`: `vocab_size`)
- იშვიათი/უცნობი სიტყვა -> `<unk>` token
- `<pad>` padding-ისთვის

ეს მხოლოდ train-ზე უნდა აიგოს, val/test-ზე არა (თორემ leakage).

---

## 5. Train/val/test split

### 5.1 group-level split (80/10/10)

split **row-level random არა** - **article/title group**-ის მიხედვით.

SQuAD-ში ერთ `context` paragraph-ს 5-10 question შეიძლება ერთვებოდეს. თუ row-level split გავაკეთებდით:
- იგივე paragraph train-ში და test-ში ერთად მოხვდებოდა
- model "იმ paragraph-ს უკვე ნახა" test-ზე - inflated metrics (leakage)

Wikipedia-ში group = article (title). იგივე title-ის paragraphs ერთ split-ში რჩება.

**Verification:** `analyze.py` ამოწმებს train/test document overlap-ს.
ორივე dataset-ზე: **overlap = 0** ✓

### 5.2 split sizes

**SQuAD:**

| split | pairs | unique docs |
|-------|-------|-------------|
| train | 68,536 | 14,605 |
| val | 9,155 | 1,921 |
| test | 7,880 | 1,721 |

**Wikipedia:**

| split | pairs | unique docs |
|-------|-------|-------------|
| train | 38,070 | 38,020 |
| val | 4,542 | 4,541 |
| test | 5,136 | 5,136 |

seed=42 (`config.yaml`) - reproducible split.

---

## 6. Quality analysis - როდის, რა ვზომავთ, რას ვფიქრობთ

### 6.1 როდის ხდება

```bash
python data/analyze.py   # preprocess-ის შემდეგ
```

`analyze.py` კითხულობს processed jsonl-ებს და წერს `data/processed/stats.json`-ს.
console-ზეც print-ავს summary + 2 example pair-ს dataset-ზე.

**analyze არ ცვლის data-ს** - read-only check-ია. თუ stats ცუდი გამოვიდა, preprocess config-ს ვცვლით და თავიდან ვაგენერირებთ.

### 6.2 რა metrics-ს ვზომავთ

| metric | meaning | why we care |
|--------|---------|-------------|
| `count` | total pairs in split | dataset size |
| `unique_queries` | distinct query strings | low dup = diverse queries |
| `unique_documents` | distinct doc strings | how many unique chunks |
| `dup_query_rate` | 1 - unique_q/count | same query, different docs → InfoNCE false negatives in batch |
| `dup_doc_rate` | 1 - unique_d/count | same doc, different queries → false negatives + eval indexing issue |
| `query_words_avg/min/max` | query length stats | too short queries = weak signal |
| `doc_words_avg/min/max` | doc length stats | check filtering worked |
| `train_test_doc_overlap` | shared docs between train & test | leakage check, must be 0 |

### 6.3 SQuAD quality - რა კარგია, რა problematic-ია

**კარგი:**

- query-ები natural questions-ია (~10 word average)
- dup_query_rate ≈ 0 - თითქმის ყველა question unique-ია
- question→paragraph retrieval პირდაპირ ემთხვევა ჩვენს search task-ს
- train/test overlap = 0

**problematic:**

- dup_doc_rate ≈ 0.79 - ერთ paragraph ~5-6 question-ს ემსახურება
- InfoNCE batch-ში: თუ 2 different query same document-ს positive-ად უკავშირდება, ერთი მეორის "negative"-ად შეიძლება ჩაითვალოს → **false negative**
- eval-ის დროს: index-ში **unique documents** უნდა ჩავსვათ, არა 68k pair rows - თორემ იგივე paragraph 5-6 ჯერ index-ში იქნება

**example false negative scenario (SQuAD):**

```
batch row 1: Q1 → DocA  (positive)
batch row 2: Q2 → DocA  (positive, different question same paragraph)
InfoNCE: Q1 embedding should be close to DocA, far from DocB...
but Q2's DocA is also in batch as positive for Q2
→ Q1 might treat Q2's positive doc as negative incorrectly
```

ეს SQuAD-ის სტრუქტურაა და არა ბაგი. მაინც SQuAD უკეთესია training-ისთვის, რადგან query-ის ხარისხი მაღალია.

### 6.4 Wikipedia quality - comparison

**კარგი:**

- dup_doc_rate = 0 - თითო paragraph unique-ია
- train/test overlap = 0

**პრობლემატური:**

- dup_query_rate ≈ 0.77 - title ერთ article-ზე რამდენიმე paragraph-ს ემსახურება
- query_words_avg = 1.8 - title "france", "python" და ა.შ. - ძალიან მოკლე, weak semantic signal
- retrieval task title-query-ით ნაკლებად რეალისტურია ნამდვილ ძიებებთან შედარებით

**example (wiki):**

```
query: "france"
doc1: "france is a country in europe..."
doc2: "the capital of france is paris..."
doc3: "french is the official language..."
→ same query, 3 valid positives, InfoNCE confused again but from query side
```

### 6.5 side-by-side comparison

| | SQuAD (train) | Wikipedia (train) |
|--|---------------|-------------------|
| pairs | 68,536 | 38,070 |
| query avg words | 10.0 | 1.8 |
| dup_query_rate | 0.0 | 0.77 |
| dup_doc_rate | 0.79 | 0.0 |
| false negative source | same doc, diff queries | same query, diff docs |
| fits our search demo | yes (natural questions) | weak (titles only) |
| fits assignment example | question-answer type | title-text type |

### 6.6 final decision

**Training: ორივე dataset-ზე, ცალ-ცალკე მოდელი. მოდელი scratch-იდან (pretrained არა).**

იგივე architecture + hyperparams, მაგრამ ორი training run:
1. **SQuAD model** - question → paragraph
2. **Wiki model** - title → paragraph

შემდეგ ორი მოდელის შედარება test set-ზე (Recall@k, MRR).

**რატომ ორივე:**
- data report-ში უკვე ვნახეთ რომ dataset quality განსხვავდება (query length, dup rates)
- training-ითაც ვნახავთ რეალურად რომელი dataset-ი იძლევა უკეთეს retrieval-ს

**scratch-ის გავლენა:**
- მეტი data გვჭირდება (ამიტომ wiki_articles 3000 -> 15000)
- vocab თვითონ ვაშენებთ, cleaning-ში lowercasing დავამატეთ
- შედეგი pretrained-ზე უარესი იქნება. შესაძლოა BM25-საც ძლივს ჯობდეს ან წააგოს -
  ეს ნორმალურია და report-ში სწორედ ამას ვაანალიზებთ (scratch model vs BM25)


---

## 7. Code structure

```
data/
├── preprocess_utils.py   # shared: clean, filter, split, write jsonl
├── preprocess_squad.py   # SQuAD → data/processed/squad/
├── preprocess_wiki.py      # Wikipedia → data/processed/wiki/
├── preprocess_jm.py        # J&M book → data/processed/corpus.jsonl
├── preprocess.py           # runs all (or --squad / --wiki / --jm)
├── analyze.py              # quality check → stats.json  ← AFTER preprocess
└── dataset.py              # PyTorch PairDataset for training
```

SQuAD და Wikipedia ცალკე script-ებადაა, რადგან:
- ცალ-ცალკე preprocess და ცალ-ცალკე training run
- squad rebuild-ისას wiki-ს ხელახლა გადმოწერა არ გვინდა
- analyze ორივეს ცალ-ცალკე ამოწმებს

---

## 8. როგორ გავუშვათ და სად ვნახოთ output

```bash
pip install -r requirements.txt

# preprocess
python data/preprocess_squad.py
python data/preprocess_wiki.py
python data/preprocess_jm.py     # demo corpus (needs data/raw/jm_book.txt)

# quality check (preprocess-ის შემდეგ)
python data/analyze.py
```

**სად ჩანს output:**

| რა | სად |
|----|-----|
| SQuAD pairs | `data/processed/squad/train.jsonl`, `val.jsonl`, `test.jsonl` |
| Wikipedia pairs | `data/processed/wiki/train.jsonl`, `val.jsonl`, `test.jsonl` |
| demo corpus | `data/processed/corpus.jsonl` (155 J&M chunks) |
| quality stats | `data/processed/stats.json` |
| analyze print | terminal-ში (summary + 2 example pair) |

jsonl ფაილი ხელით ნახვა:
```bash
head -n 2 data/processed/squad/train.jsonl
head -n 2 data/processed/wiki/train.jsonl
```

stats.json:
```bash
cat data/processed/stats.json
```

**demo corpus გაკეთდა:** J&M SLP3-ის 3 თავი (PDF) -> `pdftotext` -> `data/raw/jm_book.txt` ->
`preprocess_jm.py` -> `corpus.jsonl` (155 chunk). იხ. section 3.3.

**scratch-ზე გადასვლის შემდეგ re-run:** cleaning-ში lowercasing დავამატეთ და `wiki_articles` 3000 -> 15000.
preprocess + analyze ხელახლა გავუშვი, data განახლებულია (wiki ახლა 38,070 train pair).

**git note:** `data/raw/` და `data/processed/` gitignore-შია (book copyrighted-ია), repo-ში მხოლოდ
`stats.json` მიდის. წიგნის ხელახლა ასაგებად იხ. run ბრძანებები ზემოთ.

---

---

---

**Alex - შენ ახლა ეს:**

data მზადაა. **ორივე dataset-ზე ცალ-ცალკე დაატრენინგე**, შემდეგ მოდელების შედარება.

**SQuAD paths:**
- train: `data/processed/squad/train.jsonl`
- val: `data/processed/squad/val.jsonl`
- test: `data/processed/squad/test.jsonl`
- checkpoint: `checkpoints/squad/`

**Wiki paths:**
- train: `data/processed/wiki/train.jsonl`
- val: `data/processed/wiki/val.jsonl`
- test: `data/processed/wiki/test.jsonl`
- checkpoint: `checkpoints/wiki/`

**როგორ გაუშვა (2 run):**
1. `config.yaml`-ში `source: squad` + squad paths → `python train.py`
2. `config.yaml`-ში `source: wiki` + wiki paths → `python train.py`

**data format** (jsonl, თითო ხაზი = ერთი positive pair):
```json
{"query": "...", "document": "...", "doc_id": "...", "source": "squad", "split": "train"}
```

**მნიშვნელოვანი: pretrained მოდელი (BERT) არ ვიყენებთ.** encoder scratch-იდან, random init.

**შენ რა გააკეთო:**
1. vocab build train split-იდან (top `vocab_size` words, `<unk>`, `<pad>`) - მხოლოდ train-ზე
2. `model/model.py` - `encode()`: Embedding layer (random) → BiLSTM ან small Transformer → mean pool → projection
3. `train.py` - training loop, InfoNCE loss, PairDataset (`data/dataset.py`) - **ორი run**
4. `search/search.py` - `build_index()` - **unique documents** only
5. TensorBoard logging (ცალკე log dir თითო მოდელისთვის)

config-ში hyperparams უკვე მიწერია (`arch`, `vocab_size`, `lr`, `epochs` და ა.შ.) - შეცვალე თუ საჭიროა.
scratch-ისთვის lr მაღალია (`1e-3`) და epochs მეტი (`15`), რადგან weights random-ია.

**მე გავაკეთებ მერე:**
- `evaluate.py` - ორივე მოდელის eval (Recall@k, MRR) + BM25 baseline
- მოდელების შედარება report-ში
- demo J&M book-ზე (როცა txt ჩავდებთ)

stats ნახვა: `cat data/processed/stats.json`
data ნახვა: `head -n 2 data/processed/squad/train.jsonl`
