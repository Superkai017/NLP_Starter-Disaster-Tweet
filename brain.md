# Project Brain: NLP Disaster Tweets Classification (Keras)

## Context
Training a Keras deep learning model to classify tweets as disaster-related (1) or not (0).
Dataset matches Kaggle's "Real or Not? NLP with Disaster Tweets" (7,613 train rows, 3,263 test).

## Pipeline progress so far

### 0. Cleaning lives in `src/text_cleaner.py`
Not a notebook-local function. Both `eda.ipynb` and `text_preprocessing.ipynb` import it, so
train and test are guaranteed identical treatment. Two presets:

- `mode="lstm"` — lowercase, strip punctuation and digits. For a from-scratch `Embedding` layer.
- `mode="transformer"` — keep casing, punctuation, digits (WordPiece/BPE uses them as signal).
  Removes only URLs, @mentions, mojibake and HTML entities.

Both keep stopwords: deep sequence models (LSTM/GRU/transformer) rely on word order, unlike
TF-IDF/LDA where stopword removal helps.

Set `MODE` in cell 3 of `text_preprocessing.ipynb`. Outputs are written per-mode
(`DATA/Processed/train_lstm.csv` etc.) so switching modes cannot silently mix datasets.

### 1. EDA (`notebook/eda.ipynb`)
- `keyword` → `"missing"` (0.80% null), `location` → `"no_location"` (33.27% train / 33.86% test).
- `keyword` is URL-decoded: 36 of 222 values arrived as `body%20bags`, `airplane%20accident`.
- `location` whitespace-collapsed — 2 values carried embedded newlines that produced quoted
  multi-line CSV records.
- Writes to `DATA/Clean/` (null-filled, keyword-decoded, **text still raw**).

### 2. Encoding repair (was silently corrupting the vocabulary)
The raw text is mojibake-corrupted: `Û` in 898 rows, `å` in 120, `Ì` in 40, plus 447 rows with
un-escaped `&amp;`. No single-byte codec round-trip repairs it (all of cp1252/latin-1/mac_roman
were tested and fail), so `MOJIBAKE_MAP` in `text_cleaner.py` is an explicit map derived from
corpus context and verified case by case:

```
'RÌ©union'        -> 'Réunion'      'å£279.00' -> '£279.00'
'91å¡F'           -> '91°F'         'ÛÏAirplaneÛ\x9d' -> '"Airplane"'
'CarolinaåÊAblaze' -> 'Carolina Ablaze'   (was fusing two words into one token)
```

Measured effect across train+test: **868 junk token occurrences → 0**, `amp` **300 → 2**
(the 2 remaining are the real word). Order in the map matters — `\x89Û` is a prefix of nearly
every other entry and must be replaced last.

### 3. Deduplication and label noise (open question from before: ANSWERED)
Dedup runs *after* cleaning, so near-duplicates differing only by URL or casing collapse too.

- Duplicate rows: 1,166 in 357 groups
- **86 groups disagree on the label (319 rows).** `keep='first'` would have resolved those by
  CSV row order, i.e. arbitrarily.
- Policy: **majority label wins; groups that tie are dropped** (40 groups) as irreducible noise.
- Rows after dedup + resolution: **6,764**. Zero rows clean to empty.

### 4. Class balance (confirmed, no action needed)
```
0    0.5931
1    0.4069
```
Not severe enough for class weights or oversampling, but track F1 alongside accuracy.

### 5. `keyword` is the strongest cheap feature — use it
222 values, and **zero test keywords are unseen in train**, so no unseen-category problem.
Against a 0.41 base rate:
```
derailment 1.00   debris 1.00   wreckage 1.00   typhoon 0.97
aftershock 0.00   body bags 0.02   ruin 0.03   blazing 0.03
```
Not yet fed to a model. Options: categorical embedding, or prepend to the text.

### 6. `location` — decided: not usable as-is
3,278 distinct values; 83.7% of the non-null ones appear exactly once; 1,154 of 1,586 test
locations never appear in train. The `fillna` makes it *look* usable. Drop it or geocode it —
do not one-hot it.

### 7. Sequence length (recomputed after the new cleaner)
```
count  6764   mean 13.64   std 5.91   min 1   50% 13   75% 18   max 32
p90 22   p95 24   p99 27
```
`MAXLEN = 27` (the 99th percentile) covers 99.5% of tweets uncut. Set from the data, not hardcoded.

### 8. Split (done, saved)
Stratified 80/20, `random_state=42`, **before** tokenization. The holdout is named `X_val`, not
`X_test` — the Kaggle test set is separate and unlabelled.
```
train 5,411   val 1,353   kaggle test 3,263
```
Saved to `DATA/Processed/{train,val,test}_<MODE>.csv`.

## Next steps (not yet done)
1. **Fit the tokenizer on `X_train` only**, then transform + pad all three splits to `MAXLEN`:
   ```python
   tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
   tokenizer.fit_on_texts(X_train['text'])          # train only — never the val/test text
   ```
   Current `lstm`-mode vocabulary across train+test is 17,381 types, so `num_words=10000`
   is a real truncation — worth tuning.
2. **Not yet designed:** the architecture (Embedding + LSTM/GRU/Conv1D + Dense).
3. Decide embedding strategy: train from scratch vs. pretrained (GloVe) vs. fine-tune a
   transformer (`transformers` 5.16.1 and `keras_nlp` 0.25.1 are both installed).

## Open questions
- Which `MODE` to commit to. `lstm` is the current default; nothing downstream is built yet,
  so switching is free right now and expensive later.
- Whether to bring `keyword` into the model, and how.
