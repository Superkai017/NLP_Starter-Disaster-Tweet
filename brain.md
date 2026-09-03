# Project Brain: NLP Disaster Tweets Classification (Keras)

## Context
Training a Keras deep learning model to classify tweets as disaster-related (1) or not (0).
Dataset matches Kaggle's "Real or Not? NLP with Disaster Tweets" (7,613 original rows).

## Pipeline progress so far

### 1. Data cleaning (script: `clean_text_for_keras.py`)
- Loaded CSV with columns `text` and `target`
- Cleaning steps applied:
  - Lowercased all text
  - Removed HTML tags, URLs, email addresses
  - Removed `#` symbol but **kept the hashtag word** (e.g. `#earthquake` → `earthquake`)
  - Removed quotes and most punctuation (kept `.,!?'`)
  - Collapsed whitespace
  - **Stopwords NOT removed** — deliberate choice, since deep learning models
    (LSTM/GRU/transformer) rely on word order/context, unlike TF-IDF/LDA where
    stopword removal helps. `REMOVE_STOPWORDS = False` in the script.
  - Numbers (e.g. `13,000`) get stripped by the current regex — confirmed as
    acceptable/intentional for this task, not a bug.

### 2. Deduplication
- Original rows: 7,613
- Duplicates found: 717 (~9.4%) — matches known noisy-duplicate issue in this dataset
- Dropped via `df.drop_duplicates(subset=['text'], keep='first')`
- **Not yet checked:** whether any duplicates had conflicting labels before
  dropping (flagged as worth checking, not confirmed done):
  ```python
  dupes = df[df.duplicated(subset=['text'], keep=False)]
  conflicting = dupes.groupby('text')['target'].nunique()
  print((conflicting > 1).sum())
  ```
- Rows after dedup: **6,896**

### 3. Class balance (confirmed, no action needed)
```
target
0    0.589327
1    0.410673
```
~59/41 split. Not severe enough to need class weights/oversampling, but
recommended to track F1 alongside accuracy since it's not perfectly balanced.

### 4. Sequence length analysis (confirmed)
```
count    6896.000000
mean       13.459252
std         5.902256
min         1.000000
25%         9.000000
50%        13.000000
75%        18.000000
max        31.000000
```
- Recommended `maxlen = 25` (covers vast majority without excess padding)
- `maxlen = 31` (true max) also reasonable given tight distribution/small dataset

## Next steps (not yet done)
1. **Split before tokenizing** (avoid data leakage):
   ```python
   from sklearn.model_selection import train_test_split
   train_df, test_df = train_test_split(
       df, test_size=0.2, random_state=42, stratify=df['target']
   )
   ```
2. **Fit tokenizer on train only, then transform + pad both sets:**
   ```python
   from tensorflow.keras.preprocessing.text import Tokenizer
   from tensorflow.keras.preprocessing.sequence import pad_sequences

   tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
   tokenizer.fit_on_texts(train_df['text'])

   train_seq = tokenizer.texts_to_sequences(train_df['text'])
   test_seq  = tokenizer.texts_to_sequences(test_df['text'])

   train_padded = pad_sequences(train_seq, maxlen=25, padding='post', truncating='post')
   test_padded  = pad_sequences(test_seq, maxlen=25, padding='post', truncating='post')
   ```
3. **Not yet designed:** Keras model architecture (Embedding + LSTM/GRU/Conv1D + Dense output).
   User was offered help sketching this next.

## Open questions / things to verify when resuming
- Confirm whether any of the 717 duplicates had conflicting labels (see snippet above)
- Decide final `maxlen` (25 vs 31)
- Design and train the actual model architecture
- Decide on embedding strategy: train embeddings from scratch vs. pretrained (GloVe, etc.)
