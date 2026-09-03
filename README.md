# NLP_Starter-Disaster-Tweet

## Disaster Tweets NLP Classification using KerasNLP

A modular data science repository for classifying whether a given tweet is about a real disaster or not, built for the Kaggle **"Natural Language Processing with Disaster Tweets"** competition. This project uses fine-tuned Transformer backbones (DistilBERT/RoBERTa) via `KerasNLP` and `JAX`/`TensorFlow`.

---

## Architecture

This project fine-tunes a Transformer encoder (e.g. BERT/DistilBERT) for **sequence classification**: the tweet text is tokenized, passed through the pre-trained encoder, and the pooled `[CLS]` representation is fed into a classification head to predict disaster vs. non-disaster.

![BERT classification architecture](https://www.cse.chalmers.se/~richajo/nlp2019/l5/bert_class.png)
*Diagram: BERT-based sequence classification architecture (source: [Chalmers NLP course, Richard Johansson](https://www.cse.chalmers.se/~richajo/nlp2019/l5/)).*

---

## Project Structure

```
NLP_Starter-Disaster-Tweet/
├── DATA/
│   ├── Raw/                        # Original, unmodified Kaggle CSVs
│   │   ├── train.csv
│   │   └── test.csv
│   ├── Clean/                      # Null-filled + keyword-decoded (text still raw)
│   ├── Processed/                  # Cleaned text + stratified split, per cleaning mode
│   │   ├── train_lstm.csv
│   │   ├── val_lstm.csv
│   │   └── test_lstm.csv
│   └── sample_submission.csv
│
├── notebook/
│   ├── eda.ipynb                   # Distributions, missing values, keyword signal
│   └── text_preprocessing.ipynb    # Cleaning, dedup, label resolution, split
│
├── src/
│   └── text_cleaner.py             # Shared cleaning: mojibake/HTML repair, two modes
│
├── brain.md                        # Running log of pipeline state and decisions
├── CLAUDE.md                       # Guidance for Claude Code
└── README.md
```

### Not built yet

`dataset.py`, `model.py`, `metrics.py`, `models/`, `submissions/` and `requirements.txt` are
part of the target design below but do not exist yet. The example code further down this file
describes that intended end state, not the current repository.


---

## Directory & File Descriptions

* **`DATA/`**: `Raw/` holds the untouched Kaggle files. `Clean/` holds the null-filled frames
  (tweet text is still raw at that stage — the name is historical). `Processed/` holds the
  fully cleaned, deduplicated, split data actually used for modelling.
* **`notebook/`**:
  * `eda.ipynb`: class balance, missing-value rates, `keyword` predictive power, `location`
    cardinality, and a survey of the dataset's encoding corruption.
  * `text_preprocessing.ipynb`: applies the shared cleaner to **both** train and test, resolves
    duplicate label conflicts by majority vote, derives `MAXLEN`, and writes a stratified split.
* **`src/text_cleaner.py`**: the single source of truth for preprocessing. Repairs mojibake and
  HTML entities, strips URLs and @mentions, and exposes two presets — `mode="lstm"` (aggressive)
  and `mode="transformer"` (light). Stdlib only; no `ftfy` dependency.

---

## Quickstart Setup

### 1. Clone the Repository & Setup Environment

```bash
git clone https://github.com/your-username/disaster-tweets-nlp.git
cd disaster-tweets-nlp

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Add the Kaggle Dataset

Download the competition data from Kaggle and place the raw files here:

```
DATA/Raw/train.csv
DATA/Raw/test.csv
DATA/sample_submission.csv
```

You can either download them manually from the [competition page](https://www.kaggle.com/competitions/nlp-getting-started/data), or use the Kaggle CLI:

```bash
kaggle competitions download -c nlp-getting-started -p DATA/Raw
unzip DATA/Raw/nlp-getting-started.zip -d DATA/Raw
```

### 3. Run the Notebooks

Start Jupyter and work through the notebooks in order:

```bash
jupyter notebook
```

1. **`notebook/eda.ipynb`** — class balance, missing values, keyword signal, encoding artifacts.
2. **`notebook/text_preprocessing.ipynb`** — clean both frames, resolve label conflicts, split.
3. *(not built yet)* modeling — tokenize on the training split only, then fine-tune and generate `submission.csv`.

### 4. Use the `src/` Modules Directly (Optional)

The notebooks are thin wrappers around reusable code in `src/`, so you can also script training end-to-end:

```python
import pandas as pd
from src.text_cleaner import clean_text
from src.dataset import make_dataset
from src.model import build_model
from src.metrics import evaluate_model

# Load and clean data
df = pd.read_csv("DATA/Raw/train.csv")
df["text"] = df["text"].apply(clean_text)

# Build train/val tf.data.Dataset pipelines
train_ds, val_ds = make_dataset(df, batch_size=32, val_split=0.2)

# Initialize and train the model
model = build_model(preset="distil_bert_base_en_uncased")
model.fit(train_ds, validation_data=val_ds, epochs=3)

# Evaluate
evaluate_model(model, val_ds)
```

---

## Generating a Kaggle Submission

After training, run inference on `DATA/Processed/test_<MODE>.csv` and format the output to match `sample_submission.csv`:

```python
import pandas as pd

test_df = pd.read_csv("DATA/Processed/test_lstm.csv")  # already cleaned
test_df["text"] = test_df["text"].apply(clean_text)

preds = model.predict(test_df["text"])
submission = pd.DataFrame({
    "id": test_df["id"],
    "target": preds.argmax(axis=1)
})
submission.to_csv("submissions/submission.csv", index=False)
```

---

## Requirements

Key dependencies (see `requirements.txt` for the full pinned list):

* `tensorflow`
* `keras-nlp`
* `jax` (optional backend for KerasNLP/Keras 3)
* `pandas`
* `numpy`
* `scikit-learn`

---

## License

See `LICENSE`.
