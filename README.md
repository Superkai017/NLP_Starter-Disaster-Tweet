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
disaster-tweets-nlp/
├── data/
│   ├── raw/                   # Original, unmodified Kaggle CSV files
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── processed/             # Preprocessed text feeds or train/val splits
│
├── notebooks/                  # Sequential experiment notebooks
│   ├── 01_eda_and_cleaning.ipynb    # Data distribution, word frequency, text cleaning
│   └── 02_kerasnlp_modeling.ipynb   # Model training, fine-tuning, and inference
│
├── src/                        # Reusable Python source modules
│   ├── __init__.py
│   ├── text_cleaner.py         # Custom NLP preprocessing routines (Regex, URLs, Emojis)
│   ├── dataset.py              # tf.data.Dataset creation and pipeline batching
│   ├── model.py                # KerasNLP model architecture and presets initialization
│   └── metrics.py              # Custom evaluation functions (F1-Score, Confusion Matrix)
│
├── models/                     # Saved weights, artifacts, and exported Keras models
│   └── best_disaster_model.keras
│
├── submissions/                 # Formatted output CSV files for Kaggle submission
│   └── submission.csv
│
├── .gitignore                   # Excludes data/, models/, and cache from git tracking
├── requirements.txt             # Python dependency requirements
└── README.md                    # Project overview and instructions
```

---

## Directory & File Descriptions

* **`data/`**: Holds all dataset files. Raw files remain untouched in `raw/`, while transformed datasets are stored in `processed/`.
* **`notebooks/`**: Houses step-by-step interactive Jupyter notebooks:
  * `01_eda_and_cleaning.ipynb`: Exploratory Data Analysis (EDA), target class distribution checks, and missing value checks.
  * `02_kerasnlp_modeling.ipynb`: End-to-end model training using pre-trained transformer presets, validation tracking, and submission file generation.
* **`src/`**: Modular, importable Python code used across notebooks or stand-alone scripts:
  * `text_cleaner.py`: Standardizes text by stripping broken URLs, HTML entities, and unnecessary whitespace.
  * `dataset.py`: Converts Pandas DataFrames into optimized `tf.data.Dataset` streams with prefetching and batching.
  * `model.py`: Wraps `keras_nlp.models.BertClassifier` initialization and optimizer compilation.
  * `metrics.py`: Generates classification reports, macro/micro F1 scores, and confusion matrices.
* **`models/`**: Checkpoint directory where `.keras` weight files are saved during training.
* **`submissions/`**: Contains generated `submission.csv` files formatted with `id` and `target` columns for Kaggle evaluation.

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
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

You can either download them manually from the [competition page](https://www.kaggle.com/competitions/nlp-getting-started/data), or use the Kaggle CLI:

```bash
kaggle competitions download -c nlp-getting-started -p data/raw
unzip data/raw/nlp-getting-started.zip -d data/raw
```

### 3. Run the Notebooks

Start Jupyter and work through the notebooks in order:

```bash
jupyter notebook
```

1. **`01_eda_and_cleaning.ipynb`** — explore class balance, inspect missing values, and clean raw tweet text.
2. **`02_kerasnlp_modeling.ipynb`** — build the `tf.data` pipeline, fine-tune a KerasNLP transformer backbone, evaluate on a validation split, and generate `submissions/submission.csv`.

### 4. Use the `src/` Modules Directly (Optional)

The notebooks are thin wrappers around reusable code in `src/`, so you can also script training end-to-end:

```python
import pandas as pd
from src.text_cleaner import clean_text
from src.dataset import make_dataset
from src.model import build_model
from src.metrics import evaluate_model

# Load and clean data
df = pd.read_csv("data/raw/train.csv")
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

After training, run inference on `data/raw/test.csv` and format the output to match `sample_submission.csv`:

```python
import pandas as pd

test_df = pd.read_csv("data/raw/test.csv")
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

This project is provided as a starter template for the Kaggle "NLP with Disaster Tweets" competition. Add your preferred license (e.g., MIT) here.
