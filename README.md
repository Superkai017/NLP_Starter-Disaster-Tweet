# NLP_Starter-Disaster-Tweet
# Disaster Tweets NLP Classification using KerasNLP

A modular data science repository for classifying whether a given tweet is about a real disaster or not, built for the Kaggle **"Natural Language Processing with Disaster Tweets"** competition. This project uses fine-tuned Transformer backbones (DistilBERT/RoBERTa) via `KerasNLP` and `JAX`/`TensorFlow`.

---

## Project Structure

```text
disaster-tweets-nlp/
├── data/
│   ├── raw/                   # Original, unmodified Kaggle CSV files
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── processed/             # Preprocessed text feeds or train/val splits
│
├── notebooks/                 # Sequential experiment notebooks
│   ├── 01_eda_and_cleaning.ipynb    # Data distribution, word frequency, text cleaning
│   └── 02_kerasnlp_modeling.ipynb  # Model training, fine-tuning, and inference
│
├── src/                       # Reusable Python source modules
│   ├── __init__.py
│   ├── text_cleaner.py        # Custom NLP preprocessing routines (Regex, URLs, Emojis)
│   ├── dataset.py             # tf.data.Dataset creation and pipeline batching
│   ├── model.py               # KerasNLP model architecture and presets initialization
│   └── metrics.py             # Custom evaluation functions (F1-Score, Confusion Matrix)
│
├── models/                    # Saved weights, artifacts, and exported Keras models
│   └── best_disaster_model.keras
│
├── submissions/               # Formatted output CSV files for Kaggle submission
│   └── submission.csv
│
├── .gitignore                 # Excludes data/, models/, and cache from git tracking
├── requirements.txt           # Python dependency requirements
└── README.md                  # Project overview and instructions


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
* **`models/`**: Checkpoint directory where `.keras` weight files are saved during model training.
* **`submissions/`**: Contains generated `submission.csv` files formatted with `id` and `target` columns for Kaggle evaluation.

---

## Quickstart Setup

### 1. Clone the Repository & Setup Environment
```bash
git clone [https://github.com/your-username/disaster-tweets-nlp.git](https://github.com/your-username/disaster-tweets-nlp.git)
cd disaster-tweets-nlp

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
