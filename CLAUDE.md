# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Kaggle "Natural Language Processing with Disaster Tweets" project (binary classification: tweet is about a real disaster = 1, or not = 0). It is currently **notebook-only** — there is no package, no test suite, no build step, and no `requirements.txt`. All work happens in Jupyter notebooks under `notebook/`.

## Environment

Python 3.10.12 in the `tf-env` virtualenv at `~/tf-env`, registered as a Jupyter kernel named `tf-env`. Both notebooks are saved with that kernel; keep it selected so outputs stay reproducible.

```bash
source ~/tf-env/bin/activate
jupyter lab            # or: jupyter notebook
```

To execute a notebook headlessly (e.g. to verify a change runs end to end):

```bash
jupyter nbconvert --to notebook --execute --inplace notebook/eda.ipynb
```

Everything runs under WSL2 on `/mnt/d`. Notebook paths are hardcoded absolute WSL paths
(`/mnt/d/NLP-Starter/NLP_Starter-Disaster-Tweet/DATA/...`) — a leftover from the native-Windows → WSL
migration (commit `ed0ef65`). If you touch those cells, keep them absolute and WSL-style, or convert
the whole notebook to repo-relative paths in one pass rather than mixing conventions.

## Data flow

The pipeline is a chain of CSVs on disk, one notebook per stage:

1. `DATA/Raw/{train,test}.csv` + `DATA/sample_submission.csv` — untouched Kaggle downloads. Never edit.
2. `notebook/eda.ipynb` — class-balance and missing-value inspection, then fills `keyword` → `"missing"` and `location` → `"no_location"` (`location` is 33% null, `keyword` 0.8%). Writes `cleaned_train_data.csv` / `cleaned_test_data.csv`.
   **Gotcha:** the final cell writes to the notebook's CWD (`notebook/`), not to `DATA/Clean/`. The committed files in `DATA/Clean/` were moved there by hand. Fix the path rather than re-moving files if you rerun this.
3. `notebook/text_preprocessing.ipynb` — reads `DATA/Clean/cleaned_train_data.csv`, applies `clean_text()` (strips URLs, `@mentions`, the `#` symbol while keeping the hashtag word, all punctuation, all digits, then lowercases), drops duplicate `text` rows and rows that cleaned to empty, and does a stratified 80/20 `train_test_split`.
4. Modeling — **not written yet.**

Note the naming: `DATA/Clean/cleaned_*.csv` holds only the *null-filled* data with raw text. The regex text cleaning in step 3 is applied in memory and is not currently persisted.

## Decisions already made (don't relitigate)

Recorded in `brain.md`, which is the running scratchpad of pipeline state — read it before resuming work, and update it when a stage lands.

- **Stopwords are deliberately kept.** The target models are sequence models (LSTM/GRU/Transformer) that depend on word order, unlike TF-IDF/LDA.
- **Numbers are stripped** by `clean_text` — intentional for this task, not a bug.
- **717 duplicate texts (~9.4% of 7,613) are dropped**, `keep='first'`, leaving 6,896 rows. Whether any duplicate pair carried conflicting labels was never checked — still open.
- **Class balance is ~59/41**, judged not skewed enough for class weights or resampling. Track F1 alongside accuracy.
- **Token-length distribution:** mean 13.5, max 31 words → `maxlen` of 25 or 31; not yet decided.
- **Split before tokenizing.** Fit the tokenizer on the training split only, then transform both splits, to avoid vocabulary leakage.

## README caveat

`README.md` documents an aspirational layout (`src/` modules, `models/`, `submissions/`, `requirements.txt`, a KerasNLP/BERT fine-tuning flow) that **does not exist in the repo**. Treat it as a target design, not a description of the code. `brain.md` and the notebooks are the source of truth for current state; note that `brain.md` also describes the cleaning as living in a `clean_text_for_keras.py` script that was never created — the code is in `text_preprocessing.ipynb`.
