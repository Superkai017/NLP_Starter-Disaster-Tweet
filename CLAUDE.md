# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Kaggle "Natural Language Processing with Disaster Tweets" project (binary classification: tweet is about a real disaster = 1, or not = 0). There is no test suite and no build step. Work happens in the two notebooks under `notebook/`, with shared preprocessing factored into `src/`. No modeling code exists yet.

## Environment

Python 3.10.12 in the `tf-env` virtualenv at `~/tf-env`, registered as a Jupyter kernel named `tf-env`. Both notebooks are saved with that kernel; keep it selected.

```bash
source ~/tf-env/bin/activate
jupyter lab

# run a notebook headlessly to verify it still executes end to end
jupyter nbconvert --to notebook --execute --inplace notebook/eda.ipynb
```

Installed and available: `tensorflow` 2.21, `keras_nlp` 0.25.1, `transformers` 5.16.1, `tokenizers` 0.23.1, `sklearn` 1.7.2, `seaborn`. **`ftfy` is not installed** — `src/text_cleaner.py` is deliberately stdlib-only, so don't reach for it.

There is no `requirements.txt`. Everything runs under WSL2 on `/mnt/d`.

## Architecture

Cleaning is **not** notebook-local. `src/text_cleaner.py` is the single source of truth, imported by both notebooks so train and test provably receive identical treatment — applying cleaning to only one split was a real bug here and produces a good validation score with a bad leaderboard score.

Notebooks resolve the repo root themselves and can be run from either the repo root or `notebook/`:

```python
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
```

### `clean_text(text, mode=...)` has two presets

The project has not committed to a model family yet, so the cleaner serves both:

- `mode="lstm"` — lowercase, strip punctuation and digits. For a from-scratch `Embedding` layer.
- `mode="transformer"` — keeps casing, punctuation and digits, which WordPiece/BPE use as signal. Removes only URLs, @mentions, mojibake and HTML entities.

`MODE` is set in cell 3 of `text_preprocessing.ipynb` and is baked into the output filenames (`train_lstm.csv`), so the two modes cannot silently mix on disk. Stopwords are kept in both modes — both target architectures depend on word order.

### `MOJIBAKE_MAP` — do not "simplify" this

The dataset's mojibake is not repairable by any codec round-trip (cp1252, latin-1, mac_roman and every other stdlib codec were tested and fail). The map is explicit, derived from corpus context, and **order-dependent**: `\x89Û` is a prefix of nearly every other entry and must be replaced last. Replacing it with a generic `.encode().decode()` will silently reintroduce 868 junk tokens.

## Data flow

Each stage reads the previous stage's directory. Nothing writes to its own input.

1. `DATA/Raw/{train,test}.csv` + `DATA/sample_submission.csv` — untouched Kaggle downloads. Never edit.
2. `notebook/eda.ipynb` → `DATA/Clean/` — null-filling (`keyword`→`"missing"`, `location`→`"no_location"`), keyword URL-decoding, whitespace normalisation. **Tweet text is still raw at this stage**; the directory name is historical.
3. `notebook/text_preprocessing.ipynb` → `DATA/Processed/` — applies `clean_text` to *both* frames, resolves duplicate label conflicts, computes `MAXLEN` from the data, and writes a stratified 80/20 split as `{train,val,test}_<MODE>.csv`.
4. Modeling — **not written yet.** Tokenization belongs here, fitted on `X_train` only.

The holdout is `X_val`, not `X_test`. `test` always means the unlabelled Kaggle set.

## Decisions already made (don't relitigate)

`brain.md` is the running scratchpad of pipeline state — read it before resuming, update it when a stage lands. Key settled points:

- **Duplicate label conflicts are resolved by majority vote, ties dropped.** 86 of 357 duplicate groups disagree; `keep='first'` would decide those by CSV row order. 40 tied groups are dropped as irreducible noise → 6,764 rows.
- **Stopwords kept**, in both modes.
- **`location` is not usable as-is** — 83.7% of non-null values are unique, and 1,154 of 1,586 test locations are unseen in train. It is filled for completeness, not for use.
- **`keyword` is the strongest cheap feature** — 222 values, zero unseen in test, several perfectly separating against a 0.41 base rate. Not yet wired into a model.
- **`MAXLEN` is computed as the 99th percentile** (currently 27), not hardcoded.
- **Split before tokenizing**, `random_state=42`, stratified.

## README caveat

`README.md`'s "Project Structure" describes modules that are still aspirational (`dataset.py`, `model.py`, `metrics.py`, `models/`, `submissions/`, `requirements.txt`). Only `src/text_cleaner.py` exists. Treat the rest as a target design; `brain.md` and the notebooks are the source of truth for current state.
