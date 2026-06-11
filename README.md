# Career Path Transformer — Demo

Local web app for sense-checking the career-path models (**item2vec** and
**BERT4Rec**). Select a real held-out resume or build one by hand, run it
through both models, compare the predicted next job titles, and explore a 2-D
map of where the resume's job-title / education vectors sit in each model's
embedding space.

![architecture](https://img.shields.io/badge/python-3.11%20(dwh--ai--py311)-blue)

## What it shows

Two top-level views, switched from the header: **Predict** and **Job title flow**.

### Predict

- **Sample resumes** — real held-out pairs from the eval predictions CSVs in
  `datawarehouse-ai-analysis/career_path_transformer/model_eval_csv/`. Each has
  the candidate's career history (context) and the *actual* next job title
  (target), so you can see exactly where the true answer ranks.
- **Resume builder** — compose education / work entries with autocomplete from
  the model vocabulary, and see what each model predicts. Out-of-vocabulary
  values are flagged and ignored, just as in production inference.
- **Embedding space** — PCA projection of the model's vectors:
  - grey points: job titles (the query's cosine neighbourhood in *local* view,
    the most frequent titles in *global* view)
  - coloured numbered points joined by a dotted line: the resume's tokens in
    career order (blue = work, green = education)
  - ★ the model's query vector (item2vec: mean of the last 8 context vectors;
    BERT4Rec: the transformer hidden state at the `[MASK]` position)
  - orange diamonds: the top-K predicted next titles

### Job title flow (Sankey)

Select one or more starting job titles and see where careers flow next, as a
Sankey diagram of **empirical** title→title transitions (counted from the real
career sequences, not model output — independent of item2vec/BERT4Rec).

- For each selected title, its **top-K** transitions are drawn explicitly; the
  rest are summed into that title's own **Other** node.
- A destination is drawn explicitly whenever *any* selected title has it in its
  top-K — so shared destinations become a single converging node, which is how
  the **overlap** between selected careers shows up (e.g. *software engineer*
  and *data scientist* both flow into *machine learning engineer*).
- **Top-K** (3–20) and **Depth** (1–5) are adjustable. Depth > 1 expands each
  named destination into the next layer. Each (title, layer) is its own node,
  so looping careers (A→B→A) stay a clean left-to-right flow. Very wide/deep
  requests are capped for browser performance and flagged as "truncated".
- **Flow is conserved across layers**: each stem starts with its real outgoing
  volume, and every node forwards the flow it *received*, split by that title's
  empirical transition probabilities. So a deeper node's size is proportional
  to the share of the starting cohort that actually reaches it — the cohort
  fragments fast and the Other buckets swell, which is the whole point: it
  visualises why predicting the next title is hard.
- **Click any title node to drill in** — it's toggled as a starting title
  (added as a new stem, or removed if already selected) and the flow redraws.

Built from `artifacts/transitions.json` (see Setup).

## Prerequisites

- conda env **`dwh-ai-py311`** (gensim, torch, sklearn, flask, pandas — no
  extra installs needed; Plotly is loaded from a CDN in the browser)
- the `dwh` repos checked out side by side — the demo reads the eval CSVs from
  `../datawarehouse-ai-analysis/career_path_transformer/model_eval_csv/` by
  default (override with `CPT_EVAL_CSV_DIR` / `CPT_ANALYSIS_DIR`)

## Setup

```bash
# 1. Stage the model run's artifacts from MLflow into incoming/, then import
#    them (copies the item2vec .bin + vocab.csv, converts bert4rec model.pth):
conda run -n dwh-ai-py311 python scripts/import_mlflow_artifacts.py

# 2. Sample resumes for the picker (from that run's predictions CSV). By
#    default only resumes whose held-out next title is a taxonomy L3 title
#    (is_taxonomy_l3 in vocab.csv) are kept; pass --all-targets to disable.
conda run -n dwh-ai-py311 python scripts/prepare_samples.py \
  --csv incoming/career_path_transformer_20260611_035118.csv --n 300

# 3. Job-title transitions for the Sankey view. A larger run gives many more
#    titles and richer tails:
conda run -n dwh-ai-py311 python scripts/build_transitions.py \
  --csv ../datawarehouse-ai-analysis/career_path_transformer/model_eval_csv/career_path_transformer_predictions_20260521_053039.csv

# 4. Run the app
./run.sh          # http://127.0.0.1:5050
```

Note: every eval CSV is dominated by software/engineering careers (that's the
evaluation set), so the flow view's title list is tech-centric.

### Staging the MLflow artifacts

The models and vocab come from an MLflow run (experiment 51). `mlflow` isn't on
the CLI here, so download via the Python client (the dwh-ai-py311 env has the
`mlflow` package; connection details come from
`datawarehouse-configurations/sj/ai/ai.conf`, as the inference notebook uses)
into `incoming/`:

- `career_path_transformer_<ts>.bin` — item2vec gensim model
- `career_path_transformer_vocab_<ts>.csv` — shared vocab + `in_ranking_domain`
- `model.pth` (under `models/m-…/artifacts/data/`) — bert4rec pickled module
- `career_path_transformer_<ts>.csv` (item2vec & bert4rec predictions)

`scripts/import_mlflow_artifacts.py` then stages them into `artifacts/`.

## Model artifacts

Both models are the **real trained artifacts** from the same MLflow run, staged
by `scripts/import_mlflow_artifacts.py`:

**item2vec** — `artifacts/item2vec.bin` (gensim Word2Vec). Override the path
with `CPT_ITEM2VEC_BIN`. Verified to reproduce the run's predictions CSV exactly
(150/150 top-1 and top-3) once ranking is restricted to the SJ domain (below).

**BERT4Rec** — `artifacts/bert4rec/{model.pt, vocab.json, config.json}`,
converted from the run's `model.pth`. `mlflow.pytorch` logs the whole pickled
`nn.Module` under the original training module path and never persists the
vocab, so the importer registers the demo's identical `BERT4Rec` class to
unpickle it, extracts a plain `state_dict`, and rebuilds `idx2str` as
`['[PAD]','[MASK]'] + vocab-CSV tokens` (the CSV is in the model's
train-count-descending order — confirmed against the model's own predictions).
BERT4Rec's softmax over titles is nearly flat (it's genuinely uncertain), so its
exact top-1 ordering is sensitive to CPU/torch-version numerics; the loaded
weights are the production weights.

### Ranking domain

Both demo models rank the next title over the **taxonomy L3 SuperTitles only**
(the `is_taxonomy_l3` titles in `artifacts/vocab.csv` — 247 of the 4,070 trained
title tokens); all other titles are hidden from the predictions and the
embedding-space background. This is applied via `demo/ranking_domain.py`. Set
`CPT_RANKING_DOMAIN_COL=in_ranking_domain` to rank over the broader SJ domain
(464 titles) instead, or remove `vocab.csv` to rank the full vocabulary. The
full vocabulary is always available for the resume builder's autocomplete.

(The sample picker likewise only shows resumes whose held-out next title is a
taxonomy title — see step 2 above.)

`scripts/train_item2vec.py` / `train_bert4rec.py` remain for training
demo-quality models locally when no MLflow artifacts are staged.

## Layout

```
app.py                      Flask app + API (/api/predict, /api/space, …)
run.sh                      launcher (conda env + OpenMP workaround)
demo/
  config.py                 paths, env-var overrides
  tokens.py                 tokenisation mirroring build_tokens()
  ranking_domain.py         SJ ranking-domain titles from vocab.csv
  item2vec_model.py         gensim loader + cosine ranking (last-8 mean)
  bert4rec_model.py         architecture copy + checkpoint loader + ranking
  embedding_space.py        PCA maps (local neighbourhood / global)
  transitions.py            transition store + layered Sankey builder
  samples.py                sample-resume loader
scripts/
  data_common.py            shared eval-CSV → sequence loading
  import_mlflow_artifacts.py  incoming/ MLflow files → artifacts/ (bin, vocab, bert4rec)
  prepare_samples.py        eval CSV → artifacts/sample_resumes.json
  build_transitions.py      eval CSV → artifacts/transitions.json
  train_item2vec.py         eval CSV → artifacts/item2vec.bin (local fallback)
  train_bert4rec.py         eval CSV → artifacts/bert4rec/… (local fallback)
static/                     UI (vanilla JS + Plotly CDN)
artifacts/                  generated, gitignored
```

## Notes

- `KMP_DUPLICATE_LIB_OK=TRUE` is set by `run.sh`/the scripts: torch and
  gensim/sklearn each bundle their own OpenMP on macOS and abort without it.
- Ranking matches the training code exactly: item2vec averages the last 8
  known context vectors and ranks all `W_TITLE:` tokens by cosine; BERT4Rec
  left-pads the encoded context, appends `[MASK]`, and softmaxes the
  mask-position logits over title tokens only.
- The demo ranks over the model's full trained title vocabulary (no extra SJ
  allowlist gate). Resume tokens outside the vocabulary are flagged in the UI
  and ignored, exactly as production inference ignores them.
- Default data source for samples is the **newest** eval CSV; pass `--csv` to
  `prepare_samples.py` / `train_*.py` to use a different run.
