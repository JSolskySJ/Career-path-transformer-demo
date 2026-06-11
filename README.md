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
# 1. Sample resumes for the picker (from the newest eval CSV)
conda run -n dwh-ai-py311 python scripts/prepare_samples.py --n 300

# 2. Job-title transitions for the Sankey view. The newest CSV is engineering-
#    heavy; a larger run gives many more titles and richer tails:
conda run -n dwh-ai-py311 python scripts/build_transitions.py \
  --csv ../datawarehouse-ai-analysis/career_path_transformer/model_eval_csv/career_path_transformer_predictions_20260521_053039.csv

# 3. Run the app
./run.sh          # http://127.0.0.1:5050
```

Note: every eval CSV is dominated by software/engineering careers (that's the
evaluation set), so the flow view's title list is tech-centric.

## Model artifacts

**item2vec** (the focus) — loads the real trained artifact:
`datawarehouse-ai-analysis/career_path_transformer/career_path_transformer_20260528_074025.bin`
by default; override with `CPT_ITEM2VEC_BIN`. The current sample resumes (from
the 0609 eval run) align with its vocabulary: 292/300 targets are rankable and
~0.4% of context tokens are out-of-vocabulary.

**BERT4Rec** — the app loads a checkpoint from `artifacts/bert4rec/`
(`model.pt` state dict + `vocab.json` + `config.json`). Two ways to get one:

- drop in a downloaded artifact from a training run (note: inference needs the
  vocab `idx2str`, not just the torch model — `mlflow.pytorch.log_model` alone
  doesn't persist it);
- or train a demo-quality one locally on the eval-CSV sequences (~16k):
  `conda run -n dwh-ai-py311 python scripts/train_bert4rec.py --epochs 40`

There is also `scripts/train_item2vec.py` to train a demo item2vec on the same
corpus, used only as a fallback when no production `.bin` is present.

## Layout

```
app.py                      Flask app + API (/api/predict, /api/space, …)
run.sh                      launcher (conda env + OpenMP workaround)
demo/
  config.py                 paths, env-var overrides
  tokens.py                 tokenisation mirroring build_tokens()
  item2vec_model.py         gensim loader + cosine ranking (last-8 mean)
  bert4rec_model.py         architecture copy + checkpoint loader + ranking
  embedding_space.py        PCA maps (local neighbourhood / global)
  transitions.py            transition store + layered Sankey builder
  samples.py                sample-resume loader
scripts/
  data_common.py            shared eval-CSV → sequence loading
  prepare_samples.py        eval CSV → artifacts/sample_resumes.json
  build_transitions.py      eval CSV → artifacts/transitions.json
  train_item2vec.py         eval CSV → artifacts/item2vec.bin
  train_bert4rec.py         eval CSV → artifacts/bert4rec/{model.pt,vocab.json,config.json}
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
