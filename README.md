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

One command fetches everything from MLflow and stages the demo (models,
vocab, sample resumes, transitions):

```bash
# Latest FINISHED run per architecture (add --skills-only for runs trained
# with the prospective-worker skill preamble):
conda run -n dwh-ai-py311 python scripts/fetch_mlflow_artifacts.py

# Or resolve runs by their run_tag MLflow tag:
conda run -n dwh-ai-py311 python scripts/fetch_mlflow_artifacts.py --tag my_run_tag

# Or pin explicit run ids (either or both):
conda run -n dwh-ai-py311 python scripts/fetch_mlflow_artifacts.py \
  --item2vec 3e215487579b42e395a25d75b57ad731 \
  --bert4rec 6dde1506e8434550a9bb21a92a3aef1d

# Then run the app
./run.sh          # http://127.0.0.1:5050
```

Each fetched run is staged into the **run registry** —
`artifacts/runs/<run_id>/` with its model checkpoint, `run.json` (params +
key metrics), its own ranking-domain `vocab.csv` (when logged) and its own
held-out sample resumes. Several runs of one architecture can be staged side
by side (repeat the flag: `--bert4rec RUN_A --bert4rec RUN_B`) and compared
in the UI: the predictions panel has a model multi-select with a per-run
info dropdown (params + metrics), and the sample picker has a "resumes from
run" selector showing that run's ranking domain and model-side
transformations. Re-fetching a run overwrites only that run's directory;
`--no-import` downloads into `incoming/` without staging.

**Supported architectures:** `item2vec`, `bert4rec`, `modernbert` (pre-norm /
RoPE / GeGLU backbone), and `denserec` (dual-path content embeddings; either
the bert4rec or modernbert backbone, from the run's `backbone` param). Any
other architecture is staged **metadata-only** — visible in the model
dropdown / compare table with a "not displayable" note, no model download.

**Downloads are cached.** `incoming/` is a persistent download cache with
per-run filenames, so re-running the fetch (or restarting the app, which
auto-syncs) never re-downloads an artifact already on disk — only genuinely
new runs are pulled. A model download that the flaky tracking server
interrupts resumes from where it stopped on the next attempt rather than
restarting. Delete a file from `incoming/` to force its re-download.

Note: every eval CSV is dominated by software/engineering careers (that's the
evaluation set), so the flow view's title list is tech-centric.

### How artifacts are located in MLflow

Everything is queryable from the run: run-root files (predictions CSV, vocab
CSV, bert4rec's `vocab.json`) come from
`mlflow.artifacts.download_artifacts(run_id=...)`, and the model binaries live
under the run's *logged model*, resolved via
`mlflow.search_logged_models(filter_string="source_run_id='<run_id>'")`.
Tag-based lookup uses `mlflow.search_runs(filter_string="tags.run_tag = '...'
and tags.architecture = '...'")` — note recent runs launched without an
explicit `run_tag` carry the literal tag `N/A`, so run-id (or `--skills-only`
/ latest) resolution is the reliable path for them.

Per run the fetch stages into `incoming/`:

- `career_path_transformer_<ts>.bin` (+ any `.npy` gensim sidecars) — item2vec
  model, from the item2vec logged model's `artifacts/`
- `career_path_transformer_vocab_<ts>.csv` — vocab + ranking-domain flags
  (only exists when the run was trained with `log_vocab=True`; without it the
  demo ranks over the model's full title vocabulary)
- `model.pth` — bert4rec pickled module, from the bert4rec logged model's `data/`
- `bert4rec_vocab.json` — the bert4rec run's logged `vocab.json` (`idx2str`);
  the **authoritative** index→token mapping, preferred over rebuilding it from
  the CSV's train-count order (which scrambles on count ties)
- `career_path_transformer_<ts>.csv` — predictions CSVs (samples + transitions
  are rebuilt from the largest one)

MLflow connection details come from
`datawarehouse-configurations/{partner}/ai/ai.conf` → `{env}.mlflow.*`
(`--env test-prod-sj` by default).

### Prospective-worker / skills runs

The demo understands the full prospective-worker token set: `S_SKILL` (a
candidate-level preamble prepended to the sequence), `W_DURATION`,
`W_COMPANY`, `W_SPEC` and `E_LEVEL`, alongside the original work/education
tokens. The resume builder has a **+ Skills** entry (comma-separated,
autocompleted, always emitted first, matching training) and the new work /
education fields.

Caveat: item2vec runs logged **before** the `_log_model` sidecar fix in
datawarehouse-ai are unrecoverable when their vocab was large enough that
gensim split the vectors into `.npy` sidecar files (the training code only
uploaded the `.bin`). The importer verifies the staged model loads and removes
it with a clear message if not — retrain to get a loadable skills item2vec.

## Model artifacts

Both models are the **real trained artifacts** from the same MLflow run, staged
by `scripts/import_mlflow_artifacts.py`:

**item2vec** — `artifacts/item2vec.bin` (gensim Word2Vec). Override the path
with `CPT_ITEM2VEC_BIN`. Verified to reproduce the run's predictions CSV exactly
(150/150 top-1 and top-3) once ranking is restricted to the SJ domain (below).

**BERT4Rec** — `artifacts/bert4rec/{model.pt, vocab.json, config.json}`,
converted from the run's `model.pth`. `mlflow.pytorch` logs the whole pickled
`nn.Module` under the original training module path, so the importer registers
the demo's identical `BERT4Rec` class to unpickle it and extracts a plain
`state_dict`. The `idx2str` mapping is taken from the run's logged
`bert4rec_vocab.json` (`idx2str`) — the exact mapping the model was trained
with. (Earlier runs were reconstructed as `['[PAD]','[MASK]'] + vocab-CSV
tokens`, which only matches when the CSV's train-count order has no tie-breaking
differences from the model's; these runs scramble heavily at ties, so using the
logged mapping is required — the importer falls back to the CSV rebuild only
when no logged vocab is staged.) BERT4Rec's softmax over titles is nearly flat
(it's genuinely uncertain), so its exact top-1 ordering is sensitive to
CPU/torch-version numerics; the loaded weights are the production weights.

### Ranking domain

Both demo models rank the next title over the **taxonomy L3 SuperTitles only**
(the `is_taxonomy_l3` titles in `artifacts/vocab.csv` — 1,440 of the 14,341
trained title tokens); all other titles are hidden from the predictions and the
embedding-space background. This is applied via `demo/ranking_domain.py`. Set
`CPT_RANKING_DOMAIN_COL=in_ranking_domain` to rank over the broader SJ domain
(2,080 titles) instead, or remove `vocab.csv` to rank the full vocabulary. The
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
  fetch_mlflow_artifacts.py MLflow (tag / run id / latest) → incoming/ → full staging
  import_mlflow_artifacts.py  incoming/ MLflow files → artifacts/ (bin, vocab, bert4rec)
  prepare_samples.py        eval CSV → artifacts/sample_resumes.json
  build_transitions.py      eval CSV → artifacts/transitions.json
  train_item2vec.py         eval CSV → artifacts/item2vec.bin (local fallback)
  train_bert4rec.py         eval CSV → artifacts/bert4rec/… (local fallback)
static/                     UI (vanilla JS + Plotly CDN)
artifacts/                  generated, gitignored
  runs/<run_id>/            run registry: model + run.json + vocab.csv + samples
  transitions.json          global (Sankey view)
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
