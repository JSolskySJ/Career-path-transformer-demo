"""Paths and defaults for the demo. Everything is overridable via env vars so
the demo works on any machine that has the dwh repos checked out side by side."""

import os

DEMO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DWH_ROOT  = os.path.dirname(DEMO_ROOT)

ANALYSIS_DIR = os.environ.get(
    'CPT_ANALYSIS_DIR',
    os.path.join(DWH_ROOT, 'datawarehouse-ai-analysis', 'career_path_transformer'),
)

# Trained item2vec artifact (gensim Word2Vec .bin). Preference order:
#   1. CPT_ITEM2VEC_BIN env var
#   2. artifacts/item2vec.bin — staged from MLflow by import_mlflow_artifacts.py
#   3. the older production artifact in datawarehouse-ai-analysis
_ARTIFACT_ITEM2VEC = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts', 'item2vec.bin'))
_PROD_ITEM2VEC = os.path.join(ANALYSIS_DIR, 'career_path_transformer_20260528_074025.bin')
ITEM2VEC_BIN = os.environ.get('CPT_ITEM2VEC_BIN') or (
    _ARTIFACT_ITEM2VEC if os.path.exists(_ARTIFACT_ITEM2VEC) else _PROD_ITEM2VEC
)

# Eval predictions CSVs (context_tokens / correct_target) used for sample
# resumes and for training the demo BERT4Rec
EVAL_CSV_DIR = os.environ.get('CPT_EVAL_CSV_DIR', os.path.join(ANALYSIS_DIR, 'model_eval_csv'))

ARTIFACTS_DIR    = os.path.join(DEMO_ROOT, 'artifacts')
SAMPLES_JSON     = os.path.join(ARTIFACTS_DIR, 'sample_resumes.json')
BERT4REC_DIR     = os.path.join(ARTIFACTS_DIR, 'bert4rec')
TRANSITIONS_JSON = os.path.join(ARTIFACTS_DIR, 'transitions.json')
# Shared vocab + ranking domain (token, token_type, train_count, is_sj_title,
# in_ranking_domain) exported alongside the model run.
VOCAB_CSV        = os.path.join(ARTIFACTS_DIR, 'vocab.csv')

CONTEXT_LAST_N = 8   # item2vec ranking window (matches training default)
DEFAULT_TOP_K  = 10
