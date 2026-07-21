"""Run registry — every staged MLflow run the demo can serve.

Layout (written by scripts/fetch_mlflow_artifacts.py):

    artifacts/runs/<run_id>/
      run.json               run name/tag, architecture, params, key metrics
      vocab.csv              per-run ranking domain (only when the run logged one)
      sample_resumes.json    resumes held out from THIS run's eval set
      bert4rec/{model.pt, vocab.json, config.json}   (bert4rec runs)
      item2vec.bin (+ .npy sidecars)                 (item2vec runs)

Several runs of the same architecture can be staged side by side — that's the
whole point (comparing bert4rec versions). When no registry runs exist, the
legacy single-slot artifacts (artifacts/item2vec.bin, artifacts/bert4rec/) are
registered as pseudo-runs so locally-trained checkpoints still work.

Model-side transformation params surfaced to the UI — what was done to the
sequences before training, i.e. why two runs' resumes/vocab differ.
"""

import json
import os

from demo import config

RUNS_DIR = os.path.join(config.ARTIFACTS_DIR, 'runs')

TRANSFORMATION_PARAMS = (
    'taxonomy_standardisation', 'normalise_education', 'collapse_consecutive_titles',
    'add_duration_buckets', 'add_skills', 'skill_cap', 'prospective_worker',
    'min_count', 'min_title_count', 'holdout_frac', 'context_last_n', 'dataset',
    'data_run_id', 'sample',
)
KEY_METRICS = ('test_recall_at_1', 'test_recall_at_5', 'test_recall_at_10',
               'test_mrr', 'vocab_size', 'train_sequences', 'test_pairs')


def _load_model(run_dir, architecture, vocab_csv):
    no_ckpt = ('no model checkpoint staged — the MLflow run has no logged model '
               '(still RUNNING, or it finished without logging one); re-fetch '
               'once the run completes')
    if architecture in ('bert4rec', 'modernbert'):
        # same wrapper for both — config.json's `backbone` picks the module
        from demo.bert4rec_model import Bert4RecModel
        model = Bert4RecModel.load_if_available(
            os.path.join(run_dir, architecture), vocab_csv=vocab_csv)
        if model is None:
            raise FileNotFoundError(no_ckpt)
        return model
    if architecture == 'denserec':
        from demo.denserec_model import DenseRecModel
        model = DenseRecModel.load_if_available(
            os.path.join(run_dir, 'denserec'), vocab_csv=vocab_csv)
        if model is None:
            raise FileNotFoundError(no_ckpt)
        return model
    if architecture == 'item2vec':
        from demo.item2vec_model import Item2VecModel
        return Item2VecModel(os.path.join(run_dir, 'item2vec.bin'), vocab_csv=vocab_csv)
    raise ValueError(f'architecture {architecture!r} is not displayable in the '
                     f'demo (supported: item2vec, bert4rec, modernbert, denserec)')


def _entry(run_id, run_dir, meta):
    """One registry entry: metadata always, model loaded eagerly (errors kept)."""
    vocab_csv = os.path.join(run_dir, 'vocab.csv')
    vocab_csv = vocab_csv if os.path.exists(vocab_csv) else None
    samples = os.path.join(run_dir, 'sample_resumes.json')
    entry = {
        'run_id': run_id,
        'run_name': meta.get('run_name') or run_id[:8],
        'run_tag': meta.get('run_tag'),
        'architecture': meta.get('architecture', 'unknown'),
        'label': f"{meta.get('architecture', '?')} · {meta.get('run_name') or run_id[:8]}",
        'params': meta.get('params', {}),
        'metrics': meta.get('metrics', {}),
        'start_time': meta.get('start_time'),
        'samples_path': samples if os.path.exists(samples) else None,
        'model': None,
        'error': None,
    }
    try:
        entry['model'] = _load_model(run_dir, entry['architecture'], vocab_csv)
    except Exception as exc:
        entry['error'] = str(exc)
    return entry


def discover_runs() -> dict:
    """{run_id: entry} for every staged run; legacy single-slot artifacts are
    registered as pseudo-runs only when the registry is empty."""
    runs = {}
    if os.path.isdir(RUNS_DIR):
        for run_id in sorted(os.listdir(RUNS_DIR)):
            run_dir = os.path.join(RUNS_DIR, run_id)
            meta_path = os.path.join(run_dir, 'run.json')
            if not os.path.isfile(meta_path):
                continue
            with open(meta_path) as f:
                runs[run_id] = _entry(run_id, run_dir, json.load(f))
    if runs:
        return runs

    # Legacy single-slot fallback (locally-trained / pre-registry stagings)
    for arch, exists in (
        ('item2vec', os.path.exists(os.path.join(config.ARTIFACTS_DIR, 'item2vec.bin'))),
        ('bert4rec', os.path.isdir(config.BERT4REC_DIR)),
    ):
        if not exists:
            continue
        entry = {
            'run_id': f'local-{arch}', 'run_name': 'local', 'run_tag': None,
            'architecture': arch, 'label': f'{arch} · local',
            'params': {}, 'metrics': {}, 'start_time': None,
            'samples_path': config.SAMPLES_JSON if os.path.exists(config.SAMPLES_JSON) else None,
            'model': None, 'error': None,
        }
        try:
            if arch == 'item2vec':
                from demo.item2vec_model import Item2VecModel
                entry['model'] = Item2VecModel()
            else:
                from demo.bert4rec_model import Bert4RecModel
                entry['model'] = Bert4RecModel.load_if_available()
                if entry['model'] is None:
                    continue
        except Exception as exc:
            entry['error'] = str(exc)
        runs[entry['run_id']] = entry
    return runs


def transformations(params: dict) -> dict:
    """The model-side transformation subset of a run's params, for display."""
    return {k: params[k] for k in TRANSFORMATION_PARAMS if k in params}


def key_metrics(metrics: dict) -> dict:
    return {k: metrics[k] for k in KEY_METRICS if k in metrics}
