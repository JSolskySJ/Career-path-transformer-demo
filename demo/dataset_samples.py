"""Build held-out sample resumes for a run straight from its dataset parquet,
reusing the training repo's OWN code (datawarehouse-ai sits alongside this
repo): the exact `_load_sequences` load/tokenisation path and the exact
`_split_train_holdout` split, driven by the run's logged MLflow params. With
the deterministic per-candidate `split` column (new datasets) the resulting
test pairs are identical to the ones the run evaluated on — so this only needs
re-running when a new dataset is generated.

The stub subclass trains nothing; the mlflow metric calls inside the real load
path are pointed at a throwaway local store so nothing touches the tracking
server.
"""

import json
import os
import sys
import tempfile

from demo import config

_DWH_AI = os.path.join(config.DWH_ROOT, 'datawarehouse-ai')


def build_samples_for_run(params: dict, data_run_id: str, out_path: str,
                          n: int = 300, env: str = 'test-prod-sj',
                          sample_seed: int = 42) -> dict:
    """params: the run's logged MLflow params (strings, as staged in run.json).
    Writes the demo sample_resumes.json to out_path; returns build stats."""
    if not os.path.isdir(_DWH_AI):
        raise FileNotFoundError(f'datawarehouse-ai repo not found at {_DWH_AI}')
    os.environ.setdefault('SPARK_MODE', 'local')
    # Don't trust an inherited SPARK_CONFIG_HOME blindly — shell profiles have
    # been seen exporting a mangled value; fall back to the sibling checkout.
    if not os.path.isdir(os.environ.get('SPARK_CONFIG_HOME', '')):
        os.environ['SPARK_CONFIG_HOME'] = os.path.join(
            config.DWH_ROOT, 'datawarehouse-configurations')
    if _DWH_AI not in sys.path:
        sys.path.insert(0, _DWH_AI)

    from modules.config_utils import generate_config
    generate_config(env)                      # config + AWS env creds for get_data
    import mlflow
    from models import career_path_transformer_common as C

    class _Sampler(C.CareerPathModelBase):
        """Instantiable stub — reuses the real load + split, trains nothing."""
        def _model_params(self):
            return {}

        def _train_model(self, train_seqs):
            raise NotImplementedError

        def _rank_titles(self, model, context):
            raise NotImplementedError

    # shared_model_kwargs is the exact coercion the training factories apply to
    # airflow string args — the MLflow params are those same strings.
    kwargs = C.shared_model_kwargs(env, data_run_id,
                                   int(params.get('seed', 123) or 123), None, params)
    obj = _Sampler(architecture='demo_sampler', **kwargs)

    # The real load path logs mlflow metrics — point them at a throwaway local
    # store so nothing is written to the tracking server.
    mlflow.set_tracking_uri(f'file://{tempfile.mkdtemp()}/mlruns')
    with mlflow.start_run():
        sequences = obj._load_sequences()
    _, _, test_pairs = obj._split_train_holdout(sequences)
    if not test_pairs:
        raise ValueError('the reconstructed split produced no test pairs — '
                         'check data_run_id and the run params')

    import numpy as np
    from scripts.prepare_samples import categorise, label_for
    rng = np.random.default_rng(sample_seed)
    idx = rng.choice(len(test_pairs), size=min(n, len(test_pairs)), replace=False)
    samples = []
    for i in sorted(idx):
        context, target = test_pairs[i]
        samples.append({
            'id': len(samples),
            'label': label_for(list(context)),
            'category': categorise(list(context)),
            'context_tokens': list(context),
            'target': target,
        })
    samples.sort(key=lambda s: (s['category'], s['label']))
    for i, s in enumerate(samples):
        s['id'] = i

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(samples, f, indent=1)
    return {
        'n_samples': len(samples),
        'n_test_pairs': len(test_pairs),
        'split_column': obj._seq_splits is not None,
        'data_run_id': data_run_id,
    }
