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
DATASETS_DIR = os.path.join(config.ARTIFACTS_DIR, 'datasets')


def ensure_eval_slice(data_run_id: str) -> str:
    """Local VAL+TEST slice of a dataset — all columns, only the held-out
    candidates' rows (a few % of the data), downloaded ONCE per dataset and
    reused by every run trained on it. Returns the local parquet path, or
    None when the dataset has no `split` column (can't slice deterministically).
    Assumes generate_config() has already set the AWS env credentials."""
    path = os.path.join(DATASETS_DIR, f'{data_run_id}.parquet')
    if os.path.exists(path):
        return path
    import pyarrow.dataset as pads
    import pyarrow.parquet as pq
    from modules.utils import get_dataset_input
    location = f's3://{get_dataset_input()}/career_path_transformer/{data_run_id}/data'
    ds = pads.dataset(location, format='parquet')
    if 'split' not in ds.schema.names:
        return None
    print(f'[slice] downloading VAL+TEST rows of {data_run_id} '
          f'(one-time, reused for all runs on this dataset)...', flush=True)
    table = ds.to_table(filter=pads.field('split').isin(['VAL', 'TEST']))
    os.makedirs(DATASETS_DIR, exist_ok=True)
    pq.write_table(table, path)
    print(f'[slice] {table.num_rows:,} rows -> {path} '
          f'({os.path.getsize(path) / 1e6:.0f} MB)', flush=True)
    _ensure_title_stats(data_run_id, ds)
    return path


def _title_stats_path(data_run_id):
    return os.path.join(DATASETS_DIR, f'{data_run_id}.titlestats.json')


def _ensure_title_stats(data_run_id, ds) -> str:
    """FULL-corpus title statistics stored beside the slice — the SJ/taxonomy
    eligibility sets and min-count frequencies are corpus-global (a title is
    SJ-eligible or min-count-eligible based on ALL rows, not just held-out
    ones), so the slice alone cannot reproduce them. One narrow-column scan."""
    path = _title_stats_path(data_run_id)
    if os.path.exists(path):
        return path
    print('[slice] computing full-corpus title stats (narrow scan)...', flush=True)
    cols = [c for c in ('experience_type', 'work_title_name',
                        'taxonomy_normalised_job_title', 'is_sj_title')
            if c in ds.schema.names]
    df = ds.to_table(columns=cols).to_pandas()
    df = df[df['experience_type'] == 'WORK']
    raw = df['work_title_name'].astype('string').str.strip().str.lower()
    raw = raw.replace('', None)
    if 'taxonomy_normalised_job_title' in df.columns:
        l3col = df['taxonomy_normalised_job_title'].astype('string').str.strip().str.lower()
        l3 = l3col.replace('', None).fillna(raw)
        l3_values = sorted(l3col.dropna().unique())
    else:
        l3, l3_values = raw, []
    sj = df['is_sj_title'].fillna(False).astype(bool) if 'is_sj_title' in df.columns \
        else raw.notna() & False
    stats = {
        'freq_freetext': raw.value_counts().to_dict(),
        'freq_l3': l3.value_counts().to_dict(),
        'sj_freetext': sorted(raw[sj].dropna().unique()),
        'sj_l3': sorted(l3[sj].dropna().unique()),
        'l3_values': l3_values,
    }
    with open(path, 'w') as f:
        json.dump(stats, f)
    print(f'[slice] title stats: {len(stats["freq_l3"]):,} L3 titles, '
          f'{len(stats["sj_l3"]):,} SJ -> {path}', flush=True)
    return path


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

    # Drop any leftover staging shim (a fake `models` package registered while
    # unpickling checkpoints) so the real datawarehouse-ai package imports.
    for mod in list(sys.modules):
        if (mod == 'models' or mod.startswith('models.')) and \
                getattr(sys.modules[mod], '__file__', None) is None:
            del sys.modules[mod]

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

    # Resume-parsing (personResume) dataset support: description feeding is a
    # DenseRec-level flag, not a shared param, so shared_model_kwargs doesn't
    # carry it — the base load paths read these attrs via getattr. Setting
    # them makes built resumes include the run's W_DESC tokens.
    if str(params.get('add_descriptions', 'False')).strip().lower() in ('true', '1', 'yes'):
        obj._description_col = params.get('description_col', 'work_description')
        obj._description_chars = int(params.get('description_chars', 300) or 300)

    # Prefer the cached local VAL+TEST slice over a full S3 read. Corpus-global
    # quantities the slice can't reproduce — min-count title frequencies and
    # the SJ/taxonomy eligibility sets — come from the stored full-corpus
    # title stats, making the slice path exact. Runs whose preprocessing ran
    # over the full corpus in ways the slice can't mirror (candidate sampling,
    # education normalisation's common-major allowlist) fall back to a full read.
    source = 's3_full'
    stats = None
    if obj._sample >= 1.0 and not obj._normalise_education:
        slice_path = ensure_eval_slice(data_run_id)
        if slice_path:
            if not os.path.exists(_title_stats_path(data_run_id)):
                import pyarrow.dataset as pads
                from modules.utils import get_dataset_input
                _ensure_title_stats(data_run_id, pads.dataset(
                    f's3://{get_dataset_input()}/career_path_transformer/{data_run_id}/data',
                    format='parquet'))
            with open(_title_stats_path(data_run_id)) as f:
                stats = json.load(f)
            import pandas as pd
            # Serve the slice through the loader's own get_data(cols=…) call —
            # reading a column SUBSET, exactly like training. Reading all
            # columns would also drag in junk timestamp columns (year-0 dates
            # in the personResume store) that pandas can't represent.
            orig_get_data = C.get_data
            C.get_data = lambda location, cols=None, **kw: pd.read_parquet(
                slice_path, columns=cols)
            obj._split_col_present = True
            obj._batched_read = False
            source = 'local_slice'

    # The real load path logs mlflow metrics — point them at a throwaway local
    # store so nothing is written to the tracking server.
    mlflow.set_tracking_uri(f'file://{tempfile.mkdtemp()}/mlruns')
    freq = None
    if stats:
        # min-count eligibility from FULL-corpus frequencies (module-level
        # function, patched for the duration of the sequence build)
        freq = stats['freq_l3' if obj._taxonomy_standardisation else 'freq_freetext']
        orig_eligible = C._min_count_eligible
        C._min_count_eligible = lambda rows, m: (
            None if m <= 1 else
            {C.W_TITLE_PREFIX + t for t, c in freq.items() if c >= m})
    try:
        with mlflow.start_run():
            sequences = obj._load_sequences()
    finally:
        if stats:
            C._min_count_eligible = orig_eligible
            C.get_data = orig_get_data

    if stats:
        # SJ / taxonomy target-eligibility sets from the FULL corpus — a title
        # is eligible if flagged anywhere, including TRAIN rows the slice lacks.
        tax = obj._taxonomy_standardisation
        if obj._rank_taxonomy_only:
            names = stats['l3_values']
        elif obj._rank_all_titles:
            names = list(freq.keys())
        else:
            names = stats['sj_l3' if tax else 'sj_freetext']
        obj._sj_title_set = {C.W_TITLE_PREFIX + t for t in names}

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
        'source': source,
    }
