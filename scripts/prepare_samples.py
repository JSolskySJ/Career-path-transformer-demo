"""Build artifacts/sample_resumes.json from a model-eval predictions CSV, or
directly from the source dataset parquet (which carries per-job tenure).

Two sources:

  --csv PATH            the eval predictions CSV (default). Real held-out pairs
                        — context_tokens (the resume the model saw) and
                        correct_target (the actual next title) — but NO tenure.

  --dataset-run-id ID   the source dataset parquet on S3 (e.g. the run the
                        models were trained on). Rebuilds the held-out resumes
                        with the repo's own build_tokens (so the L3
                        tokenisation matches the model) AND attaches each work
                        experience's tenure from experience_duration_days. Use
                        this to get tenure into the resume view.

Usage:
    python scripts/prepare_samples.py [--csv PATH] [--n 300] [--seed 42]
    python scripts/prepare_samples.py --dataset-run-id 97c13d80-... [--n 300]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from demo import config
from demo.tokens import parse_token_string, token_type, token_value, W_TITLE_PREFIX
from scripts.data_common import newest_eval_csv


def tenure_label(days) -> str:
    """Human tenure from a day count, or None when missing/non-positive."""
    if days is None:
        return None
    try:
        d = int(days)
    except (TypeError, ValueError):
        return None
    if d != d or d <= 0:  # NaN or non-positive
        return None
    yrs, rem = divmod(d, 365)
    mos = rem // 30
    if yrs and mos:
        return f'{yrs} yr {mos} mo'
    if yrs:
        return f'{yrs} yr'
    if mos:
        return f'{mos} mo'
    return '<1 mo'


def categorise(tokens: list) -> str:
    has_edu  = any(t.startswith('E_') for t in tokens)
    has_work = any(t.startswith('W_') for t in tokens)
    if has_work and has_edu:
        return 'mixed'
    if has_work:
        return 'work_only'
    return 'education_only' if has_edu else 'other'


def label_for(tokens: list) -> str:
    titles = [token_value(t) for t in tokens if t.startswith(W_TITLE_PREFIX)]
    majors = [token_value(t) for t in tokens if token_type(t) == 'E_MAJOR']
    n_work = len(titles)
    n_edu  = sum(1 for t in tokens if t.startswith('E_TYPE:')) or (1 if majors else 0)
    if titles:
        head = titles[-1]
    elif majors:
        head = f'edu: {majors[0]}'
    else:
        head = 'education only'
    return f'{head}  ({n_work} work, {n_edu} edu)'


def taxonomy_titles(vocab_csv=None):
    """W_TITLE tokens flagged is_taxonomy_l3 in the vocab CSV (the L3
    SuperTitles), or None if no vocab CSV is staged."""
    path = vocab_csv or config.VOCAB_CSV
    if not os.path.exists(path):
        return None
    v = pd.read_csv(path)
    if 'is_taxonomy_l3' not in v.columns:
        return None
    return set(v.loc[v['is_taxonomy_l3'] == True, 'token'])


def _experiences_with_tenure(group, build_tokens):
    """One candidate's experience rows (sorted) -> ordered list of
    (experience_type, duration_days, token_bundle), using the repo's build_tokens
    with taxonomy standardisation so the W_TITLE matches the trained model."""
    out = []
    for row in group.to_dict('records'):
        bundle = build_tokens(row, taxonomy_standardisation=True)
        if not bundle:
            continue
        dur = row.get('experience_duration_days')
        out.append((row['experience_type'], dur, bundle))
    return out


def build_from_dataset(run_id, n, seed, taxonomy_only, conf_path, env):
    """Build tenure-carrying samples straight from the source dataset parquet.

    Holds out each candidate's last work title (the next-job target), keeps the
    preceding experiences as context, and records the tenure (days) of every
    work experience in that context — aligned, in order, to the work experiences
    the demo will render from the context tokens.
    """
    import pyarrow.dataset as pads
    import pyarrow.fs as pafs
    from pyhocon import ConfigFactory

    sys.path.insert(0, os.path.abspath(os.path.join(config.DEMO_ROOT, '..', 'datawarehouse-ai')))
    from models.career_path_transformer_common import build_tokens, _duration_bucket

    conf = ConfigFactory.parse_file(conf_path)
    io = f'{env}.io'
    s3_uri = conf.get_string(f'{io}.datasetInput')
    s3 = pafs.S3FileSystem(access_key=conf.get_string(f'{io}.amazonAccessKey'),
                           secret_key=conf.get_string(f'{io}.amazonSecretKey'),
                           region=conf.get_string(f'{io}.amazonRegion'))
    path = s3_uri.replace('s3a://', '') + f'/career_path_transformer/{run_id}/data'
    print(f'Reading parquet: {path}')

    cols = ['candidate_id', 'experience_index', 'experience_type',
            'work_title_role', 'work_title_sub_role', 'work_title_name',
            'work_company_industry', 'is_sj_title',
            'edu_degrees', 'edu_majors', 'edu_school_type',
            'experience_duration_days', 'taxonomy_normalised_job_title']
    ds = pads.dataset(path, filesystem=s3, format='parquet')
    df = ds.to_table(columns=cols).to_pandas()
    print(f'{len(df):,} experience rows, {df["candidate_id"].nunique():,} candidates')

    tax = None if taxonomy_only is False else taxonomy_titles()

    rng = np.random.default_rng(seed)
    cand_ids = df['candidate_id'].dropna().unique()
    rng.shuffle(cand_ids)

    df = df.sort_values(['candidate_id', 'experience_index'])
    groups = dict(tuple(df.groupby('candidate_id', sort=False)))

    samples, seen = [], set()
    for cid in cand_ids:
        if len(samples) >= n:
            break
        recs = _experiences_with_tenure(groups[cid], build_tokens)
        # work experiences in order; need >=2 so one can be held out as target
        work_idx = [i for i, (etype, _, b) in enumerate(recs)
                    if etype == 'WORK' and any(t.startswith(W_TITLE_PREFIX) for t in b)]
        if len(work_idx) < 2:
            continue
        last = work_idx[-1]
        target = next(t for t in recs[last][2] if t.startswith(W_TITLE_PREFIX))
        if tax is not None and target not in tax:
            continue
        ctx_recs = recs[:last]                       # everything before the held-out title
        context_tokens = [t for _, _, b in ctx_recs for t in b]
        if not context_tokens:
            continue
        key = ' | '.join(context_tokens)
        if key in seen:
            continue
        seen.add(key)
        # tenure per work experience in context, in render order
        tenures = []
        for etype, dur, _b in ctx_recs:
            if etype != 'WORK':
                continue
            d = None if dur is None or (isinstance(dur, float) and dur != dur) else int(dur)
            tenures.append({'days': d, 'label': tenure_label(d),
                            'bucket': _duration_bucket(d)})
        tokens = parse_token_string(key)
        samples.append({
            'id': len(samples),
            'label': label_for(tokens),
            'category': categorise(tokens),
            'context_tokens': tokens,
            'work_tenures': tenures,
            'target': target,
        })

    samples.sort(key=lambda s: (s['category'], s['label']))
    for i, s in enumerate(samples):
        s['id'] = i
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    with open(config.SAMPLES_JSON, 'w') as f:
        json.dump(samples, f, indent=1)
    counts = pd.Series([s['category'] for s in samples]).value_counts().to_dict()
    with_tenure = sum(1 for s in samples for t in s['work_tenures'] if t['label'])
    print(f'Wrote {len(samples)} samples to {config.SAMPLES_JSON}  {counts}')
    print(f'  {with_tenure} work experiences carry a tenure label')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default=None, help='eval CSV (default: newest in model_eval_csv)')
    parser.add_argument('--dataset-run-id', default=None,
                        help='source dataset parquet run_id — build resumes WITH per-job tenure')
    parser.add_argument('--env', default='test-prod-sj', help='config env for the dataset parquet')
    parser.add_argument('--ai-conf', default=None,
                        help='path to ai.conf (default: ../datawarehouse-configurations/{partner}/ai/ai.conf)')
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--all-targets', action='store_true',
                        help='do not restrict to taxonomy (is_taxonomy_l3) next titles')
    parser.add_argument('--out', default=config.SAMPLES_JSON,
                        help='output JSON path (default: artifacts/sample_resumes.json)')
    parser.add_argument('--vocab-csv', default=None,
                        help='vocab CSV for the taxonomy filter (default: artifacts/vocab.csv)')
    args = parser.parse_args()

    if args.dataset_run_id:
        partner = args.env.split('-')[-1]   # test-prod-sj -> sj
        conf_path = args.ai_conf or os.path.abspath(os.path.join(
            config.DEMO_ROOT, '..', 'datawarehouse-configurations', partner, 'ai', 'ai.conf'))
        build_from_dataset(args.dataset_run_id, args.n, args.seed,
                           taxonomy_only=(not args.all_targets), conf_path=conf_path, env=args.env)
        return

    csv_path = args.csv or newest_eval_csv()
    print(f'Reading {csv_path}')
    df = pd.read_csv(csv_path, usecols=['context_tokens', 'correct_target'])
    df = df.drop_duplicates(subset='context_tokens')
    print(f'{len(df):,} unique held-out pairs')

    # Only show resumes whose next (held-out) title is a taxonomy L3 title.
    if not args.all_targets:
        tax = taxonomy_titles(args.vocab_csv)
        if tax is None:
            print('  (no vocab CSV with is_taxonomy_l3 — keeping all targets)')
        else:
            before = len(df)
            df = df[df['correct_target'].isin(tax)]
            print(f'  taxonomy filter: {len(df):,}/{before:,} pairs have a taxonomy next title')

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df), size=min(args.n, len(df)), replace=False)
    rows = df.iloc[sorted(idx)]

    samples = []
    for i, (_, row) in enumerate(rows.iterrows()):
        tokens = parse_token_string(row['context_tokens'])
        samples.append({
            'id': i,
            'label': label_for(tokens),
            'category': categorise(tokens),
            'context_tokens': tokens,
            'target': row['correct_target'],
        })
    # Group by category so the picker reads naturally
    samples.sort(key=lambda s: (s['category'], s['label']))
    for i, s in enumerate(samples):
        s['id'] = i

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(samples, f, indent=1)
    counts = pd.Series([s['category'] for s in samples]).value_counts().to_dict()
    print(f'Wrote {len(samples)} samples to {args.out}  {counts}')


if __name__ == '__main__':
    main()
