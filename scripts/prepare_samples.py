"""Build artifacts/sample_resumes.json from a model-eval predictions CSV.

The eval CSVs (datawarehouse-ai-analysis/career_path_transformer/model_eval_csv)
hold real held-out pairs: context_tokens (the resume the model saw) and
correct_target (the actual next job title). Sampling from them gives the demo
realistic resumes with a ground-truth answer for sense-checking.

Usage:
    python scripts/prepare_samples.py [--csv PATH] [--n 300] [--seed 42]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default=None, help='eval CSV (default: newest in model_eval_csv)')
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    csv_path = args.csv or newest_eval_csv()
    print(f'Reading {csv_path}')
    df = pd.read_csv(csv_path, usecols=['context_tokens', 'correct_target'])
    df = df.drop_duplicates(subset='context_tokens')
    print(f'{len(df):,} unique held-out pairs')

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

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    with open(config.SAMPLES_JSON, 'w') as f:
        json.dump(samples, f, indent=1)
    counts = pd.Series([s['category'] for s in samples]).value_counts().to_dict()
    print(f'Wrote {len(samples)} samples to {config.SAMPLES_JSON}  {counts}')


if __name__ == '__main__':
    main()
