"""Shared data loading for the demo scripts: career sequences come from the
local eval predictions CSVs (context_tokens + correct_target = the candidate's
full held-out career sequence)."""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from demo import config
from demo.tokens import parse_token_string


def newest_eval_csv() -> str:
    paths = glob.glob(os.path.join(config.EVAL_CSV_DIR, '*.csv'))
    if not paths:
        raise SystemExit(f'No eval CSVs found in {config.EVAL_CSV_DIR}')
    return max(paths, key=os.path.getmtime)


def load_sequences(csv_path: str) -> list:
    print(f'Reading {csv_path}')
    df = pd.read_csv(csv_path, usecols=['context_tokens', 'correct_target'])
    df = df.drop_duplicates(subset='context_tokens')
    seqs = [parse_token_string(c) + [t]
            for c, t in zip(df['context_tokens'], df['correct_target'])]
    print(f'{len(seqs):,} sequences, avg length {np.mean([len(s) for s in seqs]):.1f}')
    return seqs
