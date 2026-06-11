"""Build artifacts/transitions.json — empirical job-title transition counts.

Each career sequence (from the eval CSVs: context_tokens + correct_target) is
reduced to its ordered W_TITLE tokens, and every consecutive title→title change
is counted. Self-transitions (same title twice in a row, e.g. the same role at
a new company) are excluded — they aren't a title change and would show as
loops in the Sankey.

Output:
    {
      "transitions": { "<from W_TITLE>": { "<to W_TITLE>": count, ... }, ... },
      "source_freq": { "<from W_TITLE>": total_outgoing, ... },
      "meta": { "source_csv": ..., "n_sequences": ..., "n_sources": ... }
    }

Usage:
    python scripts/build_transitions.py [--csv PATH]
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo import config
from demo.tokens import W_TITLE_PREFIX
from scripts.data_common import load_sequences, newest_eval_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default=None, help='eval CSV (default: newest in model_eval_csv)')
    args = parser.parse_args()

    csv_path = args.csv or newest_eval_csv()
    seqs = load_sequences(csv_path)

    trans = defaultdict(Counter)
    n_pairs = 0
    for seq in seqs:
        titles = [t for t in seq if t.startswith(W_TITLE_PREFIX)]
        for a, b in zip(titles, titles[1:]):
            if a == b:
                continue
            trans[a][b] += 1
            n_pairs += 1

    transitions = {a: dict(c) for a, c in trans.items()}
    source_freq = {a: sum(c.values()) for a, c in trans.items()}

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    out = config.TRANSITIONS_JSON
    with open(out, 'w') as f:
        json.dump({
            'transitions': transitions,
            'source_freq': source_freq,
            'meta': {
                'source_csv': os.path.basename(csv_path),
                'n_sequences': len(seqs),
                'n_transitions': n_pairs,
                'n_sources': len(transitions),
            },
        }, f)
    print(f'{n_pairs:,} title transitions, {len(transitions):,} distinct source titles')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
