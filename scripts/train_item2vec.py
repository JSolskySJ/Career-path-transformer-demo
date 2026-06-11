"""Train a demo item2vec (gensim Word2Vec) checkpoint the app can load.

Trains with the production hyperparameters (skip-gram, window 2, 128 dims,
10 negatives) on the same local sequences as scripts/train_bert4rec.py, so the
two demo models share a corpus and vocabulary domain and their predictions are
directly comparable. Saves to artifacts/item2vec.bin, which the app prefers
over the (engineering-subset) artifact in datawarehouse-ai-analysis.

Usage:
    python scripts/train_item2vec.py [--csv PATH] [--epochs 10]
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo import config
from scripts.data_common import load_sequences, newest_eval_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default=None, help='eval CSV (default: newest in model_eval_csv)')
    parser.add_argument('--vector-size', type=int, default=128)
    parser.add_argument('--window',      type=int, default=2)
    parser.add_argument('--epochs',      type=int, default=10)
    parser.add_argument('--negative',    type=int, default=10)
    parser.add_argument('--min-count',   type=int, default=3)
    parser.add_argument('--seed',        type=int, default=123)
    args = parser.parse_args()

    from gensim.models import Word2Vec

    seqs = load_sequences(args.csv or newest_eval_csv())
    model = Word2Vec(
        sentences=seqs,
        vector_size=args.vector_size,
        window=args.window,
        sg=1,
        negative=args.negative,
        min_count=args.min_count,
        epochs=args.epochs,
        seed=args.seed,
        workers=4,
    )
    n_titles = sum(1 for t in model.wv.index_to_key if t.startswith('W_TITLE:'))
    print(f'Vocab: {len(model.wv):,} tokens ({n_titles:,} titles)')

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    out = os.path.join(config.ARTIFACTS_DIR, 'item2vec.bin')
    model.save(out)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
