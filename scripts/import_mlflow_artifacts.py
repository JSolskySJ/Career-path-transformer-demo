"""Stage MLflow-downloaded artifacts (in incoming/) into the demo's artifacts/.

Run the download first (see README), leaving these in incoming/:
  career_path_transformer_<ts>.bin                  item2vec gensim model
  career_path_transformer_vocab_<ts>.csv            shared vocab + ranking domain
  career_path_transformer_<ts>.csv (x2)             item2vec / bert4rec predictions
  model.pth                                         bert4rec (pickled nn.Module)

This script:
  - copies the item2vec .bin            -> artifacts/item2vec.bin
  - copies the vocab CSV                -> artifacts/vocab.csv
  - converts model.pth                  -> artifacts/bert4rec/{model.pt, vocab.json, config.json}

model.pth is a whole pickled BERT4Rec module logged by mlflow.pytorch under the
original training module path (models.career_path_transformer_bert4rec). We
register the demo's identical BERT4Rec class under that name so it unpickles,
then extract a plain state_dict + the architecture dims read off the module, so
the app loads it with no MLflow / shim dependency. The vocab idx→token mapping
is rebuilt as ['[PAD]','[MASK]'] + vocab-CSV tokens (the CSV is in the model's
train-count-descending order — verified against the model's own predictions).

Usage:
    python scripts/import_mlflow_artifacts.py [--incoming incoming]
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import glob
import json
import shutil
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from demo import config

PAD_TOKEN, MASK_TOKEN = '[PAD]', '[MASK]'


def _one(incoming, pattern, label):
    hits = glob.glob(os.path.join(incoming, pattern))
    if not hits:
        raise SystemExit(f'Missing {label}: no {pattern} in {incoming}')
    return max(hits, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incoming', default=os.path.join(config.DEMO_ROOT, 'incoming'))
    args = parser.parse_args()
    inc = args.incoming

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)

    # ── bert4rec model.pth (load first, to drive vocab selection) ─────────────
    import torch
    import demo.bert4rec_model as b4

    pkg = types.ModuleType('models'); pkg.__path__ = []
    sys.modules['models'] = pkg
    shim = types.ModuleType('models.career_path_transformer_bert4rec')
    shim.BERT4Rec = b4.BERT4Rec
    sys.modules['models.career_path_transformer_bert4rec'] = shim

    pth = _one(inc, 'model.pth', 'bert4rec model.pth')
    model = torch.load(pth, map_location='cpu', weights_only=False).eval()
    sd = model.state_dict()
    d_model = sd['item_emb.weight'].shape[1]
    vocab_size = sd['item_emb.weight'].shape[0]
    max_len = sd['pos_emb.weight'].shape[0]
    n_layers = len({k.split('.')[2] for k in sd if k.startswith('encoder.layers.')})
    n_heads = model.encoder.layers[0].self_attn.num_heads

    # ── vocab CSV ─────────────────────────────────────────────────────────────
    # A run may bundle a vocab CSV whose token count doesn't equal the model's
    # (the two models here come from sibling runs). Pick the one that matches the
    # bert4rec embedding table (len + 2 specials) so idx2str decodes correctly;
    # the runs share an identical taxonomy / ranking domain, so any matching CSV
    # serves for both models. The vocab CSV is in train-count-descending order,
    # matching how the training Vocab is built, so idx2str = [PAD, MASK] + tokens.
    vocab_csvs = glob.glob(os.path.join(inc, 'career_path_transformer_vocab_*.csv'))
    if not vocab_csvs:
        raise SystemExit(f'Missing vocab CSV in {inc}')
    chosen = None
    for c in sorted(vocab_csvs, key=os.path.getmtime, reverse=True):
        if len(pd.read_csv(c, usecols=['token'])) + 2 == vocab_size:
            chosen = c
            break
    if chosen is None:
        raise SystemExit(
            f'No vocab CSV matches bert4rec vocab_size={vocab_size} (+2). '
            f'Candidates: {[os.path.basename(c) for c in vocab_csvs]}')
    shutil.copy(chosen, config.VOCAB_CSV)
    v = pd.read_csv(config.VOCAB_CSV)
    idx2str = [PAD_TOKEN, MASK_TOKEN] + list(v['token'])
    n_dom = int((v.get('in_ranking_domain') == True).sum()) if 'in_ranking_domain' in v else 0
    n_tax = int((v.get('is_taxonomy_l3') == True).sum()) if 'is_taxonomy_l3' in v else 0
    print(f'vocab: {os.path.basename(chosen)}  {len(v):,} tokens  '
          f'({n_tax:,} taxonomy, {n_dom:,} ranking domain)  -> {config.VOCAB_CSV}')

    # ── item2vec .bin ─────────────────────────────────────────────────────────
    # the .bin file (not the *_vocab_*.csv, not the predictions *.csv)
    bins = glob.glob(os.path.join(inc, 'career_path_transformer_*.bin'))
    if not bins:
        raise SystemExit(f'Missing item2vec .bin in {inc}')
    item2vec_bin = max(bins, key=os.path.getmtime)
    dst_bin = os.path.join(config.ARTIFACTS_DIR, 'item2vec.bin')
    shutil.copy(item2vec_bin, dst_bin)
    print(f'item2vec: {os.path.basename(item2vec_bin)} -> {dst_bin}')

    # ── bert4rec -> state_dict + vocab.json + config.json ─────────────────────
    out = config.BERT4REC_DIR
    os.makedirs(out, exist_ok=True)
    torch.save({k: t.cpu() for k, t in sd.items()}, os.path.join(out, 'model.pt'))
    with open(os.path.join(out, 'vocab.json'), 'w') as f:
        json.dump(idx2str, f)
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump({
            'd_model': int(d_model), 'n_layers': int(n_layers), 'n_heads': int(n_heads),
            'max_len': int(max_len), 'dropout': 0.2,
            'vocab_size': int(vocab_size),
            'source': os.path.basename(pth), 'origin': 'mlflow',
        }, f, indent=2)
    print(f'bert4rec: vocab_size={vocab_size} d_model={d_model} '
          f'max_len={max_len} n_layers={n_layers} n_heads={n_heads} -> {out}')
    print('Done. Restart the app to load the new artifacts.')


if __name__ == '__main__':
    main()
