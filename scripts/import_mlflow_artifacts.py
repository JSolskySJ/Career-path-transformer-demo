"""Stage MLflow-downloaded model artifacts into demo-loadable checkpoints.

Exposes the two staging conversions, used by scripts/fetch_mlflow_artifacts.py
to stage each run into the registry (artifacts/runs/<run_id>/):

  stage_bert4rec(pth, vocab_json_dir, vocab_csvs, out_dir)
      model.pth (pickled nn.Module) -> out_dir/{model.pt, vocab.json, config.json}
  stage_item2vec(bin_path, out_dir)
      gensim .bin (+ .npy sidecars) -> out_dir/item2vec.bin*, load-verified

Run directly, it stages incoming/ into the LEGACY single-slot artifacts
(artifacts/item2vec.bin, artifacts/bert4rec/, artifacts/vocab.csv) — the
layout scripts/train_*.py also write and the app falls back to when no
registry runs exist.

model.pth is a whole pickled BERT4Rec module logged by mlflow.pytorch under the
original training module path (models.career_path_transformer_bert4rec). We
register the demo's identical BERT4Rec class under that name so it unpickles,
then extract a plain state_dict + the architecture dims read off the module, so
the app loads it with no MLflow / shim dependency. The vocab idx→token mapping
is taken from the run's logged vocab json ('idx2str' — authoritative). If that
isn't staged we fall back to rebuilding it as ['[PAD]','[MASK]'] + vocab-CSV
tokens (only correct when the CSV's train-count order has no tie-breaking
differences from the model's).

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


def _load_bert4rec_module(pth):
    """Unpickle an mlflow-logged torch career-path module (BERT4Rec or its
    DenseRec extension), shimming the original training module paths onto the
    demo's weight-compatible classes."""
    import torch
    import demo.bert4rec_model as b4
    import demo.denserec_model as dr
    import demo.modernbert_model as mb

    # The shim must not permanently shadow a real `models` package (e.g. the
    # datawarehouse-ai one used by demo.dataset_samples in the same process) —
    # save whatever is loaded and restore it after unpickling.
    saved = {k: sys.modules[k] for k in list(sys.modules)
             if k == 'models' or k.startswith('models.')}
    try:
        pkg = types.ModuleType('models'); pkg.__path__ = []
        sys.modules['models'] = pkg
        shim = types.ModuleType('models.career_path_transformer_bert4rec')
        shim.BERT4Rec = b4.BERT4Rec
        sys.modules['models.career_path_transformer_bert4rec'] = shim
        dshim = types.ModuleType('models.career_path_transformer_denserec')
        dshim.DenseRecBERT4Rec = dr.DenseRecBERT4Rec
        dshim.DenseRecModernBERT4Rec = dr.DenseRecModernBERT4Rec
        sys.modules['models.career_path_transformer_denserec'] = dshim
        mshim = types.ModuleType('models.career_path_transformer_modernbert')
        mshim.ModernBERT4Rec = mb.ModernBERT4Rec
        mshim.RotaryEmbedding = mb.RotaryEmbedding
        mshim.ModernBertBlock = mb.ModernBertBlock
        sys.modules['models.career_path_transformer_modernbert'] = mshim
        return torch.load(pth, map_location='cpu', weights_only=False).eval()
    finally:
        for k in list(sys.modules):
            if k == 'models' or k.startswith('models.'):
                del sys.modules[k]
        sys.modules.update(saved)


def _logged_idx2str(vocab_json_candidates, vocab_size):
    """The run's logged vocab json ('idx2str' key or a bare list), when staged
    and consistent with the model's embedding table. None otherwise."""
    for path in vocab_json_candidates:
        with open(path) as f:
            blob = json.load(f)
        cand = blob.get('idx2str') if isinstance(blob, dict) else blob
        if cand and len(cand) == vocab_size:
            print(f'idx2str: logged {os.path.basename(path)} '
                  f'({len(cand):,} tokens) — authoritative')
            return cand
        if cand:
            print(f'idx2str: logged vocab len {len(cand)} != model vocab_size '
                  f'{vocab_size}; ignoring')
    return None


def stage_bert4rec(pth, vocab_json_candidates, vocab_csvs, out_dir):
    """model.pth -> out_dir/{model.pt, vocab.json, config.json}. Returns the
    vocab CSV whose token count matches the model (or None).
    vocab_json_candidates: possible logged-vocab json paths, tried in order."""
    import torch

    model = _load_bert4rec_module(pth)
    sd = model.state_dict()
    d_model = sd['item_emb.weight'].shape[1]
    vocab_size = sd['item_emb.weight'].shape[0]
    max_len = sd['pos_emb.weight'].shape[0]
    # backbone detection: ModernBERT blocks live at encoder.{i}.* with a
    # final_norm; the stock stack at encoder.layers.{i}.*
    modern = 'final_norm.weight' in sd
    if modern:
        n_layers = len({k.split('.')[1] for k in sd if k.startswith('encoder.')})
        n_heads = model.encoder[0].n_heads
    else:
        n_layers = len({k.split('.')[2] for k in sd if k.startswith('encoder.layers.')})
        n_heads = model.encoder.layers[0].self_attn.num_heads

    matching_csv = next(
        (c for c in sorted(vocab_csvs, key=os.path.getmtime, reverse=True)
         if len(pd.read_csv(c, usecols=['token'])) + 2 == vocab_size), None)

    idx2str = _logged_idx2str(vocab_json_candidates, vocab_size)
    if idx2str is None:
        if matching_csv is None:
            raise SystemExit(
                f'bert4rec: no logged vocab json and no vocab CSV matches '
                f'vocab_size={vocab_size} (+2) — cannot recover the token map. '
                f'Stage the run\'s vocab.json as bert4rec_vocab.json.')
        idx2str = [PAD_TOKEN, MASK_TOKEN] + list(pd.read_csv(matching_csv)['token'])
        print('idx2str: rebuilt from vocab CSV train-count order (no logged '
              'vocab json staged)')

    os.makedirs(out_dir, exist_ok=True)
    torch.save({k: t.cpu() for k, t in sd.items()}, os.path.join(out_dir, 'model.pt'))
    with open(os.path.join(out_dir, 'vocab.json'), 'w') as f:
        json.dump(idx2str, f)
    cfg = {
        'd_model': int(d_model), 'n_layers': int(n_layers), 'n_heads': int(n_heads),
        'max_len': int(max_len), 'dropout': 0.2,
        'vocab_size': int(vocab_size),
        'backbone': 'modernbert' if modern else 'bert4rec',
        'source': os.path.basename(pth), 'origin': 'mlflow',
    }
    arch = 'modernbert' if modern else 'bert4rec'
    if 'content' in sd:                    # DenseRec extension buffers present
        arch = 'denserec'
        cfg.update({
            'content_dim': int(sd['content'].shape[1]),
            'dense_path_p': float(getattr(model, 'dense_path_p', 0.5)),
        })
    cfg['architecture'] = arch
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'{arch} ({cfg["backbone"]} backbone): vocab_size={vocab_size} '
          f'd_model={d_model} max_len={max_len} n_layers={n_layers} '
          f'n_heads={n_heads} -> {out_dir}')
    return matching_csv


def stage_item2vec(bin_path, out_dir) -> bool:
    """Copy the gensim .bin (+ any .npy sidecars, renamed to match) into
    out_dir/item2vec.bin* and verify it loads. Removes the staged files and
    returns False when the model is unloadable (e.g. a run logged without its
    sidecars — arrays > 10MB are stored separately by gensim)."""
    os.makedirs(out_dir, exist_ok=True)
    dst_bin = os.path.join(out_dir, 'item2vec.bin')
    for stale in glob.glob(dst_bin + '.*'):
        os.remove(stale)
    for src in [bin_path] + glob.glob(bin_path + '.*'):
        dst = dst_bin + src[len(bin_path):]
        shutil.copy(src, dst)
        print(f'item2vec: {os.path.basename(src)} -> {dst}')
    try:
        from gensim.models import Word2Vec
        wv = Word2Vec.load(dst_bin).wv
        print(f'item2vec: verified ({len(wv):,} tokens, dim={wv.vector_size})')
        return True
    except Exception as exc:
        for f in [dst_bin] + glob.glob(dst_bin + '.*'):
            os.remove(f)
        print(f'item2vec: staged model FAILED to load ({exc}) — removed. '
              f'The run was probably logged without its gensim sidecar files; '
              f'retrain with the fixed _log_model in datawarehouse-ai.')
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incoming', default=os.path.join(config.DEMO_ROOT, 'incoming'))
    args = parser.parse_args(argv)
    inc = args.incoming

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    staged = []

    vocab_csvs = glob.glob(os.path.join(inc, 'career_path_transformer_vocab_*.csv'))

    # ── bert4rec model.pth (optional; drives vocab-CSV selection) ─────────────
    vocab_csv = None
    pths = glob.glob(os.path.join(inc, 'model.pth'))
    if pths:
        vocab_jsons = (glob.glob(os.path.join(inc, 'bert4rec_vocab.json'))
                       or glob.glob(os.path.join(inc, '*bert4rec*vocab*.json')))
        vocab_csv = stage_bert4rec(max(pths, key=os.path.getmtime), vocab_jsons,
                                   vocab_csvs, config.BERT4REC_DIR)
        staged.append('bert4rec')
    if vocab_csv is None and vocab_csvs:
        # No CSV matches the bert4rec vocab (or no bert4rec staged) — the newest
        # CSV still carries the ranking-domain flags, which ranking intersects
        # with each model's own titles.
        vocab_csv = max(vocab_csvs, key=os.path.getmtime)

    # ── vocab CSV (optional — absent on runs trained with log_vocab=False) ────
    if vocab_csv:
        shutil.copy(vocab_csv, config.VOCAB_CSV)
        v = pd.read_csv(config.VOCAB_CSV)
        n_dom = int((v.get('in_ranking_domain') == True).sum()) if 'in_ranking_domain' in v else 0
        n_tax = int((v.get('is_taxonomy_l3') == True).sum()) if 'is_taxonomy_l3' in v else 0
        print(f'vocab: {os.path.basename(vocab_csv)}  {len(v):,} tokens  '
              f'({n_tax:,} taxonomy, {n_dom:,} ranking domain)  -> {config.VOCAB_CSV}')
        staged.append('vocab.csv')
    else:
        if os.path.exists(config.VOCAB_CSV):
            os.remove(config.VOCAB_CSV)
            print('vocab: no vocab CSV staged (run trained with log_vocab=False?) — '
                  'removed stale artifacts/vocab.csv; ranking falls back to the '
                  'full title vocabulary')
        else:
            print('vocab: no vocab CSV staged — ranking uses the full title vocabulary')

    # ── item2vec .bin (optional) ───────────────────────────────────────────────
    bins = glob.glob(os.path.join(inc, 'career_path_transformer_*.bin'))
    if bins:
        if stage_item2vec(max(bins, key=os.path.getmtime), config.ARTIFACTS_DIR):
            staged.append('item2vec')

    if not staged:
        raise SystemExit(f'Nothing to import: no model.pth or *.bin in {inc}')
    print(f'Done ({", ".join(staged)}). Restart the app to load the new artifacts.')


if __name__ == '__main__':
    main()
