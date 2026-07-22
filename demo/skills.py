"""Counterfactual skill suggestions — demo bridge to the training repo.

The canonical implementation is CareerPathDenseRecModel.suggest_skills in
datawarehouse-ai (exact brute force through _rank_titles_batch, window guard,
_PositionedSeq-safe preamble insertion). This module loads the run's PICKLED
training module + logged vocab straight from the fetch script's incoming/
cache — the same artifacts the importer stages, so no hand-rolled download
logic — and exposes suggest() for the /api/suggest_skills endpoint.

DenseRec runs only: ranking needs the unseen-token catch-all / anchored-
positions plumbing that lives on the DenseRec class.
"""

import json
import os
import sys

from demo import config

_DWH_AI = os.path.join(config.DWH_ROOT, 'datawarehouse-ai')
INCOMING = os.path.join(os.path.dirname(config.ARTIFACTS_DIR), 'incoming')

_CACHE = {}   # run_id -> (shell, (module, vocab))


def _load(run_id, run_entry):
    if run_id in _CACHE:
        return _CACHE[run_id]
    if run_entry['architecture'] != 'denserec':
        raise ValueError('skill suggestions need a DenseRec run — ranking '
                         'variants uses its unseen-token catch-all plumbing')
    short = run_id[:8]
    model_path = os.path.join(INCOMING, f'model_{short}.pth')
    vocab_path = os.path.join(INCOMING, f'bert4rec_vocab_{short}.json')
    for p in (model_path, vocab_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f'{os.path.basename(p)} not in the incoming/ cache — re-run '
                f'scripts/fetch_mlflow_artifacts.py for this run')

    # The pickle references the real training package — import it from the
    # sibling checkout (the staging shim must not be active here).
    if _DWH_AI not in sys.path:
        sys.path.insert(0, _DWH_AI)
    for mod in list(sys.modules):
        if (mod == 'models' or mod.startswith('models.')) and \
                getattr(sys.modules[mod], '__file__', None) is None:
            del sys.modules[mod]
    import torch
    from models.career_path_transformer_attention import Vocab
    from models.career_path_transformer_denserec import CareerPathDenseRecModel

    module = torch.load(model_path, map_location='cpu', weights_only=False)
    module.eval()
    with open(vocab_path) as f:
        vocab = Vocab.from_idx2str(json.load(f)['idx2str'])

    params = run_entry.get('params', {})
    shell = CareerPathDenseRecModel.__new__(CareerPathDenseRecModel)
    shell._max_len = int(params.get('max_len', 64) or 64)
    shell._anchored_positions = str(params.get(
        'anchored_positions', 'False')).strip().lower() in ('true', '1', 'yes')
    shell._unseen_catch_all = str(params.get(
        'unseen_catch_all', 'True')).strip().lower() in ('true', '1', 'yes')
    shell._unseen_vector_cache = {}
    # Rank over the same domain the demo displays for this run, so the
    # baseline rank here matches the prediction panel.
    demo_model = run_entry.get('model')
    shell._sj_title_set = (set(demo_model.title_vocab)
                           if demo_model is not None and
                           getattr(demo_model, 'restricted', False) else None)
    _CACHE[run_id] = (shell, (module, vocab))
    return _CACHE[run_id]


def suggest(run_id, run_entry, tokens, target, top_k=10, limit=0):
    """Top-k skills whose addition most improves ``target``'s rank for this
    context. limit > 0 prunes the candidate set (by vocab frequency order)
    for a faster pass; 0 scores every trained skill."""
    shell, mv = _load(run_id, run_entry)
    _, vocab = mv
    candidates = None
    if limit:
        skills = [t for t in vocab.idx2str if t.startswith('S_SKILL:')]
        candidates = skills[:limit]
    base_rank, rows = shell.suggest_skills(
        mv, list(tokens), target, top_k=top_k, candidates=candidates)
    n_scored = len(candidates) if candidates is not None else \
        sum(1 for t in vocab.idx2str if t.startswith('S_SKILL:'))
    return {
        'baseline_rank': base_rank,
        'n_candidates': n_scored,
        'suggestions': [{'skill': s, 'delta': round(d, 6), 'new_rank': r}
                        for s, d, r in rows],
    }
