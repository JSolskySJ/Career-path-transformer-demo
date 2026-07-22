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

    from models.career_path_skill_suggestion import make_ranker_shell
    params = run_entry.get('params', {})
    _bool = lambda v, d: str(v if v is not None else d).strip().lower() \
        in ('true', '1', 'yes')
    # Rank over the same domain the demo displays for this run, so the
    # baseline rank here matches the prediction panel.
    demo_model = run_entry.get('model')
    sj = (set(demo_model.title_vocab) if demo_model is not None and
          getattr(demo_model, 'restricted', False) else None)
    shell = make_ranker_shell(
        vocab, max_len=int(params.get('max_len', 64) or 64),
        anchored_positions=_bool(params.get('anchored_positions'), 'False'),
        unseen_catch_all=_bool(params.get('unseen_catch_all'), 'True'),
        sj_title_set=sj)
    _CACHE[run_id] = (shell, (module, vocab))
    return _CACHE[run_id]


def suggest(run_id, run_entry, tokens, target, top_k=10, alpha=0.9,
            min_count=50):
    """Two-stage per spec: the fast Bayes-ratio scorer picks the top-k
    candidates (2 forward passes over ALL skills), then the exact brute force
    verifies just those k (+1 baseline) so every number shown is an exact
    model output. Returns lift + verified rank movement per skill."""
    shell, mv = _load(run_id, run_entry)
    module, vocab = mv
    from models.career_path_skill_suggestion import (
        skill_delta_exact, suggest_skills_fast, _skill_ids)
    fast = suggest_skills_fast(
        module, vocab, list(tokens), target, top_k=top_k, alpha=alpha,
        min_train_count=min_count, anchored=shell._anchored_positions)
    base_rank, base_prob, exact = skill_delta_exact(
        shell, mv, list(tokens), target, [s for s, _, _ in fast])
    lifts = {s: (l, p) for s, l, p in fast}
    return {
        'baseline_rank': base_rank,
        'baseline_prob': round(base_prob, 6),
        'n_candidates': len(_skill_ids(vocab, min_count)),
        # exact order (verified deltas) — the fast path only chose candidates
        'suggestions': [{'skill': s, 'delta': round(d, 6), 'new_rank': r,
                         'lift': round(lifts[s][0], 2),
                         'p_given_t': round(lifts[s][1], 6)}
                        for s, d, r in exact],
    }
