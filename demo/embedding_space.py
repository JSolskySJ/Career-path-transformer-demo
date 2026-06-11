"""2-D maps of the model's embedding space for a given resume.

Both wrapped models expose the same surface (title_vocab, title_matrix with
unit rows, vector(token), context_vector(context)), so one mapper serves both.

Two views:
  local  — PCA fitted on the context vector's cosine neighbourhood plus the
           resume's own tokens. Zoomed-in: shows where the resume sits among
           the titles the model considers nearby. Default.
  global — PCA fitted once on the full title matrix (cached per model). Shows
           the resume against the overall shape of the title space, with the
           most frequent titles as background.

Note: for BERT4Rec the context vector is the transformer hidden state at the
[MASK] position. With weight tying the logits are h . item_emb, so it shares a
dot-product space with the token embeddings, but it is not itself a token
embedding — treat its position as indicative.
"""

import numpy as np
from sklearn.decomposition import PCA

from demo.tokens import (token_type, token_value, W_TITLE_PREFIX,
                         group_token_bundles, bundle_kind)

_GLOBAL_PCA_CACHE = {}   # id(model) -> (fitted PCA, fixed axis ranges)

GLOBAL_FIT_CAP = 5000    # titles used to fit the global PCA (frequency order)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _global_pca(model):
    """Fitted PCA plus fixed axis ranges, cached per model — the global view
    keeps a constant basis AND frame so the query's movement between resumes
    is real movement, not autorange."""
    key = id(model)
    if key not in _GLOBAL_PCA_CACHE:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(model.title_matrix[:GLOBAL_FIT_CAP])
        lo, hi = proj.min(axis=0), proj.max(axis=0)
        pad = 0.08 * (hi - lo)
        ranges = {'x': [float(lo[0] - pad[0]), float(hi[0] + pad[0])],
                  'y': [float(lo[1] - pad[1]), float(hi[1] + pad[1])]}
        _GLOBAL_PCA_CACHE[key] = (pca, ranges)
    return _GLOBAL_PCA_CACHE[key]


def map_space(model, resume_tokens: list, predictions: list,
              mode: str = 'local', n_background: int = 120,
              n_landmarks: int = 10) -> dict:
    """Build the point set for the embedding-space plot.

    predictions: [{'token': ..., 'score': ...}] from rank_titles (already top-k).
    Returns {'points': [...], 'explained_variance': [...], 'mode': mode}.
    """
    ctx_vec, used_tokens, _ = model.context_vector(resume_tokens)
    used_set = set(used_tokens or [])

    # One vector per experience: each bundle's known token vectors are unit-
    # normalised and averaged (the same composition the item2vec query uses),
    # so the career path shows one point per experience, not per token.
    resume_known = [t for t in resume_tokens if model.knows(t)]
    bundle_points = []
    for bundle in group_token_bundles(resume_tokens):
        known = [t for t in bundle if model.knows(t)]
        if not known:
            continue
        vec = _unit(np.mean([_unit(model.vector(t)) for t in known], axis=0))
        title_tok = next((t for t in bundle if t.startswith(W_TITLE_PREFIX)), None)
        bundle_points.append({
            'vec': vec,
            'kind': bundle_kind(bundle),
            'label': token_value(title_tok or known[0]),
            'tokens': bundle,
            'in_window': any(t in used_set for t in known),
        })

    pred_tokens = [p['token'] for p in predictions]
    pred_scores = {p['token']: p['score'] for p in predictions}
    title_index = {t: i for i, t in enumerate(model.title_vocab)}

    # ── Choose background titles and the PCA basis ───────────────────────────
    axis_ranges = None
    if mode == 'global' or ctx_vec is None:
        pca, axis_ranges = _global_pca(model)
        background = model.title_vocab[:n_background]
    else:
        sims  = model.title_matrix @ _unit(ctx_vec)
        order = np.argsort(-sims)[:n_background]
        background = [model.title_vocab[i] for i in order]
        fit_rows = [model.title_matrix[i] for i in order]
        fit_rows += [b['vec'] for b in bundle_points]
        fit_rows.append(_unit(ctx_vec))
        pca = PCA(n_components=2)
        pca.fit(np.stack(fit_rows))

    # ── Project every point of interest ──────────────────────────────────────
    points = []
    seen = set()

    def add(token, vec, kind, **extra):
        if (token, kind) in seen or vec is None:
            return
        seen.add((token, kind))
        x, y = pca.transform(vec.reshape(1, -1))[0]
        points.append({
            'token': token, 'value': token_value(token), 'type': token_type(token),
            'x': float(x), 'y': float(y), 'kind': kind, **extra,
        })

    if ctx_vec is not None:
        add('[CONTEXT]', _unit(ctx_vec), 'context')
    for i, b in enumerate(bundle_points):
        add(f'[EXPERIENCE:{i}]', b['vec'], 'resume', order=i,
            value=b['label'], type=b['kind'], tokens=b['tokens'],
            in_window=b['in_window'])
    for rank, tok in enumerate(pred_tokens, start=1):
        idx = title_index.get(tok)
        vec = model.title_matrix[idx] if idx is not None else None
        add(tok, vec, 'prediction', rank=rank, score=pred_scores[tok])

    # The most common titles as fixed, labelled landmarks — a stable reference
    # frame for reading where the query and predictions sit. Skipped when the
    # title is already highlighted as a resume token or prediction.
    skip = set(resume_known) | set(pred_tokens)
    for tok in model.title_vocab[:n_landmarks]:
        if tok not in skip:
            add(tok, model.title_matrix[title_index[tok]], 'landmark')
            skip.add(tok)

    for tok in background:
        if tok not in skip:
            add(tok, model.title_matrix[title_index[tok]], 'background')

    return {
        'points': points,
        'explained_variance': [float(v) for v in pca.explained_variance_ratio_],
        'mode': mode if ctx_vec is not None else 'global',
        'axis_ranges': axis_ranges,
    }
