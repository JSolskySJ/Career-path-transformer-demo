"""Counterfactual skill suggestions + live evaluation — demo-native.

Runs entirely on the demo's own staged models (demo/denserec_model.py, loaded
from state_dicts by the registry) — no sibling-repo import, no pickled
training module, nothing to run first. The canonical at-scale implementation
lives in datawarehouse-ai/models/career_path_skill_suggestion.py; this is the
same maths on the demo wrappers.

Two-stage product path (see skill_delta_maths_by_hand.pptx):
  * fast — Bayes' rule: P(T|ctx+s)/P(T|ctx) = P(s|ctx,T)/P(s|ctx). Two
    masked-skill forward passes score ALL skills at once (~2,500x cheaper
    than brute force). lift = P(s|ctx,T)^alpha / P(s|ctx).
  * exact — each fast-picked skill is inserted into the preamble and the full
    title ranking recomputed through the demo model's own rank_titles (content
    injection included), so every displayed number is an exact model output.

Also ``live_eval``: the fidelity/bias evaluation run interactively over a
run's held-out sample resumes (LOSO recovery vs baselines, genericity,
cross-title discrimination, fidelity vs exact, self-consistency + ship
gates). All correlational — surface as skills *associated* with the role.
"""

import json
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
import torch

from demo import config

S_SKILL = 'S_SKILL:'
INCOMING = os.path.join(os.path.dirname(config.ARTIFACTS_DIR), 'incoming')

_COUNTS_CACHE = {}


def _norm(value):
    """Mirror of the training-side _normalise: lowercase, collapse whitespace."""
    return re.sub(r'\s+', ' ', str(value).strip().lower())


def load_counts(run_id):
    """Token training counts from the fetch cache's logged vocab.json (the
    staged demo vocab strips them). {} when unavailable — the frequency floor
    is then skipped with a note."""
    if run_id not in _COUNTS_CACHE:
        path = os.path.join(INCOMING, f'bert4rec_vocab_{run_id[:8]}.json')
        counts = {}
        if os.path.exists(path):
            with open(path) as f:
                counts = json.load(f).get('counts', {}) or {}
        _COUNTS_CACHE[run_id] = counts
    return _COUNTS_CACHE[run_id]


def _held_skills(tokens):
    return {S_SKILL + _norm(t[len(S_SKILL):])
            for t in tokens if t.startswith(S_SKILL)}


def build_pass_rows(vocab, tokens, target, max_len):
    """The two scoring rows: [pad..., preamble, [MASK], recent exps, title
    slot] — Pass A's slot holds T, Pass B's holds [MASK] (replaced, not
    dropped: the rows differ ONLY at the title slot). The preamble + mask slot
    always survive truncation; the remaining budget takes the MOST RECENT
    experiences (deliberately not the ranking path's tail-window rule, which
    would evict the preamble first). OOV tokens are dropped here; the exact
    path keeps them via the demo model's content injection."""
    n_pre = next((i for i, t in enumerate(tokens)
                  if not t.startswith(S_SKILL)), len(tokens))
    pre = [t for t in tokens[:n_pre] if t in vocab.str2idx]
    exp = [t for t in tokens[n_pre:] if t in vocab.str2idx]
    if len(pre) + 2 > max_len:
        pre = pre[-(max_len - 8):]
    exp = exp[-(max_len - len(pre) - 2):]
    body = ([vocab.str2idx[t] for t in pre] + [vocab.mask_id]
            + [vocab.str2idx[t] for t in exp])
    pad_n = max_len - len(body) - 1
    row_a = [vocab.pad_id] * pad_n + body + [vocab.str2idx[target]]
    row_b = [vocab.pad_id] * pad_n + body + [vocab.mask_id]
    return row_a, row_b, pad_n + len(pre)


def _skill_candidates(vocab, counts, min_count):
    out = [(i, t) for i, t in enumerate(vocab.idx2str) if t.startswith(S_SKILL)]
    if min_count and counts:
        kept = [(i, t) for i, t in out if counts.get(t, 0) >= min_count]
        if kept:
            return kept
    return out


def fast_scores(m, tokens, target, alpha=0.9, min_count=50, counts=None):
    """(candidate tokens, lift vector, pass-A probability vector) for one
    worker — 2 forward passes, all skills scored at once."""
    vocab, model = m.vocab, m.model
    if target not in vocab.str2idx:
        raise ValueError(f'{target!r} is not in the model vocabulary')
    cand = _skill_candidates(vocab, counts or {}, min_count)
    row_a, row_b, mask_col = build_pass_rows(vocab, list(tokens), target,
                                             m._max_len)
    ids = torch.tensor([i for i, _ in cand])
    model.eval()
    with torch.no_grad():
        hidden = model._encode(torch.tensor([row_a, row_b]))[:, mask_col, :]
        p = torch.softmax(hidden @ model.item_emb.weight[ids].t(), dim=-1)
    p_a, p_b = p[0].numpy(), p[1].numpy()
    lift = p_a ** alpha / np.clip(p_b, 1e-9, None)
    return [t for _, t in cand], lift, p_a


def insert_skill(tokens, skill):
    """Skill inserted at the end of the preamble (before the first non-skill
    token) — how build_sequences would have emitted it."""
    at = next((i for i, t in enumerate(tokens)
               if not t.startswith(S_SKILL)), len(tokens))
    variant = list(tokens)
    variant.insert(at, skill)
    return variant


def _prob_rank(m, tokens, target):
    """(prob, rank) of the target over the model's full ranking domain —
    exactly what the predictions panel shows."""
    ranked = m.rank_titles(list(tokens), top_k=len(m.title_vocab))
    for r, p in enumerate(ranked['predictions'], start=1):
        if p['token'] == target:
            return p['score'], r
    return 0.0, None


def suggest(m, run_id, tokens, target, top_k=10, alpha=0.9, min_count=50):
    """Fast selection + exact verification: top-k by lift, then each verified
    with a real re-ranking. Every displayed delta/rank is an exact output."""
    counts = load_counts(run_id)
    # Window confound guard for the exact stage: baseline pre-truncated so it
    # and every 1-longer variant see identical experience tokens.
    tokens = list(tokens)
    if len(tokens) + 1 > m._max_len - 1:
        tokens = tokens[-(m._max_len - 2):]

    cand, lift, p_a = fast_scores(m, tokens, target, alpha, min_count, counts)
    held = _held_skills(tokens)
    order = [j for j in np.argsort(-lift)
             if S_SKILL + _norm(cand[j][len(S_SKILL):]) not in held][:top_k]

    base_prob, base_rank = _prob_rank(m, tokens, target)
    if base_rank is None:
        raise ValueError(f'{target!r} is not in this model\'s ranking domain')
    rows = []
    for j in order:
        prob, rank = _prob_rank(m, insert_skill(tokens, cand[j]), target)
        rows.append({'skill': cand[j], 'delta': round(prob - base_prob, 6),
                     'new_rank': rank, 'lift': round(float(lift[j]), 2),
                     'p_given_t': round(float(p_a[j]), 6)})
    rows.sort(key=lambda r: -r['delta'])
    return {'baseline_rank': base_rank, 'baseline_prob': round(base_prob, 6),
            'n_candidates': len(cand), 'suggestions': rows}


# ── Live evaluation ──────────────────────────────────────────────────────────

def live_eval(m, run_id, samples, n=30, alpha=0.9, min_count=50,
              n_fidelity=5, prune=150, seed=42):
    """The fidelity/bias eval, interactive-sized. Baselines:
    B1 popularity · B2 co-occurrence P(s|T) · B3 pass-A only · B4 lift.
    Co-occurrence comes from the sample pairs themselves (proxy — flagged)."""
    rng = random.Random(seed)
    counts = load_counts(run_id)
    vocab = m.vocab
    pairs = [(list(s['context_tokens']), s['target']) for s in samples
             if s.get('target') in vocab.str2idx]
    rng.shuffle(pairs)
    pairs = pairs[:n]
    if not pairs:
        raise ValueError('no usable (worker, target) pairs for this run')

    cand = _skill_candidates(vocab, counts, min_count)
    tokens_ = [t for _, t in cand]
    col = {t: j for j, t in enumerate(tokens_)}
    pop = np.array([counts.get(t, 0) for t in tokens_], dtype=float)

    cooc = defaultdict(Counter)                     # proxy from the eval pairs
    for ctx, target in pairs:
        for s in _held_skills(ctx):
            cooc[target][s] += 1
    skill_titles = defaultdict(Counter)
    for t, d in cooc.items():
        for s, c in d.items():
            skill_titles[s][t] += c

    def genericity(skill):
        v = np.array(list(skill_titles.get(skill, {}).values()), dtype=float)
        if v.sum() == 0:
            return 0.0
        p = v / v.sum()
        return float(-(p * np.log(p + 1e-12)).sum())

    def top10(vec, ctx):
        held = _held_skills(ctx)
        out = []
        for j in np.argsort(-vec):
            if S_SKILL + _norm(tokens_[j][len(S_SKILL):]) in held:
                continue
            out.append(tokens_[j])
            if len(out) >= 10:
                break
        return out

    def spearman(a, b):
        ra = np.argsort(np.argsort(-np.asarray(a))).astype(float)
        rb = np.argsort(np.argsort(-np.asarray(b))).astype(float)
        if ra.std() == 0 or rb.std() == 0:
            return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])

    fast = {}                                        # i -> (lift, p_a)
    for i, (ctx, target) in enumerate(pairs):
        c, lift, p_a = fast_scores(m, ctx, target, alpha, min_count, counts)
        fast[i] = (lift, p_a)

    # LOSO: remove one held skill, does each baseline re-suggest it?
    loso = {b: [] for b in ('B1', 'B2', 'B3', 'B4')}
    for i, (ctx, target) in enumerate(pairs):
        held = [t for t in ctx if t.startswith(S_SKILL) and t in col]
        if not held:
            continue
        removed = rng.choice(held)
        ctx2 = [t for t in ctx if t != removed]
        _, lift2, pa2 = fast_scores(m, ctx2, target, alpha, min_count, counts)
        b2 = np.array([cooc[target].get(t, 0) for t in tokens_], dtype=float)
        b2[col[removed]] = max(b2[col[removed]] - 1, 0)   # LOSO its own count
        for b, vec in (('B1', pop), ('B2', b2), ('B3', pa2), ('B4', lift2)):
            held2 = _held_skills(ctx2)
            order = [tokens_[j] for j in np.argsort(-vec)
                     if S_SKILL + _norm(tokens_[j][len(S_SKILL):]) not in held2]
            rank = order.index(removed) + 1 if removed in order else None
            loso[b].append({'mrr': 1 / rank if rank else 0.0,
                            'hit10': 1.0 if rank and rank <= 10 else 0.0})
    loso_out = {b: {'mrr': round(float(np.mean([x['mrr'] for x in v])), 4),
                    'hit10': round(float(np.mean([x['hit10'] for x in v])), 4)}
                for b, v in loso.items() if v}

    gen = {b: [] for b in ('B3', 'B4')}
    for i, (ctx, _) in enumerate(pairs):
        lift, p_a = fast[i]
        gen['B3'].extend(genericity(t) for t in top10(p_a, ctx))
        gen['B4'].extend(genericity(t) for t in top10(lift, ctx))
    gen_out = {b: round(float(np.mean(v)), 4) for b, v in gen.items() if v}

    jac = []
    for i, (ctx, target) in enumerate(pairs[:20]):
        other = pairs[(i + len(pairs) // 2) % len(pairs)][1]
        if other == target:
            continue
        _, lift_x, _ = fast_scores(m, ctx, other, alpha, min_count, counts)
        a, b = set(top10(fast[i][0], ctx)), set(top10(lift_x, ctx))
        jac.append(len(a & b) / max(len(a | b), 1))
    jaccard = round(float(np.mean(jac)), 4) if jac else None

    # Fidelity vs exact + self-consistency (the slow bit — n_fidelity pairs)
    ov10, spear, selfc = [], [], []
    for i, (ctx, target) in enumerate(pairs[:n_fidelity]):
        ctx = ctx[-(m._max_len - 2):] if len(ctx) + 1 > m._max_len - 1 else ctx
        base_prob, base_rank = _prob_rank(m, ctx, target)
        if base_rank is None:
            continue
        b2 = np.array([cooc[target].get(t, 0) for t in tokens_])
        pruned = [tokens_[j] for j in np.argsort(-b2)[:prune]]
        exact = []
        for s in pruned:
            prob, _ = _prob_rank(m, insert_skill(ctx, s), target)
            exact.append((s, prob - base_prob))
        exact.sort(key=lambda r: -r[1])
        exact_by = dict(exact)
        lift = fast[i][0]
        fast_order = [t for t in
                      (tokens_[j] for j in np.argsort(-lift))
                      if t in exact_by][:len(pruned)]
        ov10.append(len(set(fast_order[:10])
                        & {s for s, _ in exact[:10]}) / 10)
        common = fast_order
        if len(common) >= 5:
            spear.append(spearman([float(lift[col[t]]) for t in common],
                                  [exact_by[t] for t in common]))
        top1 = top10(lift, ctx)[:1]
        if top1:
            d_top, _ = _prob_rank(m, insert_skill(ctx, top1[0]), target)
            rand_s = rng.choice(tokens_)
            d_rand, _ = _prob_rank(m, insert_skill(ctx, rand_s), target)
            selfc.append(1.0 if d_top - base_prob > 0 and
                         d_top > d_rand else 0.0)

    fidelity = {'overlap10': round(float(np.mean(ov10)), 4) if ov10 else None,
                'spearman': round(float(np.mean(spear)), 4) if spear else None,
                'self_consistency': round(float(np.mean(selfc)), 4)
                if selfc else None}
    gates = {
        'overlap10 ≥ 0.6': bool(ov10) and np.mean(ov10) >= 0.6,
        'spearman ≥ 0.7': bool(spear) and np.mean(spear) >= 0.7,
        'B4 beats B1 on LOSO MRR': bool(loso_out) and
            loso_out.get('B4', {}).get('mrr', 0) > loso_out.get('B1', {}).get('mrr', 1),
        'B4 beats B2 on LOSO MRR': bool(loso_out) and
            loso_out.get('B4', {}).get('mrr', 0) > loso_out.get('B2', {}).get('mrr', 1),
        'genericity B4 < B3': bool(gen_out) and
            gen_out.get('B4', 1) < gen_out.get('B3', 0),
        'self-consistency > 0.5': bool(selfc) and np.mean(selfc) > 0.5,
    }
    return {
        'n_pairs': len(pairs), 'n_loso': len(next(iter(loso.values()), [])),
        'n_fidelity': len(ov10), 'alpha': alpha, 'min_count': min_count,
        'n_candidates': len(tokens_), 'cooc_proxy': True,
        'loso': loso_out, 'genericity': gen_out,
        'cross_title_jaccard': jaccard, 'fidelity': fidelity,
        'gates': {k: bool(v) for k, v in gates.items()},
    }
