"""Confidence summary for a ranking score distribution.

Quantifies how decisive a model's next-title ranking is for one resume:
  - top1 / top2       : the two highest scores (native units)
  - margin            : top1 - top2 (how far ahead the winner is)
  - entropy           : Shannon entropy of the score distribution, normalised
                        to 0..1 (0 = all mass on one title, 1 = uniform over the
                        whole ranking domain). High = the model is spreading its
                        probability thinly across many titles, i.e. unsure.

BERT4Rec scores are already a probability distribution (softmax over the title
domain), so its entropy is the model's own uncertainty. item2vec scores are
cosine similarities sitting in a narrow range, so a temperature-1 softmax over
them is always near-uniform (uninformative). We instead scale by the score std
before the softmax, making the entropy a scale-free measure of how far the top
titles stand out from the crowd. `unit` records which so the UI can format the
top-1/margin numbers accordingly.
"""

import numpy as np


def distribution_confidence(scores, unit: str):
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return None
    order = np.sort(s)[::-1]
    top1 = float(order[0])
    top2 = float(order[1]) if order.size > 1 else None

    if unit == 'prob':
        p = s / (s.sum() + 1e-12)
    else:  # cosine similarities -> std-scaled softmax (scale-free concentration)
        sd = s.std()
        z = s / sd if sd > 0 else s
        e = np.exp(z - z.max())
        p = e / e.sum()
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum())
    norm_entropy = entropy / np.log(len(s)) if len(s) > 1 else 0.0

    return {
        'top1': top1,
        'top2': top2,
        'margin': (top1 - top2) if top2 is not None else None,
        'entropy': float(norm_entropy),
        'n': int(s.size),
        'unit': unit,
    }
