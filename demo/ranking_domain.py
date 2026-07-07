"""The ranking domain — the set of W_TITLE tokens a model is allowed to rank.

For this demo we rank over the taxonomy L3 SuperTitles only (the
`is_taxonomy_l3` column of the exported vocab CSV): every prediction is a clean
taxonomy title, and all other titles are hidden. If that column is missing we
fall back to the SwipeJobs ranking domain (`in_ranking_domain`), and if there's
no vocab CSV at all the models rank over their entire title vocabulary.

The chosen column is configurable via CPT_RANKING_DOMAIN_COL (default
`is_taxonomy_l3`).
"""

import os

from demo import config

_CACHE = {}
_PREFERRED_COL = os.environ.get('CPT_RANKING_DOMAIN_COL', 'is_taxonomy_l3')
_FALLBACK_COLS = ['in_ranking_domain']


def ranking_domain(vocab_csv: str = None):
    """Domain from the given vocab CSV (default: the legacy global one).
    Cached per path. None when the CSV is missing or has no flag columns."""
    path = vocab_csv or config.VOCAB_CSV
    if path not in _CACHE:
        _CACHE[path] = _load(path)
    return _CACHE[path]


def _load(path):
    if not os.path.exists(path):
        return None
    import pandas as pd
    df = pd.read_csv(path)
    for col in [_PREFERRED_COL, *_FALLBACK_COLS]:
        if col in df.columns:
            domain = set(df.loc[df[col] == True, 'token'])
            if domain:
                return domain
    return None
