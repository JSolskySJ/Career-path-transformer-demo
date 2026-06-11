"""The SJ ranking domain — the set of W_TITLE tokens a model is allowed to rank.

Production restricts next-title ranking to the SwipeJobs-recommendable titles
(the `in_ranking_domain` column of the exported vocab CSV), not the full trained
title vocabulary. Both demo models apply the same restriction so their live
output matches the production predictions CSVs.

Returns None when no vocab CSV is staged, in which case the models fall back to
ranking over their entire title vocabulary (the demo's original behaviour).
"""

import os

from demo import config

_CACHE = {}


def ranking_domain():
    if 'domain' not in _CACHE:
        _CACHE['domain'] = _load()
    return _CACHE['domain']


def _load():
    if not os.path.exists(config.VOCAB_CSV):
        return None
    import pandas as pd
    df = pd.read_csv(config.VOCAB_CSV)
    if 'in_ranking_domain' not in df.columns:
        return None
    domain = set(df.loc[df['in_ranking_domain'] == True, 'token'])
    return domain or None
