"""SJ / taxonomy-level flags for job titles, from artifacts/title_flags.json
(built by scripts/build_title_flags.py against the raw taxonomy exports).

Lookup keys use the ETL's taxonomy normalisation so free-text and L3-rolled
titles both resolve: strip parenthesised text and non-ASCII, lowercase,
non-alphanumeric -> space, collapse whitespace.
"""

import json
import os
import re

from demo import config

FLAGS_JSON = os.path.join(config.ARTIFACTS_DIR, 'title_flags.json')

_cache = None


def normalise_title(value) -> str:
    """Python port of normaliseForTaxonomy in CareerPathTransformer.scala."""
    if value is None:
        return ''
    s = re.sub(r'\([^)]*\)', '', str(value))
    s = re.sub(r'[^\x00-\x7F]+', '', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s.lower())
    return re.sub(r'\s+', ' ', s).strip()


def _load():
    global _cache
    if _cache is None:
        if os.path.exists(FLAGS_JSON):
            with open(FLAGS_JSON) as f:
                blob = json.load(f)
            _cache = {'tax': blob.get('tax', {}), 'sj': set(blob.get('sj', []))}
        else:
            _cache = {'tax': {}, 'sj': set()}
    return _cache


def flags_for(title_value: str) -> dict:
    """{'sj': bool, 'tax': 'L3'|'L4'|'L5'|None} for one title (no prefix)."""
    flags = _load()
    key = normalise_title(title_value)
    return {'sj': key in flags['sj'], 'tax': flags['tax'].get(key)}
