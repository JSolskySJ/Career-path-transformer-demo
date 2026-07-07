"""Item2vec (gensim Word2Vec) loading and ranking.

Ranking mirrors CareerPathItem2VecModel._rank_titles in datawarehouse-ai:
mean of the last N known context-token vectors, cosine similarity against the
L2-normalised W_TITLE matrix. Hubness correction (center_titles) is off, as in
the training default.
"""

import numpy as np

from demo import config
from demo.tokens import W_TITLE_PREFIX, ALL_PREFIXES
from demo.ranking_domain import ranking_domain
from demo.confidence import distribution_confidence


class Item2VecModel:

    def __init__(self, bin_path: str = None, context_last_n: int = None,
                 vocab_csv: str = None):
        from gensim.models import Word2Vec
        self.bin_path = bin_path or config.ITEM2VEC_BIN
        self._model = Word2Vec.load(self.bin_path)
        self._context_last_n = context_last_n or config.CONTEXT_LAST_N

        # Rank over the run's own ranking domain when available (matches
        # production), else the full trained title vocabulary.
        domain = ranking_domain(vocab_csv)
        all_titles = [t for t in self._model.wv.index_to_key
                      if t.startswith(W_TITLE_PREFIX)]
        self.full_title_count = len(all_titles)
        self.restricted = domain is not None
        self.title_vocab = ([t for t in all_titles if t in domain]
                            if domain is not None else all_titles)
        vecs = np.stack([self._model.wv[t] for t in self.title_vocab])
        # L2-normalised: rows are unit vectors so dot product = cosine
        self.title_matrix = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self._model.wv)

    @property
    def vector_size(self) -> int:
        return self._model.wv.vector_size

    def vocab_by_prefix(self) -> dict:
        """{'W_TITLE': [values...], ...} in vocab (frequency) order, for the
        resume-builder autocomplete."""
        out = {p.rstrip(':'): [] for p in ALL_PREFIXES}
        for tok in self._model.wv.index_to_key:
            if ':' not in tok:
                continue
            ttype, value = tok.split(':', 1)
            if ttype in out:
                out[ttype].append(value)
        return out

    def knows(self, token: str) -> bool:
        return token in self._model.wv

    def vector(self, token: str):
        return self._model.wv[token] if token in self._model.wv else None

    # ── Ranking ──────────────────────────────────────────────────────────────

    def context_vector(self, context: list):
        """Mean of the last N known token vectors (the ranking query vector),
        L2-normalised. Returns (vec, used_tokens, unknown_tokens)."""
        known   = [t for t in context if t in self._model.wv]
        unknown = [t for t in context if t not in self._model.wv]
        ctx = known[-self._context_last_n:]
        if not ctx:
            return None, [], unknown
        vec = np.mean([self._model.wv[t] for t in ctx], axis=0)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec, ctx, unknown

    def rank_titles(self, context: list, top_k: int = 10, allowed: set = None) -> dict:
        """Rank titles by cosine. ``allowed`` (optional set of W_TITLE tokens)
        projects the ranking onto that subset — for cosine this is an exact
        post-filter, no renormalisation needed."""
        vec, used, unknown = self.context_vector(context)
        if vec is None:
            return {'predictions': [], 'used_tokens': [], 'unknown_tokens': unknown,
                    'confidence': None, 'n_ranked': 0}
        scores = self.title_matrix @ vec
        vocab = self.title_vocab
        if allowed is not None:
            idx = [i for i, t in enumerate(vocab) if t in allowed]
            vocab = [vocab[i] for i in idx]
            scores = scores[idx]
        if not vocab:
            return {'predictions': [], 'used_tokens': used, 'unknown_tokens': unknown,
                    'confidence': None, 'n_ranked': 0}
        order = np.argsort(-scores)[:top_k]
        preds = [{'token': vocab[i], 'score': float(scores[i])} for i in order]
        return {'predictions': preds, 'used_tokens': used, 'unknown_tokens': unknown,
                'confidence': distribution_confidence(scores, 'cosine'),
                'n_ranked': len(vocab)}
