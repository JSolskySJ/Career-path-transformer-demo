"""BERT4Rec architecture + demo checkpoint loading.

The architecture is an exact copy of BERT4Rec in
datawarehouse-ai/models/career_path_transformer_bert4rec.py so checkpoints are
weight-compatible. The demo loads a plain local checkpoint —
artifacts/bert4rec/{model.pt, vocab.json, config.json} — written either by
scripts/import_mlflow_artifacts.py (from a production run's model.pth +
logged vocab.json) or by scripts/train_bert4rec.py (local fallback).

Ranking mirrors CareerPathBERT4RecModel._rank_titles: left-pad the encoded
context to max_len - 1, append [MASK], softmax the mask-position logits over
W_TITLE tokens only.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn

from demo import config
from demo.tokens import W_TITLE_PREFIX, ALL_PREFIXES
from demo.ranking_domain import ranking_domain
from demo.confidence import distribution_confidence

PAD_TOKEN  = '[PAD]'
MASK_TOKEN = '[MASK]'


class BERT4Rec(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=2,
                 max_len=64, dropout=0.2, pad_id=0):
        super().__init__()
        self.pad_id    = pad_id
        self.item_emb  = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.norm      = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.item_emb.weight   # weight tying

    def forward(self, ids):
        pos = torch.arange(ids.size(1), device=ids.device)
        h   = self.dropout(self.norm(self.item_emb(ids) + self.pos_emb(pos)))
        h   = self.encoder(h, src_key_padding_mask=(ids == self.pad_id))
        return self.head(h)


class Vocab:
    def __init__(self, idx2str: list):
        self.idx2str = idx2str
        self.str2idx = {tok: idx for idx, tok in enumerate(idx2str)}
        self.pad_id  = self.str2idx[PAD_TOKEN]
        self.mask_id = self.str2idx[MASK_TOKEN]

    @classmethod
    def build(cls, sequences, min_count: int = 5):
        from collections import Counter
        counts = Counter(tok for seq in sequences for tok in seq)
        items  = [tok for tok, n in counts.most_common() if n >= min_count]
        return cls([PAD_TOKEN, MASK_TOKEN] + items)

    @property
    def size(self) -> int:
        return len(self.idx2str)

    def encode(self, tokens: list) -> list:
        return [self.str2idx[t] for t in tokens if t in self.str2idx]


class Bert4RecModel:
    """Inference wrapper around a demo checkpoint directory."""

    def __init__(self, artifact_dir: str = None, vocab_csv: str = None):
        artifact_dir = artifact_dir or config.BERT4REC_DIR
        self._vocab_csv = vocab_csv
        with open(os.path.join(artifact_dir, 'config.json')) as f:
            self.params = json.load(f)
        with open(os.path.join(artifact_dir, 'vocab.json')) as f:
            self.vocab = Vocab(json.load(f))

        self._max_len = self.params['max_len']
        self.model = BERT4Rec(
            vocab_size=self.vocab.size,
            d_model=self.params['d_model'],
            n_layers=self.params['n_layers'],
            n_heads=self.params['n_heads'],
            max_len=self._max_len,
            dropout=self.params['dropout'],
            pad_id=self.vocab.pad_id,
        )
        state = torch.load(os.path.join(artifact_dir, 'model.pt'),
                           map_location='cpu', weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        # Rank over the run's own ranking domain when available (matches
        # production), else all trained title tokens.
        domain = ranking_domain(vocab_csv)
        all_titles = [t for t in self.vocab.idx2str if t.startswith(W_TITLE_PREFIX)]
        self.full_title_count = len(all_titles)
        self.restricted = domain is not None
        self.title_vocab = ([t for t in all_titles if t in domain]
                            if domain is not None else all_titles)
        self._title_ids  = torch.tensor([self.vocab.str2idx[t] for t in self.title_vocab])
        with torch.no_grad():
            vecs = self.model.item_emb.weight[self._title_ids].numpy().copy()
        # L2-normalised: rows are unit vectors so dot product = cosine
        self.title_matrix = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    @classmethod
    def load_if_available(cls, artifact_dir: str = None, vocab_csv: str = None):
        artifact_dir = artifact_dir or config.BERT4REC_DIR
        if all(os.path.exists(os.path.join(artifact_dir, f))
               for f in ('model.pt', 'vocab.json', 'config.json')):
            return cls(artifact_dir, vocab_csv=vocab_csv)
        return None

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self.vocab.size

    @property
    def vector_size(self) -> int:
        return self.params['d_model']

    def vocab_by_prefix(self) -> dict:
        out = {p.rstrip(':'): [] for p in ALL_PREFIXES}
        for tok in self.vocab.idx2str:
            if ':' not in tok:
                continue
            ttype, value = tok.split(':', 1)
            if ttype in out:
                out[ttype].append(value)
        return out

    def knows(self, token: str) -> bool:
        return token in self.vocab.str2idx

    def vector(self, token: str):
        """Input embedding (item_emb row) — the token's position in the
        model's learned item space."""
        idx = self.vocab.str2idx.get(token)
        if idx is None:
            return None
        with torch.no_grad():
            return self.model.item_emb.weight[idx].numpy().copy()

    # ── Ranking ──────────────────────────────────────────────────────────────

    def context_vector(self, context: list):
        """Transformer hidden state at the [MASK] position — the model's query
        vector for "what comes next". Returns (vec, used_tokens, unknown_tokens)."""
        known   = [t for t in context if t in self.vocab.str2idx]
        unknown = [t for t in context if t not in self.vocab.str2idx]
        if not known:
            return None, [], unknown
        used = known[-(self._max_len - 1):]
        ids  = self.vocab.encode(used)
        seq  = [self.vocab.pad_id] * (self._max_len - 1 - len(ids)) + ids + [self.vocab.mask_id]
        ids_t = torch.tensor(seq).unsqueeze(0)
        with torch.no_grad():
            pos = torch.arange(ids_t.size(1))
            h = self.model.dropout(self.model.norm(
                self.model.item_emb(ids_t) + self.model.pos_emb(pos)))
            h = self.model.encoder(h, src_key_padding_mask=(ids_t == self.vocab.pad_id))
        return h[0, -1].numpy().copy(), used, unknown

    def rank_titles(self, context: list, top_k: int = 10, allowed: set = None) -> dict:
        """Rank titles by [MASK] softmax. ``allowed`` (optional set of W_TITLE
        tokens) PROJECTS the softmax onto that subset — the logits are masked
        to the allowed titles and renormalised, a true restricted-domain
        prediction rather than a post-hoc filter."""
        known   = [t for t in context if t in self.vocab.str2idx]
        unknown = [t for t in context if t not in self.vocab.str2idx]
        if not known:
            return {'predictions': [], 'used_tokens': [], 'unknown_tokens': unknown,
                    'confidence': None, 'n_ranked': 0}
        used = known[-(self._max_len - 1):]
        ids  = self.vocab.encode(used)
        seq  = [self.vocab.pad_id] * (self._max_len - 1 - len(ids)) + ids + [self.vocab.mask_id]

        title_ids = self._title_ids
        if allowed is not None:
            keep = [i for i, t in zip(self._title_ids.tolist(), self.title_vocab)
                    if t in allowed]
            if not keep:
                return {'predictions': [], 'used_tokens': used, 'unknown_tokens': unknown,
                        'confidence': None, 'n_ranked': 0}
            title_ids = torch.tensor(keep)
        title_mask = torch.zeros(self.vocab.size, dtype=torch.bool)
        title_mask[title_ids] = True
        with torch.no_grad():
            logits = self.model(torch.tensor(seq).unsqueeze(0))[0, -1]
            logits = logits.masked_fill(~title_mask, float('-inf'))
            probs  = torch.softmax(logits, dim=-1).numpy()

        # Order over the (possibly projected) title ids only — a full-vocab
        # argsort would let zero-prob masked entries pad out a small domain.
        tid = title_ids.numpy()
        order = tid[np.argsort(-probs[tid])][:top_k]
        preds = [{'token': self.vocab.idx2str[i], 'score': float(probs[i])} for i in order]
        # Confidence over the (possibly projected) domain — probs sum to 1 there.
        title_probs = probs[tid]
        return {'predictions': preds, 'used_tokens': used, 'unknown_tokens': unknown,
                'confidence': distribution_confidence(title_probs, 'prob'),
                'n_ranked': len(title_ids)}
