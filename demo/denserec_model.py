"""DenseRec architecture + demo checkpoint loading.

Weight-compatible copy of DenseRecBERT4Rec in
datawarehouse-ai/models/career_path_transformer_denserec.py — BERT4Rec plus a
frozen MiniLM content matrix (one 384-d vector per renderable vocab token), a
learned 384→d_model projection, and dual-path training. At inference the
in-vocab path is IDENTICAL to BERT4Rec; the DenseRec difference the demo
surfaces is the **unseen-token catch-all**: renderable out-of-vocabulary
context tokens are injected as projected MiniLM vectors instead of being
dropped, so brand-new titles/skills actually influence the prediction.

The MiniLM encoder loads lazily via `transformers` on first unseen token;
without the package the model degrades gracefully to BERT4Rec behaviour
(OOV dropped) and reports why.
"""

import numpy as np
import torch
import torch.nn as nn

from demo.bert4rec_model import BERT4Rec, Bert4RecModel
from demo.confidence import distribution_confidence

_CONTENT_ENCODER = 'sentence-transformers/all-MiniLM-L6-v2'

# Natural-language renderings — keep in sync with the training module.
_PREFIX_RENDERINGS = {
    'W_TITLE:':    'job title: ',
    'S_SKILL:':    'skill: ',
    'E_MAJOR:':    'field of study: ',
    'E_DEGREE:':   'degree: ',
    'E_TYPE:':     'school type: ',
    'E_LEVEL:':    'education level: ',
    'W_COMPANY:':  'company: ',
    'W_INDUSTRY:': 'industry: ',
    'W_SPEC:':     'specialisation: ',
    'W_ROLE:':     'job role: ',
    'W_SUBROLE:':  'job role: ',
    'W_DESC:':     'job description: ',
}


def _render_token(token):
    for prefix, phrase in _PREFIX_RENDERINGS.items():
        if token.startswith(prefix):
            return phrase + token[len(prefix):]
    return None


class _DensePathMixin:
    """The dense members shared by both backbones — mirrors training's
    DenseRecBERT4Rec / DenseRecModernBERT4Rec (which duplicate the body; the
    demo uses a mixin since unpickling maps class NAMES to these classes and
    only the state-dict layout must match). At eval the dual path never fires
    (training-only randomness); what remains is the unseen-content injection,
    carried on an attribute stash consumed by the _input_embeddings hook so it
    composes with ANY backbone's _encode."""

    def _init_dense(self, vocab_size: int, content_dim: int, dense_path_p: float):
        d_model = self.item_emb.embedding_dim
        self.register_buffer('content', torch.zeros(vocab_size, content_dim))
        self.register_buffer('has_content', torch.zeros(vocab_size, dtype=torch.bool))
        self.proj = nn.Linear(content_dim, d_model)
        self.dense_path_p = float(dense_path_p)

    def _input_embeddings(self, ids):
        emb = self.item_emb(ids)
        unseen = getattr(self, '_unseen_stash', None)
        if unseen is not None:
            positions, vectors = unseen
            emb = emb.clone()
            emb[positions] = self.proj(vectors.to(emb))
        return emb

    def encode_with_injection(self, ids, unseen_content=None):
        self._unseen_stash = unseen_content
        try:
            return self._encode(ids)
        finally:
            self._unseen_stash = None

    def forward(self, ids, unseen_content=None):
        self._unseen_stash = unseen_content
        try:
            return super().forward(ids)
        finally:
            self._unseen_stash = None


class DenseRecBERT4Rec(_DensePathMixin, BERT4Rec):
    """DenseRec on the stock post-norm backbone."""

    def __init__(self, *, content_dim: int = 384, dense_path_p: float = 0.5,
                 vocab_size: int = 2, **kw):
        super().__init__(vocab_size=vocab_size, **kw)
        self._init_dense(vocab_size, content_dim, dense_path_p)


from demo.modernbert_model import ModernBERT4Rec


class DenseRecModernBERT4Rec(_DensePathMixin, ModernBERT4Rec):
    """DenseRec on the ModernBERT backbone (pre-norm/RoPE/GeGLU)."""

    def __init__(self, *, content_dim: int = 384, dense_path_p: float = 0.5,
                 vocab_size: int = 2, **kw):
        super().__init__(vocab_size=vocab_size, **kw)
        self._init_dense(vocab_size, content_dim, dense_path_p)


class DenseRecModel(Bert4RecModel):
    """Inference wrapper — Bert4RecModel plus the unseen-token catch-all."""

    ARCH_CLS = DenseRecBERT4Rec
    _minilm = None                       # (tokenizer, encoder), lazy

    def __init__(self, artifact_dir: str, vocab_csv: str = None):
        super().__init__(artifact_dir, vocab_csv=vocab_csv)
        self._unseen_cache = {}          # token -> np content vector

    # Bert4RecModel builds a plain BERT4Rec from config.json; give it the
    # DenseRec module instead (backbone + extra ctor args read off the config).
    def _build_module(self):
        cls = (DenseRecModernBERT4Rec if self.params.get('backbone') == 'modernbert'
               else DenseRecBERT4Rec)
        return cls(
            vocab_size=self.vocab.size,
            d_model=self.params['d_model'],
            n_layers=self.params['n_layers'],
            n_heads=self.params['n_heads'],
            max_len=self.params['max_len'],
            dropout=self.params['dropout'],
            pad_id=self.vocab.pad_id,
            content_dim=self.params.get('content_dim', 384),
            dense_path_p=self.params.get('dense_path_p', 0.5),
        )

    # ── Unseen-token catch-all ───────────────────────────────────────────────

    def _encode_unseen(self, tokens: list):
        """(len(tokens), d_c) MiniLM vectors for renderable tokens (cached).
        Raises ImportError when transformers isn't installed."""
        missing = [t for t in dict.fromkeys(tokens) if t not in self._unseen_cache]
        if missing:
            if DenseRecModel._minilm is None:
                from transformers import AutoModel, AutoTokenizer
                try:
                    # local HF cache first — no hub round-trips on every load
                    DenseRecModel._minilm = (
                        AutoTokenizer.from_pretrained(_CONTENT_ENCODER, local_files_only=True),
                        AutoModel.from_pretrained(_CONTENT_ENCODER, local_files_only=True).eval())
                except OSError:
                    DenseRecModel._minilm = (
                        AutoTokenizer.from_pretrained(_CONTENT_ENCODER),
                        AutoModel.from_pretrained(_CONTENT_ENCODER).eval())
            tokenizer, encoder = DenseRecModel._minilm
            with torch.no_grad():
                batch = tokenizer([_render_token(t) for t in missing],
                                  padding=True, truncation=True, return_tensors='pt')
                hidden = encoder(**batch).last_hidden_state
                mask = batch['attention_mask'].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                vecs = torch.nn.functional.normalize(pooled, dim=-1).numpy()
            self._unseen_cache.update(zip(missing, vecs))
        return np.stack([self._unseen_cache[t] for t in tokens])

    def _context_row(self, context: list):
        """Encode a context keeping renderable OOV tokens as injectable
        positions. Returns (seq_ids, injection or None, used, injected,
        unknown) — injection = (bool positions (1, max_len), vectors)."""
        window = self._max_len - 1
        kept = []                                    # (id, token-to-inject | None)
        unknown = []
        for t in context:
            if t in self.vocab.str2idx:
                kept.append((self.vocab.str2idx[t], None))
            elif _render_token(t) is not None:
                kept.append((self.vocab.mask_id, t))
            else:
                unknown.append(t)
        kept = kept[-window:]
        injected = [t for _, t in kept if t is not None]
        if injected:
            try:
                vectors = torch.from_numpy(self._encode_unseen(injected))
            except ImportError:
                # no transformers — degrade to BERT4Rec behaviour (drop OOV)
                unknown += injected
                kept = [(i, None) for i, t in kept if t is None]
                injected = []
        n_pad = self._max_len - 1 - len(kept)
        seq = ([self.vocab.pad_id] * n_pad + [i for i, _ in kept] + [self.vocab.mask_id])
        injection = None
        if injected:
            positions = torch.zeros(1, self._max_len, dtype=torch.bool)
            for col, (_, t) in enumerate(kept):
                if t is not None:
                    positions[0, n_pad + col] = True
            injection = (positions, vectors)
        used = [t if t is not None else self.vocab.idx2str[i] for i, t in kept]
        return seq, injection, used, injected, unknown

    def _mask_hidden(self, context: list):
        """[MASK] hidden state with unseen-token injection.
        Returns (h, used, injected, unknown) — h is None with no usable input."""
        seq, injection, used, injected, unknown = self._context_row(context)
        if len(used) == 0:
            return None, [], [], unknown
        with torch.no_grad():
            h = self.model.encode_with_injection(torch.tensor(seq).unsqueeze(0),
                                                 unseen_content=injection)
        return h[0, -1], used, injected, unknown

    # ── Overrides (injection-aware) ──────────────────────────────────────────

    def context_vector(self, context: list):
        h, used, injected, unknown = self._mask_hidden(context)
        if h is None:
            return None, [], unknown
        return h.numpy().copy(), used, unknown

    def rank_titles(self, context: list, top_k: int = 10, allowed: set = None,
                    scoring: str = 'softmax') -> dict:
        h, used, injected, unknown = self._mask_hidden(context)
        base = {'used_tokens': used, 'unknown_tokens': unknown,
                'injected_tokens': injected, 'scoring': scoring}
        if h is None:
            return {**base, 'predictions': [], 'confidence': None, 'n_ranked': 0}

        title_ids = self._title_ids
        if allowed is not None:
            keep = [i for i, t in zip(self._title_ids.tolist(), self.title_vocab)
                    if t in allowed]
            if not keep:
                return {**base, 'predictions': [], 'confidence': None, 'n_ranked': 0}
            title_ids = torch.tensor(keep)
        tid = title_ids.numpy()

        with torch.no_grad():
            emb = self.model.item_emb.weight[title_ids]
            if scoring == 'cosine':
                q = h / (h.norm() + 1e-9)
                emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-9)
                scores = (emb @ q).numpy()
                unit, conf = 'cosine', scores
            else:
                logits = emb @ h
                scores = torch.softmax(logits, dim=-1).numpy()
                unit, conf = 'prob', scores
        order = np.argsort(-scores)[:top_k]
        preds = [{'token': self.vocab.idx2str[tid[j]], 'score': float(scores[j])}
                 for j in order]
        return {**base, 'predictions': preds,
                'confidence': distribution_confidence(conf, unit),
                'n_ranked': len(title_ids)}
