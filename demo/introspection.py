"""Model drill-down for a (resume, title) pair — what the model is doing
internally when it scores one candidate title.

bert4rec — two views:
  * logit lens: the [MASK] hidden state after the embedding and after every
    encoder layer, projected through the (tied) output head. Shows the title's
    logit / in-domain probability / rank forming layer by layer, plus each
    layer's top titles — where a weird recommendation first appears.
  * attention: every layer's per-head attention matrix over the non-pad
    positions (queries × keys), so you can see what the [MASK] position (and
    every context token) is paying attention to.

The encoder forward is re-run manually (nn.TransformerEncoderLayer doesn't
return attention weights), reproducing the post-norm layer maths in eval mode;
the result is asserted against the real encoder output ('faithful' in the
response) so the displayed internals are guaranteed to be the model's own.

item2vec has no layers — the drill-down is the per-context-token cosine
against the title vector (which tokens pull the ranking toward this title).
"""

import numpy as np


def _short(token):
    return token.split(':', 1)[1] if ':' in token else token


def inspect_model(model, architecture, tokens, title):
    if architecture == 'bert4rec':
        return inspect_bert4rec(model, tokens, title)
    if architecture == 'item2vec':
        return inspect_item2vec(model, tokens, title)
    return {'error': f'no drill-down for architecture {architecture!r}'}


# ── bert4rec ─────────────────────────────────────────────────────────────────

def _layer_forward(layer, x, kpm):
    """One post-norm TransformerEncoderLayer step, returning (x, attn_weights).
    Identical to layer(x, src_key_padding_mask=kpm) in eval mode, but with
    need_weights=True so the per-head attention comes back."""
    import torch  # noqa: F401  (torch types flow through)
    sa_out, w = layer.self_attn(x, x, x, key_padding_mask=kpm,
                                need_weights=True, average_attn_weights=False)
    x = layer.norm1(x + layer.dropout1(sa_out))
    ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
    x = layer.norm2(x + layer.dropout2(ff))
    return x, w


def _logit_lens(model, hidden_mask, title_id, title_ids, idx2str, top_k=5):
    """Project one [MASK] hidden state through the tied head: the title's
    logit, its in-domain softmax probability and rank, and the layer's top
    titles."""
    import torch
    logits = model.head(hidden_mask)
    dom = logits[title_ids]
    probs = torch.softmax(dom, dim=-1)
    title_pos = (title_ids == title_id).nonzero()
    in_domain = len(title_pos) > 0
    entry = {'title_logit': round(float(logits[title_id]), 4)}
    if in_domain:
        p = int(title_pos[0, 0])
        entry['title_prob'] = round(float(probs[p]), 5)
        entry['title_rank'] = int((dom > dom[p]).sum()) + 1
    order = torch.argsort(dom, descending=True)[:top_k]
    entry['top_titles'] = [
        {'title': _short(idx2str[int(title_ids[i])]), 'prob': round(float(probs[i]), 5)}
        for i in order]
    return entry


def inspect_bert4rec(m, tokens, title):
    import torch

    vocab, model = m.vocab, m.model
    title_id = vocab.str2idx.get(title)
    if title_id is None:
        return {'error': f'{title} is not in this model\'s vocabulary'}

    known = [t for t in tokens if t in vocab.str2idx]
    if not known:
        return {'error': 'no in-vocabulary context tokens'}
    used = known[-(m._max_len - 1):]
    ids = vocab.encode(used)
    pad_n = m._max_len - 1 - len(ids)
    seq = [vocab.pad_id] * pad_n + ids + [vocab.mask_id]
    ids_t = torch.tensor(seq).unsqueeze(0)
    kpm = ids_t == vocab.pad_id
    labels = [_short(t) for t in used] + ['[MASK]']
    types = [t.split(':', 1)[0] for t in used] + ['MASK']

    model.eval()
    with torch.no_grad():
        pos = torch.arange(m._max_len)
        x = model.dropout(model.norm(model.item_emb(ids_t) + model.pos_emb(pos)))
        x0 = x
        states, attns = [x], []
        for layer in model.encoder.layers:
            x, w = _layer_forward(layer, x, kpm)
            states.append(x)
            attns.append(w[0])                        # (heads, L, L)
        # The displayed internals must be the model's own numbers.
        faithful = bool(torch.allclose(
            x, model.encoder(x0, src_key_padding_mask=kpm), atol=1e-4))

        title_ids = m._title_ids
        layers = []
        for i, h in enumerate(states):
            entry = {'layer': i,
                     'name': 'embedding' if i == 0 else f'encoder layer {i}'}
            entry.update(_logit_lens(model, h[0, -1], title_id, title_ids, vocab.idx2str))
            if i > 0:
                # per-head attention over the non-pad positions only
                w = attns[i - 1][:, pad_n:, pad_n:].numpy()
                entry['attention'] = np.round(w, 4).tolist()
            layers.append(entry)

    return {
        'architecture': 'bert4rec',
        'title': title,
        'labels': labels,          # axis labels for the attention matrices
        'token_types': types,
        'n_heads': int(model.encoder.layers[0].self_attn.num_heads),
        'faithful': faithful,
        'layers': layers,
    }


# ── item2vec ─────────────────────────────────────────────────────────────────

def inspect_item2vec(m, tokens, title):
    wv = m._model.wv
    if title not in wv:
        return {'error': f'{title} is not in this model\'s vocabulary'}
    known = [t for t in tokens if t in wv]
    ctx = known[-m._context_last_n:]
    if not ctx:
        return {'error': 'no in-vocabulary context tokens'}

    def unit(v):
        return v / (np.linalg.norm(v) + 1e-9)

    tv = unit(wv[title])
    contributions = [{'token': t, 'type': t.split(':', 1)[0],
                      'value': _short(t),
                      'cosine': round(float(unit(wv[t]) @ tv), 4)}
                     for t in ctx]
    ctx_vec = unit(np.mean([wv[t] for t in ctx], axis=0))
    return {
        'architecture': 'item2vec',
        'title': title,
        'score': round(float(ctx_vec @ tv), 4),   # the ranking cosine itself
        'contributions': contributions,           # per-token pull toward the title
    }
