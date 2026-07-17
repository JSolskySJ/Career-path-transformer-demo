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
    """One post-norm TransformerEncoderLayer step, returning
    (x_out, attn_weights, attn_block_out, x_after_attn) — identical maths to
    layer(x, src_key_padding_mask=kpm) in eval mode, but with need_weights=True
    and the residual-stream intermediates exposed for the trace."""
    import torch  # noqa: F401  (torch types flow through)
    sa_out, w = layer.self_attn(x, x, x, key_padding_mask=kpm,
                                need_weights=True, average_attn_weights=False)
    x1 = layer.norm1(x + layer.dropout1(sa_out))
    ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x1))))
    x2 = layer.norm2(x1 + layer.dropout2(ff))
    return x2, w, sa_out, x1


def _head_weights(layer, d_model, n_heads):
    """Per-head slices of the attention projections, display-shaped (head_dim ×
    d_model): W_Q, W_K, W_V rows for this head, and the head's W_O columns
    (transposed) — the matrix that writes the head's output back into the
    residual stream."""
    import numpy as np
    hd = d_model // n_heads
    W = layer.self_attn.in_proj_weight.detach().numpy()
    Wq, Wk, Wv = W[:d_model], W[d_model:2 * d_model], W[2 * d_model:]
    Wo = layer.self_attn.out_proj.weight.detach().numpy()
    out = []
    for h in range(n_heads):
        s = slice(h * hd, (h + 1) * hd)
        out.append({
            'wq': np.round(Wq[s], 3).tolist(),
            'wk': np.round(Wk[s], 3).tolist(),
            'wv': np.round(Wv[s], 3).tolist(),
            'wo_t': np.round(Wo[:, s].T, 3).tolist(),
        })
    return out


def _mask_value_weighted(layer, x_in, attn, pad_n, d_model, n_heads):
    """What each source token ACTUALLY contributes to the [MASK] query through
    each head: attention weight × the norm of the token's value vector
    projected through the head's W_O — attention alone ignores how big the
    written vector is. Returns (heads, L_used) norms."""
    import numpy as np
    import torch
    hd = d_model // n_heads
    sa = layer.self_attn
    Wv = sa.in_proj_weight[2 * d_model:]
    bv = sa.in_proj_bias[2 * d_model:] if sa.in_proj_bias is not None else 0
    with torch.no_grad():
        v = x_in[0] @ Wv.t() + bv                        # (L_full, d)
        rows = []
        for h in range(n_heads):
            s = slice(h * hd, (h + 1) * hd)
            proj = v[:, s] @ sa.out_proj.weight[:, s].t()  # what j writes if fully attended
            norms = proj.norm(dim=1)                       # (L_full,)
            contrib = attn[h, -1] * norms                  # × [MASK]-row attention
            rows.append(np.round(contrib[pad_n:].numpy(), 4).tolist())
    return rows


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

    d_model = model.item_emb.weight.shape[1]
    n_heads = int(model.encoder.layers[0].self_attn.num_heads)

    model.eval()
    with torch.no_grad():
        pos = torch.arange(m._max_len)
        x = model.dropout(model.norm(model.item_emb(ids_t) + model.pos_emb(pos)))
        x0 = x
        # Residual-stream trace: (stage name, (1, L_full, d)) at every point in
        # the forward pass, so any token can be followed through the model.
        trace_stages = [('embedding + position', x0)]
        states, attns, layer_extras = [x], [], []
        for li, layer in enumerate(model.encoder.layers, start=1):
            x_in = x
            x, w, sa_out, x_after_attn = _layer_forward(layer, x, kpm)
            states.append(x)
            attns.append(w[0])                        # (heads, L, L)
            trace_stages += [(f'L{li} · attention output', sa_out),
                             (f'L{li} · after attention', x_after_attn),
                             (f'L{li} · after FFN', x)]
            layer_extras.append({
                'heads': _head_weights(layer, d_model, n_heads),
                'mask_value_weighted': _mask_value_weighted(
                    layer, x_in, w[0], pad_n, d_model, n_heads),
                'ffn': {
                    'w1': np.round(layer.linear1.weight.detach().numpy(), 3).tolist(),
                    'w2': np.round(layer.linear2.weight.detach().numpy(), 3).tolist(),
                },
            })
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
                entry.update(layer_extras[i - 1])
            layers.append(entry)

        # Per-token trace payload: each stage's hidden vectors (pad-trimmed)
        # plus a per-token logit-lens readout (top title at that depth).
        trace = {'stages': [], 'vectors': [], 'top1': []}
        for name, h in trace_stages:
            vecs = h[0, pad_n:]                       # (L_used, d)
            logits = model.head(vecs)[:, title_ids]   # (L_used, |titles|)
            probs = torch.softmax(logits, dim=-1)
            top = probs.argmax(dim=-1)
            trace['stages'].append(name)
            trace['vectors'].append(np.round(vecs.numpy(), 3).tolist())
            trace['top1'].append([
                {'title': _short(vocab.idx2str[int(title_ids[int(t)])]),
                 'prob': round(float(probs[j, int(t)]), 4)}
                for j, t in enumerate(top)])

        # Leave-one-out influence: re-run the model with each context token
        # removed (one batched forward) and measure the change in the clicked
        # title's logit and in-domain probability. SIGNED and faithful:
        # positive = the token pushes the prediction TOWARD this title,
        # negative = it pushes it away.
        variants = []
        for j in range(len(ids)):
            v = ids[:j] + ids[j + 1:]
            variants.append([vocab.pad_id] * (m._max_len - 1 - len(v)) + v + [vocab.mask_id])
        batch = torch.tensor([seq] + variants)
        out = model(batch)[:, -1, :]                  # [MASK] logits per variant
        dom = out[:, title_ids]
        dom_probs = torch.softmax(dom, dim=-1)
        t_pos = (title_ids == title_id).nonzero()
        t_col = int(t_pos[0, 0]) if len(t_pos) else None
        base_logit = float(out[0, title_id])
        base_prob = float(dom_probs[0, t_col]) if t_col is not None else None
        ablation = {
            'base_logit': round(base_logit, 4),
            'base_prob': round(base_prob, 5) if base_prob is not None else None,
            'tokens': [{
                'i': j,      # index into labels/token_types (context positions)
                'delta_logit': round(base_logit - float(out[1 + j, title_id]), 4),
                'delta_prob': (round(base_prob - float(dom_probs[1 + j, t_col]), 5)
                               if t_col is not None else None),
            } for j in range(len(ids))],
        }

    return {
        'architecture': 'bert4rec',
        'title': title,
        'labels': labels,          # axis labels for the attention matrices
        'token_types': types,
        'n_heads': n_heads,
        'd_model': int(d_model),
        'faithful': faithful,
        'ablation': ablation,
        'layers': layers,
        'trace': trace,
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
