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
    if architecture in ('bert4rec', 'modernbert', 'denserec'):
        # modernbert/denserec are BERT4Rec-family — the full trace applies
        # (the introspection dispatches per block type internally); denserec's
        # injected OOV tokens are not traced (in-vocab view only).
        result = inspect_bert4rec(model, tokens, title)
        if architecture == 'denserec' and 'error' not in result:
            result['denserec'] = _denserec_alignment(model, tokens, title)
        return result
    if architecture == 'item2vec':
        return inspect_item2vec(model, tokens, title)
    return {'error': f'architecture {architecture!r} has no drill-down in the demo',
            'not_displayable': True}


def _denserec_alignment(m, tokens, title):
    """DenseRec dual-path view: for each in-vocab context token (and the
    clicked title), the cosine between its learned ID embedding and its
    projected MiniLM content vector — how far the two paths agree. Low
    alignment on a title means the content path would place it somewhere very
    different from where training placed its ID."""
    import torch
    model, vocab = m.model, m.vocab
    rows = []
    with torch.no_grad():
        proj = model.proj(model.content)                 # (|V|, d)
        for tok in list(dict.fromkeys(tokens)) + [title]:
            idx = vocab.str2idx.get(tok)
            if idx is None or not bool(model.has_content[idx]):
                continue
            a = model.item_emb.weight[idx]
            b = proj[idx]
            cos = float((a @ b) / (a.norm() * b.norm() + 1e-9))
            rows.append({'token': tok, 'type': tok.split(':', 1)[0],
                         'is_title': tok == title, 'cosine': round(cos, 4)})
    return {'dense_path_p': m.params.get('dense_path_p'),
            'content_dim': m.params.get('content_dim'),
            'alignment': rows}


# ── bert4rec ─────────────────────────────────────────────────────────────────

def _layer_forward(layer, x, kpm):
    """One post-norm TransformerEncoderLayer step, returning
    (x_out, attn_weights, attn_block_out, x_after_attn, qkv) — identical maths
    to layer(x, src_key_padding_mask=kpm) in eval mode, but with
    need_weights=True and the residual-stream intermediates exposed for the
    trace. qkv is the per-head (1, H, L, hd) projected queries/keys/values the
    attention actually used (recomputed from in_proj; the weights returned by
    self_attn are asserted faithful downstream)."""
    import torch  # noqa: F401  (torch types flow through)
    sa_out, w = layer.self_attn(x, x, x, key_padding_mask=kpm,
                                need_weights=True, average_attn_weights=False)
    sa = layer.self_attn
    d = x.shape[-1]
    H = sa.num_heads
    b = sa.in_proj_bias if sa.in_proj_bias is not None else torch.zeros(3 * d)
    heads = []
    for i, (W, bb) in enumerate(zip(sa.in_proj_weight.chunk(3), b.chunk(3))):
        t = (x @ W.t() + bb).view(x.shape[0], x.shape[1], H, d // H)
        heads.append(t.transpose(1, 2))                # (1, H, L, hd)
    x1 = layer.norm1(x + layer.dropout1(sa_out))
    ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x1))))
    x2 = layer.norm2(x1 + layer.dropout2(ff))
    return x2, w, sa_out, x1, tuple(heads)


def _is_modern_block(layer) -> bool:
    return hasattr(layer, 'qkv')          # ModernBertBlock; stock has self_attn


def _attn_projections(layer, d_model):
    """(W_Q, W_K, W_V, W_O, value_bias) for either backbone's attention —
    stock nn.MultiheadAttention packs QKV rows in in_proj_weight, ModernBERT
    in the bias-free qkv Linear."""
    if _is_modern_block(layer):
        W = layer.qkv.weight
        return W[:d_model], W[d_model:2 * d_model], W[2 * d_model:], \
            layer.attn_out.weight, None
    sa = layer.self_attn
    W = sa.in_proj_weight
    bv = sa.in_proj_bias[2 * d_model:] if sa.in_proj_bias is not None else None
    return W[:d_model], W[d_model:2 * d_model], W[2 * d_model:], \
        sa.out_proj.weight, bv


def _modern_layer_forward(blk, x, attn_bias):
    """One pre-norm ModernBertBlock step with the attention weights exposed —
    identical maths to blk(x, attn_bias) in eval mode (softmax attention
    computed manually since SDPA doesn't return weights)."""
    import torch
    from torch.nn import functional as F
    B, L, _ = x.shape
    xn = blk.attn_norm(x)
    q, k, v = blk.qkv(xn).chunk(3, dim=-1)
    shape = (B, L, blk.n_heads, blk.head_dim)
    q = blk.rope(q.view(shape).transpose(1, 2))
    k = blk.rope(k.view(shape).transpose(1, 2))
    v = v.view(shape).transpose(1, 2)
    w = torch.softmax(q @ k.transpose(-1, -2) / blk.head_dim ** 0.5 + attn_bias, dim=-1)
    sa_out = blk.attn_out((w @ v).transpose(1, 2).reshape(B, L, -1))
    x1 = x + sa_out                                    # dropout = identity in eval
    gate, val = blk.mlp_in(blk.mlp_norm(x1)).chunk(2, dim=-1)
    x2 = x1 + blk.mlp_out(F.gelu(gate) * val)
    return x2, w, sa_out, x1, (q, k, v)                # q,k post-RoPE — as attended


def _head_weights(layer, d_model, n_heads):
    """Per-head slices of the attention projections, display-shaped (head_dim ×
    d_model): W_Q, W_K, W_V rows for this head, and the head's W_O columns
    (transposed) — the matrix that writes the head's output back into the
    residual stream. Backbone-agnostic."""
    import numpy as np
    hd = d_model // n_heads
    Wq, Wk, Wv, Wo, _ = (t.detach().numpy() if t is not None else None
                         for t in _attn_projections(layer, d_model))
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
    written vector is. Returns (heads, L_used) norms. For the pre-norm
    ModernBERT block the value input is LN(x), matching its forward."""
    import numpy as np
    import torch
    hd = d_model // n_heads
    _, _, Wv, Wo, bv = _attn_projections(layer, d_model)
    with torch.no_grad():
        v_in = layer.attn_norm(x_in) if _is_modern_block(layer) else x_in
        v = v_in[0] @ Wv.t() + (bv if bv is not None else 0)   # (L_full, d)
        rows = []
        for h in range(n_heads):
            s = slice(h * hd, (h + 1) * hd)
            proj = v[:, s] @ Wo[:, s].t()   # what j writes if fully attended
            norms = proj.norm(dim=1)        # (L_full,)
            contrib = attn[h, -1] * norms   # × [MASK]-row attention
            rows.append(np.round(contrib[pad_n:].numpy(), 4).tolist())
    return rows


def _attn_steps(qkv, attn, pad_n):
    """The matrix-operation steps of attention for one layer, per head — the
    data actually flowing through, pad positions trimmed:
      q, k, v      (L_used, head_dim)  projected (and RoPE-rotated) vectors
      scores       (L_used, L_used)    QKᵀ/√d — pre-softmax alignment
      weights      (L_used, L_used)    softmax(scores) — from the real forward
      head_out     (L_used, head_dim)  weights @ V — the head's mixed output
    softmax(displayed scores) equals the displayed weights up to the padding
    mask, so the arithmetic can be followed end to end."""
    import numpy as np
    q, k, v = (t[0] for t in qkv)                     # (H, L, hd)
    hd = q.shape[-1]
    scores = (q @ k.transpose(-1, -2)) / hd ** 0.5    # (H, L, L)
    head_out = attn @ v                                # (H, L, hd)
    out = []
    for h in range(q.shape[0]):
        out.append({
            'q': np.round(q[h, pad_n:].numpy(), 3).tolist(),
            'k': np.round(k[h, pad_n:].numpy(), 3).tolist(),
            'v': np.round(v[h, pad_n:].numpy(), 3).tolist(),
            'scores': np.round(scores[h, pad_n:, pad_n:].numpy(), 3).tolist(),
            'head_out': np.round(head_out[h, pad_n:].numpy(), 3).tolist(),
        })
    return out


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

    # DenseRec keeps renderable out-of-vocabulary tokens (e.g. W_DESC job
    # descriptions) as content-injected positions instead of dropping them, so
    # the inspector must analyse the SAME input the prediction used. _row builds
    # the id sequence + the MiniLM injection for a context; plain models fall
    # back to the in-vocab-only row.
    is_dense = hasattr(m, '_context_row')

    def _row(context):
        """(seq ids, injection|None, used token-strings) for one context."""
        if is_dense:
            seq, injection, used, _injected, _unknown = m._context_row(context)
            return seq, injection, used
        kept = [t for t in context if t in vocab.str2idx][-(m._max_len - 1):]
        p = m._max_len - 1 - len(kept)
        return ([vocab.pad_id] * p + [vocab.str2idx[t] for t in kept]
                + [vocab.mask_id]), None, kept

    def _embed(ids_t, injection):
        """Input embeddings with the DenseRec content injection applied — the
        hook the capture forward embeds from."""
        if is_dense and injection is not None:
            model._unseen_stash = injection
            try:
                return model._input_embeddings(ids_t)
            finally:
                model._unseen_stash = None
        return model.item_emb(ids_t)

    def _mask_logits(ids_t, injection):
        """Full-vocab [MASK] logits via the module's real forward (injection
        included) — for the faithfulness-preserving ablation."""
        out = (model(ids_t, unseen_content=injection) if is_dense
               else model(ids_t))
        return out[0, -1]

    seq, injection, used = _row(tokens)
    if not used:
        return {'error': 'no usable context tokens'}
    pad_n = m._max_len - 1 - len(used)
    ids_t = torch.tensor(seq).unsqueeze(0)
    kpm = ids_t == vocab.pad_id
    labels = [_short(t) for t in used] + ['[MASK]']
    types = [t.split(':', 1)[0] for t in used] + ['MASK']

    d_model = model.item_emb.weight.shape[1]
    # ModernBERT backbone: pre-norm ModuleList blocks + final_norm, RoPE
    # (no absolute positions). Stock: post-norm nn.TransformerEncoder.
    modern = hasattr(model, 'final_norm')
    blocks = list(model.encoder) if modern else list(model.encoder.layers)
    n_heads = int(blocks[0].n_heads if modern else blocks[0].self_attn.num_heads)
    # Pre-norm residuals are only head-comparable through the final LayerNorm —
    # the logit lens applies it per stage; the stock backbone lenses raw h.
    lens_h = model.final_norm if modern else (lambda h: h)

    model.eval()
    with torch.no_grad():
        emb = _embed(ids_t, injection)
        if modern:
            x = model.dropout(model.norm(emb))                     # RoPE, no abs pos
            attn_bias = model._attn_bias(ids_t, x.dtype)
        else:
            pos = torch.arange(m._max_len)
            x = model.dropout(model.norm(emb + model.pos_emb(pos)))
        x0 = x
        # Residual-stream trace: (stage name, (1, L_full, d)) at every point in
        # the forward pass, so any token can be followed through the model.
        trace_stages = [('embedding' + ('' if modern else ' + position'), x0)]
        states, attns, layer_extras = [x], [], []
        for li, layer in enumerate(blocks, start=1):
            x_in = x
            if modern:
                x, w, sa_out, x_after_attn, qkv = _modern_layer_forward(layer, x, attn_bias)
            else:
                x, w, sa_out, x_after_attn, qkv = _layer_forward(layer, x, kpm)
            states.append(x)
            attns.append(w[0])                        # (heads, L, L)
            trace_stages += [(f'L{li} · attention output', sa_out),
                             (f'L{li} · after attention', x_after_attn),
                             (f'L{li} · after FFN', x)]
            ffn = ({'w1': layer.mlp_in.weight, 'w2': layer.mlp_out.weight} if modern
                   else {'w1': layer.linear1.weight, 'w2': layer.linear2.weight})
            layer_extras.append({
                'heads': _head_weights(layer, d_model, n_heads),
                'attn_steps': _attn_steps(qkv, w[0], pad_n),
                'mask_value_weighted': _mask_value_weighted(
                    layer, x_in, w[0], pad_n, d_model, n_heads),
                'ffn': {k: np.round(v.detach().numpy(), 3).tolist()
                        for k, v in ffn.items()},
            })
        # The displayed internals must be the model's own numbers — the manual
        # re-run is asserted against the module's real encoder output (with the
        # same content injection).
        final = model.final_norm(x) if modern else x
        real = (model.encode_with_injection(ids_t, injection) if is_dense
                else model._encode(ids_t))
        faithful = bool(torch.allclose(final, real, atol=1e-4))

        title_ids = m._title_ids
        layers = []
        for i, h in enumerate(states):
            entry = {'layer': i,
                     'name': 'embedding' if i == 0 else f'encoder layer {i}'}
            entry.update(_logit_lens(model, lens_h(h)[0, -1], title_id,
                                     title_ids, vocab.idx2str))
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
            vecs = h[0, pad_n:]                       # (L_used, d) raw residual
            logits = model.head(lens_h(h)[0, pad_n:])[:, title_ids]   # lens view
            probs = torch.softmax(logits, dim=-1)
            top = probs.argmax(dim=-1)
            trace['stages'].append(name)
            trace['vectors'].append(np.round(vecs.numpy(), 3).tolist())
            trace['top1'].append([
                {'title': _short(vocab.idx2str[int(title_ids[int(t)])]),
                 'prob': round(float(probs[j, int(t)]), 4)}
                for j, t in enumerate(top)])

        # Leave-one-out influence: re-run the model with each context token
        # removed and measure the change in the clicked title's logit and
        # in-domain probability. Ablation is over `used` (which includes any
        # DenseRec-injected description token), removing one token at a time and
        # re-encoding the variant WITH its injection — so a description's own
        # influence is measured, not silently dropped. SIGNED: positive = the
        # token pushes the prediction TOWARD this title, negative = away.
        # Looped rather than batched because each variant carries a different
        # injection tensor; the model is tiny so it stays instant.
        def _dom(ids_t, inj):
            return _mask_logits(ids_t, inj)[title_ids]

        base_dom = _dom(ids_t, injection)
        base_probs = torch.softmax(base_dom, dim=-1)
        t_pos = (title_ids == title_id).nonzero()
        t_col = int(t_pos[0, 0]) if len(t_pos) else None
        base_logit = float(_mask_logits(ids_t, injection)[title_id])
        base_prob = float(base_probs[t_col]) if t_col is not None else None
        tok_rows = []
        for j in range(len(used)):
            v_seq, v_inj, _ = _row(used[:j] + used[j + 1:])
            v_out = _mask_logits(torch.tensor(v_seq).unsqueeze(0), v_inj)
            entry = {'i': j, 'delta_logit': round(base_logit - float(v_out[title_id]), 4)}
            if t_col is not None:
                v_prob = float(torch.softmax(v_out[title_ids], dim=-1)[t_col])
                entry['delta_prob'] = round(base_prob - v_prob, 5)
            else:
                entry['delta_prob'] = None
            tok_rows.append(entry)
        ablation = {
            'base_logit': round(base_logit, 4),
            'base_prob': round(base_prob, 5) if base_prob is not None else None,
            'tokens': tok_rows,
        }

    return {
        'architecture': 'bert4rec',
        'backbone': 'modernbert' if modern else 'bert4rec',
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


def inspect_model_static(m, architecture, sample_contexts=None):
    """Whole-model view — no prediction needed. The learned, input-independent
    internals: per-layer/head attention projections and FFN weights, plus an
    aggregate attention profile (what token TYPE each head attends to from the
    [MASK] query, averaged over held-out sample resumes) so a head's function
    is visible without picking one resume."""
    if architecture == 'item2vec':
        return {'architecture': 'item2vec', 'static': True,
                'vocab_size': m.vocab_size, 'vector_size': m.vector_size,
                'note': 'item2vec has no layers or attention — one embedding '
                        'table; drill into a prediction below to see per-token '
                        'cosine pulls.'}
    import torch
    model = m.model
    d_model = model.item_emb.weight.shape[1]
    modern = hasattr(model, 'final_norm')
    blocks = list(model.encoder) if modern else list(model.encoder.layers)
    n_heads = int(blocks[0].n_heads if modern else blocks[0].self_attn.num_heads)

    layers = []
    for li, layer in enumerate(blocks, start=1):
        ffn = ({'w1': layer.mlp_in.weight, 'w2': layer.mlp_out.weight} if modern
               else {'w1': layer.linear1.weight, 'w2': layer.linear2.weight})
        layers.append({
            'name': f'encoder layer {li}',
            'heads': _head_weights(layer, d_model, n_heads),
            'ffn': {k: np.round(v.detach().numpy(), 3).tolist()
                    for k, v in ffn.items()},
        })

    # Aggregate [MASK] attention by token type, averaged over sample resumes:
    # rows = token types, one matrix per (layer, head). A slim attention-only
    # forward (same maths as the per-resume inspector, no lens/ablation).
    profile = None
    if sample_contexts:
        vocab = m.vocab
        is_dense = hasattr(m, '_context_row')

        def _attn_rows(context):
            """(token types incl. [MASK], [layer][head] mask-row attention)."""
            if is_dense:
                seq, injection, used = m._context_row(context)[:3]
            else:
                used = [t for t in context if t in vocab.str2idx][-(m._max_len - 1):]
                p = m._max_len - 1 - len(used)
                seq = ([vocab.pad_id] * p + [vocab.str2idx[t] for t in used]
                       + [vocab.mask_id])
                injection = None
            if not used:
                return None, None
            pad_n = m._max_len - 1 - len(used)
            ids_t = torch.tensor(seq).unsqueeze(0)
            with torch.no_grad():
                if is_dense and injection is not None:
                    model._unseen_stash = injection
                try:
                    emb = (model._input_embeddings(ids_t) if is_dense
                           else model.item_emb(ids_t))
                finally:
                    if is_dense:
                        model._unseen_stash = None
                if modern:
                    x = model.dropout(model.norm(emb))
                    bias = model._attn_bias(ids_t, x.dtype)
                else:
                    pos = torch.arange(m._max_len)
                    x = model.dropout(model.norm(emb + model.pos_emb(pos)))
                    kpm = ids_t == vocab.pad_id
                per_layer = []
                for layer in blocks:
                    if modern:
                        x, w, *_ = _modern_layer_forward(layer, x, bias)
                    else:
                        x, w, *_ = _layer_forward(layer, x, kpm)
                    per_layer.append(w[0, :, -1, pad_n:])   # [MASK] row per head
            types = [t.split(':', 1)[0] for t in used] + ['MASK']
            return types, per_layer

        model.eval()
        sums, counts = {}, {}
        n_used = 0
        for context in sample_contexts:
            types, per_layer = _attn_rows(context)
            if types is None:
                continue
            n_used += 1
            for li, rows in enumerate(per_layer, start=1):
                for h in range(n_heads):
                    for typ, wgt in zip(types, rows[h].tolist()):
                        key = (li, h, typ)
                        sums[key] = sums.get(key, 0.0) + wgt
                        counts[key] = counts.get(key, 0) + 1
        if n_used:
            types = sorted({k[2] for k in sums})
            profile = {
                'n_resumes': n_used,
                'types': types,
                # [layer][head][type] mean attention from the [MASK] query
                'mean': [[[round(sums.get((li, h, t), 0.0) /
                                 max(counts.get((li, h, t), 1), 1), 4)
                           for t in types]
                          for h in range(n_heads)]
                         for li in range(1, len(blocks) + 1)],
            }

    return {
        'architecture': architecture,
        'backbone': 'modernbert' if modern else 'bert4rec',
        'static': True,
        'n_layers': len(blocks),
        'n_heads': n_heads,
        'd_model': int(d_model),
        'vocab_size': m.vocab_size,
        'title_count': len(m.title_vocab),
        'layers': layers,
        'type_attention': profile,
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
