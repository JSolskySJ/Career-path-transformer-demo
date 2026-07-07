"""Career Path Transformer demo app.

Serves a local UI to run item2vec and BERT4Rec career-path predictions against
sample resumes or hand-built ones, and to map the embedding space around the
input resume. Every staged MLflow run (see demo/registry.py) is loaded side by
side, so several versions of the same architecture can be compared.

Run:  ./run.sh   (or: conda run -n dwh-ai-py311 python app.py)
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # macOS torch/gensim OpenMP clash

import json
import traceback

from flask import Flask, jsonify, request, send_from_directory

from demo import config, registry
from demo.embedding_space import map_space
from demo.tokens import (TOKEN_TYPES, group_into_experiences, tokens_from_resume)

app = Flask(__name__, static_folder='static', static_url_path='/static')

RUNS = {}                                # run_id -> registry entry (model + meta)
TRANSITIONS = {'store': None, 'error': None}


def load_models():
    global RUNS
    try:
        RUNS = registry.discover_runs()
    except Exception as e:
        traceback.print_exc()
        RUNS = {}
    for rid, r in RUNS.items():
        if r['model'] is not None:
            print(f"loaded {r['label']} ({rid[:8]}): {r['model'].vocab_size:,} tokens, "
                  f"{len(r['model'].title_vocab):,} rankable titles")
        else:
            print(f"FAILED {r['label']} ({rid[:8]}): {r['error']}")

    try:
        from demo.transitions import TransitionStore
        store = TransitionStore.load_if_available()
        if store is not None:
            TRANSITIONS['store'] = store
            print(f'transitions loaded: {store.meta.get("n_transitions", 0):,} transitions, '
                  f'{store.meta.get("n_sources", 0):,} source titles')
        else:
            TRANSITIONS['error'] = ('No artifacts/transitions.json. '
                                    'Run: python scripts/build_transitions.py')
    except Exception as e:
        TRANSITIONS['error'] = str(e)
        traceback.print_exc()


def _loaded(run_id):
    r = RUNS.get(run_id)
    return r['model'] if r and r['model'] is not None else None


def _any_loaded_model():
    for r in RUNS.values():
        if r['model'] is not None:
            return r['model']
    return None


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/inspect')
def inspect_page():
    """Full-page model inspector, opened in a new tab from a prediction row."""
    return send_from_directory('static', 'inspect.html')


@app.route('/api/status')
def status():
    """Everything the UI needs about staged runs: model state, run params
    (with the model-side transformation subset), and key metrics."""
    models = {}
    for rid, r in RUNS.items():
        info = {
            'run_id': rid,
            'run_name': r['run_name'],
            'run_tag': r['run_tag'],
            'architecture': r['architecture'],
            'label': r['label'],
            'loaded': r['model'] is not None,
            'has_samples': r['samples_path'] is not None,
            'params': r['params'],
            'transformations': registry.transformations(r['params']),
            'metrics': registry.key_metrics(r['metrics']),
            'start_time': r['start_time'],
        }
        m = r['model']
        if m is not None:
            info.update({
                'vocab_size': m.vocab_size,
                'title_count': len(m.title_vocab),          # rankable titles
                'full_title_count': getattr(m, 'full_title_count', len(m.title_vocab)),
                'ranking_restricted': getattr(m, 'restricted', False),
                'vector_size': m.vector_size,
            })
        else:
            info['error'] = r['error']
        models[rid] = info
    store = TRANSITIONS['store']
    transitions = ({'loaded': True, **store.meta} if store is not None
                   else {'loaded': False, 'error': TRANSITIONS['error']})
    return jsonify({'models': models, 'token_types': TOKEN_TYPES, 'transitions': transitions})


def _with_experiences(sample: dict) -> dict:
    """Attach the rendered experience list, and (when present) zip each work
    experience's tenure onto it in order — work_tenures aligns 1:1 with the
    WORK experiences, in career order."""
    experiences = group_into_experiences(sample['context_tokens'])
    tenures = sample.get('work_tenures') or []
    it = iter(tenures)
    for exp in experiences:
        if exp.get('type') == 'WORK':
            t = next(it, None)
            if t and t.get('label'):
                exp['tenure'] = t['label']
                exp['tenure_bucket'] = t.get('bucket')
    return {**sample, 'experiences': experiences}


@app.route('/api/samples')
def samples():
    """Samples for one staged run: /api/samples?run=<run_id>. Without run, the
    first run that has samples is used."""
    run_id = request.args.get('run')
    if run_id:
        r = RUNS.get(run_id)
    else:
        run_id, r = next(((rid, r) for rid, r in RUNS.items() if r['samples_path']),
                         (None, None))
    if r is None or not r['samples_path']:
        return jsonify({'run': run_id, 'samples': []})
    with open(r['samples_path']) as f:
        items = json.load(f)
    return jsonify({'run': run_id, 'samples': [_with_experiences(s) for s in items]})


@app.route('/api/vocab')
def vocab():
    """Autocomplete: /api/vocab?type=W_TITLE&q=soft&limit=15[&run=<run_id>]"""
    model = _loaded(request.args.get('run', '')) or _any_loaded_model()
    if model is None:
        return jsonify([])
    ttype = request.args.get('type', 'W_TITLE')
    q     = request.args.get('q', '').strip().lower()
    limit = int(request.args.get('limit', 15))
    values = model.vocab_by_prefix().get(ttype, [])
    if q:
        starts = [v for v in values if v.startswith(q)]
        contains = [v for v in values if q in v and not v.startswith(q)]
        values = starts + contains
    return jsonify(values[:limit])


_DOMAIN_CACHE = {}   # (run_id, domain) -> set of allowed W_TITLE tokens


def _domain_tokens(rid, model, domain):
    """The projection set for domain 'sj' / 'taxonomy' (None = model default).
    Intersects the model's own title vocabulary with the title-flags lookup."""
    if domain in (None, '', 'all'):
        return None
    key = (rid, domain)
    if key not in _DOMAIN_CACHE:
        from demo.title_flags import flags_for
        sel = set()
        for t in model.title_vocab:
            f = flags_for(t.split(':', 1)[-1])
            if (domain == 'sj' and f['sj']) or (domain == 'taxonomy' and f['tax']):
                sel.add(t)
        _DOMAIN_CACHE[key] = sel
    return _DOMAIN_CACHE[key]


def _resolve_tokens(payload: dict) -> list:
    """Accept either raw tokens or builder experiences."""
    if payload.get('tokens'):
        return list(payload['tokens'])
    if payload.get('experiences'):
        return tokens_from_resume(payload['experiences'])
    return []


@app.route('/api/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)
    tokens  = _resolve_tokens(payload)
    if not tokens:
        return jsonify({'error': 'No tokens provided'}), 400
    top_k  = int(payload.get('top_k', config.DEFAULT_TOP_K))
    target = payload.get('target')
    domain = payload.get('domain', 'all')
    run_ids = payload.get('models') or [rid for rid, r in RUNS.items()
                                        if r['model'] is not None]

    results = {}
    for rid in run_ids:
        r = RUNS.get(rid)
        if r is None or r['model'] is None:
            results[rid] = {'error': (r or {}).get('error', 'not loaded'),
                            'label': (r or {}).get('label', rid)}
            continue
        model = r['model']
        allowed = _domain_tokens(rid, model, domain)
        if allowed is not None and not allowed:
            results[rid] = {'error': f'no {domain} titles resolvable — is '
                                     f'artifacts/title_flags.json staged? '
                                     f'(scripts/build_title_flags.py)',
                            'label': r['label']}
            continue
        ranked = model.rank_titles(tokens, top_k=top_k, allowed=allowed)
        # SJ / taxonomy-level badges per predicted title
        from demo.title_flags import flags_for
        for p in ranked['predictions']:
            p.update(flags_for(p['token'].split(':', 1)[-1]))
        if target:
            # Sense check: rank of the known correct answer over the same
            # (possibly projected) domain the predictions used
            full = model.rank_titles(tokens, top_k=len(model.title_vocab),
                                     allowed=allowed)
            all_tokens = [p['token'] for p in full['predictions']]
            ranked['target_rank'] = (all_tokens.index(target) + 1
                                     if target in all_tokens else None)
        ranked['label'] = r['label']
        ranked['architecture'] = r['architecture']
        results[rid] = ranked
    return jsonify({'tokens': tokens, 'target': target, 'results': results})


@app.route('/api/inspect', methods=['POST'])
def inspect():
    """Drill-down for one (model, resume, title): bert4rec logit lens +
    per-layer/head attention matrices; item2vec per-token cosine pull."""
    payload = request.get_json(force=True)
    tokens  = _resolve_tokens(payload)
    if not tokens:
        return jsonify({'error': 'No tokens provided'}), 400
    rid = payload.get('model', '')
    title = payload.get('title')
    if not title:
        return jsonify({'error': 'No title provided'}), 400
    r = RUNS.get(rid)
    if r is None or r['model'] is None:
        return jsonify({'error': f'{rid} not loaded'}), 400
    from demo.introspection import inspect_model
    result = inspect_model(r['model'], r['architecture'], tokens, title)
    result['model'] = rid
    result['label'] = r['label']
    return jsonify(result)


@app.route('/api/transition_titles')
def transition_titles():
    store = TRANSITIONS['store']
    if store is None:
        return jsonify({'error': TRANSITIONS['error'], 'titles': []})
    return jsonify({'titles': store.source_titles(), 'meta': store.meta})


@app.route('/api/sankey', methods=['POST'])
def sankey():
    store = TRANSITIONS['store']
    if store is None:
        return jsonify({'error': TRANSITIONS['error'] or 'transitions not loaded'}), 400
    payload = request.get_json(force=True)
    selected = payload.get('titles') or []
    if not selected:
        return jsonify({'error': 'No titles selected'}), 400
    top_k = max(1, min(int(payload.get('top_k', 10)), 25))
    depth = max(1, min(int(payload.get('depth', 1)), 5))
    return jsonify(store.build_sankey(selected, top_k=top_k, depth=depth))


@app.route('/api/space', methods=['POST'])
def space():
    payload = request.get_json(force=True)
    tokens  = _resolve_tokens(payload)
    if not tokens:
        return jsonify({'error': 'No tokens provided'}), 400
    rid = payload.get('model', '')
    model = _loaded(rid)
    if model is None:
        return jsonify({'error': f'{rid} not loaded'}), 400
    top_k = int(payload.get('top_k', config.DEFAULT_TOP_K))
    mode  = payload.get('mode', 'local')
    allowed = _domain_tokens(rid, model, payload.get('domain', 'all'))
    ranked = model.rank_titles(tokens, top_k=top_k, allowed=allowed)
    result = map_space(model, tokens, ranked['predictions'], mode=mode)
    result['model'] = rid
    return jsonify(result)


if __name__ == '__main__':
    load_models()
    port = int(os.environ.get('CPT_DEMO_PORT', 5050))
    print(f'\n  Career Path Transformer demo: http://127.0.0.1:{port}\n')
    app.run(host='127.0.0.1', port=port, debug=False)
