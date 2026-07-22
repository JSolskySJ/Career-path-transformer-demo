"""Career Path Transformer demo app.

Serves a local UI to run item2vec and BERT4Rec career-path predictions against
sample resumes or hand-built ones, and to map the embedding space around the
input resume. Every staged MLflow run (see demo/registry.py) is loaded side by
side, so several versions of the same architecture can be compared.

Run:  ./run.sh   (or: conda run -n dwh-ai-py311 python app.py)
"""

import inspect
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # macOS torch/gensim OpenMP clash

import json
import traceback

from flask import Flask, jsonify, request, send_from_directory

from demo import config, registry
from demo.embedding_space import map_space
from demo.tokens import (TOKEN_TYPES, group_into_experiences, rollup_titles,
                         tokens_from_resume)

app = Flask(__name__, static_folder='static', static_url_path='/static')

RUNS = {}                                # run_id -> registry entry (model + meta)
TRANSITIONS = {'store': None, 'error': None}


def load_models():
    global RUNS
    # Auto-sync: stage any FINISHED MLflow runs newer than the registry, so
    # fresh training runs appear on every start. CPT_AUTO_SYNC=0 disables.
    if os.environ.get('CPT_AUTO_SYNC', '1') != '0':
        try:
            from scripts.fetch_mlflow_artifacts import sync_new_runs
            n = sync_new_runs()
            print(f'auto-sync: {n} new run(s) staged from MLflow' if n
                  else 'auto-sync: registry up to date')
        except Exception as e:
            print(f'auto-sync skipped ({e})')
    try:
        RUNS = registry.discover_runs()
    except Exception as e:
        traceback.print_exc()
        RUNS = {}
    # Label runs with their experiment code from the docs vault (read-only) —
    # e.g. R4, M1 — so models in the UI line up with the experiment logs.
    try:
        from demo.experiments import match_runs
        n = match_runs(RUNS)
        print(f'experiment logs: {n}/{len(RUNS)} staged runs matched to vault codes')
    except Exception as e:
        print(f'experiment-log scan skipped ({e})')
    for rid, r in RUNS.items():
        if r['model'] is not None:
            print(f"loaded {r['label']} ({rid[:8]}): {r['model'].vocab_size:,} tokens, "
                  f"{len(r['model'].title_vocab):,} rankable titles")
        else:
            print(f"FAILED {r['label']} ({rid[:8]}): {r['error']}")

    # Ensure the VAL+TEST dataset slice exists for the latest few runs' datasets
    # (Dataset view). New datasets download once; existing ones are a no-op.
    if os.environ.get('CPT_SYNC_DATASETS', '1') != '0':
        try:
            from demo.dataset_samples import sync_datasets
            sync_datasets(RUNS, n=3)
        except Exception as e:
            print(f'dataset sync skipped ({e})')

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
            'experiment': r.get('experiment'),
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


@app.route('/api/build_samples', methods=['POST'])
def build_samples():
    """Rebuild a run's sample resumes straight from its dataset parquet, using
    the training repo's own load/tokenise/split code. Slow (full parquet read)
    — only needed when a new dataset is generated (deterministic split)."""
    payload = request.get_json(force=True)
    rid = payload.get('run', '')
    r = RUNS.get(rid)
    if r is None:
        return jsonify({'error': f'unknown run {rid}'}), 400
    dataset = payload.get('dataset') or r['params'].get('data_run_id')
    if not dataset or str(dataset).lower() == 'none':
        return jsonify({'error': 'no dataset run id — this run predates data_run_id '
                                 'param logging; pass one explicitly'}), 400
    n = int(payload.get('n', 300))
    out = os.path.join(registry.RUNS_DIR, rid, 'sample_resumes.json')
    try:
        from demo.dataset_samples import build_samples_for_run
        stats = build_samples_for_run(r['params'], str(dataset), out, n=n)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    r['samples_path'] = out
    return jsonify({'ok': True, 'run': rid, **stats})


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
    """Accept either raw tokens or builder experiences, then apply the optional
    title rollup ('rollup': none | consecutive | all). Applied server-side so
    the returned tokens — and everything downstream (inspector, space plot) —
    show exactly what the model saw."""
    if payload.get('tokens'):
        tokens = list(payload['tokens'])
    elif payload.get('experiences'):
        tokens = tokens_from_resume(payload['experiences'])
    else:
        return []
    return rollup_titles(tokens, payload.get('rollup', 'none'))


@app.route('/api/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)
    tokens  = _resolve_tokens(payload)
    if not tokens:
        return jsonify({'error': 'No tokens provided'}), 400
    top_k  = int(payload.get('top_k', config.DEFAULT_TOP_K))
    target = payload.get('target')
    domain = payload.get('domain', 'all')
    scoring = payload.get('scoring', 'softmax')   # bert4rec only; item2vec is cosine-native
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
        kw = {'scoring': scoring} if 'scoring' in inspect.signature(model.rank_titles).parameters else {}
        ranked = model.rank_titles(tokens, top_k=top_k, allowed=allowed, **kw)
        # SJ / taxonomy-level badges per predicted title
        from demo.title_flags import flags_for
        for p in ranked['predictions']:
            p.update(flags_for(p['token'].split(':', 1)[-1]))
        if target:
            # Sense check: rank of the known correct answer over the same
            # (possibly projected) domain the predictions used
            full = model.rank_titles(tokens, top_k=len(model.title_vocab),
                                     allowed=allowed, **kw)
            all_tokens = [p['token'] for p in full['predictions']]
            ranked['target_rank'] = (all_tokens.index(target) + 1
                                     if target in all_tokens else None)
        ranked['label'] = r['label']
        ranked['architecture'] = r['architecture']
        results[rid] = ranked
    return jsonify({'tokens': tokens, 'target': target, 'results': results})


@app.route('/api/inspect', methods=['POST'])
def inspect_endpoint():   # named to avoid shadowing the stdlib `inspect` module
    """Drill-down for one (model, resume, title): bert4rec logit lens +
    per-layer/head attention matrices; item2vec per-token cosine pull."""
    payload = request.get_json(force=True)
    rid = payload.get('model', '')
    r = RUNS.get(rid)
    if r is None or r['model'] is None:
        return jsonify({'error': f'{rid} not loaded'}), 400
    title = payload.get('title')
    if not title:
        # Whole-model view: learned weights per layer/head plus the aggregate
        # [MASK]-attention-by-token-type profile over held-out sample resumes.
        contexts = []
        if r['samples_path']:
            with open(r['samples_path']) as f:
                contexts = [s['context_tokens'] for s in json.load(f)[:12]]
        from demo.introspection import inspect_model_static
        result = inspect_model_static(r['model'], r['architecture'], contexts)
    else:
        tokens = _resolve_tokens(payload)
        if not tokens:
            return jsonify({'error': 'No tokens provided'}), 400
        from demo.introspection import inspect_model
        result = inspect_model(r['model'], r['architecture'], tokens, title)
    result['model'] = rid
    result['label'] = r['label']
    result['experiment'] = r.get('experiment')
    return jsonify(result)


@app.route('/api/datasets')
def datasets():
    """Sliced datasets available for the Dataset view, with the runs on each."""
    from demo.dataset_samples import available_datasets
    return jsonify({'datasets': available_datasets(RUNS)})


@app.route('/api/dataset')
def dataset():
    """A page of raw rows: /api/dataset?id=<data_run_id>&offset=0&limit=50"""
    did = request.args.get('id', '')
    offset = max(0, int(request.args.get('offset', 0)))
    limit = max(1, min(int(request.args.get('limit', 50)), 500))
    filters = json.loads(request.args.get('filters') or '{}')
    try:
        from demo.dataset_samples import read_slice
        return jsonify(read_slice(did, offset, limit, filters))
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404


@app.route('/skills')
def skills_page():
    """Counterfactual skill suggestions for one worker, opened per sample."""
    return send_from_directory('static', 'skills.html')


@app.route('/api/suggest_skills', methods=['POST'])
def suggest_skills():
    """Top-k skills whose addition most improves the target title's rank.
    DenseRec runs only. Brute force over the skill vocabulary — slow-ish
    (a few chunked forward passes), so the page shows progress copy."""
    payload = request.get_json(force=True)
    rid = payload.get('model', '')
    r = RUNS.get(rid)
    if r is None:
        return jsonify({'error': f'unknown run {rid}'}), 400
    tokens = payload.get('tokens') or []
    target = payload.get('target', '')
    if not tokens or not target:
        return jsonify({'error': 'tokens and target are required'}), 400
    try:
        from demo.skills import suggest
        result = suggest(rid, r, tokens, target,
                         top_k=int(payload.get('top_k', 10)),
                         limit=int(payload.get('limit', 0)))
    except (ValueError, FileNotFoundError) as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    result['model'] = rid
    result['label'] = r['label']
    result['experiment'] = r.get('experiment')
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
