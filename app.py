"""Career Path Transformer demo app.

Serves a local UI to run item2vec and BERT4Rec career-path predictions against
sample resumes or hand-built ones, and to map the embedding space around the
input resume.

Run:  ./run.sh   (or: conda run -n dwh-ai-py311 python app.py)
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # macOS torch/gensim OpenMP clash

import traceback

from flask import Flask, jsonify, request, send_from_directory

from demo import config
from demo.embedding_space import map_space
from demo.samples import load_samples
from demo.tokens import (TOKEN_TYPES, group_into_experiences, tokens_from_resume)

app = Flask(__name__, static_folder='static', static_url_path='/static')

MODELS = {}
MODEL_ERRORS = {}
TRANSITIONS = {'store': None, 'error': None}


def load_models():
    try:
        from demo.item2vec_model import Item2VecModel
        MODELS['item2vec'] = Item2VecModel()
        print(f'item2vec loaded: {MODELS["item2vec"].vocab_size:,} tokens '
              f'({len(MODELS["item2vec"].title_vocab):,} titles)')
    except Exception as e:
        MODEL_ERRORS['item2vec'] = str(e)
        traceback.print_exc()

    try:
        from demo.bert4rec_model import Bert4RecModel
        model = Bert4RecModel.load_if_available()
        if model is not None:
            MODELS['bert4rec'] = model
            print(f'bert4rec loaded: {model.vocab_size:,} tokens '
                  f'({len(model.title_vocab):,} titles)')
        else:
            MODEL_ERRORS['bert4rec'] = (
                'No checkpoint at artifacts/bert4rec/. '
                'Run: python scripts/train_bert4rec.py')
    except Exception as e:
        MODEL_ERRORS['bert4rec'] = str(e)
        traceback.print_exc()

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


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/status')
def status():
    info = {}
    for name, m in MODELS.items():
        info[name] = {
            'loaded': True,
            'vocab_size': m.vocab_size,
            'title_count': len(m.title_vocab),          # rankable titles
            'full_title_count': getattr(m, 'full_title_count', len(m.title_vocab)),
            'ranking_restricted': getattr(m, 'restricted', False),
            'vector_size': m.vector_size,
        }
        if name == 'bert4rec':
            info[name]['source'] = m.params.get('source', 'local')
            info[name]['origin'] = m.params.get('origin', 'local')
        if name == 'item2vec':
            info[name]['source'] = os.path.basename(m.bin_path)
    for name, err in MODEL_ERRORS.items():
        info[name] = {'loaded': False, 'error': err}
    store = TRANSITIONS['store']
    transitions = ({'loaded': True, **store.meta} if store is not None
                   else {'loaded': False, 'error': TRANSITIONS['error']})
    return jsonify({'models': info, 'token_types': TOKEN_TYPES, 'transitions': transitions})


@app.route('/api/samples')
def samples():
    items = load_samples()
    return jsonify([
        {**s, 'experiences': group_into_experiences(s['context_tokens'])}
        for s in items
    ])


@app.route('/api/vocab')
def vocab():
    """Autocomplete: /api/vocab?type=W_TITLE&q=soft&limit=15"""
    model = MODELS.get('item2vec') or MODELS.get('bert4rec')
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
    names  = payload.get('models') or list(MODELS)

    results = {}
    for name in names:
        model = MODELS.get(name)
        if model is None:
            results[name] = {'error': MODEL_ERRORS.get(name, 'not loaded')}
            continue
        ranked = model.rank_titles(tokens, top_k=top_k)
        if target:
            # Sense check: rank of the known correct answer over the full title list
            full = model.rank_titles(tokens, top_k=len(model.title_vocab))
            all_tokens = [p['token'] for p in full['predictions']]
            ranked['target_rank'] = (all_tokens.index(target) + 1
                                     if target in all_tokens else None)
        results[name] = ranked
    return jsonify({'tokens': tokens, 'target': target, 'results': results})


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
    name = payload.get('model', 'item2vec')
    model = MODELS.get(name)
    if model is None:
        return jsonify({'error': MODEL_ERRORS.get(name, f'{name} not loaded')}), 400
    top_k = int(payload.get('top_k', config.DEFAULT_TOP_K))
    mode  = payload.get('mode', 'local')
    ranked = model.rank_titles(tokens, top_k=top_k)
    result = map_space(model, tokens, ranked['predictions'], mode=mode)
    result['model'] = name
    return jsonify(result)


if __name__ == '__main__':
    load_models()
    port = int(os.environ.get('CPT_DEMO_PORT', 5050))
    print(f'\n  Career Path Transformer demo: http://127.0.0.1:{port}\n')
    app.run(host='127.0.0.1', port=port, debug=False)
