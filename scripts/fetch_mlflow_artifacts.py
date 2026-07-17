"""Fetch career-path model artifacts straight from MLflow and stage each run
into the demo's run registry (artifacts/runs/<run_id>/) — model checkpoint,
per-run vocab.csv, run.json (params + key metrics) and per-run sample resumes.
Several runs of one architecture can be staged side by side and compared in
the UI (e.g. two bert4rec versions).

Runs are resolved three ways (first match wins):

  --item2vec RUN_ID / --bert4rec RUN_ID   explicit run ids (repeatable —
                                          pass --bert4rec twice to stage two
                                          bert4rec versions)
  --tag RUN_TAG                           newest FINISHED run per architecture
                                          with tags.run_tag = RUN_TAG
  (default)                               newest FINISHED run per architecture
                                          (add --skills-only to require
                                          add_skills=True runs)

Per resolved run this downloads (into incoming/, then stages):
  - the predictions CSV        career_path_transformer_<ts>.csv       (run root)
  - the vocab CSV              career_path_transformer_vocab_<ts>.csv (if logged)
  - bert4rec: the run's vocab.json (authoritative idx2str) and the logged
    pytorch model's data/model.pth
  - item2vec: the logged pyfunc model's gensim .bin (+ .npy sidecars)

Global artifacts/transitions.json is rebuilt from the largest predictions CSV.
Existing registry runs are left untouched — re-fetching a run overwrites just
that run's directory. --no-import downloads only.

MLflow connection comes from datawarehouse-configurations ai.conf
({env}.mlflow.*), the same source the training runners and notebooks use.

Usage:
    python scripts/fetch_mlflow_artifacts.py                      # latest finished runs
    python scripts/fetch_mlflow_artifacts.py --skills-only        # latest skills runs
    python scripts/fetch_mlflow_artifacts.py --tag my_run_tag
    python scripts/fetch_mlflow_artifacts.py --bert4rec RUN_A --bert4rec RUN_B
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import glob
import json
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo import config

EXPERIMENT = '{partner}_career_path_transformer'
ARCHITECTURES = ('item2vec', 'bert4rec')
PREDICTIONS_RE = re.compile(r'^career_path_transformer_\d{8}_\d{6}\.csv$')
VOCAB_CSV_RE = re.compile(r'^career_path_transformer_vocab_\d{8}_\d{6}\.csv$')
# Files this script manages inside incoming/ — cleared before each fetch so
# staging never mixes artifacts from different fetches.
MANAGED_GLOBS = ('career_path_transformer_*.csv', 'career_path_transformer_*.bin*',
                 'model*.pth', 'bert4rec_vocab*.json', 'RUN_INFO.json')


def connect(env, ai_conf):
    import mlflow
    from pyhocon import ConfigFactory
    conf = ConfigFactory.parse_file(ai_conf)
    mlflow.set_tracking_uri(conf.get_string(f'{env}.mlflow.mlflow_url'))
    os.environ['MLFLOW_TRACKING_USERNAME'] = conf.get_string(f'{env}.mlflow.mlflow_user')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = conf.get_string(f'{env}.mlflow.mlflow_password')
    print(f'MLflow: {conf.get_string(f"{env}.mlflow.mlflow_url")}')
    return mlflow


def resolve_runs(mlflow, experiment, args) -> list:
    """[(architecture, run), ...] to fetch. Explicit run ids win (repeatable
    flags); then --tag; else the newest FINISHED run per architecture."""
    client = mlflow.MlflowClient()
    explicit = [('item2vec', rid) for rid in (args.item2vec or [])] + \
               [('bert4rec', rid) for rid in (args.bert4rec or [])]
    if explicit:
        return [(a, client.get_run(rid)) for a, rid in explicit]

    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f'No MLflow experiment named {experiment}')
    runs = []
    for arch in ARCHITECTURES:
        clauses = [f"tags.architecture = '{arch}'", "attributes.status = 'FINISHED'"]
        if args.tag:
            clauses.append(f"tags.run_tag = '{args.tag}'")
        if args.skills_only:
            clauses.append("params.add_skills = 'True'")
        df = mlflow.search_runs(experiment_ids=[exp.experiment_id],
                                filter_string=' and '.join(clauses),
                                order_by=['start_time DESC'], max_results=1)
        if len(df):
            runs.append((arch, client.get_run(df.iloc[0]['run_id'])))
        else:
            print(f'WARNING: no finished {arch} run matches '
                  f'({" and ".join(clauses)}) — skipping {arch}')
    if not runs:
        raise SystemExit('No runs resolved — nothing to fetch.')
    return runs


def _retry(fn, attempts=6, label=''):
    """The tracking server intermittently drops larger transfers ('Response
    ended prematurely') — retry with a short backoff before giving up."""
    import time
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if i == attempts - 1:
                raise
            print(f'  retrying {label} ({i + 1}/{attempts - 1}): {exc}')
            time.sleep(3 * (i + 1))


def _http_download(url, dst, expected_size=None, attempts=30):
    """Raw streaming download with Range-resume — the fallback for files the
    MLflow client keeps failing on. The server cuts long transfers mid-stream
    and only *sometimes* honours Range, so this keeps whatever prefix it has
    (including a partial left behind by a failed MLflow-client attempt) and
    grinds forward until the byte count matches."""
    import time
    import requests
    auth = (os.environ['MLFLOW_TRACKING_USERNAME'], os.environ['MLFLOW_TRACKING_PASSWORD'])
    pos = os.path.getsize(dst) if os.path.exists(dst) else 0
    if expected_size and pos >= expected_size:
        return dst
    for i in range(attempts):
        headers = {'Range': f'bytes={pos}-'} if pos else {}
        try:
            with requests.get(url, stream=True, auth=auth, headers=headers, timeout=120) as r:
                r.raise_for_status()
                if pos and r.status_code != 206:
                    pos = 0        # server ignored Range — restart from scratch
                with open(dst, 'ab' if pos else 'wb') as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        pos += len(chunk)
            if expected_size is None or pos >= expected_size:
                return dst
            print(f'  http resume ({i + 1}/{attempts}): {pos:,}/{expected_size:,} bytes')
        except Exception as exc:
            print(f'  http retry ({i + 1}/{attempts}) at {pos:,} bytes: {exc}')
            time.sleep(2)
    raise RuntimeError(f'download failed after {attempts} attempts: {url}')


def _run_artifact_http_url(mlflow, run, artifact_path) -> str:
    """Map a run's mlflow-artifacts:/ URI to the tracking server's HTTP
    artifact endpoint for one file."""
    root = run.info.artifact_uri  # e.g. mlflow-artifacts:/51/<run_id>/artifacts
    base = mlflow.get_tracking_uri().rstrip('/')
    return root.replace('mlflow-artifacts:/', f'{base}/api/2.0/mlflow-artifacts/artifacts/') \
        + f'/{artifact_path}'


def _find_artifacts(repo, pattern) -> list:
    """BFS an artifact repository for all files matching pattern; returns
    their repo-relative paths."""
    import fnmatch
    hits, queue = [], [None]
    while queue:
        for a in repo.list_artifacts(queue.pop(0)):
            if a.is_dir:
                queue.append(a.path)
            elif fnmatch.fnmatch(os.path.basename(a.path), pattern):
                hits.append(a.path)
    return hits


def fetch_run_files(mlflow, arch, run, inc) -> dict:
    """Download one run's demo artifacts into incoming/. Returns a manifest:
    {'predictions': [csv, ...], 'vocab_csv': path|None,
     'vocab_json': path|None, 'model_file': .bin/.pth path|None}."""
    rid = run.info.run_id
    name = run.data.tags.get('mlflow.runName', '')
    print(f'\n[{arch}] run {rid} ({name})')
    files = {'predictions': [], 'vocab_csv': None, 'vocab_json': None, 'model_file': None}

    for a in mlflow.artifacts.list_artifacts(run_id=rid):
        base = os.path.basename(a.path)
        if PREDICTIONS_RE.match(base) or VOCAB_CSV_RE.match(base):
            # CSVs are nice-to-have (samples/transitions) — don't let one flaky
            # transfer kill the fetch of the model itself. Fall back to a raw
            # resumable GET when the MLflow client keeps dropping the transfer.
            try:
                local = _retry(lambda p=a.path: mlflow.artifacts.download_artifacts(
                    run_id=rid, artifact_path=p, dst_path=inc), attempts=3, label=base)
            except Exception:
                try:
                    print(f'  {base}: MLflow client failed — raw HTTP fallback')
                    local = _http_download(_run_artifact_http_url(mlflow, run, a.path),
                                           os.path.join(inc, base), a.file_size)
                except Exception as exc:
                    print(f'  WARNING: giving up on {base}: {exc}')
                    continue
            print(f'  {base}  ({a.file_size or 0:,} bytes)')
            if PREDICTIONS_RE.match(base):
                files['predictions'].append(local)
            else:
                files['vocab_csv'] = local
        elif arch == 'bert4rec' and base == 'vocab.json':
            local = _retry(lambda p=a.path: mlflow.artifacts.download_artifacts(
                run_id=rid, artifact_path=p, dst_path=inc), label=base)
            # per-run name — two bert4rec runs in one fetch must not collide
            dst = os.path.join(inc, f'bert4rec_vocab_{rid[:8]}.json')
            shutil.move(local, dst)
            files['vocab_json'] = dst
            print(f'  vocab.json -> {os.path.basename(dst)} (authoritative idx2str)')

    # The model binary lives under the run's logged model (older runs) OR
    # directly under the run's own artifact tree (newer runs log the model as a
    # run artifact, not a separate logged-model entity — search_logged_models
    # returns nothing for those). Try the logged model first, fall back to the
    # run root.
    from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
    models = mlflow.search_logged_models(experiment_ids=[run.info.experiment_id],
                                         filter_string=f"source_run_id='{rid}'",
                                         output_format='list')
    named = [m for m in models if m.name == arch] or models
    # Download ONLY the model binary — the logged model dir also holds large
    # files the demo doesn't need (python_model.pkl, env specs), and pulling
    # everything makes the fetch slow and flaky. For item2vec that's the .bin
    # PLUS any gensim .npy sidecars (large models store vectors separately as
    # <name>.bin.wv.vectors.npy etc.).
    pattern = '*.bin*' if arch == 'item2vec' else '*.pth'
    if named:
        source = f'logged model {named[0].model_id}'
        repo = get_artifact_repository(mlflow.get_logged_model(named[0].model_id).artifact_location)
    else:
        source = 'run artifacts'
        repo = get_artifact_repository(run.info.artifact_uri)
    paths = _find_artifacts(repo, pattern)
    if not paths:
        print(f'  WARNING: no {pattern} in {source} for run {rid} — model binary skipped')
        return files
    tmp = os.path.join(inc, f'_model_{arch}')
    os.makedirs(tmp, exist_ok=True)
    for path in paths:
        local = _retry(lambda p=path: repo.download_artifacts(p, tmp),
                       label=os.path.basename(path))
        base = os.path.basename(local)
        if base == 'model.pth':                       # per-run name (see vocab.json)
            base = f'model_{rid[:8]}.pth'
        dst = os.path.join(inc, base)
        shutil.move(local, dst)
        if dst.endswith(('.bin', '.pth')):
            files['model_file'] = dst
        print(f'  {base}  ({source})')
    shutil.rmtree(tmp, ignore_errors=True)
    return files


def stage_run(arch, run, files, run_dir, n_samples):
    """Stage one fetched run into the registry: model checkpoint, vocab.csv,
    run.json (params + metrics), and per-run sample resumes."""
    from scripts import import_mlflow_artifacts as staging

    os.makedirs(run_dir, exist_ok=True)
    rid = run.info.run_id

    if files['model_file'] is None:
        print(f'  [{arch} {rid[:8]}] no model binary — run NOT staged as a model')
    elif arch == 'bert4rec':
        staging.stage_bert4rec(files['model_file'],
                               [files['vocab_json']] if files['vocab_json'] else [],
                               [files['vocab_csv']] if files['vocab_csv'] else [],
                               os.path.join(run_dir, 'bert4rec'))
    else:
        staging.stage_item2vec(files['model_file'], run_dir)

    vocab_csv = None
    if files['vocab_csv']:
        vocab_csv = os.path.join(run_dir, 'vocab.csv')
        shutil.copy(files['vocab_csv'], vocab_csv)

    # Params + headline metrics for the UI's run-properties / model-info views.
    metrics = {k: v for k, v in run.data.metrics.items()
               if not k.startswith(('track_', 'finetune_track_'))}
    with open(os.path.join(run_dir, 'run.json'), 'w') as f:
        json.dump({
            'run_id': rid,
            'run_name': run.data.tags.get('mlflow.runName'),
            'run_tag': run.data.tags.get('run_tag'),
            'architecture': arch,
            'start_time': str(run.info.start_time),
            'params': dict(run.data.params),
            'metrics': metrics,
        }, f, indent=2)

    if files['predictions']:
        csv = max(files['predictions'], key=os.path.getsize)
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = [sys.executable, os.path.join(scripts_dir, 'prepare_samples.py'),
               '--csv', csv, '--n', str(n_samples),
               '--out', os.path.join(run_dir, 'sample_resumes.json')]
        if vocab_csv:
            cmd += ['--vocab-csv', vocab_csv]
        subprocess.run(cmd, check=True)
    else:
        print(f'  [{arch} {rid[:8]}] no predictions CSV — run has no sample resumes')


def sync_new_runs(env='test-prod-sj', ai_conf=None, n_samples=300, max_runs=25):
    """Stage every FINISHED experiment run newer than the newest staged run that
    isn't in the registry yet — called by the demo app at startup so fresh
    training runs appear without a manual fetch. Per-run failures are non-fatal.
    Returns the number of newly staged runs."""
    runs_dir = os.path.join(config.ARTIFACTS_DIR, 'runs')
    staged = set(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else set()

    # Newest staged start_time (ms) minus an hour of slack; 14 days if empty.
    since = 0
    for rid in staged:
        try:
            with open(os.path.join(runs_dir, rid, 'run.json')) as f:
                since = max(since, int(json.load(f).get('start_time') or 0))
        except Exception:
            pass
    since = since - 3600_000 if since else _now_ms() - 14 * 86400_000

    partner = env.split('-')[-1]
    ai_conf = ai_conf or os.path.abspath(os.path.join(
        config.DEMO_ROOT, '..', 'datawarehouse-configurations', partner, 'ai', 'ai.conf'))
    mlflow = connect(env, ai_conf)
    exp = mlflow.get_experiment_by_name(EXPERIMENT.format(partner=partner))
    if exp is None:
        return 0
    df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"attributes.status = 'FINISHED' and attributes.start_time > {since}",
        order_by=['start_time DESC'], max_results=max_runs)
    client = mlflow.MlflowClient()
    inc = os.path.join(config.DEMO_ROOT, 'incoming')
    os.makedirs(inc, exist_ok=True)

    n_staged = 0
    for rid in ([] if not len(df) else df['run_id'].tolist()):
        if rid in staged:
            continue
        run = client.get_run(rid)
        arch = run.data.tags.get('architecture')
        if arch not in ARCHITECTURES:
            print(f'[sync] {rid[:8]} ({run.data.tags.get("mlflow.runName")}): '
                  f'architecture {arch!r} not supported by the demo — skipped')
            continue
        try:
            files = fetch_run_files(mlflow, arch, run, inc)
            stage_run(arch, run, files, os.path.join(runs_dir, rid), n_samples)
            n_staged += 1
        except Exception as exc:
            print(f'[sync] {rid[:8]} FAILED: {exc}')
    return n_staged


def _now_ms():
    import time
    return int(time.time() * 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--item2vec', action='append',
                        help='explicit item2vec run id (repeatable)')
    parser.add_argument('--bert4rec', action='append',
                        help='explicit bert4rec run id (repeatable)')
    parser.add_argument('--tag', help='resolve runs by tags.run_tag')
    parser.add_argument('--skills-only', action='store_true',
                        help="only consider runs trained with add_skills=True")
    parser.add_argument('--env', default='test-prod-sj')
    parser.add_argument('--ai-conf', default=None,
                        help='path to ai.conf (default: ../datawarehouse-configurations/{partner}/ai/ai.conf)')
    parser.add_argument('--no-import', action='store_true',
                        help='download into incoming/ only; skip import/samples/transitions')
    parser.add_argument('--n-samples', type=int, default=300)
    args = parser.parse_args()

    partner = args.env.split('-')[-1]   # test-prod-sj -> sj
    ai_conf = args.ai_conf or os.path.abspath(os.path.join(
        config.DEMO_ROOT, '..', 'datawarehouse-configurations', partner, 'ai', 'ai.conf'))
    mlflow = connect(args.env, ai_conf)

    runs = resolve_runs(mlflow, EXPERIMENT.format(partner=partner), args)

    inc = os.path.join(config.DEMO_ROOT, 'incoming')
    os.makedirs(inc, exist_ok=True)
    for pattern in MANAGED_GLOBS:
        for f in glob.glob(os.path.join(inc, pattern)):
            os.remove(f)
    for d in glob.glob(os.path.join(inc, '_model_*')):
        shutil.rmtree(d, ignore_errors=True)

    predictions = []
    fetched = []
    for arch, run in runs:
        files = fetch_run_files(mlflow, arch, run, inc)
        predictions += files['predictions']
        fetched.append((arch, run, files))
        if not args.no_import:
            print(f'\n── staging {arch} run {run.info.run_id[:8]} into the registry ──')
            stage_run(arch, run, files,
                      os.path.join(config.ARTIFACTS_DIR, 'runs', run.info.run_id),
                      args.n_samples)

    with open(os.path.join(inc, 'RUN_INFO.json'), 'w') as f:
        json.dump([{
            'run_id': run.info.run_id,
            'architecture': arch,
            'run_name': run.data.tags.get('mlflow.runName'),
            'run_tag': run.data.tags.get('run_tag'),
            'start_time': str(run.info.start_time),
            'params': dict(run.data.params),
        } for arch, run, _ in fetched], f, indent=2)
    print(f'\nRun provenance -> {os.path.join(inc, "RUN_INFO.json")}')

    if args.no_import:
        print('--no-import: done. Files left in incoming/.')
        return

    if predictions:
        # Global transitions (Sankey view) from the largest predictions CSV.
        csv = max(predictions, key=os.path.getsize)
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        print(f'\n── transitions from {os.path.basename(csv)} ──')
        subprocess.run([sys.executable, os.path.join(scripts_dir, 'build_transitions.py'),
                        '--csv', csv], check=True)
    else:
        print('\nWARNING: no predictions CSV downloaded — transitions not rebuilt.')

    print('\nAll staged. Restart the app (./run.sh) to load the new artifacts.')


if __name__ == '__main__':
    main()
