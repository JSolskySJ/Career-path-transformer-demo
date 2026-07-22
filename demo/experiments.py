"""Experiment codes from the Obsidian vault (read-only).

Each vault page Experiments/<CODE>.md declares its MLflow run in a blockquote
header line:

    > R-series · run `dazzling-roo-991` (`3fc79c6317…`) · FINISHED · …

On startup the demo scans those headers and labels staged runs with their
experiment code (R4, M1, …) so models can be tracked against the docs. Run
names are the primary key; run ids (sometimes truncated with a trailing `…`
in the docs) are the fallback. The vault is never written.
"""

import os
import re

VAULT_DIR = os.environ.get('CPT_VAULT_DIR',
                           os.path.expanduser('~/VAULTS/Career-Path-Transformer'))

_RUN_REF = re.compile(r'run `([^`]+)` \(`([0-9a-f]+)(…?)`')


def scan_vault(vault_dir: str = None) -> list:
    """[(code, run_name, run_id, id_truncated), …] from Experiments/*.md
    blockquote headers. Empty when the vault isn't present (demo still works)."""
    exp_dir = os.path.join(vault_dir or VAULT_DIR, 'Experiments')
    if not os.path.isdir(exp_dir):
        return []
    refs = []
    for fn in sorted(os.listdir(exp_dir)):
        if not fn.endswith('.md'):
            continue
        code = fn[:-3]
        with open(os.path.join(exp_dir, fn)) as f:
            for line in f:
                if line.startswith('>') and (m := _RUN_REF.search(line)):
                    refs.append((code, m.group(1), m.group(2), m.group(3) == '…'))
    return refs


def match_runs(runs: dict, vault_dir: str = None) -> int:
    """Label registry entries with their experiment code: entry['experiment'].
    Matches on run_name first, then run_id (prefix match when the doc
    truncated it). A run referenced by several pages gets 'A1/B2'."""
    refs = scan_vault(vault_dir)
    n = 0
    for entry in runs.values():
        codes = []
        for code, name, rid, trunc in refs:
            hit = (entry.get('run_name') == name
                   or (entry['run_id'].startswith(rid) if trunc
                       else entry['run_id'] == rid))
            if hit and code not in codes:
                codes.append(code)
        entry['experiment'] = '/'.join(codes) if codes else None
        n += bool(codes)
    return n
