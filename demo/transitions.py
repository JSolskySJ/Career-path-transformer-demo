"""Empirical job-title transition store + Sankey builder.

Loads artifacts/transitions.json (built by scripts/build_transitions.py) and
turns a set of selected source titles into a layered Sankey graph.

Sankey construction rules (matching the demo brief):
  - For each selected source, take its top-K most frequent outgoing
    transitions. The "named" target set for a layer is the UNION of every
    source's top-K — so if title X is in source A's top-K but not B's, B's
    flow into X is still drawn as its own link (X is a named node), which is
    exactly how the overlap between selected titles becomes visible.
  - Any of a source's transitions that land outside the named set are summed
    into a per-source "Other" bucket (one Other node per source).
  - depth > 1 expands the named targets into the next layer and repeats.

Nodes are identified per (layer, title) so the diagram stays an acyclic
left-to-right flow even when careers loop (A→B→A appears as A@0, B@1, A@2).
"""

import json

from demo import config
from demo.tokens import token_value, W_TITLE_PREFIX

# Safety caps so deep / wide requests can't melt the browser. Truncation is
# reported back so the UI can note it.
MAX_SOURCES_PER_LAYER = 24
MAX_LINKS             = 500


class TransitionStore:

    def __init__(self, path: str = None):
        with open(path or config.TRANSITIONS_JSON) as f:
            data = json.load(f)
        self.transitions = data['transitions']
        self.source_freq = data['source_freq']
        self.meta        = data.get('meta', {})

    @classmethod
    def load_if_available(cls, path: str = None):
        import os
        path = path or config.TRANSITIONS_JSON
        return cls(path) if os.path.exists(path) else None

    # ── Source title list (for the picker) ────────────────────────────────────

    def source_titles(self) -> list:
        """All titles that have outgoing transitions, most active first."""
        out = []
        for title, freq in sorted(self.source_freq.items(), key=lambda kv: -kv[1]):
            out.append({
                'title': title,
                'value': token_value(title),
                'out_count': freq,
                'degree': len(self.transitions.get(title, {})),
            })
        return out

    # ── Sankey ─────────────────────────────────────────────────────────────────

    def _top_k_targets(self, title: str, top_k: int) -> list:
        trans = self.transitions.get(title, {})
        return [t for t, _ in sorted(trans.items(), key=lambda kv: -kv[1])[:top_k]]

    def build_sankey(self, selected: list, top_k: int = 10, depth: int = 1) -> dict:
        selected = [s for s in dict.fromkeys(selected) if s in self.transitions]
        nodes, node_index = [], {}
        truncated = False

        def node(layer: int, title: str, is_other: bool = False, other_of: str = None) -> int:
            key = (layer, 'OTHER::' + other_of) if is_other else (layer, title)
            if key not in node_index:
                node_index[key] = len(nodes)
                nodes.append({
                    'label': 'Other' if is_other else token_value(title),
                    'title': title,
                    'layer': layer,
                    'is_other': is_other,
                    'is_selected': (layer == 0 and not is_other),
                })
            return node_index[key]

        links = {}   # (src_idx, tgt_idx) -> value

        current = list(selected)
        for s in current:
            node(0, s)

        for layer in range(max(1, depth)):
            if not current:
                break
            # Named set for this layer = union of each current source's top-K.
            named = set()
            for s in current:
                named.update(self._top_k_targets(s, top_k))

            for s in current:
                trans = self.transitions.get(s, {})
                if not trans:
                    continue
                src_idx = node(layer, s)
                other_total = 0
                for to, cnt in trans.items():
                    if to in named:
                        tgt = node(layer + 1, to)
                        links[(src_idx, tgt)] = links.get((src_idx, tgt), 0) + cnt
                    else:
                        other_total += cnt
                if other_total:
                    oidx = node(layer + 1, s, is_other=True, other_of=s)
                    links[(src_idx, oidx)] = links.get((src_idx, oidx), 0) + other_total

            # Next layer's sources = the named targets, most active first, capped.
            ranked = sorted(named, key=lambda t: -self.source_freq.get(t, 0))
            if len(ranked) > MAX_SOURCES_PER_LAYER:
                ranked = ranked[:MAX_SOURCES_PER_LAYER]
                truncated = True
            current = ranked

        link_items = sorted(links.items(), key=lambda kv: -kv[1])
        if len(link_items) > MAX_LINKS:
            link_items = link_items[:MAX_LINKS]
            truncated = True
        link_list = [{'source': s, 'target': t, 'value': v} for (s, t), v in link_items]

        return {
            'nodes': nodes,
            'links': link_list,
            'selected': selected,
            'top_k': top_k,
            'depth': max(1, depth),
            'truncated': truncated,
        }
