"""Token utilities mirroring datawarehouse-ai's career_path_transformer_common.

The demo keeps its own copy of the tiny tokenisation rules (prefixes, ordering,
normalisation) so it has no import-path dependency on the training repo, but
the behaviour must match build_tokens() exactly: W_TITLE is emitted first in a
work bundle, values are stripped/lowercased, empty values dropped.
"""

W_TITLE_PREFIX    = 'W_TITLE:'
W_ROLE_PREFIX     = 'W_ROLE:'
W_SUBROLE_PREFIX  = 'W_SUBROLE:'
W_INDUSTRY_PREFIX = 'W_INDUSTRY:'
E_MAJOR_PREFIX    = 'E_MAJOR:'
E_DEGREE_PREFIX   = 'E_DEGREE:'
E_TYPE_PREFIX     = 'E_TYPE:'

ALL_PREFIXES = (
    W_TITLE_PREFIX, W_ROLE_PREFIX, W_SUBROLE_PREFIX, W_INDUSTRY_PREFIX,
    E_MAJOR_PREFIX, E_DEGREE_PREFIX, E_TYPE_PREFIX,
)

# Display metadata for the UI
TOKEN_TYPES = {
    'W_TITLE':    {'label': 'Job title',  'group': 'work'},
    'W_ROLE':     {'label': 'Role',       'group': 'work'},
    'W_SUBROLE':  {'label': 'Sub-role',   'group': 'work'},
    'W_INDUSTRY': {'label': 'Industry',   'group': 'work'},
    'E_MAJOR':    {'label': 'Major',      'group': 'education'},
    'E_DEGREE':   {'label': 'Degree',     'group': 'education'},
    'E_TYPE':     {'label': 'School type','group': 'education'},
}


def _normalise(value):
    if value is None:
        return None
    s = str(value).strip().lower()
    return s or None


def _emit(prefix, value):
    v = _normalise(value)
    return f'{prefix}{v}' if v else None


def tokens_from_experience(exp: dict) -> list:
    """One UI experience entry -> ordered token bundle (same order as
    build_tokens in the training repo: W_TITLE first within a work bundle)."""
    if exp.get('type') == 'WORK':
        candidates = [
            _emit(W_TITLE_PREFIX,    exp.get('title')),
            _emit(W_ROLE_PREFIX,     exp.get('role')),
            _emit(W_SUBROLE_PREFIX,  exp.get('subrole')),
            _emit(W_INDUSTRY_PREFIX, exp.get('industry')),
        ]
    elif exp.get('type') == 'EDUCATION':
        candidates = [
            _emit(E_MAJOR_PREFIX,  exp.get('major')),
            _emit(E_DEGREE_PREFIX, exp.get('degree')),
            _emit(E_TYPE_PREFIX,   exp.get('school_type')),
        ]
    else:
        candidates = []
    return [t for t in candidates if t]


def tokens_from_resume(experiences: list) -> list:
    """Ordered list of UI experience entries -> flat token sequence."""
    out = []
    for exp in experiences:
        out.extend(tokens_from_experience(exp))
    return out


def parse_token_string(s: str) -> list:
    """'A | B | C' -> ['A', 'B', 'C'] (the eval-CSV context format)."""
    return [t.strip() for t in str(s).split('|') if t.strip()]


def token_type(token: str) -> str:
    return token.split(':', 1)[0] if ':' in token else 'UNKNOWN'


def token_value(token: str) -> str:
    return token.split(':', 1)[1] if ':' in token else token


_FIELD_BY_TYPE = {
    'W_TITLE': 'title', 'W_ROLE': 'role', 'W_SUBROLE': 'subrole', 'W_INDUSTRY': 'industry',
    'E_MAJOR': 'major', 'E_DEGREE': 'degree', 'E_TYPE': 'school_type',
}

# Canonical emission order within one experience (build_tokens emits in this
# order): work = title→role→subrole→industry; education = major(s)→degree→type.
_CANON_RANK = {
    'W_TITLE': 0, 'W_ROLE': 1, 'W_SUBROLE': 2, 'W_INDUSTRY': 3,
    'E_MAJOR': 0, 'E_DEGREE': 1, 'E_TYPE': 2,
}
# E_MAJOR is the only field that may repeat within one experience (a double /
# triple major), so repeated E_MAJOR tokens stay in the same bundle.
_REPEATABLE = {'E_MAJOR'}


def group_token_bundles(tokens: list) -> list:
    """Split a flat token sequence into per-experience token bundles.

    Within one experience the tokens climb the canonical order above, so a new
    bundle starts when the order *resets* — the next token's rank is lower than
    the last (e.g. a new W_TITLE after a W_INDUSTRY, or an E_MAJOR after an
    E_DEGREE) — or when the work/education kind switches. The one exception is
    a repeated E_MAJOR (same rank, repeatable), which keeps multiple majors of
    one degree together as a single education experience.
    """
    bundles = []
    cur_kind, cur_toks, prev_rank, prev_type = None, None, None, None
    for tok in tokens:
        ttype = token_type(tok)
        if ttype not in _CANON_RANK:
            continue
        kind = 'EDUCATION' if ttype.startswith('E_') else 'WORK'
        rank = _CANON_RANK[ttype]
        same_repeat = (rank == prev_rank and ttype == prev_type and ttype in _REPEATABLE)
        new_bundle = (
            cur_toks is None
            or kind != cur_kind
            or rank < prev_rank
            or (rank == prev_rank and not same_repeat)
        )
        if new_bundle:
            cur_kind, cur_toks = kind, []
            bundles.append(cur_toks)
        cur_toks.append(tok)
        prev_rank, prev_type = rank, ttype
    return bundles


def bundle_kind(bundle: list) -> str:
    return 'EDUCATION' if token_type(bundle[0]).startswith('E_') else 'WORK'


def group_into_experiences(tokens: list) -> list:
    """Inverse of tokens_from_resume, for displaying a flat context as a resume.

    Repeated fields within one experience (e.g. a double major) are joined."""
    experiences = []
    for bundle in group_token_bundles(tokens):
        exp = {'type': bundle_kind(bundle)}
        for tok in bundle:
            field = _FIELD_BY_TYPE[token_type(tok)]
            val = token_value(tok)
            exp[field] = f'{exp[field]}, {val}' if field in exp else val
        experiences.append(exp)
    return experiences
