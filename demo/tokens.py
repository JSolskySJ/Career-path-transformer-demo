"""Token utilities mirroring datawarehouse-ai's career_path_transformer_common.

The demo keeps its own copy of the tiny tokenisation rules (prefixes, ordering,
normalisation) so it has no import-path dependency on the training repo, but
the behaviour must match build_tokens() exactly: W_TITLE is emitted first in a
work bundle, values are stripped/lowercased, empty values dropped.

Prospective-worker runs add W_DURATION / W_COMPANY / W_SPEC / E_LEVEL to the
experience bundles and a candidate-level S_SKILL preamble PREPENDED to the
sequence (skills aren't tied to a date, so they sit before the first
experience — matching build_sequences in the training repo).
"""

W_TITLE_PREFIX    = 'W_TITLE:'
W_DURATION_PREFIX = 'W_DURATION:'
W_ROLE_PREFIX     = 'W_ROLE:'
W_SUBROLE_PREFIX  = 'W_SUBROLE:'
W_INDUSTRY_PREFIX = 'W_INDUSTRY:'
W_COMPANY_PREFIX  = 'W_COMPANY:'
W_SPEC_PREFIX     = 'W_SPEC:'
E_MAJOR_PREFIX    = 'E_MAJOR:'
E_DEGREE_PREFIX   = 'E_DEGREE:'
E_TYPE_PREFIX     = 'E_TYPE:'
E_LEVEL_PREFIX    = 'E_LEVEL:'
S_SKILL_PREFIX    = 'S_SKILL:'

ALL_PREFIXES = (
    W_TITLE_PREFIX, W_DURATION_PREFIX, W_ROLE_PREFIX, W_SUBROLE_PREFIX,
    W_INDUSTRY_PREFIX, W_COMPANY_PREFIX, W_SPEC_PREFIX,
    E_MAJOR_PREFIX, E_DEGREE_PREFIX, E_TYPE_PREFIX, E_LEVEL_PREFIX,
    S_SKILL_PREFIX,
)

# Display metadata for the UI
TOKEN_TYPES = {
    'W_TITLE':    {'label': 'Job title',      'group': 'work'},
    'W_DURATION': {'label': 'Tenure',         'group': 'work'},
    'W_ROLE':     {'label': 'Role',           'group': 'work'},
    'W_SUBROLE':  {'label': 'Sub-role',       'group': 'work'},
    'W_INDUSTRY': {'label': 'Industry',       'group': 'work'},
    'W_COMPANY':  {'label': 'Company',        'group': 'work'},
    'W_SPEC':     {'label': 'Specialisation', 'group': 'work'},
    'E_MAJOR':    {'label': 'Major',          'group': 'education'},
    'E_DEGREE':   {'label': 'Degree',         'group': 'education'},
    'E_TYPE':     {'label': 'School type',    'group': 'education'},
    'E_LEVEL':    {'label': 'Education level','group': 'education'},
    'S_SKILL':    {'label': 'Skill',          'group': 'skills'},
}


def _normalise(value):
    if value is None:
        return None
    s = str(value).strip().lower()
    return s or None


def _emit(prefix, value):
    v = _normalise(value)
    return f'{prefix}{v}' if v else None


def _emit_list(prefix, values):
    """Comma-separated string or list -> de-duped token list, order kept."""
    if values is None:
        return []
    if isinstance(values, str):
        values = values.split(',')
    seen, out = set(), []
    for v in values:
        tok = _emit(prefix, v)
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def tokens_from_experience(exp: dict) -> list:
    """One UI experience entry -> ordered token bundle (same order as
    build_tokens in the training repo: W_TITLE first, duration right after)."""
    if exp.get('type') == 'WORK':
        candidates = [
            _emit(W_TITLE_PREFIX,    exp.get('title')),
            _emit(W_DURATION_PREFIX, exp.get('duration')),
            _emit(W_ROLE_PREFIX,     exp.get('role')),
            _emit(W_SUBROLE_PREFIX,  exp.get('subrole')),
            _emit(W_INDUSTRY_PREFIX, exp.get('industry')),
            _emit(W_COMPANY_PREFIX,  exp.get('company')),
        ] + _emit_list(W_SPEC_PREFIX, exp.get('spec'))
    elif exp.get('type') == 'EDUCATION':
        candidates = [
            _emit(E_MAJOR_PREFIX,  exp.get('major')),
            _emit(E_DEGREE_PREFIX, exp.get('degree')),
            _emit(E_TYPE_PREFIX,   exp.get('school_type')),
            _emit(E_LEVEL_PREFIX,  exp.get('level')),
        ]
    elif exp.get('type') == 'SKILLS':
        candidates = _emit_list(S_SKILL_PREFIX, exp.get('skills'))
    else:
        candidates = []
    return [t for t in candidates if t]


def tokens_from_resume(experiences: list) -> list:
    """Ordered list of UI experience entries -> flat token sequence. SKILLS
    entries are emitted first regardless of position (training prepends the
    skill preamble to the sequence)."""
    out = []
    for exp in experiences:
        if exp.get('type') == 'SKILLS':
            out = tokens_from_experience(exp) + out
        else:
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
    'W_TITLE': 'title', 'W_DURATION': 'duration', 'W_ROLE': 'role',
    'W_SUBROLE': 'subrole', 'W_INDUSTRY': 'industry', 'W_COMPANY': 'company',
    'W_SPEC': 'spec',
    'E_MAJOR': 'major', 'E_DEGREE': 'degree', 'E_TYPE': 'school_type',
    'E_LEVEL': 'level',
    'S_SKILL': 'skills',
}

# Canonical emission order within one experience (build_tokens emits in this
# order): work = title→duration→role→subrole→industry→company→spec(s);
# education = major(s)→degree→type→level; skills = a flat preamble bundle.
_CANON_RANK = {
    'W_TITLE': 0, 'W_DURATION': 1, 'W_ROLE': 2, 'W_SUBROLE': 3,
    'W_INDUSTRY': 4, 'W_COMPANY': 5, 'W_SPEC': 6,
    'E_MAJOR': 0, 'E_DEGREE': 1, 'E_TYPE': 2, 'E_LEVEL': 3,
    'S_SKILL': 0,
}
# Fields that may repeat within one experience (double majors, several
# specialisations, the whole skill preamble).
_REPEATABLE = {'E_MAJOR', 'W_SPEC', 'S_SKILL'}


def _token_kind(ttype: str) -> str:
    if ttype.startswith('E_'):
        return 'EDUCATION'
    if ttype.startswith('S_'):
        return 'SKILLS'
    return 'WORK'


def group_token_bundles(tokens: list) -> list:
    """Split a flat token sequence into per-experience token bundles.

    Within one experience the tokens climb the canonical order above, so a new
    bundle starts when the order *resets* — the next token's rank is lower than
    the last (e.g. a new W_TITLE after a W_INDUSTRY, or an E_MAJOR after an
    E_DEGREE) — or when the work/education/skills kind switches. The exception
    is a repeated repeatable type (same rank), which keeps multiple majors /
    specialisations / skills together as a single bundle.
    """
    bundles = []
    cur_kind, cur_toks, prev_rank, prev_type = None, None, None, None
    for tok in tokens:
        ttype = token_type(tok)
        if ttype not in _CANON_RANK:
            continue
        kind = _token_kind(ttype)
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
    return _token_kind(token_type(bundle[0]))


def _self_check():
    """Smallest checks that fail if bundling / skills-prepending breaks."""
    seq = ['S_SKILL:welding', 'S_SKILL:forklift',
           'E_MAJOR:physics', 'E_MAJOR:maths', 'E_DEGREE:bachelors', 'E_LEVEL:university',
           'W_TITLE:engineer', 'W_DURATION:1-2y', 'W_ROLE:engineering',
           'W_COMPANY:acme', 'W_SPEC:welding', 'W_SPEC:pipes',
           'W_TITLE:manager']
    kinds = [bundle_kind(b) for b in group_token_bundles(seq)]
    assert kinds == ['SKILLS', 'EDUCATION', 'WORK', 'WORK'], kinds
    exps = group_into_experiences(seq)
    assert exps[0] == {'type': 'SKILLS', 'skills': 'welding, forklift'}, exps[0]
    assert exps[2]['duration'] == '1-2y' and exps[2]['spec'] == 'welding, pipes', exps[2]
    rebuilt = tokens_from_resume([
        {'type': 'WORK', 'title': 'engineer', 'duration': '1-2y', 'role': 'engineering',
         'company': 'acme', 'spec': 'welding, pipes'},
        {'type': 'SKILLS', 'skills': 'welding, forklift'},   # prepended despite position
    ])
    assert rebuilt[:2] == ['S_SKILL:welding', 'S_SKILL:forklift'], rebuilt
    assert rebuilt[2:] == ['W_TITLE:engineer', 'W_DURATION:1-2y', 'W_ROLE:engineering',
                           'W_COMPANY:acme', 'W_SPEC:welding', 'W_SPEC:pipes'], rebuilt
    print('tokens self-check: OK')


def group_into_experiences(tokens: list) -> list:
    """Inverse of tokens_from_resume, for displaying a flat context as a resume.

    Repeated fields within one experience (e.g. a double major, several
    skills/specialisations) are joined."""
    experiences = []
    for bundle in group_token_bundles(tokens):
        exp = {'type': bundle_kind(bundle)}
        for tok in bundle:
            field = _FIELD_BY_TYPE[token_type(tok)]
            val = token_value(tok)
            exp[field] = f'{exp[field]}, {val}' if field in exp else val
        experiences.append(exp)
    return experiences


if __name__ == '__main__':
    _self_check()
