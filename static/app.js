/* Career Path Transformer demo frontend. Vanilla JS + Plotly. */

const state = {
  samples: [],
  selectedSample: null,
  entries: [],            // builder entries
  entrySeq: 0,
  activeTab: 'samples',
  lastPrediction: null,
  runs: {},               // run_id -> /api/status model entry
  selectedModels: new Set(),   // run_ids used for prediction
  resumeRun: null,        // run_id the sample resumes come from
  modelsLoaded: [],       // loaded run_ids
  // job title flow (Sankey)
  view: 'predict',
  flowTitles: [],         // [{title, value, out_count, degree}]
  flowSelected: [],       // selected W_TITLE tokens, in selection order
  flowLoaded: false,
  topK: 10,
  depth: 1,
  sankeyNodes: [],
  hoveredNode: null,
};

const $ = (sel) => document.querySelector(sel);

const FIELD_SPECS = {
  WORK: [
    ['title',    'W_TITLE',    'Job title *'],
    ['duration', 'W_DURATION', 'Tenure bucket (e.g. 1-2y)'],
    ['role',     'W_ROLE',     'Role (e.g. engineering)'],
    ['subrole',  'W_SUBROLE',  'Sub-role (e.g. software)'],
    ['industry', 'W_INDUSTRY', 'Industry'],
    ['company',  'W_COMPANY',  'Company'],
    ['spec',     'W_SPEC',     'Specialisations (comma-separated)'],
  ],
  EDUCATION: [
    ['major',       'E_MAJOR',  'Major (e.g. computer science)'],
    ['degree',      'E_DEGREE', 'Degree (e.g. bachelors)'],
    ['school_type', 'E_TYPE',   'School type'],
    ['level',       'E_LEVEL',  'Education level'],
  ],
  SKILLS: [
    ['skills', 'S_SKILL', 'Skills (comma-separated, e.g. forklift, welding)'],
  ],
};

// Comma-separated multi-value fields → one token per value.
const MULTI_FIELDS = new Set(['spec', 'skills']);

/* ── Token building (mirrors demo/tokens.py) ────────────────────────────── */

function norm(v) { return (v || '').trim().toLowerCase(); }

function tokensFromEntry(e) {
  const out = [];
  for (const [field, prefix] of FIELD_SPECS[e.type]) {
    const values = MULTI_FIELDS.has(field)
      ? (e.values[field] || '').split(',') : [e.values[field]];
    for (const raw of values) {
      const v = norm(raw);
      if (v && !out.includes(`${prefix}:${v}`)) out.push(`${prefix}:${v}`);
    }
  }
  return out;
}

function tokensFromEntries(entries) {
  // SKILLS entries are emitted first regardless of position — training
  // prepends the skill preamble to the sequence.
  const skills = entries.filter(e => e.type === 'SKILLS');
  const rest = entries.filter(e => e.type !== 'SKILLS');
  return [...skills, ...rest].flatMap(tokensFromEntry);
}

function currentTokens() {
  if (state.activeTab === 'samples') {
    return state.selectedSample ? state.selectedSample.context_tokens : [];
  }
  return tokensFromEntries(state.entries);
}

function currentTarget() {
  return (state.activeTab === 'samples' && state.selectedSample)
    ? state.selectedSample.target : null;
}

/* ── Status ─────────────────────────────────────────────────────────────── */

// Run start date as YYYY-MM-DD ('' when unknown, e.g. locally-trained runs).
function runDate(info) {
  const ms = +info.start_time;
  return ms ? new Date(ms).toISOString().slice(0, 10) : '';
}

// Newest run first everywhere — makes the most recent run obvious.
function runsByDate() {
  return Object.entries(state.runs)
    .sort((a, b) => (+b[1].start_time || 0) - (+a[1].start_time || 0));
}

async function loadStatus() {
  const res = await fetch('/api/status');
  const data = await res.json();
  state.runs = data.models;
  const box = $('#model-status');
  box.innerHTML = '';
  const spaceSel = $('#space-model');
  spaceSel.innerHTML = '';
  state.titleCounts = {};
  for (const [rid, info] of runsByDate()) {
    const chip = document.createElement('span');
    chip.className = 'chip' + (info.loaded ? '' : ' err');
    chip.textContent = info.loaded
      ? `${info.label} · ${info.title_count.toLocaleString()} titles`
      : `${info.label} unavailable`;
    if (!info.loaded) chip.title = info.error;
    box.appendChild(chip);
    if (info.loaded) {
      state.modelsLoaded.push(rid);
      state.selectedModels.add(rid);
      state.titleCounts[rid] = info.title_count;
      const opt = document.createElement('option');
      opt.value = rid;
      opt.textContent = runDate(info) ? `${info.label} · ${runDate(info)}` : info.label;
      spaceSel.appendChild(opt);
    }
  }
  // Default the embedding-space view to a BERT4Rec run when available.
  const b4 = state.modelsLoaded.find(r => state.runs[r].architecture === 'bert4rec');
  if (b4) spaceSel.value = b4;
  renderModelSelect();
}

/* ── Model selection (compare model versions side by side) ─────────────── */

function fmtMetric(v) {
  return Number.isInteger(v) ? v.toLocaleString() : Number(v).toFixed(4);
}

function kvTable(obj, fmt = (v) => v) {
  const rows = Object.entries(obj)
    .map(([k, v]) => `<tr><td>${k}</td><td>${fmt(v)}</td></tr>`).join('');
  return rows ? `<table class="kv">${rows}</table>` : '<p class="hint">none recorded</p>';
}

function modelInfoHtml(info) {
  const ranking = info.loaded
    ? `ranks over <b>${info.title_count.toLocaleString()}</b>` +
      (info.ranking_restricted
        ? ` of ${info.full_title_count.toLocaleString()} trained titles (ranking domain)`
        : ' titles (full trained vocabulary)')
    : '';
  return `
    <p class="hint">${info.architecture} · run <code>${info.run_id}</code>
      ${runDate(info) ? `· trained ${runDate(info)}` : ''}
      ${info.run_tag && info.run_tag !== 'N/A' ? `· tag <code>${info.run_tag}</code>` : ''}
      ${ranking ? `<br>${ranking}` : ''}</p>
    <h4>Key metrics</h4>
    ${kvTable(info.metrics, fmtMetric)}
    <h4>Model-side transformations</h4>
    ${kvTable(info.transformations)}
    <details class="all-params"><summary>All run params (${Object.keys(info.params).length})</summary>
      ${kvTable(info.params)}</details>`;
}

function renderModelSelect() {
  const box = $('#model-select');
  box.innerHTML = '';
  for (const rid of state.modelsLoaded) {
    const info = state.runs[rid];
    const row = document.createElement('div');
    row.className = 'model-row';
    const id = `model-check-${rid}`;
    row.innerHTML = `
      <label for="${id}">
        <input type="checkbox" id="${id}" ${state.selectedModels.has(rid) ? 'checked' : ''}>
        <b>${info.architecture}</b> · ${info.run_name}
        ${runDate(info) ? `<span class="run-date">${runDate(info)}</span>` : ''}
        ${info.metrics.test_recall_at_10 !== undefined
          ? `<span class="hint">R@10 ${info.metrics.test_recall_at_10.toFixed(3)}</span>` : ''}
      </label>
      <details class="model-info"><summary>info</summary>${modelInfoHtml(info)}</details>`;
    row.querySelector('input').onchange = (e) => {
      e.target.checked ? state.selectedModels.add(rid) : state.selectedModels.delete(rid);
    };
    box.appendChild(row);
  }
}

/* ── Samples ────────────────────────────────────────────────────────────── */

async function loadSamples(runId) {
  const res = await fetch('/api/samples' + (runId ? `?run=${encodeURIComponent(runId)}` : ''));
  const data = await res.json();
  state.samples = data.samples;
  state.resumeRun = data.run;
  state.selectedSample = null;
  renderResumeSource();
  renderRunProps();
  renderSampleList();
}

function renderResumeSource() {
  const sel = $('#resume-source');
  sel.innerHTML = '';
  const sources = runsByDate().map(([, r]) => r).filter(r => r.has_samples);
  for (const r of sources) {
    const opt = document.createElement('option');
    opt.value = r.run_id;
    const date = runDate(r);
    opt.textContent = `${r.label}${date ? ' · ' + date : ''} (${r.run_id.slice(0, 8)})`;
    sel.appendChild(opt);
  }
  if (state.resumeRun) sel.value = state.resumeRun;
  sel.disabled = sources.length <= 1;
}

function renderRunProps() {
  const box = $('#run-props');
  const info = state.runs[state.resumeRun];
  if (!info) { box.innerHTML = ''; return; }
  const ranking = info.loaded
    ? (info.ranking_restricted
        ? `ranks over <b>${info.title_count.toLocaleString()}</b> of
           ${info.full_title_count.toLocaleString()} trained titles (ranking domain)`
        : `ranks over its full <b>${info.title_count.toLocaleString()}</b>-title vocabulary`)
    : '<span class="warn">model not loaded</span>';
  const chips = Object.entries(info.transformations)
    .map(([k, v]) => `<span class="chip-sm prop" title="${k}">${k}=${v}</span>`)
    .join('');
  box.innerHTML = `
    <p class="hint">Held-out resumes from <b>${info.label}</b> — the model ${ranking}.</p>
    <div class="chips-inline">${chips || '<span class="hint">no transformation params recorded</span>'}</div>`;
}

function renderSampleList() {
  const cat = $('#sample-category').value;
  const q = norm($('#sample-search').value);
  const list = $('#sample-list');
  list.innerHTML = '';
  const items = state.samples.filter(s =>
    (!cat || s.category === cat) &&
    (!q || s.label.includes(q) || s.context_tokens.join(' ').includes(q)));
  if (!items.length) {
    list.innerHTML = '<div class="sample-item">No samples — run scripts/prepare_samples.py</div>';
    return;
  }
  for (const s of items) {
    const div = document.createElement('div');
    div.className = 'sample-item' +
      (state.selectedSample && state.selectedSample.id === s.id ? ' selected' : '');
    const nSkills = s.context_tokens.filter(t => t.startsWith('S_SKILL:')).length;
    const skills = nSkills ? `<span class="skill-count" title="${nSkills} skills">🛠${nSkills}</span>` : '';
    div.innerHTML = `<span>${s.label}</span><span class="tags">${skills}<span class="cat">${s.category.replace('_', ' ')}</span></span>`;
    div.onclick = () => { state.selectedSample = s; renderSampleList(); onResumeChanged(); };
    list.appendChild(div);
  }
}

/* ── Builder ────────────────────────────────────────────────────────────── */

function addEntry(type) {
  state.entries.push({ id: ++state.entrySeq, type, values: {} });
  renderBuilder();
  onResumeChanged();
}

function renderBuilder() {
  const box = $('#builder-entries');
  box.innerHTML = '';
  state.entries.forEach((entry, i) => {
    const div = document.createElement('div');
    div.className = 'entry ' + entry.type.toLowerCase();
    const head = document.createElement('div');
    head.className = 'entry-head';
    const badge = { WORK: '💼 WORK', EDUCATION: '🎓 EDUCATION', SKILLS: '🛠 SKILLS' }[entry.type];
    head.innerHTML = `<span class="badge">${badge} #${i + 1}</span>`;
    const btns = document.createElement('span');
    for (const [label, fn, enabled] of [
      ['↑', () => moveEntry(i, -1), i > 0],
      ['↓', () => moveEntry(i, 1), i < state.entries.length - 1],
      ['✕', () => removeEntry(i), true],
    ]) {
      const b = document.createElement('button');
      b.className = 'icon';
      b.textContent = label;
      b.disabled = !enabled;
      b.onclick = fn;
      btns.appendChild(b);
    }
    head.appendChild(btns);
    div.appendChild(head);

    const fields = document.createElement('div');
    fields.className = 'fields';
    for (const [field, prefix, placeholder] of FIELD_SPECS[entry.type]) {
      const wrap = document.createElement('div');
      wrap.className = 'field';
      const input = document.createElement('input');
      input.type = 'text';
      input.placeholder = placeholder;
      input.value = entry.values[field] || '';
      input.oninput = () => {
        entry.values[field] = input.value;
        onResumeChanged(false);
        autocomplete(input, wrap, prefix, entry, field);
      };
      input.onblur = () => setTimeout(() => clearSuggestions(wrap), 180);
      wrap.appendChild(input);
      fields.appendChild(wrap);
    }
    div.appendChild(fields);
    box.appendChild(div);
  });
}

function moveEntry(i, d) {
  const [e] = state.entries.splice(i, 1);
  state.entries.splice(i + d, 0, e);
  renderBuilder();
  onResumeChanged();
}

function removeEntry(i) {
  state.entries.splice(i, 1);
  renderBuilder();
  onResumeChanged();
}

let acTimer = null;
function autocomplete(input, wrap, prefix, entry, field) {
  clearTimeout(acTimer);
  // Multi-value fields autocomplete the segment after the last comma.
  const multi = MULTI_FIELDS.has(field);
  const parts = multi ? input.value.split(',') : [input.value];
  const q = norm(parts[parts.length - 1]);
  if (!q) { clearSuggestions(wrap); return; }
  acTimer = setTimeout(async () => {
    const res = await fetch(`/api/vocab?type=${prefix}&q=${encodeURIComponent(q)}&limit=12`);
    const values = await res.json();
    clearSuggestions(wrap);
    if (!values.length) return;
    const sug = document.createElement('div');
    sug.className = 'suggestions';
    for (const v of values) {
      const item = document.createElement('div');
      item.textContent = v;
      item.onmousedown = () => {
        const next = multi ? [...parts.slice(0, -1), ` ${v}`].join(',').replace(/^ /, '') : v;
        entry.values[field] = next;
        input.value = next;
        clearSuggestions(wrap);
        onResumeChanged(false);
      };
      sug.appendChild(item);
    }
    wrap.appendChild(sug);
  }, 150);
}

function clearSuggestions(wrap) {
  wrap.querySelectorAll('.suggestions').forEach(el => el.remove());
}

/* ── Resume display + token preview ─────────────────────────────────────── */

// Repeated fields (skills, specs, double majors) arrive comma-joined.
function splitMulti(v) {
  return (v || '').split(',').map(s => s.trim()).filter(Boolean);
}

function chipRow(values, cls) {
  return values.length
    ? `<span class="chips-inline">${values.map(v => `<span class="chip-sm ${cls}">${v}</span>`).join('')}</span>`
    : '';
}

function renderSkillsExperience(exp) {
  // Skill preambles are long — a counter with a native dropdown.
  const skills = splitMulti(exp.skills);
  const el = document.createElement('details');
  el.className = 'exp skills';
  el.innerHTML = `
    <summary><span class="badge">🛠</span>
      <b>Skills</b> <span class="count">${skills.length}</span>
      <span class="meta">candidate-level, fed to the model before the first experience</span>
    </summary>
    <div class="skill-list">${chipRow(skills, 'skill')}</div>`;
  return el;
}

function renderResumeView() {
  const box = $('#resume-view');
  box.innerHTML = '';
  if (state.activeTab !== 'samples' || !state.selectedSample) return;
  for (const exp of state.selectedSample.experiences) {
    if (exp.type === 'SKILLS') {
      box.appendChild(renderSkillsExperience(exp));
      continue;
    }
    const div = document.createElement('div');
    const isWork = exp.type === 'WORK';
    div.className = 'exp ' + exp.type.toLowerCase();
    const main = isWork ? (exp.title || '—') : (exp.major || exp.degree || '—');
    const meta = isWork
      ? [exp.role, exp.subrole, exp.industry].filter(Boolean).join(' · ')
      : [exp.degree, exp.school_type, exp.level].filter(Boolean).join(' · ');
    const tenure = isWork && (exp.tenure || exp.duration)
      ? `<span class="tenure" title="Tenure in this role">⏱ ${exp.tenure || exp.duration}</span>` : '';
    const company = isWork && exp.company
      ? `<span class="company" title="Company">🏢 ${exp.company}</span>` : '';
    const specs = isWork ? chipRow(splitMulti(exp.spec), 'spec') : '';
    div.innerHTML = `
      <span class="badge">${isWork ? '💼' : '🎓'}</span>
      <span class="body"><b>${main}</b>${tenure}${company}<br>
        <span class="meta">${meta}</span>${specs}</span>`;
    box.appendChild(div);
  }
  const t = document.createElement('p');
  t.className = 'hint';
  t.innerHTML = `Actual next role (held out): <b>${state.selectedSample.target.replace('W_TITLE:', '')}</b>`;
  box.appendChild(t);
}

function renderTokenPreview(oov = []) {
  const tokens = currentTokens();
  const box = $('#token-preview');
  box.innerHTML = '';
  for (const tok of tokens) {
    const span = document.createElement('span');
    const type = tok.split(':')[0];
    span.className = 'tok ' + type + (oov.includes(tok) ? ' oov' : '');
    span.textContent = tok;
    if (oov.includes(tok)) span.title = 'Not in model vocabulary — ignored';
    box.appendChild(span);
  }
  $('#predict-btn').disabled = tokens.length === 0;
}

function onResumeChanged(rerenderView = true) {
  if (rerenderView) renderResumeView();
  renderTokenPreview();
  $('#sense-check').innerHTML = '';
}

/* ── Prediction ─────────────────────────────────────────────────────────── */

async function predict() {
  const tokens = currentTokens();
  if (!tokens.length) return;
  $('#predict-btn').disabled = true;
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tokens,
        target: currentTarget(),
        top_k: parseInt($('#top-k').value || '10', 10),
        models: [...state.selectedModels],
        domain: $('#rank-domain').value,
      }),
    });
    state.lastPrediction = await res.json();
    renderPredictions();
    const oov = new Set();
    for (const r of Object.values(state.lastPrediction.results)) {
      (r.unknown_tokens || []).forEach(t => oov.add(t));
    }
    renderTokenPreview([...oov]);
    await drawSpace();
  } finally {
    $('#predict-btn').disabled = false;
  }
}

function renderPredictions() {
  const { results, target } = state.lastPrediction;
  const box = $('#predictions');
  box.innerHTML = '';
  const sense = $('#sense-check');
  sense.innerHTML = '';

  for (const [rid, r] of Object.entries(results)) {
    const name = r.label || rid;
    const card = document.createElement('div');
    card.className = 'pred-card';
    if (r.error) {
      card.innerHTML = `<h3>${name}</h3><p class="warn">${r.error}</p>`;
      box.appendChild(card);
      continue;
    }
    const scoreLabel = r.architecture === 'item2vec' ? 'cosine' : 'probability';
    const nRanked = r.n_ranked ? ` · ranked over ${r.n_ranked.toLocaleString()} titles` : '';
    let html = `<h3>${name} <span class="hint">(${scoreLabel}${nRanked})</span></h3>`;
    html += confidenceHtml(r.confidence);
    const max = Math.max(...r.predictions.map(p => p.score), 1e-9);
    r.predictions.forEach((p, i) => {
      const title = p.token.replace('W_TITLE:', '');
      const hit = target && p.token === target;
      const flags = (p.sj ? ' <span class="flag">(SJ)</span>' : '')
                  + (p.tax ? ` <span class="flag">(${p.tax})</span>` : '');
      html += `
        <div class="pred-row${hit ? ' hit' : ''}">
          <span class="rank">${i + 1}</span>
          <span class="title">${title}${flags}${hit ? ' ✓' : ''}</span>
          <span class="bar-wrap"><span class="bar" style="width:${Math.max(3, 100 * p.score / max)}%"></span></span>
          <span class="score">${p.score.toFixed(4)}</span>
        </div>`;
    });
    if (r.unknown_tokens && r.unknown_tokens.length) {
      html += `<p class="warn">Ignored ${r.unknown_tokens.length} out-of-vocabulary token(s)</p>`;
    }
    card.innerHTML = html;
    // Drill-down: click a predicted title to open the full-page inspector
    // (bert4rec logit lens + attention; item2vec token contributions).
    card.querySelectorAll('.pred-row').forEach((row, i) => {
      row.classList.add('clickable');
      row.title = 'Click to inspect what the model is doing for this title (new tab)';
      row.onclick = () => openInspect(rid, r.predictions[i].token);
    });
    box.appendChild(card);

    if (target && r.target_rank !== undefined) {
      const div = document.createElement('div');
      const rank = r.target_rank;
      const n = r.n_ranked || (state.titleCounts ? state.titleCounts[rid] : null);
      const cls = rank === null ? 'bad' : rank <= 10 ? 'good' : rank <= 100 ? 'mid' : 'bad';
      const msg = rank === null
        ? 'not in this model\'s ranking domain (not ranked)'
        : `ranked <b>#${rank}</b> of ${n ? n.toLocaleString() + ' ' : ''}rankable titles`;
      div.className = 'sense ' + cls;
      div.innerHTML = `<b>${name}</b> — actual next role
        “${target.replace('W_TITLE:', '')}” ${msg}`;
      sense.appendChild(div);
    }
  }
}

/* ── Title drill-down — opens the full-page inspector in a new tab ───────── */

function openInspect(rid, title) {
  // The resume token list is too long for a URL — hand the payload to the new
  // tab via localStorage (shared across same-origin tabs), keyed by timestamp.
  for (const k of Object.keys(localStorage)) {          // prune old payloads
    if (k.startsWith('cpt_inspect_') && Date.now() - +k.split('_')[2] > 3600e3) {
      localStorage.removeItem(k);
    }
  }
  const key = `cpt_inspect_${Date.now()}`;
  localStorage.setItem(key, JSON.stringify({
    model: rid,
    title,
    tokens: state.lastPrediction.tokens,
    target: state.lastPrediction.target,
    label: (state.runs[rid] || {}).label || rid,
  }));
  window.open(`/inspect#${key}`, '_blank');
}

// Confidence summary: how decisive the model's ranking is for this resume.
// High entropy / tiny margin = the model is spreading probability thinly and is
// effectively guessing (the BERT4Rec story).
function confidenceHtml(c) {
  if (!c) return '';
  const ent = c.entropy;                 // 0 = certain, 1 = uniform
  const certainty = 1 - ent;             // flip so the bar reads "how sure"
  const label = ent >= 0.9 ? 'very unsure' : ent >= 0.75 ? 'unsure'
              : ent >= 0.5 ? 'moderate' : 'focused';
  const cls = ent >= 0.75 ? 'lo' : ent >= 0.5 ? 'mid' : 'hi';
  let top1, margin;
  if (c.unit === 'prob') {
    top1 = `top-1 ${(c.top1 * 100).toFixed(1)}%`;
    margin = c.margin != null ? `margin ${(c.margin * 100).toFixed(1)} pp` : '';
  } else {
    top1 = `top-1 cos ${c.top1.toFixed(2)}`;
    margin = c.margin != null ? `margin ${c.margin.toFixed(2)}` : '';
  }
  return `
    <div class="conf ${cls}" title="Normalised entropy of the model's score distribution over ${c.n} rankable titles. Higher = probability spread thinly across many titles (less sure).">
      <div class="conf-head">
        <span>confidence: <b>${label}</b></span>
        <span class="conf-meta">${top1} · ${margin} · spread ${(ent * 100).toFixed(0)}%</span>
      </div>
      <div class="conf-bar-wrap"><span class="conf-bar" style="width:${Math.max(2, certainty * 100).toFixed(0)}%"></span></div>
    </div>`;
}

/* ── Embedding space plot ───────────────────────────────────────────────── */

const KIND_STYLE = {
  background: { color: '#b8c2cf', size: 5,  symbol: 'circle' },
  prediction: { color: '#d97706', size: 11, symbol: 'diamond' },
  context:    { color: '#dc2626', size: 16, symbol: 'star' },
};
const TYPE_COLOR = { WORK: '#2563eb', EDUCATION: '#059669', SKILLS: '#b45309' };

async function drawSpace() {
  if ($('#space-section').classList.contains('hidden')) return;   // hidden — skip the work
  const tokens = currentTokens();
  if (!tokens.length || !state.modelsLoaded.length) return;
  const model = $('#space-model').value || state.modelsLoaded[0];
  const mode = $('#space-mode').value;
  const res = await fetch('/api/space', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      tokens, model, mode,
      top_k: parseInt($('#top-k').value || '10', 10),
      domain: $('#rank-domain').value,
    }),
  });
  const data = await res.json();
  if (data.error) {
    $('#space-info').textContent = data.error;
    $('#space-info').title = data.error;
    return;
  }
  const pts = data.points;
  const by = (kind) => pts.filter(p => p.kind === kind);

  // Most common job titles, always labelled — the shared reference frame.
  // Built per plot (traces can't be reused across Plotly divs).
  const landmarks = by('landmark');
  const landmarkTrace = () => ({
    x: landmarks.map(p => p.x), y: landmarks.map(p => p.y),
    text: landmarks.map(p => p.value),
    mode: 'markers+text', type: 'scatter', name: 'common titles',
    textposition: 'middle right', textfont: { size: 10.5, color: '#475569' },
    marker: { color: '#475569', size: 8, symbol: 'square-open', line: { width: 1.5 } },
    hovertemplate: '%{text}<extra>common title</extra>',
  });

  // ── Graph 1: the resume's tokens among nearby titles ────────────────────
  const resumeTraces = [];
  const bg = by('background');
  resumeTraces.push({
    x: bg.map(p => p.x), y: bg.map(p => p.y),
    text: bg.map(p => p.value),
    mode: 'markers', type: 'scatter', name: 'titles',
    marker: KIND_STYLE.background,
    hovertemplate: '%{text}<extra>title</extra>',
  });
  resumeTraces.push(landmarkTrace());   // after background → drawn above it

  const resume = by('resume').sort((a, b) => a.order - b.order);
  if (resume.length) {
    resumeTraces.push({
      x: resume.map(p => p.x), y: resume.map(p => p.y),
      mode: 'lines', type: 'scatter', name: 'career path',
      line: { color: '#64748b', dash: 'dot', width: 1.5 },
      hoverinfo: 'skip', showlegend: true,
    });
    resumeTraces.push({
      x: resume.map(p => p.x), y: resume.map(p => p.y),
      text: resume.map((p, i) => `${i + 1}. ${p.value}`),
      customdata: resume.map(p =>
        `${(p.tokens || [p.token]).join('<br>')}${p.in_window ? '' : '<br>(outside context window)'}`),
      mode: 'markers+text', type: 'scatter', name: 'experiences',
      textposition: 'top center', textfont: { size: 10, color: '#334155' },
      marker: {
        size: 13,
        color: resume.map(p => TYPE_COLOR[p.type] || '#475569'),
        opacity: resume.map(p => p.in_window ? 1 : 0.45),
        line: { width: 1, color: '#fff' },
      },
      hovertemplate: '%{customdata}<extra>experience</extra>',
    });
  }

  // ── Graph 2: the query vector and the predicted titles ──────────────────
  const resultTraces = [landmarkTrace()];

  const preds = by('prediction').sort((a, b) => a.rank - b.rank);
  resultTraces.push({
    x: preds.map(p => p.x), y: preds.map(p => p.y),
    text: preds.map(p => `${p.rank}`),
    customdata: preds.map(p => `#${p.rank} ${p.value} (${p.score.toFixed(4)})`),
    mode: 'markers+text', type: 'scatter', name: 'predicted next titles',
    textposition: 'bottom center', textfont: { size: 10, color: '#92400e' },
    marker: { ...KIND_STYLE.prediction, line: { width: 1, color: '#fff' } },
    hovertemplate: '%{customdata}<extra>prediction</extra>',
  });

  const ctx = by('context');
  resultTraces.push({
    x: ctx.map(p => p.x), y: ctx.map(p => p.y),
    mode: 'markers', type: 'scatter', name: 'query vector',
    marker: KIND_STYLE.context,
    hovertemplate: 'query vector<extra></extra>',
  });

  const layout = () => {
    const xaxis = { title: 'PC1', zeroline: false };
    const yaxis = { title: 'PC2', zeroline: false };
    if (data.axis_ranges) {          // global view: fixed frame across queries
      xaxis.range = data.axis_ranges.x;
      yaxis.range = data.axis_ranges.y;
      xaxis.autorange = false;
      yaxis.autorange = false;
    }
    // uirevision keeps the user's zoom/pan across predictions; it only resets
    // when the model or view mode changes (the axes mean something different).
    return {
      margin: { t: 36, r: 10, b: 45, l: 50 },
      xaxis, yaxis,
      legend: { orientation: 'h', y: 1.02, yanchor: 'bottom', x: 0 },
      hovermode: 'closest',
      uirevision: `${model}|${mode}`,
    };
  };
  const cfg = { responsive: true, displaylogo: false };
  Plotly.react('space-plot', resumeTraces, layout(), cfg);
  Plotly.react('space-plot-results', resultTraces, layout(), cfg);

  const ev = data.explained_variance;
  $('#space-info').textContent =
    `${data.model} · ${data.mode} view · PCA explains ${(100 * (ev[0] + ev[1])).toFixed(0)}% of variance`;
}

/* ── Job title flow (Sankey) ────────────────────────────────────────────── */

// Stable palette for selected source titles (so a title keeps its colour
// across layers, making downstream flow easy to trace).
const FLOW_PALETTE = [
  '#2563eb', '#059669', '#d97706', '#7c3aed', '#db2777',
  '#0891b2', '#65a30d', '#dc2626', '#4f46e5', '#0d9488',
];

async function ensureFlowLoaded() {
  if (state.flowLoaded) return;
  const res = await fetch('/api/transition_titles');
  const data = await res.json();
  state.flowTitles = data.titles || [];
  state.flowLoaded = true;
  const note = $('#flow-note');
  if (data.error) {
    note.textContent = data.error;
  } else if (data.meta) {
    note.textContent = `${data.meta.n_transitions.toLocaleString()} transitions `
      + `from ${data.meta.n_sources.toLocaleString()} titles `
      + `(source: ${data.meta.source_csv})`;
  }
  renderTitleList();
}

function renderTitleList() {
  const q = ($('#title-search').value || '').trim().toLowerCase();
  const list = $('#title-list');
  const selected = new Set(state.flowSelected);
  const items = state.flowTitles
    .filter(t => !q || t.value.includes(q))
    .slice(0, 300);
  list.innerHTML = '';
  for (const t of items) {
    const div = document.createElement('div');
    div.className = 'title-item' + (selected.has(t.title) ? ' selected' : '');
    div.innerHTML = `<span>${t.value}</span><span class="cnt">${t.out_count.toLocaleString()} →</span>`;
    div.onclick = () => toggleTitle(t.title);
    list.appendChild(div);
  }
  if (!items.length) list.innerHTML = '<div class="title-item">No matches</div>';
}

function renderChips() {
  const box = $('#flow-selected');
  box.innerHTML = '';
  state.flowSelected.forEach((title, i) => {
    const v = title.replace('W_TITLE:', '');
    const chip = document.createElement('span');
    chip.className = 'chip-sel';
    chip.style.borderColor = FLOW_PALETTE[i % FLOW_PALETTE.length];
    chip.innerHTML = `<span>${v}</span>`;
    const x = document.createElement('button');
    x.textContent = '✕';
    x.onclick = () => toggleTitle(title);
    chip.appendChild(x);
    box.appendChild(chip);
  });
}

function toggleTitle(title) {
  const i = state.flowSelected.indexOf(title);
  if (i >= 0) state.flowSelected.splice(i, 1);
  else state.flowSelected.push(title);
  renderChips();
  renderTitleList();
  drawSankey();
}

let sankeyTimer = null;
function drawSankey() {
  clearTimeout(sankeyTimer);
  sankeyTimer = setTimeout(drawSankeyNow, 250);
}

async function drawSankeyNow() {
  const empty = $('#sankey-empty');
  if (!state.flowSelected.length) {
    Plotly.purge('sankey-plot');
    empty.style.display = '';
    empty.textContent = 'Select job titles on the left to draw the transition flow.';
    return;
  }
  const res = await fetch('/api/sankey', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      titles: state.flowSelected, top_k: state.topK, depth: state.depth,
    }),
  });
  const data = await res.json();
  if (data.error) {
    empty.style.display = '';
    empty.textContent = data.error;
    return;
  }
  empty.style.display = 'none';

  // Colour each selected source from the palette; propagate that colour to the
  // nodes/links it feeds so a selected title's downstream flow stays traceable.
  const selColor = {};
  state.flowSelected.forEach((t, i) => { selColor[t] = FLOW_PALETTE[i % FLOW_PALETTE.length]; });

  const nodeColors = data.nodes.map(n => {
    if (n.is_other) return '#cbd5e1';
    if (n.is_selected) return selColor[n.title] || '#475569';
    return '#93b4dc';
  });
  const labels = data.nodes.map(n => n.label);
  const hover  = data.nodes.map(n =>
    n.is_other ? 'Other (long-tail transitions)' : n.label);

  const linkColors = data.links.map(l => {
    const c = nodeColors[l.source];
    return hexA(c, 0.4);
  });

  const layers = data.nodes.map(n => n.layer);
  const maxLayerCount = {};
  layers.forEach(l => { maxLayerCount[l] = (maxLayerCount[l] || 0) + 1; });
  const busiest = Math.max(...Object.values(maxLayerCount), 1);
  const height = Math.max(560, busiest * 26);

  // Keep the rendered nodes so the click handler can map a clicked node index
  // back to its title.
  state.sankeyNodes = data.nodes;

  Plotly.react('sankey-plot', [{
    type: 'sankey',
    orientation: 'h',
    arrangement: 'snap',
    node: {
      label: labels, customdata: hover,
      color: nodeColors,
      pad: 14, thickness: 16,
      line: { color: '#fff', width: 0.5 },
      hovertemplate: '%{customdata}<br>%{value:.0f} flow<br><i>click to drill in</i><extra></extra>',
    },
    link: {
      source: data.links.map(l => l.source),
      target: data.links.map(l => l.target),
      value:  data.links.map(l => l.value),
      color:  linkColors,
      hovertemplate: '%{source.label} → %{target.label}<br>%{value:.0f} flow<extra></extra>',
    },
  }], {
    margin: { t: 10, r: 10, b: 10, l: 10 },
    height,
    font: { size: 11 },
  }, { responsive: true, displaylogo: false }).then(bindSankeyClick);

  const note = $('#flow-note');
  note.textContent = `${data.nodes.length} nodes · ${data.links.length} links`
    + ` · top-${data.top_k} · depth ${data.depth}`
    + (data.truncated ? ' · truncated for size' : '');
}

// Click a job-title node to drill into its flow: it's toggled as a starting
// title (added as a new stem, or removed if already selected), then redrawn.
//
// Plotly Sankey swallows real `click` events on its nodes (it preventDefaults
// the mousedown for node dragging, so the browser never synthesises a click) —
// verified with a headless browser. But plotly_hover fires reliably, and raw
// pointerdown/pointerup still reach the div. So: remember which node is hovered
// at pointerdown, and on pointerup treat it as a click if the pointer didn't
// move (a small drag tolerance keeps node dragging usable). Bound once — the
// handlers live on the graph div and survive Plotly.react().
function bindSankeyClick() {
  if (state.sankeyClickBound) return;
  const gd = document.getElementById('sankey-plot');
  if (!gd || !gd.on) return;

  const nodeFromPoint = (pt) => {
    if (!pt) return null;
    // Link hovers carry source & target node objects; nodes don't.
    if (pt.source !== undefined && pt.target !== undefined) return null;
    const nodes = state.sankeyNodes || [];
    const idx = (pt.pointNumber !== undefined) ? pt.pointNumber : pt.index;
    let node = (idx !== undefined) ? nodes[idx] : null;
    // Fallback: match by label (titles repeat across layers but map to the
    // same token, which is all toggleTitle needs).
    if ((!node || node.label !== pt.label) && pt.label && pt.label !== 'Other') {
      node = nodes.find(n => !n.is_other && n.label === pt.label) || node;
    }
    return (node && !node.is_other) ? node : null;   // Other isn't drillable
  };

  gd.on('plotly_hover', (ev) => {
    state.hoveredNode = nodeFromPoint(ev.points && ev.points[0]);
    gd.style.cursor = state.hoveredNode ? 'pointer' : '';
  });
  gd.on('plotly_unhover', () => {
    state.hoveredNode = null;
    gd.style.cursor = '';
  });

  let down = null;   // {x, y, node} captured at pointerdown
  gd.addEventListener('pointerdown', (e) => {
    down = { x: e.clientX, y: e.clientY, node: state.hoveredNode };
  });
  gd.addEventListener('pointerup', (e) => {
    if (!down || !down.node) { down = null; return; }
    const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
    const node = down.node;
    down = null;
    if (moved <= 4) toggleTitle(node.title);   // a click, not a drag
  });
  state.sankeyClickBound = true;
}

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

/* ── Wiring ─────────────────────────────────────────────────────────────── */

document.querySelectorAll('.view-btn').forEach(btn => {
  btn.onclick = async () => {
    document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.view = btn.dataset.view;
    $('#view-predict').classList.toggle('hidden', state.view !== 'predict');
    $('#view-flow').classList.toggle('hidden', state.view !== 'flow');
    if (state.view === 'flow') {
      await ensureFlowLoaded();
      Plotly.Plots.resize('sankey-plot');
    }
  };
});

$('#title-search').oninput = renderTitleList;
$('#topk-slider').oninput = (e) => {
  state.topK = parseInt(e.target.value, 10);
  $('#topk-val').textContent = state.topK;
  drawSankey();
};
$('#depth-slider').oninput = (e) => {
  state.depth = parseInt(e.target.value, 10);
  $('#depth-val').textContent = state.depth;
  drawSankey();
};

document.querySelectorAll('.tab').forEach(tab => {
  tab.onclick = () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    state.activeTab = tab.dataset.tab;
    $('#tab-samples').classList.toggle('hidden', state.activeTab !== 'samples');
    $('#tab-builder').classList.toggle('hidden', state.activeTab !== 'builder');
    onResumeChanged();
  };
});

$('#sample-category').onchange = renderSampleList;
$('#sample-search').oninput = renderSampleList;
$('#resume-source').onchange = async (e) => {
  await loadSamples(e.target.value);
  onResumeChanged();
};
$('#add-work').onclick = () => addEntry('WORK');
$('#add-education').onclick = () => addEntry('EDUCATION');
$('#add-skills').onclick = () => addEntry('SKILLS');
$('#predict-btn').onclick = predict;
$('#space-model').onchange = drawSpace;
$('#space-mode').onchange = drawSpace;
$('#rank-domain').onchange = () => { if (state.lastPrediction) predict(); };
$('#toggle-space').onclick = () => {
  const hidden = $('#space-section').classList.toggle('hidden');
  $('#toggle-space').textContent = hidden ? 'Show' : 'Hide';
  if (!hidden) drawSpace();   // redraw fresh — plots sized while hidden collapse
};

(async function init() {
  await loadStatus();
  await loadSamples();
  // Start the builder with a sensible template
  state.entries = [
    { id: ++state.entrySeq, type: 'EDUCATION', values: {} },
    { id: ++state.entrySeq, type: 'WORK', values: {} },
  ];
  renderBuilder();
})();
