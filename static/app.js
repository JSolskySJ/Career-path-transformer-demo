/* Career Path Transformer demo frontend. Vanilla JS + Plotly. */

const state = {
  samples: [],
  selectedSample: null,
  entries: [],            // builder entries
  entrySeq: 0,
  builderTarget: null,    // held-out target carried over from an edited sample
  activeTab: 'samples',
  lastPrediction: null,
  runs: {},               // run_id -> /api/status model entry
  selectedModels: new Set(),   // run_ids used for prediction
  // model-list date filter — default: only runs trained in the last 14 days
  modelDateCutoff: new Date(Date.now() - 14 * 864e5).toISOString().slice(0, 10),
  loggedOnly: false,      // model filter: only runs with a vault experiment code
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
  // raw dataset viewer
  datasets: [],
  datasetId: null,
  datasetOffset: 0,
  datasetLoaded: false,
  datasetColOrder: [],       // column display order (drag-reorderable)
  datasetHidden: new Set(),  // hidden columns
  datasetFilters: {},        // column -> substring filter
  datasetPage: null,         // last /api/dataset response
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
    ['spec',     'W_SPEC',     'Specialisations (separate with |)'],
    ['description', 'W_DESC',  'Job description (freetext — DenseRec-only, injected via MiniLM)'],
  ],
  EDUCATION: [
    ['major',       'E_MAJOR',  'Major(s) (separate with |)'],
    ['degree',      'E_DEGREE', 'Degree (e.g. bachelors)'],
    ['school_type', 'E_TYPE',   'School type'],
    ['level',       'E_LEVEL',  'Education level'],
  ],
  SKILLS: [
    ['skills', 'S_SKILL', 'Skills (separate with |, e.g. forklift | welding)'],
  ],
};

// Multi-value fields → one token per value. '|' is the ONLY separator (real
// values routinely contain commas, so commas stay literal).
const MULTI_FIELDS = new Set(['spec', 'skills', 'major']);
function splitMultiValue(v) {
  return (v || '').split('|');
}

/* ── Token building (mirrors demo/tokens.py) ────────────────────────────── */

function norm(v) { return (v || '').trim().toLowerCase(); }

function tokensFromEntry(e) {
  const out = [];
  for (const [field, prefix] of FIELD_SPECS[e.type]) {
    const values = MULTI_FIELDS.has(field)
      ? splitMultiValue(e.values[field]) : [e.values[field]];
    for (const raw of values) {
      // descriptions are kept verbatim (case AND whitespace — a stored value
      // can legitimately end in a space after training's strip-then-truncate)
      const v = field === 'description'
        ? ((raw || '').trim() ? raw : '') : norm(raw);
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
  if (state.activeTab === 'samples') {
    return state.selectedSample ? state.selectedSample.target : null;
  }
  // Builder: an edited sample keeps its held-out target, so token edits show
  // how the actual answer's rank moves.
  return state.builderTarget;
}

// Load the selected sample's experiences into the builder for token editing —
// every field (title, description, skills, …) becomes editable, and the
// held-out target rides along for the sense check.
function editSelectedSample() {
  const s = state.selectedSample;
  if (!s) return;
  const fields = new Set(Object.values(FIELD_SPECS).flat().map(([f]) => f));
  state.entries = s.experiences.map(exp => ({
    id: ++state.entrySeq,
    type: exp.type,
    values: Object.fromEntries(Object.entries(exp)
      .filter(([k, v]) => k !== 'type' && fields.has(k) && v)),
  }));
  state.builderTarget = s.target;
  renderBuilder();
  document.querySelector('.tab[data-tab="builder"]').click();
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
  const spaceSel = $('#space-model');
  spaceSel.innerHTML = '';
  state.titleCounts = {};
  const defaultPerArch = new Set();   // newest loaded run per architecture
  for (const [rid, info] of runsByDate()) {
    if (info.loaded) {
      state.modelsLoaded.push(rid);
      state.titleCounts[rid] = info.title_count;
      if (![...defaultPerArch].some(r => state.runs[r].architecture === info.architecture)) {
        defaultPerArch.add(rid);
      }
      const opt = document.createElement('option');
      opt.value = rid;
      opt.textContent = runDate(info) ? `${info.label} · ${runDate(info)}` : info.label;
      spaceSel.appendChild(opt);
    }
  }
  // Sensible default with many staged versions: the newest run per architecture.
  state.selectedModels = defaultPerArch;
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

// Curated param order for the comparison table — the ones that tell runs
// apart; any other params that DIFFER across runs are appended automatically.
const COMPARE_METRICS = ['test_recall_at_10', 'test_recall_at_5', 'test_recall_at_1', 'test_mrr'];
const COMPARE_HIDDEN = new Set(['run_tag']);   // noise (usually N/A)

function compareTableHtml(rids) {
  const runs = rids.map(r => state.runs[r]);
  const header = runs.map(r =>
    `<th title="${r.run_id}">${r.experiment ? `<span class="exp-badge">${r.experiment}</span> ` : ''}${r.run_name}<br><span class="hint">${r.architecture}${runDate(r) ? ' · ' + runDate(r) : ''}${r.loaded ? '' : ' · unavailable'}</span></th>`).join('');
  const paramKeys = [...new Set(runs.flatMap(r => Object.keys(r.params)))]
    .filter(k => !COMPARE_HIDDEN.has(k)).sort();
  const row = (label, values, cls = '') => {
    const differs = new Set(values.map(v => String(v ?? '—'))).size > 1;
    return `<tr class="${cls}${differs ? ' diff' : ''}"><td>${label}</td>${
      values.map(v => `<td>${v ?? '—'}</td>`).join('')}</tr>`;
  };
  const metricRows = COMPARE_METRICS.map(k =>
    row(k, runs.map(r => r.metrics[k] !== undefined ? r.metrics[k].toFixed(4) : null), 'metric'));
  const paramRows = paramKeys.map(k => row(k, runs.map(r => r.params[k])));
  return `<div class="compare-wrap"><table class="kv compare">
    <tr><th></th>${header}</tr>${metricRows.join('')}${paramRows.join('')}
  </table></div>
  <p class="hint">Highlighted rows differ between runs — that's what tells the models apart.</p>`;
}

function modelSummaryText() {
  const n = state.selectedModels.size;
  const names = [...state.selectedModels].map(r => state.runs[r].run_name).join(', ');
  return n ? `Models — ${n} selected: ${names}` : 'Models — none selected';
}

function renderModelSelect(keepOpen = false) {
  const box = $('#model-select');
  box.innerHTML = '';
  const dd = document.createElement('details');
  dd.className = 'dropdown';
  dd.open = keepOpen;
  dd.innerHTML = `<summary>${modelSummaryText()}</summary><div class="dropdown-body"></div>`;
  const body = dd.querySelector('.dropdown-body');

  // Date filter — STRICTLY hide runs trained before the cutoff (undated
  // legacy checkpoints excepted). A selected model that falls outside the
  // filter stays selected (and predicted) but is hidden like the rest — the
  // hint calls that out so the filter never looks broken.
  const cutoffMs = state.modelDateCutoff ? Date.parse(state.modelDateCutoff) : 0;
  const all = runsByDate();
  const ordered = all.filter(([, i]) =>
    (!cutoffMs || !+i.start_time || +i.start_time >= cutoffMs)
    && (!state.loggedOnly || i.experiment));
  const nHidden = all.length - ordered.length;
  const visibleIds = new Set(ordered.map(([rid]) => rid));
  const hiddenSelected = [...state.selectedModels]
    .filter(rid => state.runs[rid] && !visibleIds.has(rid));
  const filter = document.createElement('div');
  filter.className = 'row model-filter';
  filter.innerHTML = `
    <label>Trained after
      <input type="date" id="model-date-filter" value="${state.modelDateCutoff || ''}">
    </label>
    <label title="Only show runs that have an experiment page in the docs vault
           (Experiments/<code>.md) — matched on startup by run name / id">
      <input type="checkbox" id="model-logged-filter" ${state.loggedOnly ? 'checked' : ''}>
      logged experiments only
    </label>
    <span class="hint">${nHidden
      ? `${nHidden} older run${nHidden > 1 ? 's' : ''} hidden — clear the date to show all`
      : 'showing all runs'}${hiddenSelected.length
      ? ` · <b>${hiddenSelected.length} selected model${hiddenSelected.length > 1 ? 's' : ''}
          hidden but still predicted</b> (${hiddenSelected.map(r => state.runs[r].run_name).join(', ')})`
      : ''}</span>`;
  filter.querySelector('#model-date-filter').onchange = (e) => {
    state.modelDateCutoff = e.target.value;
    renderModelSelect(true);
  };
  filter.querySelector('#model-logged-filter').onchange = (e) => {
    state.loggedOnly = e.target.checked;
    renderModelSelect(true);
  };
  body.appendChild(filter);

  for (const [rid, info] of ordered.filter(([, i]) => i.loaded)
      .concat(ordered.filter(([, i]) => !i.loaded))) {
    const row = document.createElement('div');
    row.className = 'model-row' + (info.loaded ? '' : ' unavailable');
    const id = `model-check-${rid}`;
    row.innerHTML = `
      <label for="${id}">
        <input type="checkbox" id="${id}" ${info.loaded ? '' : 'disabled'}
               ${state.selectedModels.has(rid) ? 'checked' : ''}>
        ${info.experiment ? `<span class="exp-badge">${info.experiment}</span>` : ''}
        <b>${info.architecture}</b> · ${info.run_name}
        ${runDate(info) ? `<span class="run-date">${runDate(info)}</span>` : ''}
        ${info.metrics.test_recall_at_10 !== undefined
          ? `<span class="hint">R@10 ${info.metrics.test_recall_at_10.toFixed(3)}</span>` : ''}
        ${info.loaded ? '' : `<span class="warn" title="${info.error || ''}">unavailable — ${info.error || 'no model'}</span>`}
      </label>
      ${info.loaded ? '<a class="model-internals" href="#" title="Whole-model view:\nattention profile, learned weights, drill-down">internals ↗</a>' : ''}
      <details class="model-info"><summary>info</summary>${modelInfoHtml(info)}</details>`;
    row.querySelector('input').onchange = (e) => {
      e.target.checked ? state.selectedModels.add(rid) : state.selectedModels.delete(rid);
      dd.querySelector('summary').textContent = modelSummaryText();
    };
    const internals = row.querySelector('.model-internals');
    if (internals) internals.onclick = (e) => {
      e.preventDefault();
      const key = `cpt_model_${rid}`;
      localStorage.setItem(key, JSON.stringify({ model: rid, label: info.label }));
      window.open('/inspect#' + encodeURIComponent(key), '_blank');
    };
    body.appendChild(row);
  }

  // Full parameter comparison across every staged run (params from MLflow).
  const cmp = document.createElement('details');
  cmp.className = 'compare-params';
  cmp.innerHTML = '<summary>Compare run parameters</summary>';
  cmp.addEventListener('toggle', () => {
    if (cmp.open && !cmp.dataset.built) {
      cmp.insertAdjacentHTML('beforeend', compareTableHtml(ordered.map(([rid]) => rid)));
      cmp.dataset.built = '1';
    }
  });
  body.appendChild(cmp);
  box.appendChild(dd);
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
  // Every run is selectable — ones without staged resumes are marked, and the
  // "Build from dataset" button can create theirs.
  const sources = runsByDate().map(([, r]) => r)
    .sort((a, b) => (b.has_samples ? 1 : 0) - (a.has_samples ? 1 : 0));
  for (const r of sources) {
    const opt = document.createElement('option');
    opt.value = r.run_id;
    const date = runDate(r);
    const code = r.experiment ? `[${r.experiment}] ` : '';
    opt.textContent = `${code}${r.label}${date ? ' · ' + date : ''} (${r.run_id.slice(0, 8)})`
      + (r.has_samples ? '' : ' — no resumes');
    sel.appendChild(opt);
  }
  if (state.resumeRun) sel.value = state.resumeRun;
}

async function buildSamples() {
  const btn = $('#build-samples');
  const rid = $('#resume-source').value;
  if (!rid) return;
  const info = state.runs[rid];
  let dataset = info.params.data_run_id;
  if (!dataset || dataset === 'None') {
    dataset = prompt(`This run predates data_run_id logging.\n` +
                     `Dataset run id for ${info.run_name}:`);
    if (!dataset) return;
  }
  btn.disabled = true;
  btn.textContent = 'Building… (reads the dataset from S3, takes a few minutes)';
  try {
    const res = await fetch('/api/build_samples', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ run: rid, dataset, n: 300 }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    info.has_samples = true;
    await loadSamples(rid);
    onResumeChanged();
    btn.textContent = `✓ ${data.n_samples} resumes from ${data.n_test_pairs.toLocaleString()} test pairs`
      + (data.split_column ? ' (split column)' : ' (holdout draw)');
  } catch (e) {
    btn.textContent = '⟳ Build from dataset';
    alert(`Build failed: ${e.message}`);
  } finally {
    btn.disabled = false;
    setTimeout(() => { btn.textContent = '⟳ Build from dataset'; }, 8000);
  }
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

// Title-repetition profile of a resume — over the context titles PLUS the
// held-out target, since a target repeating a context title is the
// repeat-copy case that inflates work-history recall.
function repeatProfile(s) {
  const titles = s.context_tokens.filter(t => t.startsWith('W_TITLE:'));
  if (s.target) titles.push(s.target);
  const hasConsecutive = titles.some((t, i) => i > 0 && t === titles[i - 1]);
  const hasRepeats = new Set(titles).size < titles.length;
  return { hasConsecutive, hasRepeats };
}

function repeatFilterPass(s, mode) {
  if (!mode) return true;
  const p = repeatProfile(s);
  if (mode === 'no-consecutive') return !p.hasConsecutive;
  if (mode === 'no-repeats') return !p.hasRepeats;
  if (mode === 'has-repeats') return p.hasRepeats;
  return true;
}

function renderSampleList() {
  const cat = $('#sample-category').value;
  const rep = $('#sample-repeats').value;
  const q = norm($('#sample-search').value);
  const list = $('#sample-list');
  list.innerHTML = '';
  const items = state.samples.filter(s =>
    (!cat || s.category === cat) &&
    repeatFilterPass(s, rep) &&
    (!q || s.label.includes(q) || s.context_tokens.join(' ').includes(q)));
  if (!items.length) {
    list.innerHTML = '<div class="sample-item">No resumes staged for this run — ' +
      'use “⟳ Build from dataset” above to reconstruct its held-out test resumes.</div>';
    return;
  }
  for (const s of items) {
    const div = document.createElement('div');
    div.className = 'sample-item' +
      (state.selectedSample && state.selectedSample.id === s.id ? ' selected' : '');
    const nSkills = s.context_tokens.filter(t => t.startsWith('S_SKILL:')).length;
    const skills = nSkills ? `<span class="skill-count" title="${nSkills} skills">🛠${nSkills}</span>` : '';
    div.innerHTML = `<span>${s.label}</span><span class="tags">${skills}
      <button class="icon skill-suggest" title="Skill suggestions: which skills,
      added to this worker, most improve a target role's ranking (opens per-worker
      page; DenseRec models)">💡</button>
      <span class="cat">${s.category.replace('_', ' ')}</span></span>`;
    div.onclick = () => { state.selectedSample = s; renderSampleList(); onResumeChanged(); };
    div.querySelector('.skill-suggest').onclick = (e) => {
      e.stopPropagation();                    // don't also select the row
      const key = `cpt_skills_${state.resumeRun}_${s.id}`;
      localStorage.setItem(key, JSON.stringify({
        tokens: s.context_tokens, target: s.target, label: s.label,
        model: state.resumeRun, experiences: s.experiences || null,
      }));
      window.open('/skills#' + encodeURIComponent(key), '_blank');
    };
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
  if (state.builderTarget) {
    const chip = document.createElement('p');
    chip.className = 'hint builder-target';
    chip.innerHTML = `Editing against held-out target:
      <b>${state.builderTarget.replace('W_TITLE:', '')}</b>
      <button class="icon" title="Drop the target (rank check disappears)">✕</button>`;
    chip.querySelector('button').onclick = () => {
      state.builderTarget = null;
      renderBuilder();
      onResumeChanged();
    };
    box.appendChild(chip);
  }
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
  // Multi-value fields autocomplete the segment after the last separator.
  const multi = MULTI_FIELDS.has(field);
  const parts = multi ? input.value.split('|') : [input.value];
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
        const next = multi ? [...parts.slice(0, -1), ` ${v}`].join('|').replace(/^ /, '') : v;
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
  return splitMultiValue(v).map(s => s.trim()).filter(Boolean);
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
    // Freetext description (DenseRec W_DESC) — often long, so collapse it into
    // a native dropdown instead of dumping it inline.
    const desc = isWork && exp.description
      ? `<details class="exp-desc"><summary>📄 description</summary>${exp.description}</details>`
      : '';
    div.innerHTML = `
      <span class="badge">${isWork ? '💼' : '🎓'}</span>
      <span class="body"><b>${main}</b>${tenure}${company}<br>
        <span class="meta">${meta}</span>${specs}${desc}</span>`;
    box.appendChild(div);
  }
  const t = document.createElement('p');
  t.className = 'hint';
  t.innerHTML = `Actual next role (held out): <b>${state.selectedSample.target.replace('W_TITLE:', '')}</b>`;
  box.appendChild(t);

  const edit = document.createElement('button');
  edit.className = 'ghost';
  edit.textContent = '✎ Edit tokens';
  edit.title = 'Open this resume in the builder — change any token (title, ' +
               'description, skills, …) and re-predict to see what the model ' +
               'is relying on. The held-out target rides along for the sense check.';
  edit.onclick = editSelectedSample;
  box.appendChild(edit);
}

function renderTokenPreview(oov = [], tokensOverride = null) {
  // After a prediction, show the server-resolved tokens (post-rollup) — what
  // the model actually consumed; before one, the raw resume tokens.
  const tokens = tokensOverride || currentTokens();
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
        scoring: $('#rank-scoring').value,
        rollup: $('#rollup-mode').value,
      }),
    });
    state.lastPrediction = await res.json();
    renderFlowStrip();
    renderPredictions();
    const oov = new Set();
    for (const r of Object.values(state.lastPrediction.results)) {
      (r.unknown_tokens || []).forEach(t => oov.add(t));
    }
    renderTokenPreview([...oov], state.lastPrediction.tokens);
    await drawSpace();
  } finally {
    $('#predict-btn').disabled = false;
  }
}

// Input → model → output at a glance, above the prediction columns — the
// data flow an exec can read without knowing what a token is.
function renderFlowStrip() {
  const { tokens, results } = state.lastPrediction;
  const strip = $('#flow-strip');
  const nExp = tokens.filter(t => t.startsWith('W_TITLE:')).length;
  const nEdu = tokens.filter(t => t.startsWith('E_')).length;
  const nSkill = tokens.filter(t => t.startsWith('S_SKILL:')).length;
  const models = Object.values(results).filter(r => !r.error);
  const topK = parseInt($('#top-k').value || '10', 10);
  const domains = [...new Set(models.map(r => r.n_ranked).filter(Boolean))]
    .map(n => n.toLocaleString()).join(' / ');
  strip.classList.remove('hidden');
  strip.innerHTML = `
    <div class="flow-node">
      <div class="flow-label">Input · one career</div>
      <div class="flow-value">${nExp} role${nExp === 1 ? '' : 's'}${
        nEdu ? ` · ${nEdu} education` : ''}${
        nSkill ? ` · ${nSkill} skills` : ''}</div>
      <div class="flow-sub">${tokens.length} tokens the model reads</div>
    </div>
    <div class="flow-arrow">→</div>
    <div class="flow-node model">
      <div class="flow-label">Model${models.length === 1 ? '' : 's'}</div>
      <div class="flow-value">${models.length} run${models.length === 1 ? '' : 's'} compared</div>
      <div class="flow-sub">${domains ? `ranking ${domains} titles` : ''}</div>
    </div>
    <div class="flow-arrow">→</div>
    <div class="flow-node">
      <div class="flow-label">Output</div>
      <div class="flow-value">top ${topK} next roles</div>
      <div class="flow-sub">click any title to see why</div>
    </div>`;
}

// The target's rank in one model's ranking, as a per-column header banner —
// sits directly over that model's predictions for side-by-side comparison.
function senseHtml(r, target) {
  // Empty placeholder keeps the card's grid rows aligned across models.
  if (!target || r.target_rank === undefined) return '<div class="sense empty"></div>';
  const rank = r.target_rank;
  const n = r.n_ranked;
  const cls = rank === null ? 'bad' : rank <= 10 ? 'good' : rank <= 100 ? 'mid' : 'bad';
  const msg = rank === null
    ? 'not in this model\'s ranking domain'
    : `ranked <b>#${rank}</b> of ${n ? n.toLocaleString() + ' ' : ''}titles`;
  return `<div class="sense ${cls}">actual: “${target.replace('W_TITLE:', '')}” — ${msg}</div>`;
}

function renderPredictions() {
  const { results, target } = state.lastPrediction;
  const box = $('#predictions');
  box.innerHTML = '';

  for (const [rid, r] of Object.entries(results)) {
    const name = r.label || rid;
    const card = document.createElement('div');
    card.className = 'pred-card';
    if (r.error) {
      card.innerHTML = `<h3>${name}</h3><div class="sense empty"></div>
        <div class="conf empty"></div><p class="warn">${r.error}</p>`;
      box.appendChild(card);
      continue;
    }
    const scoreLabel = r.architecture === 'item2vec' ? 'cosine' : 'probability';
    const nRanked = r.n_ranked ? ` · ranked over ${r.n_ranked.toLocaleString()} titles` : '';
    // Trained date + logged test R@10 beside the model name, as in the dropdown
    const info = state.runs[rid] || {};
    const date = runDate(info);
    const r10 = info.metrics && info.metrics.test_recall_at_10 !== undefined
      ? `<span class="hint">R@10 ${info.metrics.test_recall_at_10.toFixed(3)}</span>` : '';
    let html = `<h3>${info.experiment ? `<span class="exp-badge">${info.experiment}</span> ` : ''}${name}
      ${date ? `<span class="run-date">${date}</span>` : ''} ${r10}
      <span class="hint">(${scoreLabel}${nRanked})</span>
      <a class="model-internals" href="#" data-run="${rid}"
         title="Model internals with THIS resume loaded — drill into any of its
         predicted titles">internals ↗</a></h3>`;
    html += senseHtml(r, target);
    html += confidenceHtml(r.confidence);
    html += '<div class="pred-list">';
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
          <span class="why" title="Trace exactly how this model produced this
            title for this resume — per-token influence, attention arithmetic,
            logit lens (new tab)">why?</span>
        </div>`;
    });
    if (r.injected_tokens && r.injected_tokens.length) {
      html += `<p class="hint" title="${r.injected_tokens.join(', ')}">
        ✚ DenseRec injected ${r.injected_tokens.length} unseen token(s) via the
        content path (MiniLM) instead of dropping them</p>`;
    }
    if (r.unknown_tokens && r.unknown_tokens.length) {
      html += `<p class="warn">Ignored ${r.unknown_tokens.length} out-of-vocabulary token(s)</p>`;
    }
    html += '</div>';   // .pred-list
    card.innerHTML = html;
    // Drill-down: click a predicted title to open the full-page inspector
    // (bert4rec logit lens + attention; item2vec token contributions).
    card.querySelectorAll('.pred-row').forEach((row, i) => {
      row.classList.add('clickable');
      row.title = 'Click to inspect what the model is doing for this title (new tab)';
      row.onclick = () => openInspect(rid, r.predictions[i].token);
    });
    // Model internals seeded with THIS prediction's resume — the drill-down
    // there lists this resume's top titles for per-title traces.
    const link = card.querySelector('.model-internals');
    if (link) link.onclick = (e) => {
      e.preventDefault();
      const key = `cpt_model_${rid}`;
      localStorage.setItem(key, JSON.stringify({
        model: rid, label: info.label || name,
        tokens: state.lastPrediction.tokens, target: state.lastPrediction.target,
      }));
      window.open('/inspect#' + encodeURIComponent(key), '_blank');
    };
    box.appendChild(card);
  }

  // Unified scrolling: scrolling one model's rows scrolls every column, so
  // rank N stays level across models. Programmatic scrollTop sets fire scroll
  // events too, but the equality guard stops any feedback loop.
  const lists = [...box.querySelectorAll('.pred-list')];
  for (const list of lists) {
    list.onscroll = () => {
      for (const other of lists) {
        if (other !== list && other.scrollTop !== list.scrollTop) {
          other.scrollTop = list.scrollTop;
        }
      }
    };
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
  if (!c) return '<div class="conf empty"></div>';   // placeholder keeps rows aligned
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
      rollup: $('#rollup-mode').value,
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

/* ── Raw dataset viewer ─────────────────────────────────────────────────── */

const escHtml = (s) => String(s ?? '')
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function datasetLimit() {
  return Math.max(1, Math.min(parseInt($('#dataset-limit').value, 10) || 50, 500));
}

async function ensureDatasetLoaded() {
  if (state.datasetLoaded) return;
  let data;
  try {
    const res = await fetch('/api/datasets');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (e) {
    // Typically a demo server process older than this endpoint (404 → HTML)
    $('#dataset-note').textContent =
      `Could not load datasets (${e.message}). If the demo server has been ` +
      'running a while, restart it (./run.sh) to pick up the Dataset view.';
    return;   // not marked loaded — retried on next tab visit
  }
  state.datasets = data.datasets || [];
  const sel = $('#dataset-select');
  sel.innerHTML = '';
  if (!state.datasets.length) {
    $('#dataset-note').textContent =
      'No dataset slices staged yet. They download on start for the latest runs; ' +
      'set CPT_SYNC_DATASETS=1 and restart, or build samples from a dataset.';
    $('#dataset-table').innerHTML = '';
    state.datasetLoaded = true;
    return;
  }
  for (const d of state.datasets) {
    const opt = document.createElement('option');
    opt.value = d.data_run_id;
    opt.textContent = `${d.data_run_id.slice(0, 8)} — ${d.rows.toLocaleString()} rows, ` +
                      `${d.columns} cols${d.used_by.length ? ' · ' + d.used_by.join(', ') : ''}`;
    sel.appendChild(opt);
  }
  state.datasetId = state.datasets[0].data_run_id;
  state.datasetOffset = 0;
  state.datasetLoaded = true;
  await loadDatasetPage();
}

async function loadDatasetPage() {
  const data = await fetchDatasetPage();
  if (!data) return;
  syncDatasetSchema(data.columns);
  renderDatasetTable(data);
}

const fmtCell = (v) => (v === null || v === undefined) ? '' : String(v);

// Visible columns in display order (colOrder minus hidden), for the active dataset.
function visibleColumns() {
  return state.datasetColOrder.filter(c => !state.datasetHidden.has(c));
}

// Reset column order/visibility/filters when the dataset's schema first appears
// or changes (adaptable — new fields just show up in schema order).
function syncDatasetSchema(columns) {
  const same = state.datasetColOrder.length === columns.length &&
    state.datasetColOrder.every(c => columns.includes(c));
  if (same) return;
  state.datasetColOrder = columns.slice();
  state.datasetHidden = new Set();
  state.datasetFilters = {};
  renderColumnPicker();
}

function renderColumnPicker() {
  const list = $('#dataset-cols-list');
  list.innerHTML = '';
  for (const c of state.datasetColOrder) {
    const id = 'colchk-' + c;
    const lbl = document.createElement('label');
    lbl.className = 'col-toggle';
    lbl.innerHTML =
      `<input type="checkbox" id="${id}" ${state.datasetHidden.has(c) ? '' : 'checked'}>` +
      `<span>${escHtml(c)}</span>`;
    lbl.querySelector('input').onchange = (e) => {
      if (e.target.checked) state.datasetHidden.delete(c);
      else state.datasetHidden.add(c);
      renderDatasetTable(state.datasetPage);
    };
    list.appendChild(lbl);
  }
}

// Full (re)render — used on load and structural changes (hide/reorder).
function renderDatasetTable(data) {
  if (!data) return;
  state.datasetPage = data;
  const from = data.total ? data.offset + 1 : 0;
  const to = Math.min(data.offset + data.limit, data.total);
  $('#dataset-info').textContent =
    `${from.toLocaleString()}–${to.toLocaleString()} of ${data.total.toLocaleString()}`;
  const d = state.datasets.find(x => x.data_run_id === data.data_run_id);
  $('#dataset-note').textContent = d && d.used_by.length
    ? `Trained on by: ${d.used_by.join(', ')}` : '';
  $('#dataset-prev').disabled = data.offset <= 0;
  $('#dataset-next').disabled = to >= data.total;

  const cols = visibleColumns();
  const labelRow = cols.map(c =>
    `<th draggable="true" data-col="${escHtml(c)}">${escHtml(c)}</th>`).join('');
  const filterRow = cols.map(c =>
    `<th><input class="col-filter" data-col="${escHtml(c)}" ` +
    `value="${escHtml(state.datasetFilters[c] || '')}" placeholder="filter…"></th>`).join('');
  $('#dataset-table').innerHTML =
    `<table><thead><tr>${labelRow}</tr><tr class="filter-row">${filterRow}</tr></thead>` +
    `<tbody>${renderDatasetBody(data, cols)}</tbody></table>`;
  wireHeaderDrag();
  wireFilterInputs();
}

function renderDatasetBody(data, cols) {
  return data.rows.map(r =>
    '<tr>' + cols.map(c => `<td title="${escHtml(fmtCell(r[c]))}">${escHtml(fmtCell(r[c]))}</td>`).join('') + '</tr>'
  ).join('');
}

// Column reorder via native HTML5 drag-and-drop on the label header cells.
function wireHeaderDrag() {
  let dragCol = null;
  $('#dataset-table').querySelectorAll('th[draggable]').forEach(th => {
    th.ondragstart = () => { dragCol = th.dataset.col; };
    th.ondragover = (e) => { e.preventDefault(); th.classList.add('drop-target'); };
    th.ondragleave = () => th.classList.remove('drop-target');
    th.ondrop = (e) => {
      e.preventDefault();
      th.classList.remove('drop-target');
      const target = th.dataset.col;
      if (!dragCol || dragCol === target) return;
      const order = state.datasetColOrder;
      order.splice(order.indexOf(dragCol), 1);
      order.splice(order.indexOf(target), 0, dragCol);
      renderDatasetTable(state.datasetPage);
    };
  });
}

// Per-field filters: debounced refetch (whole-slice), then only tbody + paging
// are swapped so the focused filter input keeps focus and caret.
let _filterTimer = null;
function wireFilterInputs() {
  $('#dataset-table').querySelectorAll('.col-filter').forEach(inp => {
    inp.oninput = () => {
      const col = inp.dataset.col;
      const v = inp.value.trim();
      if (v) state.datasetFilters[col] = v; else delete state.datasetFilters[col];
      state.datasetOffset = 0;
      clearTimeout(_filterTimer);
      _filterTimer = setTimeout(applyDatasetFilters, 250);
    };
  });
}

async function applyDatasetFilters() {
  const data = await fetchDatasetPage();
  if (!data) return;
  state.datasetPage = data;
  const cols = visibleColumns();
  const from = data.total ? data.offset + 1 : 0;
  const to = Math.min(data.offset + data.limit, data.total);
  $('#dataset-info').textContent =
    `${from.toLocaleString()}–${to.toLocaleString()} of ${data.total.toLocaleString()}`;
  $('#dataset-prev').disabled = data.offset <= 0;
  $('#dataset-next').disabled = to >= data.total;
  $('#dataset-table').querySelector('tbody').innerHTML = renderDatasetBody(data, cols);
}

async function fetchDatasetPage() {
  if (!state.datasetId) return null;
  const params = new URLSearchParams({
    id: state.datasetId,
    offset: state.datasetOffset,
    limit: datasetLimit(),
    filters: JSON.stringify(state.datasetFilters),
  });
  let data;
  try {
    const res = await fetch('/api/dataset?' + params);
    data = await res.json();
  } catch (e) {
    $('#dataset-note').textContent = `Could not load rows (${e.message}).`;
    return null;
  }
  if (data.error) { $('#dataset-note').textContent = data.error; return null; }
  return data;
}

$('#dataset-select').onchange = (e) => {
  state.datasetId = e.target.value;
  state.datasetOffset = 0;
  loadDatasetPage();
};
$('#dataset-limit').onchange = () => { state.datasetOffset = 0; loadDatasetPage(); };
$('#dataset-prev').onclick = () => {
  state.datasetOffset = Math.max(0, state.datasetOffset - datasetLimit());
  loadDatasetPage();
};
$('#dataset-next').onclick = () => {
  state.datasetOffset += datasetLimit();
  loadDatasetPage();
};
$('#dataset-cols-all').onclick = () => {
  state.datasetHidden.clear();
  renderColumnPicker();
  renderDatasetTable(state.datasetPage);
};
$('#dataset-cols-none').onclick = () => {
  state.datasetColOrder.forEach(c => state.datasetHidden.add(c));
  renderColumnPicker();
  renderDatasetTable(state.datasetPage);
};

/* ── Wiring ─────────────────────────────────────────────────────────────── */

document.querySelectorAll('.view-btn').forEach(btn => {
  btn.onclick = async () => {
    document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.view = btn.dataset.view;
    $('#view-predict').classList.toggle('hidden', state.view !== 'predict');
    $('#view-flow').classList.toggle('hidden', state.view !== 'flow');
    $('#view-dataset').classList.toggle('hidden', state.view !== 'dataset');
    if (state.view === 'flow') {
      await ensureFlowLoaded();
      Plotly.Plots.resize('sankey-plot');
    }
    if (state.view === 'dataset') await ensureDatasetLoaded();
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
$('#sample-repeats').onchange = renderSampleList;
$('#sample-search').oninput = renderSampleList;
$('#resume-source').onchange = async (e) => {
  await loadSamples(e.target.value);
  onResumeChanged();
};
$('#build-samples').onclick = buildSamples;
$('#add-work').onclick = () => addEntry('WORK');
$('#add-education').onclick = () => addEntry('EDUCATION');
$('#add-skills').onclick = () => addEntry('SKILLS');
$('#predict-btn').onclick = predict;
$('#space-model').onchange = drawSpace;
$('#space-mode').onchange = drawSpace;
$('#rank-domain').onchange = () => { if (state.lastPrediction) predict(); };
$('#rollup-mode').onchange = () => { if (state.lastPrediction) predict(); };
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
