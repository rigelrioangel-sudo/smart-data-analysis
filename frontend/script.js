/* ============================================================
   DATASENSE PRO — Frontend Script
   Handles: upload, drag-drop, language/direction, workflow steps,
   executive summary, findings, next actions, data quality, cleaning
   review, preview (search/sort/filter), charts, exports.
   All analysis is computed by the FastAPI backend; this file only
   presents it. All interface text comes from i18n.js (locales/*.json);
   dataset column names and cell values are always shown unchanged.
   ============================================================ */

'use strict';

const { t, th, msg, msgHtml } = I18N;

// Chart.js color palette
const PALETTE = [
  '#6c63ff', '#00d4aa', '#ff6b6b', '#ffc75f', '#4ecdc4',
  '#a29bfe', '#fd79a8', '#55efc4', '#fdcb6e', '#e17055',
  '#74b9ff', '#81ecec', '#fab1a0', '#dfe6e9', '#b2bec3',
];

// Active chart instances (for cleanup on re-render)
let _charts = [];

// Backend base URL (resolved in index.html <head>)
const API_BASE = window.API_BASE_RESOLVED ?? '';
// The browser's private session cookie must accompany API calls; a separate dev server needs 'include'
const FETCH_CREDENTIALS = API_BASE ? 'include' : 'same-origin';

// ─── Application state ───────────────────────────────────────
// `gen` increments on every new upload; responses that belong to an older
// generation are discarded so previous-upload data can never leak into the
// current view. Each async view also has its own request sequence number so
// only the latest response is rendered (no stale search results).

const App = {
  data: null,
  gen: 0,
  pending: {},          // cleaning decisions being edited (not yet applied)
  preview: null,
  previewSeq: 0,
  lastPreview: null,
  changes: { page: 1, column: '' },
  changesSeq: 0,
  lastChanges: null,
  issueSev: 'all',
  findingsKind: 'all',
  corrMethod: 'spearman',
  subTab: 'overview',
  visited: new Set(),
  exported: false,
  reportLang: null,     // defaults to the interface language
  busy: false,
  uploadError: null,    // last upload error (message object), re-rendered on language change
};

const DETAIL_TABS = ['overview', 'preview', 'stats', 'charts', 'correlation'];
const DECISIONS = ['pending', 'accepted', 'deferred', 'dismissed'];

function defaultPreviewState() {
  return { version: 'cleaned', page: 1, pageSize: 25, search: '', sort: null, order: 'asc', filters: [], changedOnly: false, colFilter: '' };
}

// ─── Utilities ───────────────────────────────────────────────

const escapeHtml = I18N.escapeHtml;

// A dataset value (column name, cell value): escaped, never translated, bidi-isolated
function raw(v) {
  return `<bdi dir="auto">${escapeHtml(v)}</bdi>`;
}

// Canvas can't resolve CSS variables, so read the computed value
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function fmt(num) {
  if (num === null || num === undefined) return '—';
  if (typeof num === 'number') {
    if (Number.isNaN(num) || !Number.isFinite(num)) return '—';
    if (Math.abs(num) >= 1e6) return I18N.num(num, { notation: 'compact', maximumFractionDigits: 2 });
    return I18N.num(num, { maximumFractionDigits: Math.abs(num) >= 1000 ? 2 : 4 });
  }
  return String(num);
}

function fmtInt(n) {
  if (n === null || n === undefined) return '—';
  return I18N.num(Number(n), { maximumFractionDigits: 0 });
}

function fmtQ(q) {
  if (q === null || q === undefined) return '—';
  return q < 0.001 ? `<${I18N.num(0.001, { maximumFractionDigits: 3 })}` : I18N.num(q, { maximumFractionDigits: 3 });
}

// Compact number for axis labels (3 significant digits)
function fmtShort(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  return I18N.num(Number(v.toPrecision(3)), { maximumSignificantDigits: 3, notation: Math.abs(v) >= 1e6 ? 'compact' : 'standard' });
}

// Show a cell value with visible quotes so whitespace differences are obvious
function quoted(v) {
  if (v === null || v === undefined) return t('common.missing_value');
  return `"${v}"`;
}

function destroyCharts() {
  _charts.forEach(c => { try { c.destroy(); } catch(_) {} });
  _charts = [];
}

function typeCategory(kind) {
  const k = (kind || '').toLowerCase();
  if (k === 'numeric') return 'numeric';
  if (k === 'categorical' || k === 'text') return 'text';
  if (k === 'datetime') return 'datetime';
  if (k === 'boolean') return 'boolean';
  return 'other';
}

// Map correlation value [-1, 1] to a CSS color
function corrColor(v) {
  if (v === null || v === undefined || isNaN(v)) return 'var(--surface-3)';
  const tt = (v + 1) / 2;
  if (tt < 0.5) return `rgba(255, 107, 107, ${0.1 + 0.7 * (1 - tt / 0.5)})`;
  return `rgba(108, 99, 255, ${0.1 + 0.7 * ((tt - 0.5) / 0.5)})`;
}

function el(tag, attrs = {}, html = '') {
  const e = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else e.setAttribute(k, v);
  });
  if (html) e.innerHTML = html;
  return e;
}

function emptyNote(text) {
  return `<p class="empty-note">${escapeHtml(text)}</p>`;
}

function debounce(fn, ms) {
  let tm;
  return (...args) => { clearTimeout(tm); tm = setTimeout(() => fn(...args), ms); };
}

function axisColors() {
  return { tick: cssVar('--text-muted'), grid: 'rgba(128,128,128,0.12)' };
}

// Locale and direction for every chart: mirrored category/time axes in RTL
function withDirection(config) {
  const rtl = I18N.rtl;
  const o = config.options = config.options || {};
  o.locale = I18N.locale;
  o.plugins = o.plugins || {};
  o.plugins.legend = { ...(o.plugins.legend || {}), rtl, textDirection: rtl ? 'rtl' : 'ltr' };
  o.plugins.tooltip = { rtl, textDirection: rtl ? 'rtl' : 'ltr', ...(o.plugins.tooltip || {}) };
  if (o.scales && config.type !== 'scatter') {
    if (o.scales.x) o.scales.x.reverse = rtl;
    if (o.scales.y) o.scales.y.position = rtl ? 'right' : 'left';
  }
  return config;
}

function addChart(canvas, config) {
  const chart = new Chart(canvas, withDirection(config));
  _charts.push(chart);
  return chart;
}

function badge(cls, text) {
  return `<span class="badge ${cls}">${escapeHtml(text)}</span>`;
}

function directionBadge(direction) {
  if (!['positive', 'negative', 'investigate'].includes(direction)) return '';
  const icon = { positive: '▲', negative: '▼', investigate: '◆' }[direction];
  return `<span class="badge dir-${direction}" title="${escapeHtml(t('direction.' + direction + '_help'))}">${icon} ${escapeHtml(t('direction.' + direction))}</span>`;
}

// ─── API ─────────────────────────────────────────────────────

class ApiError extends Error {
  constructor(m) { super(msg(m)); this.m = m; }
}

async function api(path, opts = {}) {
  let resp;
  try {
    resp = await fetch(API_BASE + path, { credentials: FETCH_CREDENTIALS, ...opts });
  } catch (_) {
    throw new ApiError({ key: 'err.network', params: { url: API_BASE || location.origin } });
  }
  const text = await resp.text();
  let data = null;
  try { data = JSON.parse(text); } catch (_) { data = null; }
  if (!resp.ok) {
    if (data && data.error && data.error.key) throw new ApiError(data.error);
    throw new ApiError({ key: 'err.server', params: { status: resp.status } });
  }
  if (data === null) throw new ApiError({ key: 'err.unexpected' });
  return data;
}

function postJson(path, body) {
  return api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}) });
}

function errMsg(err) {
  return err && err.m ? err.m : { key: 'err.unexpected' };
}

// ─── DOM References ──────────────────────────────────────────

const $ = id => document.getElementById(id);

const fileInput       = $('file-input');
const uploadLabel     = $('upload-label');
const emptyState      = $('empty-state');
const loadingOverlay  = $('loading-overlay');
const loadingText     = loadingOverlay.querySelector('.spinner-text');
const dashboard       = $('dashboard');
const uploadError     = $('upload-error');
const themeToggle     = $('theme-toggle');
const iconDark        = $('theme-icon-dark');
const iconLight       = $('theme-icon-light');
const fileBadge       = $('file-badge');
const dashFilename    = $('dash-filename');
const dashMeta        = $('dash-meta');
const uploadNewBtn    = $('upload-new-btn');

function showLoading(key) {
  loadingText.textContent = t(key);
  loadingOverlay.classList.remove('hidden');
}
function hideLoading() { loadingOverlay.classList.add('hidden'); }

// ─── Theme ───────────────────────────────────────────────────

(function initTheme() {
  const saved = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  updateThemeIcons(saved);
})();

function updateThemeIcons(theme) {
  iconDark.classList.toggle('hidden', theme === 'light');
  iconLight.classList.toggle('hidden', theme !== 'light');
}

themeToggle.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  updateThemeIcons(next);
  if (App.data) renderAllCharts(); // chart text colors follow the theme
});

// ─── Language ────────────────────────────────────────────────

document.querySelectorAll('.lang-btn').forEach(b => {
  b.addEventListener('click', () => { if (b.dataset.lang !== I18N.lang) I18N.setLang(b.dataset.lang); });
});

// Re-render everything from the data already in memory. Nothing is refetched
// and no dataset value passes through translation.
I18N.onChange(() => {
  if (App.uploadError) showError(App.uploadError);
  if (!App.data) return;
  if (!App.reportLang) setSegActive('report-lang', 'rlang', I18N.lang);
  renderDashboard(App.data, { keepPending: true });
  if (App.lastPreview) renderPreviewTable(App.lastPreview);
  if (App.lastChanges) renderChanges(App.lastChanges);
});

// ─── Tab Navigation ──────────────────────────────────────────

function activateTab(tab) {
  if (tab === 'analysis') tab = App.subTab;
  const detail = DETAIL_TABS.includes(tab);
  if (detail) App.subTab = tab;
  $('subtab-nav').classList.toggle('hidden', !detail);
  document.querySelectorAll('.tab-nav .tab-btn').forEach(b => {
    const on = b.dataset.tab === tab || (detail && b.dataset.tab === 'analysis');
    b.classList.toggle('active', on);
    b.setAttribute('aria-selected', on ? 'true' : 'false');
  });
  document.querySelectorAll('.subtab-nav .tab-btn').forEach(b => {
    const on = b.dataset.tab === tab;
    b.classList.toggle('active', on);
    b.setAttribute('aria-selected', on ? 'true' : 'false');
  });
  document.querySelectorAll('.tab-panel').forEach(p => {
    const on = p.id === 'tab-' + tab;
    p.classList.toggle('hidden', !on);
    p.classList.toggle('active', on);
  });
  App.visited.add(tab);
  if (App.data) renderStepper();
  if (App.data && tab === 'explore') Explore.onShow();
}

function currentTab() {
  const b = document.querySelector('.subtab-nav:not(.hidden) .tab-btn.active') || document.querySelector('.tab-nav .tab-btn.active');
  return b ? b.dataset.tab : 'summary';
}

// ─── Browser history (Back / Forward between views) ─────────
// Only user navigation creates an entry: opening a tab, the upload screen, a new upload,
// or an Analyze mode. Re-renders (cleaning, language, charts, filters) never do. Entries
// are restored by `popstate` with recording switched off, so restoring cannot loop.
// The URL hash (#tab=…&mode=…&ds=…) lets a refresh or a copied link reopen the same view
// while the dataset is still held by the server.

const Nav = {
  restoring: false,
  current() {
    const onDash = !!App.data && !dashboard.classList.contains('hidden');
    if (!onDash) return { app: 'sdap', view: 'upload' };
    const tab = currentTab();
    return { app: 'sdap', view: 'dashboard', tab, ds: App.data.dataset_id, mode: tab === 'explore' ? Explore.mode : null };
  },
  url(s) {
    const base = location.pathname + location.search;
    if (s.view !== 'dashboard') return base;
    const p = new URLSearchParams({ tab: s.tab });
    if (s.mode) p.set('mode', s.mode);
    p.set('ds', s.ds);
    return base + '#' + p.toString();
  },
  same(a, b) {
    return !!a && !!b && a.view === b.view && (a.view === 'upload' || (a.tab === b.tab && a.ds === b.ds && (a.mode || null) === (b.mode || null)));
  },
  // push a new entry for user navigation; replace = keep the URL accurate without a new entry
  record(replace = false) {
    if (this.restoring) return;
    const s = this.current();
    if (!replace && this.same(history.state, s)) return;
    try { history[replace ? 'replaceState' : 'pushState'](s, '', this.url(s)); } catch (_) { /* e.g. sandboxed file:// */ }
  },
  parse(hash) {
    const p = new URLSearchParams(String(hash || '').replace(/^#/, ''));
    const tab = p.get('tab'), ds = p.get('ds');
    if (!tab || !ds || !/^[0-9a-f]{32}$/.test(ds) || !document.getElementById('tab-' + tab)) return { app: 'sdap', view: 'upload' };
    return { app: 'sdap', view: 'dashboard', tab, ds, mode: p.get('mode') || null };
  },
};

function goTab(tab) {
  activateTab(tab);
  Nav.record();
}

function showDashboard() {
  emptyState.classList.add('hidden');
  dashboard.classList.remove('hidden');
  fileBadge.classList.remove('hidden');
}

// Show a recorded view without creating history entries
async function restoreView(s) {
  if (!s || s.view !== 'dashboard') {
    if (App.data) showEmptyState();
    return;
  }
  Nav.restoring = true;
  try {
    if (!App.data || App.data.dataset_id !== s.ds) {
      // The page no longer holds this dataset (refresh, link, older upload): reload it if the server still has it
      showLoading('loading.analyzing');
      let data;
      try { data = await api(`/datasets/${s.ds}`); } catch (_) { data = null; } finally { hideLoading(); }
      if (!data) {
        if (App.data) showEmptyState();
        try { history.replaceState({ app: 'sdap', view: 'upload' }, '', Nav.url({ view: 'upload' })); } catch (_) { /* ignore */ }
        return;
      }
      startSession(data, { record: false });
    }
    showDashboard();
    if (s.tab === 'explore' && s.mode) Explore.setMode(s.mode);
    activateTab(s.tab);
  } finally {
    Nav.restoring = false;
  }
}

window.addEventListener('popstate', e => {
  const s = e.state && e.state.app === 'sdap' ? e.state : Nav.parse(location.hash);
  restoreView(s);
});

document.querySelectorAll('.tab-nav .tab-btn, .subtab-nav .tab-btn').forEach(btn => {
  btn.addEventListener('click', () => goTab(btn.dataset.tab));
});

// Buttons/links that jump to another step
document.addEventListener('click', e => {
  const b = e.target.closest('[data-goto]');
  if (!b || !App.data) return;
  goTab(b.dataset.goto);
  if (b.dataset.card) {
    const card = document.getElementById('card-' + b.dataset.card);
    if (card) { card.scrollIntoView({ behavior: 'smooth', block: 'center' }); card.classList.add('flash'); setTimeout(() => card.classList.remove('flash'), 1600); }
  } else {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
});

// ─── Upload Handling ─────────────────────────────────────────

if (uploadLabel) {
  uploadLabel.addEventListener('click', (e) => {
    if (e.target === fileInput) return;
    fileInput.click();
  });
  uploadLabel.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  uploadLabel.addEventListener('dragover', e => { e.preventDefault(); uploadLabel.classList.add('drag-over'); });
  uploadLabel.addEventListener('dragleave', (e) => {
    if (!uploadLabel.contains(e.relatedTarget)) uploadLabel.classList.remove('drag-over');
  });
  uploadLabel.addEventListener('drop', e => {
    e.preventDefault();
    uploadLabel.classList.remove('drag-over');
    const file = e.dataTransfer?.files?.[0];
    if (file) processFile(file);
  });
}

// Prevent the browser from navigating to a file dropped outside the zone
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => e.preventDefault());

// Verify the backend connection once translations are ready, then reopen the view in the URL (refresh / link)
I18N.ready.then(() => {
  fetch(API_BASE + '/health')
    .then(r => { if (!r.ok) throw new Error(); })
    .catch(() => showError({ key: 'err.server_unreachable', params: { url: API_BASE || location.origin } }));
  const target = Nav.parse(location.hash);
  if (target.view === 'dashboard') {
    restoreView(target).then(() => Nav.record(true));
  } else {
    try { history.replaceState({ app: 'sdap', view: 'upload' }, '', Nav.url(target)); } catch (_) { /* ignore */ }
  }
});

fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (file) processFile(file);
});

uploadNewBtn.addEventListener('click', () => {
  showEmptyState();
  fileInput.value = '';
  Nav.record();   // Back returns to the dashboard; the dataset stays loaded
});

function showEmptyState() {
  emptyState.classList.remove('hidden');
  dashboard.classList.add('hidden');
  fileBadge.classList.add('hidden');
  clearError();
}

function clearError() {
  App.uploadError = null;
  uploadError.classList.add('hidden');
  uploadError.textContent = '';
}

function showError(m) {
  App.uploadError = m;
  uploadError.textContent = msg(m);
  uploadError.classList.remove('hidden');
}

async function processFile(file) {
  // Validate extension BEFORE showing any loading state
  const allowed = ['.csv', '.xlsx', '.xls', '.json'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!allowed.includes(ext)) {
    showError({ key: 'err.unsupported_ext', params: { ext } });
    return;
  }

  clearError();
  showLoading('loading.analyzing');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const data = await api('/upload', { method: 'POST', body: formData });
    if (!data.analysis || !data.dataset_id) throw new ApiError({ key: 'err.unexpected' });
    const previous = App.data && App.data.dataset_id;
    startSession(data);
    // Free the previous dataset on the server; it is no longer reachable from the UI
    if (previous && previous !== data.dataset_id) fetch(`${API_BASE}/datasets/${previous}`, { method: 'DELETE', credentials: FETCH_CREDENTIALS }).catch(() => {});
  } catch (err) {
    showError({ key: 'err.upload_failed', params: { reason: errMsg(err) } });
  } finally {
    hideLoading();
    fileInput.value = '';
  }
}

// ─── Session & Dashboard Renderer ───────────────────────────

function startSession(data, { record = true } = {}) {
  App.gen += 1;
  App.preview = defaultPreviewState();
  App.lastPreview = null;
  App.lastChanges = null;
  App.changes = { page: 1, column: '' };
  App.issueSev = 'all';
  App.findingsKind = 'all';
  App.visited = new Set();
  App.exported = false;
  App.subTab = 'overview';
  // Clear every input that could carry state from the previous upload
  ['row-search', 'table-search', 'filter-val'].forEach(id => { $(id).value = ''; });
  $('filter-op').value = 'contains';
  $('changed-only').checked = false;
  $('recipe-msg').textContent = '';
  $('clean-error').classList.add('hidden');
  App.corrMethod = 'spearman';
  setSegActive('corr-method', 'method', 'spearman');
  setSegActive('version-toggle', 'version', 'cleaned');
  setSegActive('issue-filter', 'sev', 'all');
  setSegActive('findings-filter', 'kind', 'all');
  if (data.explore_catalog) Explore.reset(data);
  renderDashboard(data, { resetTab: true });
  if (record) Nav.record();
}

function renderDashboard(data, { resetTab = false, keepPending = false } = {}) {
  destroyCharts();
  App.data = data;
  if (!keepPending) App.pending = JSON.parse(JSON.stringify(data.decisions || {}));

  const { filename, profile } = data;
  const report = data.cleaning.report;

  fileBadge.textContent = filename;
  fileBadge.classList.remove('hidden');
  dashFilename.textContent = filename;
  dashMeta.textContent = t('dash.meta', {
    rows: profile.row_count, columns: profile.column_count,
    orig_rows: report.rows.before, orig_columns: report.columns.before, changed: report.values.changed,
  });

  emptyState.classList.add('hidden');
  dashboard.classList.remove('hidden');
  const tab = resetTab ? 'summary' : currentTab();

  const highOrig = (data.profile_original.issue_counts.high || 0) + (data.profile_original.issue_counts.medium || 0);
  const qc = $('quality-count'); qc.textContent = highOrig ? fmtInt(highOrig) : ''; qc.classList.toggle('alert', highOrig > 0);
  const pend = report.pending_review.length;
  const cc = $('cleaning-count'); cc.textContent = pend ? fmtInt(pend) : ''; cc.classList.toggle('alert', pend > 0);

  renderSummaryTab(data);
  renderFindingsTab(data);
  renderActionsTab(data);
  renderOverview(data);
  renderQualityTab(data);
  renderCleaningTab(data);
  renderStatsTab(data);
  renderChartsTab(data);
  renderCorrelationTab(data);
  renderExportTab(data);
  Explore.sync(data);
  setupPreviewColumns();
  activateTab(tab);
  if (!keepPending || !App.lastPreview) loadPreview();
  if (!keepPending || !App.lastChanges) loadChanges();
}

function renderAllCharts() {
  destroyCharts();
  renderSummaryCharts(App.data);
  renderCatCharts(App.data.profile);
  renderChartsTab(App.data);
  Explore.redraw();
}

// ─── Workflow stepper ────────────────────────────────────────

function renderStepper() {
  const d = App.data;
  const pending = d.cleaning.report.pending_review.length;
  const highIssues = d.profile_original.issue_counts.high || 0;
  const v = id => App.visited.has(id);
  const steps = [
    { id: 'upload', goto: null, state: 'done' },
    { id: 'quality', goto: 'quality', state: v('quality') ? 'done' : (highIssues ? 'attention' : 'todo'),
      note: highIssues ? t('step.note_issues', { count: highIssues }) : '' },
    { id: 'cleaning', goto: 'cleaning', state: pending ? 'attention' : (v('cleaning') ? 'done' : 'todo'),
      note: pending ? t('step.note_pending', { count: pending }) : '' },
    { id: 'findings', goto: 'findings', state: v('findings') ? 'done' : 'todo' },
    { id: 'actions', goto: 'actions', state: v('actions') ? 'done' : 'todo' },
    { id: 'export', goto: 'export', state: App.exported ? 'done' : 'todo' },
  ];
  const cur = currentTab();
  const ol = $('stepper');
  ol.innerHTML = '';
  steps.forEach((s, i) => {
    const li = el('li', { class: `step ${s.state}${s.goto === cur ? ' current' : ''}` });
    const icon = s.state === 'done' ? '✓' : s.state === 'attention' ? '!' : I18N.num(i + 1);
    const btn = el('button', { class: 'step-btn', type: 'button', 'aria-current': s.goto === cur ? 'step' : 'false' });
    btn.innerHTML = `<span class="step-icon" aria-hidden="true">${icon}</span><span class="step-text"><span class="step-label">${escapeHtml(t('step.' + s.id))}</span>`
      + `<span class="step-state">${escapeHtml(s.note || t('step.state_' + s.state))}</span></span>`;
    btn.addEventListener('click', () => {
      if (s.id === 'upload') {
        if (confirm(t('step.confirm_new_upload'))) { showEmptyState(); fileInput.value = ''; Nav.record(); }
      } else {
        goTab(s.goto);
      }
    });
    li.appendChild(btn);
    ol.appendChild(li);
  });
}

// ─── Executive Summary ───────────────────────────────────────

function cardsById(data) {
  return Object.fromEntries(data.insights.map(c => [c.id, c]));
}

function renderSummaryTab(data) {
  const s = data.summary;
  const ev = $('summary-evidence');
  ev.className = `evidence-banner ${s.evidence.level}`;
  ev.innerHTML = `<strong>${escapeHtml(t('sum.evidence_label'))}:</strong> ${msgHtml(s.evidence.msg)}`;

  $('summary-description').innerHTML = s.description.map(m => `<p>${msgHtml(m)}</p>`).join('');

  const kc = $('summary-kpis');
  kc.innerHTML = '';
  s.kpis.forEach(k => {
    let value;
    if (k.value && typeof k.value === 'object') value = msgHtml(k.value);
    else if (k.unit === 'pct') value = escapeHtml(I18N.pct(k.value));
    else value = escapeHtml(fmt(k.value));
    const card = el('div', { class: `kpi-card manager dir-${k.direction}` });
    card.innerHTML = `<div class="kpi-label">${msgHtml(k.label)}</div><div class="kpi-value ${k.value && typeof k.value === 'object' ? 'small' : ''}">${value}</div>`
      + `<div class="kpi-sub">${msgHtml(k.sub)}</div>${directionBadge(k.direction)}`;
    kc.appendChild(card);
  });

  const byId = cardsById(data);
  const mini = (containerId, ids, emptyKey) => {
    const c = $(containerId);
    const cards = ids.map(id => byId[id]).filter(Boolean);
    c.innerHTML = cards.length ? '' : emptyNote(t(emptyKey));
    cards.forEach(card => {
      const item = el('div', { class: `mini-item ${card.priority}` });
      item.innerHTML = `<div class="mini-title">${msgHtml(card.title)} ${badge(card.priority, t('priority.' + card.priority))} ${directionBadge(card.direction)}</div>`
        + `<div class="mini-text">${msgHtml(card.finding)}</div>`
        + `<button class="btn-link" data-goto="findings" data-card="${escapeHtml(card.id)}">${escapeHtml(t('sum.view_evidence'))}</button>`;
      c.appendChild(item);
    });
  };
  mini('summary-findings', s.findings, 'sum.none_findings');
  mini('summary-risks', s.risks, 'sum.none_risks');
  mini('summary-opportunities', s.opportunities, 'sum.none_opportunities');
  mini('summary-warnings', s.quality_warnings, 'sum.none_warnings');

  const ol = $('summary-actions');
  ol.innerHTML = '';
  s.actions.slice(0, 5).forEach(a => {
    ol.appendChild(el('li', { class: 'action-row' }, `${badge(a.priority, t('priority.' + a.priority))} <span>${msgHtml(a.msg)}</span>`));
  });
  if (!s.actions.length) ol.innerHTML = emptyNote(t('sum.none_actions'));
  renderSummaryCharts(data);
}

function renderSummaryCharts(data) {
  const container = $('summary-charts');
  container.innerHTML = '';
  const ax = axisColors();
  data.summary.charts.forEach((ch, idx) => {
    const card = el('div', { class: 'chart-card' });
    const tip = [msg(ch.explanation), ch.note ? msg(ch.note) : ''].filter(Boolean).join(' ');
    card.innerHTML = `<div class="chart-card-title">${msgHtml(ch.title)} ${directionBadge(ch.direction)}`
      + ` <span class="info-tip" tabindex="0" role="note" title="${escapeHtml(tip)}" aria-label="${escapeHtml(tip)}">i</span></div>`
      + `<div class="chart-wrap-big"><canvas></canvas></div>`;
    container.appendChild(card);
    const canvas = card.querySelector('canvas');
    if (ch.type === 'trend') {
      addChart(canvas, {
        type: 'line',
        data: { labels: ch.labels.map(p => I18N.period(p, ch.freq)), datasets: [{ data: ch.values, borderColor: PALETTE[1], backgroundColor: PALETTE[1] + '33', fill: true, tension: 0.25, pointRadius: 2 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
          scales: { x: { ticks: { color: ax.tick, maxTicksLimit: 10 }, grid: { color: ax.grid } }, y: { beginAtZero: ch.agg !== 'mean', ticks: { color: ax.tick }, grid: { color: ax.grid } } } },
      });
    } else {
      addChart(canvas, {
        type: 'bar',
        data: { labels: ch.labels, datasets: [{ data: ch.values, backgroundColor: PALETTE[(idx + 3) % PALETTE.length] + 'bb', borderRadius: 4 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false },
            tooltip: { callbacks: { label: ctx => ch.counts ? t('chart.tooltip_n', { value: fmt(ctx.raw), count: ch.counts[ctx.dataIndex] }) : fmt(ctx.raw) } } },
          scales: { x: { ticks: { color: ax.tick, maxRotation: 30 }, grid: { display: false } }, y: { beginAtZero: true, ticks: { color: ax.tick }, grid: { color: ax.grid } } } },
      });
    }
  });
}

// ─── Findings (action-oriented insight cards) ────────────────

function insightCardHtml(c) {
  const cols = c.evidence.columns.map(col => `<span class="col">${raw(col)}</span>`).join('');
  const ev = c.evidence.items.map(e => `<li>${msgHtml(e)}</li>`).join('');
  const lim = c.limitations.map(x => `<li>${msgHtml(x)}</li>`).join('');
  return `<div class="insight-head"><div class="insight-title">${msgHtml(c.title)}</div>
      <div class="insight-badges">${badge('kind-' + c.kind, t('kind_card.' + c.kind))} ${badge(c.priority, t('priority.' + c.priority))} ${directionBadge(c.direction)}</div></div>
    <p class="insight-finding">${msgHtml(c.finding)}</p>
    <details class="more"><summary>${escapeHtml(t('card.details'))}</summary>
    <dl class="insight-body">
      <dt>${escapeHtml(t('card.why'))}</dt><dd>${msgHtml(c.why)}</dd>
      <dt>${escapeHtml(t('card.action'))}</dt><dd class="action-text">${msgHtml(c.action)}</dd>
      <dt>${escapeHtml(t('card.evidence'))}</dt><dd><ul>${ev}</ul><div class="evidence">${cols}</div></dd>
      ${lim ? `<dt>${escapeHtml(t('card.limitations'))}</dt><dd><ul>${lim}</ul></dd>` : ''}
    </dl>
    ${c.technical ? `<p class="tech-line">${msgHtml(c.technical)}</p>` : ''}</details>`;
}

function renderFindingsTab(data) {
  const c = $('findings-container');
  c.innerHTML = '';
  const kind = App.findingsKind;
  const list = data.insights.filter(i => kind === 'all' || i.kind === kind);
  const groups = [
    { key: 'findings.group_urgent', test: i => i.kind === 'data_quality' && ['urgent', 'high'].includes(i.priority) },
    { key: 'findings.group_business', test: i => i.kind === 'business' || i.category === 'no_findings' },
    { key: 'findings.group_quality_other', test: i => i.kind === 'data_quality' && !['urgent', 'high'].includes(i.priority) },
    { key: 'findings.group_routine', test: i => i.kind === 'observation' && i.category !== 'no_findings' },
  ];
  groups.forEach(g => {
    const items = list.filter(g.test);
    if (!items.length) return;
    c.appendChild(el('div', { class: 'insight-group-title', text: t(g.key) }));
    items.forEach(i => {
      c.appendChild(el('article', { class: `insight-card ${i.priority} dir-${i.direction}`, id: 'card-' + i.id }, insightCardHtml(i)));
    });
  });
  if (!list.length) c.innerHTML = emptyNote(t('findings.none'));
  $('limitations-list').innerHTML = data.eda.limitations.map(l => `<li>${msgHtml(l)}</li>`).join('');
}

$('findings-filter').addEventListener('click', e => {
  const b = e.target.closest('.seg-btn');
  if (!b || !App.data) return;
  App.findingsKind = b.dataset.kind;
  setSegActive('findings-filter', 'kind', b.dataset.kind);
  renderFindingsTab(App.data);
});

// ─── Next actions (decisions saved in this browser) ──────────

function decisionStoreKey(data) {
  return `sdap.decisions.${data.filename}|${data.original.rows}|${data.original.columns.join('\u001f')}`;
}

function loadActionDecisions(data) {
  try { return JSON.parse(localStorage.getItem(decisionStoreKey(data)) || '{}'); } catch (_) { return {}; }
}

function saveActionDecision(data, index, value) {
  const all = loadActionDecisions(data);
  all[index] = value;
  localStorage.setItem(decisionStoreKey(data), JSON.stringify(all));
  renderExportTab(data); // export links carry the decisions
}

function renderActionsTab(data) {
  const decisions = loadActionDecisions(data);
  const fix = $('actions-fix'), inv = $('actions-investigate');
  fix.innerHTML = ''; inv.innerHTML = '';
  data.summary.actions.forEach((a, i) => {
    const li = el('li', { class: 'action-row decision-row' });
    const sel = `<select class="search-input small decision-select" data-index="${i}" aria-label="${escapeHtml(t('actions.decision'))}">${DECISIONS.map(dv =>
      `<option value="${dv}" ${(decisions[i] || 'pending') === dv ? 'selected' : ''}>${escapeHtml(t('decision.' + dv))}</option>`).join('')}</select>`;
    li.innerHTML = `<div class="decision-main">${badge(a.priority, t('priority.' + a.priority))} <span>${msgHtml(a.msg)}</span>`
      + (a.insight ? ` <button class="btn-link" data-goto="findings" data-card="${escapeHtml(a.insight)}">${escapeHtml(t('sum.view_evidence'))}</button>` : '')
      + (a.insight === null ? ` <button class="btn-link" data-goto="cleaning">${escapeHtml(t('actions.open_cleaning'))}</button>` : '')
      + `</div>${sel}`;
    li.querySelector('select').addEventListener('change', e => saveActionDecision(data, i, e.target.value));
    (a.type === 'fix_data' ? fix : inv).appendChild(li);
  });
  if (!fix.children.length) fix.innerHTML = emptyNote(t('actions.none_fix'));
  if (!inv.children.length) inv.innerHTML = emptyNote(t('actions.none_investigate'));
}

// ─── Technical overview (Detailed analysis) ──────────────────

function renderOverview(data) {
  renderCleanBanner(data);
  renderKPIs(data);
  renderColumnTypes(data.profile);
  renderDataQuality(data.profile);
  renderCatCharts(data.profile);
}

function renderCleanBanner(data) {
  const report = data.cleaning.report;
  const applied = data.cleaning.operations.filter(o => o.status === 'applied');
  const safe = applied.filter(o => o.category === 'safe').length;
  const reviewed = applied.filter(o => o.category === 'review').length;
  const pending = report.pending_review.length;
  const provisional = (data.eda.provisional || []).map(p => p.column);
  $('clean-banner').innerHTML = `
    <div>
      <strong>${escapeHtml(t('banner.cleaned_copy'))}</strong>
      ${escapeHtml(t('banner.applied', { safe, reviewed, removed: report.rows.removed, changed: report.values.changed }))}
      <strong>${escapeHtml(pending ? t('banner.pending', { count: pending }) : t('banner.none_pending'))}</strong>
      ${provisional.length ? `<br><span class="muted">${th('banner.provisional', { columns: provisional })}</span>` : ''}
    </div>
    <button class="btn-secondary" data-goto="cleaning">${escapeHtml(t('banner.review'))}</button>`;
}

function kpiCard(container, k) {
  const card = el('div', { class: 'kpi-card' });
  card.style.setProperty('--kpi-color', k.color);
  card.innerHTML = `<div class="kpi-label">${escapeHtml(k.label)}</div><div class="kpi-value">${escapeHtml(String(k.value))}</div><div class="kpi-sub">${escapeHtml(k.sub)}</div>`;
  container.appendChild(card);
}

function renderKPIs(data) {
  const container = $('kpi-grid');
  container.innerHTML = '';
  const p = data.profile;
  const used = data.eda.columns_used;
  const totalMissing = p.missing_cells + p.marker_cells;
  const issues = (p.issue_counts.high || 0) + (p.issue_counts.medium || 0);
  [
    { label: t('kpi.total_rows'), value: fmtInt(p.row_count), sub: t('kpi.in_original', { count: data.cleaning.report.rows.before }), color: 'var(--accent)' },
    { label: t('kpi.columns'), value: fmtInt(p.column_count), sub: t('kpi.identifiers', { count: p.role_counts.identifier || 0 }), color: 'var(--accent-2)' },
    { label: t('kpi.measures'), value: fmtInt(used.measures.length), sub: t('kpi.quantitative'), color: 'var(--accent)' },
    { label: t('kpi.categories'), value: fmtInt(used.categories.length), sub: t('kpi.qualitative'), color: 'var(--accent-2)' },
    { label: t('kpi.missing_cells'), value: fmtInt(totalMissing), sub: t('kpi.of_data', { pct: p.missing_pct }), color: totalMissing > 0 ? 'var(--warning)' : 'var(--positive)' },
    { label: t('kpi.duplicate_rows'), value: fmtInt(p.exact_duplicate_rows), sub: t('kpi.exact_duplicates'), color: p.exact_duplicate_rows > 0 ? 'var(--danger)' : 'var(--positive)' },
    { label: t('kpi.open_issues_label'), value: fmtInt(issues), sub: t('kpi.high_medium'), color: issues > 0 ? 'var(--warning)' : 'var(--positive)' },
  ].forEach(k => kpiCard(container, k));
}

function renderColumnTypes(profile) {
  const container = $('col-types-list');
  container.innerHTML = '';
  profile.columns.forEach(c => {
    const tag = document.createElement('span');
    tag.className = `type-tag ${typeCategory(c.kind)}`;
    const role = ['identifier', 'constant', 'empty', 'text'].includes(c.role) && c.role !== c.kind ? ` · ${t('role.' + c.role)}` : '';
    tag.innerHTML = `<span class="type-tag-name" title="${escapeHtml(c.name)}">${raw(c.name)}</span><span>${escapeHtml(t('kind.' + c.kind) + role)}</span>`;
    container.appendChild(tag);
  });
}

function renderDataQuality(profile) {
  const container = $('missing-list');
  container.innerHTML = '';
  profile.columns.forEach(c => {
    const pctv = c.missing_pct || 0;
    const color = pctv === 0 ? 'var(--positive)' : pctv < 5 ? 'var(--warning)' : 'var(--danger)';
    const row = el('div', { class: 'quality-row', title: t('quality.missing_of', { count: c.missing_total, total: c.count }) });
    row.innerHTML = `
      <span class="quality-col" title="${escapeHtml(c.name)}">${raw(c.name)}</span>
      <div class="quality-bar-bg"><div class="quality-bar-fill" style="width: ${Math.min(100, pctv)}%; background: ${color};"></div></div>
      <span class="quality-pct" style="color:${color}">${escapeHtml(I18N.pct(pctv))}</span>`;
    container.appendChild(row);
  });
  if (profile.columns.length === 0) container.innerHTML = emptyNote(t('common.no_columns'));
}

function catColumns(profile) {
  return profile.columns.filter(c => (c.kind === 'categorical' || c.kind === 'boolean') && c.top_values.length && c.role !== 'identifier');
}

function renderCatCharts(profile) {
  const container = $('cat-charts-grid');
  container.innerHTML = '';
  const cats = catColumns(profile);
  if (cats.length === 0) { container.innerHTML = emptyNote(t('overview.no_categories')); return; }
  const ax = axisColors();
  cats.slice(0, 6).forEach(c => {
    const labels = c.top_values.map(tv => tv.value);
    const values = c.top_values.map(tv => tv.count);
    const more = c.unique > labels.length ? ` ${t('common.top_of', { count: labels.length, total: c.unique })}` : '';
    const card = el('div', { class: 'chart-card' });
    card.innerHTML = `<div class="chart-card-title" title="${escapeHtml(c.name)}">${raw(c.name)}${escapeHtml(more)}</div><div class="chart-wrap"><canvas></canvas></div>`;
    container.appendChild(card);
    addChart(card.querySelector('canvas'), {
      type: 'bar',
      data: { labels, datasets: [{ label: c.name, data: values, backgroundColor: PALETTE.slice(0, labels.length).map(col => col + 'cc'), borderColor: PALETTE.slice(0, labels.length), borderWidth: 1, borderRadius: 4 }] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
        scales: { x: { ticks: { color: ax.tick, font: { size: 10 }, maxRotation: 30 }, grid: { display: false } },
                  y: { beginAtZero: true, ticks: { color: ax.tick, font: { size: 10 } }, grid: { color: ax.grid } } } },
    });
  });
}

// ─── Data Quality Tab (original upload) ──────────────────────

function renderQualityTab(data) {
  const p = data.profile_original;
  const kc = $('quality-kpis');
  kc.innerHTML = '';
  [
    { label: t('kpi.rows'), value: fmtInt(p.row_count), sub: t('kpi.as_uploaded'), color: 'var(--accent)' },
    { label: t('kpi.columns'), value: fmtInt(p.column_count), sub: Object.entries(p.type_counts).map(([k, v]) => `${fmtInt(v)} ${t('kind.' + k)}`).join(t('list.separator')), color: 'var(--accent-2)' },
    { label: t('kpi.missing_cells'), value: fmtInt(p.missing_cells + p.marker_cells), sub: t('kpi.markers', { count: p.marker_cells }), color: 'var(--warning)' },
    { label: t('kpi.invalid_values'), value: fmtInt(p.invalid_cells), sub: t('kpi.wrong_type'), color: p.invalid_cells ? 'var(--danger)' : 'var(--positive)' },
    { label: t('kpi.exact_duplicates_label'), value: fmtInt(p.exact_duplicate_rows), sub: t('kpi.after_normalizing', { count: p.normalized_duplicate_rows }), color: p.exact_duplicate_rows ? 'var(--danger)' : 'var(--positive)' },
    { label: t('kpi.high_issues'), value: fmtInt(p.issue_counts.high || 0), sub: t('kpi.medium_count', { count: p.issue_counts.medium || 0 }), color: p.issue_counts.high ? 'var(--danger)' : 'var(--positive)' },
  ].forEach(k => kpiCard(kc, k));
  renderIssueList();
  renderProfileTable(p);
  renderDuplicates(p);
  renderNameIssues(p);
}

function issueMsg(i) {
  return i.msg ? msgHtml(i.msg) : escapeHtml(i.message);
}

function renderIssueList() {
  const issues = App.data.profile_original.issues;
  const sev = App.issueSev;
  const list = issues.filter(i => sev === 'all' || i.severity === sev || (sev === 'low' && (i.severity === 'low' || i.severity === 'info')));
  const c = $('issue-list');
  c.innerHTML = list.length ? '' : emptyNote(t('quality.no_issues_category'));
  list.forEach(i => {
    c.appendChild(el('div', { class: 'issue-row' }, `${badge(i.severity, t('severity.' + i.severity))}
      <span class="issue-col">${i.column === null || i.column === undefined ? escapeHtml(t('common.dataset')) : raw(i.column)}</span><span>${issueMsg(i)}</span>`));
  });
}

$('issue-filter').addEventListener('click', e => {
  const b = e.target.closest('.seg-btn');
  if (!b || !App.data) return;
  App.issueSev = b.dataset.sev;
  setSegActive('issue-filter', 'sev', b.dataset.sev);
  renderIssueList();
});

function setSegActive(groupId, attr, value) {
  $(groupId).querySelectorAll('.seg-btn').forEach(b => b.classList.toggle('active', b.dataset[attr] === value));
}

function renderProfileTable(p) {
  const wrap = $('profile-table');
  const table = el('table');
  table.innerHTML = `<thead><tr>${['column', 'type', 'role', 'missing', 'unique', 'invalid', 'issues'].map(h => `<th>${escapeHtml(t('profile.col.' + h))}</th>`).join('')}</tr></thead>`;
  const tbody = el('tbody');
  p.columns.forEach(c => {
    const tr = el('tr', { class: 'profile-row', tabindex: '0' });
    const sevs = c.issues.filter(i => i.severity === 'high').length;
    tr.innerHTML = `<td class="stat-col-name" title="${escapeHtml(c.name)}">${raw(c.name)}</td>
      <td><span class="type-tag ${typeCategory(c.kind)}">${escapeHtml(t('kind.' + c.kind))}</span></td>
      <td>${escapeHtml(t('role.' + c.role))}</td>
      <td>${fmtInt(c.missing_total)} <span class="muted">(${escapeHtml(I18N.pct(c.missing_pct))})</span></td>
      <td>${fmtInt(c.unique)} <span class="muted">${escapeHtml(t('cardinality.' + c.cardinality))}</span></td>
      <td style="color:${c.invalid_count ? 'var(--danger)' : 'inherit'}">${fmtInt(c.invalid_count)}</td>
      <td>${fmtInt(c.issues.length)}${sevs ? ` ${badge('high', t('profile.high_count', { count: sevs }))}` : ''}</td>`;
    const details = el('tr', { class: 'details-row hidden' });
    details.innerHTML = `<td colspan="7">${columnDetailsHtml(c)}</td>`;
    const toggle = () => details.classList.toggle('hidden');
    tr.addEventListener('click', toggle);
    tr.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
    tbody.appendChild(tr);
    tbody.appendChild(details);
  });
  table.appendChild(tbody);
  wrap.innerHTML = '';
  wrap.appendChild(table);
}

function columnDetailsHtml(c) {
  const blocks = [];
  blocks.push(`<div><h4>${escapeHtml(t('details.counts'))}</h4>${escapeHtml(t('details.counts_text', { count: c.count, present: c.non_missing, nulls: c.null, empty: c.empty, markers: c.marker_count }))}<br>
    ${escapeHtml(t('details.dtype'))}: <code class="raw" dir="ltr">${escapeHtml(c.dtype)}</code>
    ${c.parse_rate !== null && c.parse_rate !== undefined ? `<br>${escapeHtml(t('details.parse_rate', { kind: { key: 'kind.' + c.kind }, pct: Math.round(c.parse_rate * 1000) / 10 }))}` : ''}</div>`);
  if (c.numeric) {
    const n = c.numeric;
    blocks.push(`<div><h4>${escapeHtml(t('details.numeric'))}</h4>${escapeHtml(t('details.numeric_text', {
      mean: fmt(n.mean), median: fmt(n.median), std: fmt(n.std), min: fmt(n.min), q1: fmt(n.q1), q3: fmt(n.q3), max: fmt(n.max), skew: fmt(n.skew),
      zeros: n.zeros, negatives: n.negatives, mild: n.outliers_mild, extreme: n.outliers_extreme }))}</div>`);
  }
  if (c.datetime) {
    blocks.push(`<div><h4>${escapeHtml(t('details.dates'))}</h4>${escapeHtml(t('details.dates_text', { start: { date: c.datetime.min }, end: { date: c.datetime.max }, days: c.datetime.range_days, invalid: c.datetime.invalid_count }))}
      ${c.date_ambiguous ? ' ' + badge('medium', t('details.ambiguous')) : ''}</div>`);
  }
  if (c.top_values.length) {
    blocks.push(`<div><h4>${escapeHtml(t('details.top_values'))}</h4>${c.top_values.slice(0, 6).map(tv => `${raw(quoted(tv.value))} × ${fmtInt(tv.count)} (${escapeHtml(I18N.pct(tv.pct))})`).join('<br>')}</div>`);
  }
  if (c.invalid_examples.length) {
    blocks.push(`<div><h4>${escapeHtml(t('details.invalid'))}</h4>${c.invalid_examples.slice(0, 6).map(tv => `${raw(quoted(tv.value))} × ${fmtInt(tv.count)}`).join('<br>')}</div>`);
  }
  if (Object.keys(c.markers).length) {
    blocks.push(`<div><h4>${escapeHtml(t('details.markers'))}</h4>${Object.entries(c.markers).map(([k, v]) => `${raw(quoted(k))} × ${fmtInt(v)}`).join('<br>')}</div>`);
  }
  if (c.case_variants.length) {
    blocks.push(`<div><h4>${escapeHtml(t('details.labels'))}</h4>${c.case_variants.slice(0, 4).map(g => g.variants.map(v => `${raw(quoted(v.value))}×${fmtInt(v.count)}`).join(' / ')).join('<br>')}</div>`);
  }
  if (c.issues.length) {
    blocks.push(`<div><h4>${escapeHtml(t('details.issues'))}</h4>${c.issues.map(i => `${badge(i.severity, t('severity.' + i.severity))} ${issueMsg(i)}`).join('<br>')}</div>`);
  }
  return `<div class="details-grid">${blocks.join('')}</div>`;
}

function renderDuplicates(p) {
  const parts = [`<p class="body-text">${th('quality.dup_text', { exact: p.exact_duplicate_rows, normalized: p.normalized_duplicate_rows })}</p>`];
  const groups = p.normalized_duplicate_examples.length ? p.normalized_duplicate_examples : p.duplicate_examples;
  if (groups.length) {
    parts.push(`<p class="muted" style="margin-top:6px">${escapeHtml(t('quality.dup_examples'))} ${groups.map(g => g.map(fmtInt).join(t('list.separator'))).map(escapeHtml).join(' · ')}</p>`);
  }
  if (p.key_duplicates.length) {
    parts.push(`<p class="report-section-title">${escapeHtml(t('quality.key_dups'))}</p>`);
    p.key_duplicates.forEach(k => {
      parts.push(`<p class="body-text">${th('quality.key_dup_text', { column: k.column, count: k.duplicated_values, extra: k.extra_rows,
        examples: k.examples.map(e => ({ key: 'ex.value_times', params: { value: e.value, count: e.count } })) })}</p>`);
    });
  } else {
    parts.push(`<p class="muted" style="margin-top:6px">${escapeHtml(t('quality.no_key_dups'))}</p>`);
  }
  $('dup-info').innerHTML = parts.join('');
}

function renderNameIssues(p) {
  const issues = p.column_name_issues;
  $('name-info').innerHTML = issues.length
    ? issues.map(i => `<div class="issue-row"><span class="issue-col">${i.column === null ? escapeHtml(t('common.all_columns')) : raw(i.column)}</span><span>${issueMsg(i)}</span></div>`).join('')
    : emptyNote(t('quality.names_ok'));
}

// ─── Cleaning Tab ────────────────────────────────────────────

function actionTitle(a) { return a.title_msg ? msg(a.title_msg) : a.title; }

function renderCleaningTab(data) {
  renderPlanLists(data);
  renderQualityReport(data);
  renderCleaningLog(data);
  const sel = $('changes-column');
  sel.innerHTML = `<option value="">${escapeHtml(t('clean.all_columns'))}</option>` + data.original.columns.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('');
  sel.value = App.changes.column && data.original.columns.includes(App.changes.column) ? App.changes.column : '';
  $('clean-undo').disabled = !data.can_undo;
  updateDirtyState();
}

function renderPlanLists(data) {
  const lists = { safe: $('plan-safe'), review: $('plan-review') };
  lists.safe.innerHTML = '';
  lists.review.innerHTML = '';
  const lastGroup = { safe: null, review: null };
  data.plan.forEach(a => {
    const list = lists[a.category];
    if (lastGroup[a.category] !== a.group) {
      list.appendChild(el('div', { class: 'plan-group', text: t('group.' + a.group) }));
      lastGroup[a.category] = a.group;
    }
    list.appendChild(actionItem(a));
  });
  if (!data.plan.some(a => a.category === 'safe')) lists.safe.innerHTML = emptyNote(t('clean.no_safe'));
  if (!data.plan.some(a => a.category === 'review')) lists.review.innerHTML = emptyNote(t('clean.no_review'));
}

function exampleSide(value, m) {
  if (m) return msgHtml(m);
  return value === null || value === undefined ? escapeHtml(t('common.missing_value')) : raw(quoted(value));
}

function actionItem(a) {
  const decision = App.pending[a.id] || { enabled: a.default_enabled, options: {} };
  const item = el('div', { class: 'action-item' + (decision.enabled ? ' on' : '') });
  const cbId = 'act_' + Math.random().toString(36).slice(2);
  item.innerHTML = `
    <div class="action-head">
      <input type="checkbox" id="${cbId}" ${decision.enabled ? 'checked' : ''} aria-label="${escapeHtml(t('clean.apply_action', { title: actionTitle(a) }))}">
      <label for="${cbId}" class="action-label">
        <span class="action-title">${escapeHtml(actionTitle(a))}</span>
        ${a.column !== null && a.column !== undefined ? `<span class="action-col">${raw(a.column)}</span>` : ''}
        <span class="action-meta"> · ${escapeHtml(t('unit.' + a.unit, { count: a.affected }))}</span>
      </label>
      ${badge(a.category, t('category.' + a.category))}
    </div>
    <div class="action-desc">${a.desc ? msgHtml(a.desc) : escapeHtml(a.description)}</div>`;
  if (a.examples && a.examples.length) {
    const ex = el('div', { class: 'action-examples' });
    a.examples.slice(0, 5).forEach(e => {
      ex.appendChild(el('span', { class: 'example-chip' }, `${exampleSide(e.before, e.before_msg)}<span class="arrow">${escapeHtml(t('common.arrow'))}</span>${exampleSide(e.after, e.after_msg)}${e.count ? ` ×${escapeHtml(fmtInt(e.count))}` : ''}`));
    });
    item.appendChild(ex);
  }
  if (a.options && a.options.length) {
    const opts = el('div', { class: 'action-options' });
    a.options.forEach(o => {
      const lab = el('label');
      const current = (decision.options && decision.options[o.name]) ?? o.default;
      lab.innerHTML = `${escapeHtml(o.label_msg ? msg(o.label_msg) : o.label)}: <select class="search-input">${o.choices.map(ch =>
        `<option value="${escapeHtml(ch.value)}" ${ch.value === current ? 'selected' : ''}>${escapeHtml(ch.label_msg ? msg(ch.label_msg) : ch.label)}</option>`).join('')}</select>`;
      lab.querySelector('select').addEventListener('change', ev => {
        ensurePending(a);
        App.pending[a.id].options[o.name] = ev.target.value;
        updateDirtyState();
      });
      opts.appendChild(lab);
    });
    item.appendChild(opts);
  }
  item.querySelector('input[type=checkbox]').addEventListener('change', ev => {
    ensurePending(a);
    App.pending[a.id].enabled = ev.target.checked;
    item.classList.toggle('on', ev.target.checked);
    updateDirtyState();
  });
  return item;
}

function ensurePending(a) {
  if (!App.pending[a.id]) {
    App.pending[a.id] = { enabled: a.default_enabled, options: Object.fromEntries(a.options.map(o => [o.name, o.default])) };
  }
  App.pending[a.id].options = App.pending[a.id].options || {};
}

function isDirty() {
  if (!App.data) return false;
  return JSON.stringify(sortKeys(App.pending)) !== JSON.stringify(sortKeys(App.data.decisions));
}

function sortKeys(obj) {
  if (Array.isArray(obj) || obj === null || typeof obj !== 'object') return obj;
  return Object.keys(obj).sort().reduce((acc, k) => { acc[k] = sortKeys(obj[k]); return acc; }, {});
}

function updateDirtyState() {
  const dirty = isDirty();
  $('clean-apply').disabled = !dirty || App.busy;
  const enabled = Object.values(App.pending).filter(d => d.enabled).length;
  $('clean-status').innerHTML = dirty
    ? `<strong class="warn-text">${escapeHtml(t('clean.unapplied'))}</strong> — ${escapeHtml(t('clean.selected', { count: enabled }))}`
    : escapeHtml(t('clean.status', { count: enabled }));
}

async function runCleaningRequest(path, body, labelKey) {
  if (App.busy) return;
  App.busy = true;
  const gen = App.gen;
  const tab = currentTab();
  $('clean-error').classList.add('hidden');
  showLoading(labelKey);
  try {
    const data = await postJson(path, body);
    if (gen !== App.gen) return; // a new file was uploaded meanwhile
    App.lastPreview = null;
    App.lastChanges = null;
    renderDashboard(data);
    activateTab(tab);
    return data;
  } catch (err) {
    $('clean-error').textContent = msg(errMsg(err));
    $('clean-error').classList.remove('hidden');
    throw err;
  } finally {
    App.busy = false;
    hideLoading();
    updateDirtyState();
  }
}

$('clean-apply').addEventListener('click', () => {
  if (!App.data) return;
  // Meaning-changing actions need an explicit confirmation listing what will happen
  const newlyApproved = App.data.plan.filter(a => a.category === 'review' && App.pending[a.id]?.enabled && !App.data.decisions[a.id]?.enabled);
  if (newlyApproved.length) {
    const lines = newlyApproved.map(a => `• ${actionTitle(a)}${a.column ? ` (${a.column})` : ''}`).join('\n');
    if (!confirm(t('clean.confirm_review', { count: newlyApproved.length }) + '\n\n' + lines + '\n\n' + t('clean.confirm_note'))) return;
  }
  runCleaningRequest(`/datasets/${App.data.dataset_id}/clean`, { actions: App.pending }, 'loading.applying').catch(() => {});
});

$('clean-defaults').addEventListener('click', () => {
  if (!App.data) return;
  App.pending = Object.fromEntries(App.data.plan.map(a => [a.id, {
    enabled: a.default_enabled, options: Object.fromEntries(a.options.map(o => [o.name, o.default])),
  }]));
  renderPlanLists(App.data);
  updateDirtyState();
});

$('clean-undo').addEventListener('click', () => {
  if (!App.data) return;
  runCleaningRequest(`/datasets/${App.data.dataset_id}/undo`, {}, 'loading.undoing').catch(() => {});
});

$('clean-reset').addEventListener('click', () => {
  if (!App.data) return;
  if (!confirm(t('clean.confirm_reset'))) return;
  runCleaningRequest(`/datasets/${App.data.dataset_id}/reset`, {}, 'loading.resetting').catch(() => {});
});

function compareCard(label, before, after, sub = '') {
  return `<div class="compare-card"><div class="compare-label">${escapeHtml(label)}</div>
    <div class="compare-value">${before === null ? '' : `${escapeHtml(fmtInt(before))}<span class="arrow">${escapeHtml(t('common.arrow'))}</span>`}${escapeHtml(fmtInt(after))}</div>
    ${sub ? `<div class="compare-sub">${escapeHtml(sub)}</div>` : ''}</div>`;
}

function renderQualityReport(data) {
  const r = data.cleaning.report;
  const sep = t('list.separator');
  const colSub = [r.columns.removed.length ? t('report.removed_n', { count: r.columns.removed.length }) : '',
    r.columns.added.length ? t('report.added_n', { count: r.columns.added.length }) : '',
    Object.keys(r.columns.renamed).length ? t('report.renamed_n', { count: Object.keys(r.columns.renamed).length }) : ''].filter(Boolean).join(sep);
  const html = ['<div class="compare-grid">',
    compareCard(t('rep.metric.rows'), r.rows.before, r.rows.after, t('report.removed_n', { count: r.rows.removed })),
    compareCard(t('rep.metric.columns'), r.columns.before, r.columns.after, colSub),
    compareCard(t('rep.metric.missing'), r.missing.before.total, r.missing.after.total,
      t('report.missing_detail', { nulls: r.missing.before.null, empty: r.missing.before.empty, markers: r.missing.before.markers })
      + (r.missing.imputed ? sep + t('report.imputed_n', { count: r.missing.imputed }) : '')),
    compareCard(t('rep.metric.duplicates'), r.duplicates.exact_before, r.duplicates.remaining_exact,
      t('report.dup_detail', { exact: r.duplicates.removed_exact, key: r.duplicates.removed_by_key })),
    compareCard(t('rep.metric.invalid'), r.invalid.detected, r.invalid.unresolved,
      t('report.invalid_detail', { missing: r.invalid.set_to_missing, impossible: r.invalid.impossible_set_to_missing })),
    compareCard(t('rep.metric.changed'), null, r.values.changed, t('report.converted_n', { count: r.values.type_converted })),
    compareCard(t('rep.metric.retained'), null, r.values.retained, t('report.of_cells', { count: r.values.total_before })),
    compareCard(t('rep.metric.excluded'), null, r.values.excluded, t('report.excluded_note')),
    '</div>'];

  html.push(`<div class="report-section-title">${escapeHtml(t('report.per_column'))}</div><div class="table-scroll"><table><thead><tr>`
    + ['column', 'final_name', 'type', 'missing', 'invalid', 'changed'].map(h => `<th>${escapeHtml(t('report.col.' + h))}</th>`).join('') + '</tr></thead><tbody>');
  const arrow = escapeHtml(t('common.arrow'));
  r.per_column.forEach(c => {
    html.push(`<tr><td class="stat-col-name">${raw(c.column)}</td><td>${c.removed ? badge('rejected', t('report.removed')) : raw(c.final_name)}</td>
      <td>${escapeHtml(t('kind.' + c.type_before))}${c.type_after && c.type_after !== c.type_before ? ` ${arrow} ${escapeHtml(t('kind.' + c.type_after))}` : ''}</td>
      <td>${fmtInt(c.missing_before)} ${arrow} ${c.removed ? '—' : fmtInt(c.missing_after)}</td>
      <td>${fmtInt(c.invalid_before)} ${arrow} ${c.removed ? '—' : fmtInt(c.invalid_after)}</td><td>${fmtInt(c.changed)}</td></tr>`);
  });
  html.push('</tbody></table></div>');

  html.push(`<div class="report-section-title">${escapeHtml(t('report.unresolved'))}</div>`);
  if (r.unresolved_issues.length) {
    html.push('<div class="issue-list">' + r.unresolved_issues.slice(0, 30).map(i =>
      `<div class="issue-row">${badge(i.severity, t('severity.' + i.severity))}<span class="issue-col">${i.column === null || i.column === undefined ? escapeHtml(t('common.dataset')) : raw(i.column)}</span><span>${issueMsg(i)}</span></div>`).join('') + '</div>');
  } else {
    html.push(emptyNote(t('report.none_unresolved')));
  }
  if (r.warnings.length) html.push(`<p class="muted" style="margin-top:8px">${r.warnings.map(w => msgHtml(w)).join('<br>')}</p>`);
  $('quality-report').innerHTML = html.join('');
}

function opEffect(o) {
  if (o.status !== 'applied') return o.msg ? msg(o.msg) : (o.message || '');
  const parts = [];
  if (o.values_changed) parts.push(t('opfx.changed', { count: o.values_changed }));
  if (o.values_converted) parts.push(t('opfx.converted', { count: o.values_converted }));
  if (o.values_set_missing) parts.push(t('opfx.set_missing', { count: o.values_set_missing }));
  if (o.values_imputed) parts.push(t('opfx.imputed', { count: o.values_imputed, value: o.fill_value }));
  if (o.rows_removed) parts.push(t('opfx.rows_removed', { count: o.rows_removed }));
  if (o.columns_added) parts.push(t('opfx.added', { columns: o.columns_added }));
  if (o.columns_removed) parts.push(t('opfx.removed', { columns: o.columns_removed }));
  if (o.columns_renamed) parts.push(t('opfx.renamed', { count: o.columns_renamed }));
  if (o.outliers !== undefined) parts.push(t('opfx.outliers', { count: o.outliers, low: o.bounds[0], high: o.bounds[1] }));
  return parts.join(t('list.separator')) || t('opfx.none');
}

function renderCleaningLog(data) {
  const ops = data.cleaning.operations;
  const table = el('table');
  table.innerHTML = `<thead><tr>${['status', 'operation', 'column', 'kind', 'effect'].map(h => `<th>${escapeHtml(t('log.col.' + h))}</th>`).join('')}</tr></thead>`;
  const tb = el('tbody');
  ops.forEach(o => {
    const statusClass = o.status === 'rejected' && o.category === 'review' ? 'rejected' : o.status;
    tb.appendChild(el('tr', {}, `<td>${badge(statusClass, t('status.' + o.status))}</td><td>${escapeHtml(o.title_msg ? msg(o.title_msg) : o.title)}</td>
      <td class="mono">${o.column === null || o.column === undefined ? escapeHtml(t('common.dataset')) : raw(o.column)}</td><td>${badge(o.category, t('category.' + o.category))}</td>
      <td style="white-space:normal">${escapeHtml(opEffect(o))}</td>`));
  });
  table.appendChild(tb);
  const c = $('clean-log');
  c.innerHTML = ops.length ? '' : emptyNote(t('clean.no_actions'));
  if (ops.length) c.appendChild(table);
}

async function loadChanges() {
  if (!App.data) return;
  const seq = ++App.changesSeq;
  const gen = App.gen;
  const params = new URLSearchParams({ page: App.changes.page, page_size: 25 });
  if (App.changes.column) params.set('column', App.changes.column);
  try {
    const res = await api(`/datasets/${App.data.dataset_id}/changes?${params}`);
    if (seq !== App.changesSeq || gen !== App.gen) return;
    App.lastChanges = res;
    renderChanges(res);
  } catch (err) {
    if (seq === App.changesSeq) $('changes-table').innerHTML = emptyNote(t('clean.changes_error', { reason: errMsg(err) }));
  }
}

function renderChanges(res) {
  const total = App.data.cleaning.cell_changes_total;
  $('changes-meta').textContent = t(res.truncated ? 'clean.changes_meta_truncated' : 'clean.changes_meta', { shown: res.total, count: total });
  const c = $('changes-table');
  if (!res.changes.length) {
    c.innerHTML = emptyNote(t('clean.no_changes'));
    $('changes-pager').innerHTML = '';
    return;
  }
  const titles = Object.fromEntries(App.data.plan.map(a => [a.id, actionTitle(a)]));
  const table = el('table');
  table.innerHTML = `<thead><tr>${['row', 'column', 'before', 'after', 'action'].map(h => `<th>${escapeHtml(t('changes.col.' + h))}</th>`).join('')}</tr></thead>`;
  const tb = el('tbody');
  res.changes.forEach(ch => {
    tb.appendChild(el('tr', {}, `<td>${fmtInt(ch.row)}</td><td>${raw(ch.column)}${ch.final_column && ch.final_column !== ch.column ? ` ${escapeHtml(t('common.arrow'))} ${raw(ch.final_column)}` : ''}</td>
      <td title="${escapeHtml(quoted(ch.before))}">${ch.before === null ? escapeHtml(t('common.missing_value')) : raw(quoted(ch.before))}</td>
      <td>${ch.after === null ? escapeHtml(t('common.missing_value')) : raw(quoted(ch.after))}</td>
      <td style="font-family:inherit">${escapeHtml(titles[ch.action] || ch.action)}</td>`));
  });
  table.appendChild(tb);
  c.innerHTML = '';
  c.appendChild(table);
  renderPager($('changes-pager'), res.page, res.total_pages, p => { App.changes.page = p; loadChanges(); });
}

$('changes-column').addEventListener('change', e => {
  App.changes = { page: 1, column: e.target.value };
  loadChanges();
});

function renderPager(container, page, totalPages, go) {
  container.innerHTML = '';
  if (totalPages <= 1) return;
  const prev = el('button', { class: 'btn-secondary', text: t('pager.prev') });
  const next = el('button', { class: 'btn-secondary', text: t('pager.next') });
  prev.disabled = page <= 1;
  next.disabled = page >= totalPages;
  prev.addEventListener('click', () => go(page - 1));
  next.addEventListener('click', () => go(page + 1));
  container.append(prev, el('span', { text: t('pager.page', { page, total: totalPages }) }), next);
}

// ─── Preview Tab (server-side search / filter / sort) ────────

function previewColumns() {
  if (!App.data) return [];
  return App.preview.version === 'original' ? App.data.original.columns : App.data.columns;
}

function setupPreviewColumns() {
  const cols = previewColumns();
  App.preview.filters = App.preview.filters.filter(f => cols.includes(f.column));
  if (App.preview.sort && !cols.includes(App.preview.sort)) App.preview.sort = null;
  const current = $('filter-col').value;
  $('filter-col').innerHTML = cols.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('');
  if (cols.includes(current)) $('filter-col').value = current;
  renderFilterChips();
}

function renderFilterChips() {
  const row = $('filter-chips');
  row.innerHTML = '';
  App.preview.filters.forEach((f, i) => {
    const valueless = ['is_missing', 'not_missing'].includes(f.op);
    const chip = el('span', { class: 'filter-chip' }, `<span>${raw(f.column)} ${escapeHtml(t('op_filter.' + f.op))}${valueless ? '' : ` ${raw('"' + f.value + '"')}`}</span><button aria-label="${escapeHtml(t('preview.remove_filter'))}">×</button>`);
    chip.querySelector('button').addEventListener('click', () => {
      App.preview.filters.splice(i, 1);
      App.preview.page = 1;
      renderFilterChips();
      loadPreview();
    });
    row.appendChild(chip);
  });
}

async function loadPreview() {
  if (!App.data) return;
  const seq = ++App.previewSeq;
  const gen = App.gen;
  const p = App.preview;
  const params = new URLSearchParams({ version: p.version, page: p.page, page_size: p.pageSize, order: p.order });
  if (p.search) params.set('search', p.search);
  if (p.sort) params.set('sort', p.sort);
  if (p.filters.length) params.set('filters', JSON.stringify(p.filters));
  if (p.changedOnly) params.set('changed_only', 'true');
  $('preview-error').classList.add('hidden');
  try {
    const res = await api(`/datasets/${App.data.dataset_id}/rows?${params}`);
    if (seq !== App.previewSeq || gen !== App.gen) return; // ignore stale responses
    App.lastPreview = res;
    renderPreviewTable(res);
  } catch (err) {
    if (seq !== App.previewSeq || gen !== App.gen) return;
    $('preview-error').textContent = msg(errMsg(err));
    $('preview-error').classList.remove('hidden');
  }
}

function renderPreviewTable(res) {
  const tableDiv = $('preview-table');
  const q = App.preview.colFilter.toLowerCase();
  const allCols = res.columns.map((c, i) => ({ c, i }));
  let cols = q ? allCols.filter(x => String(x.c).toLowerCase().includes(q)) : allCols;
  if (!cols.length) cols = allCols;

  $('preview-meta').textContent = t('preview.meta', { matched: res.matched_rows, total: res.total_rows, version: { key: 'preview.' + res.version } });
  if (!res.rows.length) {
    tableDiv.innerHTML = emptyNote(res.total_rows ? t('preview.no_match') : t('preview.no_data'));
    $('preview-pager').innerHTML = '';
    return;
  }
  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const trHead = document.createElement('tr');
  trHead.appendChild(el('th', { class: 'row-num', text: '#', title: t('preview.row_number') }));
  cols.forEach(({ c }) => {
    const thEl = el('th', { class: 'sortable', title: t('preview.sort_by', { column: c }), dir: 'auto' });
    thEl.textContent = c; // column name shown exactly as in the data
    if (App.preview.sort === c) thEl.appendChild(el('span', { class: 'sort-ind', text: App.preview.order === 'asc' ? '▲' : '▼' }));
    thEl.addEventListener('click', () => {
      const p = App.preview;
      if (p.sort !== c) { p.sort = c; p.order = 'asc'; }
      else if (p.order === 'asc') p.order = 'desc';
      else { p.sort = null; p.order = 'asc'; }
      p.page = 1;
      loadPreview();
    });
    trHead.appendChild(thEl);
  });
  thead.appendChild(trHead);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  res.rows.forEach(row => {
    const tr = document.createElement('tr');
    tr.appendChild(el('td', { class: 'row-num', text: String(row.row) }));
    cols.forEach(({ i }) => {
      const td = document.createElement('td');
      const v = row.values[i];
      td.textContent = v ?? '';     // cell values are never translated or reformatted
      td.title = v ?? '';
      td.dir = 'auto';
      if (row.original && Object.prototype.hasOwnProperty.call(row.original, i)) {
        td.classList.add('changed-cell');
        td.title = t('preview.tooltip', { original: quoted(row.original[i]), now: v === '' ? t('common.missing_value') : quoted(v) });
        if (v === '') { td.textContent = '∅'; td.classList.add('missing-cell'); }
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  tableDiv.innerHTML = '';
  tableDiv.appendChild(table);

  const pager = $('preview-pager');
  renderPager(pager, res.page, res.total_pages, page => { App.preview.page = page; loadPreview(); });
  const sizeSel = el('select', { class: 'search-input small', 'aria-label': t('pager.rows_per_page') });
  sizeSel.innerHTML = [25, 50, 100, 200].map(n => `<option value="${n}" ${n === App.preview.pageSize ? 'selected' : ''}>${escapeHtml(t('pager.size', { count: n }))}</option>`).join('');
  sizeSel.addEventListener('change', () => { App.preview.pageSize = Number(sizeSel.value); App.preview.page = 1; loadPreview(); });
  pager.appendChild(sizeSel);
}

$('row-search').addEventListener('input', debounce(e => {
  if (!App.data) return;
  App.preview.search = e.target.value.trim();
  App.preview.page = 1;
  loadPreview();
}, 250));

$('table-search').addEventListener('input', e => {
  if (!App.preview) return;
  App.preview.colFilter = e.target.value;
  if (App.lastPreview) renderPreviewTable(App.lastPreview);
});

$('filter-add').addEventListener('click', () => {
  if (!App.data) return;
  const f = { column: $('filter-col').value, op: $('filter-op').value, value: $('filter-val').value.trim() };
  if (!f.column) return;
  if (!['is_missing', 'not_missing'].includes(f.op) && f.value === '') {
    $('preview-error').textContent = t('preview.need_value');
    $('preview-error').classList.remove('hidden');
    return;
  }
  App.preview.filters.push(f);
  App.preview.page = 1;
  $('filter-val').value = '';
  renderFilterChips();
  loadPreview();
});

$('filter-val').addEventListener('keydown', e => { if (e.key === 'Enter') $('filter-add').click(); });

$('changed-only').addEventListener('change', e => {
  if (!App.data) return;
  App.preview.changedOnly = e.target.checked;
  App.preview.page = 1;
  loadPreview();
});

$('version-toggle').addEventListener('click', e => {
  const b = e.target.closest('.seg-btn');
  if (!b || !App.data || b.dataset.version === App.preview.version) return;
  App.preview.version = b.dataset.version;
  App.preview.page = 1;
  setSegActive('version-toggle', 'version', b.dataset.version);
  setupPreviewColumns();
  loadPreview();
});

// ─── Statistics Tab ──────────────────────────────────────────

function renderStatsTab(data) {
  const profile = data.profile;
  const container = $('stats-table-container');
  container.innerHTML = '';
  const numericCols = profile.columns.filter(c => c.numeric);

  if (numericCols.length === 0) {
    container.innerHTML = emptyNote(t('stats.no_numeric'));
  } else {
    const heads = ['column', 'role', 'count', 'missing', 'mean', 'median', 'std', 'min', 'q1', 'q3', 'max', 'skew', 'outliers'];
    const table = el('table', { class: 'stats-table' });
    table.innerHTML = `<thead><tr>${heads.map(h => `<th>${escapeHtml(t('stats.col.' + h))}</th>`).join('')}</tr></thead>`;
    const tbody = el('tbody');
    numericCols.forEach(c => {
      const n = c.numeric;
      const tr = el('tr', {}, `<td class="stat-col-name" title="${escapeHtml(c.name)}">${raw(c.name)}</td>` + [t('role.' + c.role), fmtInt(n.count), fmtInt(c.missing_total), fmt(n.mean),
        fmt(n.median), fmt(n.std), fmt(n.min), fmt(n.q1), fmt(n.q3), fmt(n.max), fmt(n.skew), `${fmtInt(n.outliers_mild)} / ${fmtInt(n.outliers_extreme)}`]
        .map(v => `<td>${escapeHtml(v)}</td>`).join(''));
      if (c.role === 'identifier') tr.style.opacity = '0.6';
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    const wrapper = el('div', { class: 'table-scroll' });
    wrapper.appendChild(table);
    container.appendChild(wrapper);
    container.insertAdjacentHTML('beforeend', `<p class="muted" style="margin-top:8px">${escapeHtml(t('stats.note'))}</p>`);
  }

  const catContainer = $('categorical-stats-container');
  catContainer.innerHTML = '';
  const cats = profile.columns.filter(c => (c.kind === 'categorical' || c.kind === 'boolean' || c.kind === 'text') && c.top_values.length);
  if (cats.length > 0) {
    const card = el('div', { class: 'card full-card' }, `<h3 class="card-title">${escapeHtml(t('stats.categorical'))}</h3>`);
    cats.forEach(c => {
      const maxVal = Math.max(...c.top_values.map(tv => tv.count));
      const dom = c.dominant ? ` · ${th('stats.dominant', { value: c.dominant.value, pct: c.dominant.pct })}` : '';
      const block = el('div', { class: 'cat-stats-block' }, `<p class="cat-block-title">${raw(c.name)}
        <span class="cat-block-sub">— ${escapeHtml(t('stats.distinct', { count: c.unique, cardinality: { key: 'cardinality.' + c.cardinality } }))}${dom}</span></p>`);
      c.top_values.forEach(tv => {
        const pctv = maxVal > 0 ? (tv.count / maxVal * 100) : 0;
        block.appendChild(el('div', { class: 'cat-bar-row' }, `
          <span class="cat-bar-label" title="${escapeHtml(tv.value)}">${raw(tv.value)}</span>
          <div class="cat-bar-bg"><div class="cat-bar-fill" style="width:${pctv}%"></div></div>
          <span class="cat-bar-count">${fmtInt(tv.count)} · ${escapeHtml(I18N.pct(tv.pct))}</span>`));
      });
      card.appendChild(block);
    });
    catContainer.appendChild(card);
  }

  const dtContainer = $('datetime-stats-container');
  dtContainer.innerHTML = '';
  const dts = profile.columns.filter(c => c.datetime);
  if (dts.length > 0) {
    const items = dts.map(c => `<li><strong>${raw(c.name)}</strong>: ${escapeHtml(t('details.dates_text', { start: { date: c.datetime.min }, end: { date: c.datetime.max }, days: c.datetime.range_days, invalid: c.datetime.invalid_count }))}
        ${c.impossible.map(r => ` · <span class="warn-text">${escapeHtml(msg(r.label))}: ${fmtInt(r.count)}</span>`).join('')}</li>`).join('');
    dtContainer.appendChild(el('div', { class: 'card full-card' }, `<h3 class="card-title">${escapeHtml(t('stats.datetime'))}</h3><ul class="dt-list">${items}</ul>`));
  }
}

// ─── Charts Tab ──────────────────────────────────────────────

function chartCard(container, titleHtml, big = true, note = '') {
  const card = el('div', { class: 'chart-card' });
  card.innerHTML = `<div class="chart-card-title">${titleHtml}</div>
    <div class="${big ? 'chart-wrap-big' : 'chart-wrap'}"><canvas></canvas></div>${note ? `<div class="chart-note">${note}</div>` : ''}`;
  container.appendChild(card);
  return card.querySelector('canvas');
}

function section(container, key) {
  container.appendChild(el('div', { class: 'chart-section-title', text: t(key) }));
}

function boxPlotSvg(name, n) {
  const b = n.box;
  const lo = n.min, hi = n.max;
  const span = (hi - lo) || 1;
  const W = 1000, pad = 6;
  const x = v => (pad + (v - lo) / span * (W - 2 * pad)).toFixed(1);
  const ns = 'vector-effect="non-scaling-stroke"';
  const outliers = b.outliers.map(o => `<line x1="${x(o)}" y1="6" x2="${x(o)}" y2="30" stroke="var(--danger)" stroke-width="2" ${ns}><title>${escapeHtml(fmt(o))}</title></line>`).join('');
  return `<div class="box-row"><div class="box-label">${raw(name)}
      <span class="muted">${escapeHtml(t('charts.box_stats', { min: fmt(lo), q1: fmt(b.q1), median: fmt(b.median), q3: fmt(b.q3), max: fmt(hi) }))}${b.outliers.length ? ' · ' + escapeHtml(t('charts.outlier_n', { count: n.outliers_mild })) : ''}</span></div>
    <svg class="box-svg" viewBox="0 0 ${W} 36" preserveAspectRatio="none" width="100%" height="36" role="img" aria-label="${escapeHtml(t('charts.box_aria', { column: name }))}">
    <line x1="${x(b.whisker_low)}" y1="18" x2="${x(b.whisker_high)}" y2="18" stroke="var(--text-muted)" ${ns} />
    <line x1="${x(b.whisker_low)}" y1="10" x2="${x(b.whisker_low)}" y2="26" stroke="var(--text-muted)" ${ns} />
    <line x1="${x(b.whisker_high)}" y1="10" x2="${x(b.whisker_high)}" y2="26" stroke="var(--text-muted)" ${ns} />
    <rect x="${x(b.q1)}" y="6" width="${Math.max(1, x(b.q3) - x(b.q1)).toFixed(1)}" height="24" fill="rgba(108,99,255,0.25)" stroke="var(--accent)" ${ns} />
    <line x1="${x(b.median)}" y1="6" x2="${x(b.median)}" y2="30" stroke="var(--accent-2)" stroke-width="3" ${ns} />
    ${outliers}</svg></div>`;
}

function renderChartsTab(data) {
  const container = $('charts-container');
  container.innerHTML = '';
  const { profile, eda } = data;
  const by = Object.fromEntries(profile.columns.map(c => [c.name, c]));
  const ax = axisColors();
  const notes = [];
  if (eda.excluded.length) notes.push(th('charts.not_charted', { items: eda.excluded.map(e => ({ key: 'lim.excluded_item', params: { column: e.column, reason: e.reason_msg } })) }));
  (eda.provisional || []).forEach(p => notes.push(th(p.excluded_values ? 'charts.provisional_excluded' : 'charts.provisional', { column: p.column, kind: { key: 'kind.' + p.kind }, count: p.excluded_values })));
  $('charts-notes').innerHTML = notes.length ? `<strong>${escapeHtml(t('charts.notes'))}</strong><ul>${notes.map(n => `<li>${n}</li>`).join('')}</ul>` : '';

  const measures = eda.columns_used.measures.filter(m => by[m] && by[m].numeric);
  if (measures.length) {
    section(container, 'charts.distributions');
    measures.forEach((m, idx) => {
      const h = by[m].numeric.histogram;
      if (!h) return;
      const labels = h.counts.map((_, i) => `${fmtShort(h.edges[i])} – ${fmtShort(h.edges[i + 1])}`);
      const color = PALETTE[idx % PALETTE.length];
      addChart(chartCard(container, th('chart.distribution_of', { column: m }), true,
        escapeHtml(t('charts.dist_note', { count: by[m].numeric.count, mean: fmt(by[m].numeric.mean), median: fmt(by[m].numeric.median) }))), {
        type: 'bar',
        data: { labels, datasets: [{ label: m, data: h.counts, backgroundColor: color + '88', borderColor: color, borderWidth: 1, borderRadius: 3, barPercentage: 1, categoryPercentage: 1 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
          scales: { x: { ticks: { color: ax.tick, font: { size: 10 }, maxTicksLimit: 8, maxRotation: 30 }, grid: { color: ax.grid } },
                    y: { beginAtZero: true, title: { display: true, text: t('charts.count_axis'), color: ax.tick }, ticks: { color: ax.tick, font: { size: 10 } }, grid: { color: ax.grid } } } },
      });
    });
    const card = el('div', { class: 'chart-card', style: 'grid-column: 1 / -1' });
    card.innerHTML = `<div class="chart-card-title">${escapeHtml(t('charts.box_title'))} <span class="muted">${escapeHtml(t('charts.box_help'))}</span></div>`
      + measures.map(m => boxPlotSvg(m, by[m].numeric)).join('');
    container.appendChild(card);
  }

  const cats = eda.columns_used.categories.filter(c => by[c] && by[c].top_values.length);
  if (cats.length) {
    section(container, 'charts.frequencies');
    cats.slice(0, 8).forEach((c, idx) => {
      const tv = by[c].top_values;
      const more = by[c].unique > tv.length ? ' ' + escapeHtml(t('common.top_of', { count: tv.length, total: by[c].unique })) : '';
      addChart(chartCard(container, th('chart.frequency_title', { column: c }) + more), {
        type: 'bar',
        data: { labels: tv.map(x => x.value), datasets: [{ data: tv.map(x => x.count), backgroundColor: PALETTE[(idx + 1) % PALETTE.length] + 'aa', borderRadius: 4 }] },
        options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => `${fmtInt(ctx.raw)} (${I18N.pct(tv[ctx.dataIndex].pct)})` } } },
          scales: { x: { beginAtZero: true, ticks: { color: ax.tick }, grid: { color: ax.grid } }, y: { ticks: { color: ax.tick, font: { size: 10 } }, grid: { display: false } } } },
      });
    });
  }

  if (eda.trends.length) {
    section(container, 'charts.trends');
    eda.trends.forEach(tr => {
      const series = [{ name: null, values: tr.counts, agg: 'count' }]
        .concat(Object.entries(tr.measures).slice(0, 3).map(([m, v]) => ({ name: m, values: v[v.agg], agg: v.agg })));
      series.forEach((s, i) => {
        const test = tr.tests.find(x => x.series === (s.agg === 'count' ? 'record count' : s.name));
        const note = (test ? escapeHtml(t('charts.trend_test', { rho: fmt(test.spearman), q: fmtQ(test.q_value) })) + ' ' : '') + msgHtml(tr.note);
        const title = s.agg === 'count' ? th('chart.trend_title_count', { freq: { key: 'freq.' + tr.freq } })
          : th('chart.trend_title_' + s.agg, { column: s.name, freq: { key: 'freq.' + tr.freq } });
        addChart(chartCard(container, `${title} <span class="muted">(${raw(tr.date_column)})</span>`, true, note), {
          type: 'line',
          data: { labels: tr.periods.map(p => I18N.period(p, tr.freq)), datasets: [{ data: s.values, borderColor: PALETTE[i % PALETTE.length], backgroundColor: PALETTE[i % PALETTE.length] + '33', fill: true, tension: 0.25, pointRadius: 2, spanGaps: false }] },
          options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
            scales: { x: { ticks: { color: ax.tick, font: { size: 10 }, maxTicksLimit: 12 }, grid: { color: ax.grid } },
                      y: { beginAtZero: s.agg !== 'mean', ticks: { color: ax.tick }, grid: { color: ax.grid } } } },
        });
      });
    });
  }

  if (eda.groups.length) {
    section(container, 'charts.groups');
    eda.groups.slice(0, 4).forEach((g, idx) => {
      const lv = g.groups.slice(0, 15);
      const note = escapeHtml(t(g.significant ? 'charts.group_note' : 'charts.group_note_ns', { q: fmtQ(g.q_value), effect: fmt(g.effect_size) }));
      addChart(chartCard(container, th('chart.breakdown_title_median', { measure: g.measure, group: g.group }), true, note), {
        type: 'bar',
        data: { labels: lv.map(x => `${x.level} (${t('charts.n_label', { count: x.count })})`), datasets: [{ data: lv.map(x => x.median), backgroundColor: PALETTE[(idx + 3) % PALETTE.length] + 'aa', borderRadius: 4 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false },
            tooltip: { callbacks: { label: ctx => t('charts.group_tooltip', { median: fmt(lv[ctx.dataIndex].median), mean: fmt(lv[ctx.dataIndex].mean), count: lv[ctx.dataIndex].count }) } } },
          scales: { x: { ticks: { color: ax.tick, font: { size: 10 }, maxRotation: 30 }, grid: { display: false } },
                    y: { beginAtZero: true, ticks: { color: ax.tick }, grid: { color: ax.grid } } } },
      });
    });
  }

  if (eda.scatter.length) {
    section(container, 'charts.relationships');
    eda.scatter.forEach((s, idx) => {
      const note = escapeHtml(t(s.sampled ? 'charts.scatter_note_sampled' : 'charts.scatter_note', { rho: fmt(s.spearman), r: fmt(s.pearson), count: s.n, sample: s.points.length }));
      addChart(chartCard(container, th('charts.scatter_title', { x: s.x, y: s.y }), true, note), {
        type: 'scatter',
        data: { datasets: [{ data: s.points.map(([x, y]) => ({ x, y })), backgroundColor: PALETTE[idx % PALETTE.length] + '99', pointRadius: 3 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
          scales: { x: { title: { display: true, text: s.x, color: ax.tick }, ticks: { color: ax.tick }, grid: { color: ax.grid } },
                    y: { position: I18N.rtl ? 'right' : 'left', title: { display: true, text: s.y, color: ax.tick }, ticks: { color: ax.tick }, grid: { color: ax.grid } } } },
      });
    });
  }

  const missing = eda.missing.filter(m => m.missing > 0);
  section(container, 'charts.missing_outliers');
  if (missing.length) {
    addChart(chartCard(container, escapeHtml(t('chart.missing_title')), true, escapeHtml(t('charts.rows_with_missing', { count: eda.rows_with_missing }))), {
      type: 'bar',
      data: { labels: missing.map(m => m.column), datasets: [{ data: missing.map(m => m.pct), backgroundColor: cssVar('--warning') + 'aa', borderRadius: 4 }] },
      options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
        scales: { x: { min: 0, max: 100, ticks: { color: ax.tick }, grid: { color: ax.grid } }, y: { ticks: { color: ax.tick, font: { size: 10 } }, grid: { display: false } } } },
    });
  } else {
    container.appendChild(el('div', { class: 'chart-card' }, emptyNote(t('charts.no_missing'))));
  }
  if (eda.outliers.length) {
    addChart(chartCard(container, escapeHtml(t('charts.outliers_title')), true, escapeHtml(t('charts.outliers_note'))), {
      type: 'bar',
      data: { labels: eda.outliers.map(o => o.column), datasets: [
        { label: t('charts.outliers_mild'), data: eda.outliers.map(o => o.mild), backgroundColor: cssVar('--warning') + 'aa', borderRadius: 3 },
        { label: t('charts.outliers_extreme'), data: eda.outliers.map(o => o.extreme), backgroundColor: cssVar('--danger') + 'aa', borderRadius: 3 },
      ] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: ax.tick } } },
        scales: { x: { ticks: { color: ax.tick }, grid: { display: false } }, y: { beginAtZero: true, ticks: { color: ax.tick, precision: 0 }, grid: { color: ax.grid } } } },
    });
  }
}

// ─── Correlation Tab ─────────────────────────────────────────

function renderCorrelationTab(data) {
  const container = $('correlation-container');
  container.innerHTML = '';
  const corr = data.eda.correlation[App.corrMethod] || {};
  const cols = Object.keys(corr);

  if (cols.length < 2) {
    container.innerHTML = `<p class="corr-empty">${escapeHtml(t('corr.need_two'))}</p>`;
  } else {
    const table = el('table', { class: 'corr-table' });
    const thead = el('thead');
    const trHead = el('tr');
    trHead.appendChild(el('th', { text: '' }));
    cols.forEach(col => trHead.appendChild(el('th', { text: col, title: col, dir: 'auto' })));
    thead.appendChild(trHead);
    table.appendChild(thead);
    const tbody = el('tbody');
    cols.forEach(rowCol => {
      const tr = el('tr');
      tr.appendChild(el('th', { text: rowCol, title: rowCol, dir: 'auto' }));
      cols.forEach(colCol => {
        const val = corr[rowCol]?.[colCol];
        const td = el('td', { text: val !== null && val !== undefined ? I18N.num(Number(val), { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '—' });
        td.style.background = corrColor(val);
        if (rowCol === colCol) { td.style.fontWeight = '700'; td.style.color = 'var(--accent)'; }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    const wrap = el('div', { class: 'corr-table-wrap' });
    wrap.appendChild(table);
    container.appendChild(wrap);
    container.insertAdjacentHTML('beforeend', `<p class="muted" style="margin-top:8px">${escapeHtml(t(App.corrMethod === 'spearman' ? 'corr.spearman_help' : 'corr.pearson_help'))}</p>`);
  }

  const rel = data.eda.relationships;
  const rc = $('relationships-container');
  if (!rel.length) {
    rc.innerHTML = emptyNote(t('corr.no_relationships'));
  } else {
    const table = el('table');
    table.innerHTML = `<thead><tr>${['a', 'b', 'types', 'method', 'value', 'strength', 'q', 'n'].map(h => `<th>${escapeHtml(t('corr.col.' + h))}</th>`).join('')}</tr></thead>`;
    const tb = el('tbody');
    rel.slice(0, 40).forEach(r => {
      const tr = el('tr', {}, `<td>${raw(r.a)}</td><td>${raw(r.b)}</td><td style="font-family:inherit">${escapeHtml(t('relkind.' + r.kind))}</td>
        <td style="font-family:inherit">${escapeHtml(t('method.' + r.method))}</td><td>${escapeHtml(fmt(r.value))}</td>
        <td>${badge(r.strength === 'strong' ? 'applied' : r.strength === 'moderate' ? 'medium' : 'low', t('strength.' + r.strength))}</td>
        <td>${escapeHtml(fmtQ(r.q_value))}</td><td>${fmtInt(r.n)}</td>`);
      if (!r.significant) tr.style.opacity = '0.6';
      if (r.caution) tr.title = msg(r.caution);
      tb.appendChild(tr);
    });
    table.appendChild(tb);
    rc.innerHTML = '';
    rc.appendChild(table);
  }
  const ex = data.eda.excluded;
  $('corr-excluded').innerHTML = (ex.length ? th('corr.excluded', { items: ex.map(e => ({ key: 'lim.excluded_item', params: { column: e.column, reason: e.reason_msg } })) }) + ' ' : '')
    + escapeHtml(t('corr.footnote'));
}

$('corr-method').addEventListener('click', e => {
  const b = e.target.closest('.seg-btn');
  if (!b || !App.data) return;
  App.corrMethod = b.dataset.method;
  setSegActive('corr-method', 'method', App.corrMethod);
  renderCorrelationTab(App.data);
});

// ─── Export Tab ──────────────────────────────────────────────

const EXPORTS = [
  { artifact: 'executive-summary', formats: ['html', 'json'], localized: true },
  { artifact: 'analysis-report', formats: ['html', 'xlsx', 'json'], localized: true },
  { artifact: 'quality-report', formats: ['html', 'xlsx', 'json'], localized: true },
  { artifact: 'cleaning-log', formats: ['json', 'xlsx', 'csv'], localized: true },
  { artifact: 'cleaned', formats: ['csv', 'xlsx', 'json'], localized: false },
  { artifact: 'recipe', formats: ['json'], localized: false },
];

function reportLang() { return App.reportLang || I18N.lang; }

function exportUrl(data, x, f) {
  const params = new URLSearchParams({ format: f });
  if (x.localized) params.set('lang', reportLang());
  if (x.artifact === 'executive-summary') {
    const d = loadActionDecisions(data);
    if (Object.keys(d).length) params.set('decisions', JSON.stringify(d));
  }
  return `${API_BASE}/datasets/${data.dataset_id}/export/${x.artifact}?${params}`;
}

function renderExportTab(data) {
  setSegActive('report-lang', 'rlang', reportLang());
  const grid = $('export-grid');
  grid.innerHTML = '';
  EXPORTS.forEach(x => {
    const card = el('div', { class: 'export-card' + (x.artifact === 'executive-summary' ? ' featured' : '') });
    card.innerHTML = `<h3 class="card-title">${escapeHtml(t('export.' + x.artifact + '.title'))}</h3><p>${escapeHtml(t('export.' + x.artifact + '.desc'))}</p>`
      + `<div class="export-btns">${x.formats.map(f => `<a class="btn-secondary" href="${exportUrl(data, x, f)}" download data-artifact="${x.artifact}" data-format="${f}">${escapeHtml(t('format.' + f))}</a>`).join('')}</div>`
      + (x.localized ? `<p class="export-lang">${escapeHtml(t('export.in_language', { language: { key: 'language.' + reportLang() } }))}</p>` : `<p class="export-lang">${escapeHtml(t('export.data_unchanged'))}</p>`);
    card.querySelectorAll('a').forEach(a => a.addEventListener('click', () => { App.exported = true; renderStepper(); }));
    grid.appendChild(card);
  });
}

$('report-lang').addEventListener('click', e => {
  const b = e.target.closest('.seg-btn');
  if (!b || !App.data) return;
  App.reportLang = b.dataset.rlang;
  renderExportTab(App.data);
});

$('recipe-import-btn').addEventListener('click', () => $('recipe-input').click());

$('recipe-input').addEventListener('change', async e => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file || !App.data) return;
  const box = $('recipe-msg');
  try {
    if (file.size > 2 * 1024 * 1024) throw new ApiError({ key: 'recipe.too_large' });
    let recipe;
    try { recipe = JSON.parse(await file.text()); } catch (_) { throw new ApiError({ key: 'recipe.not_json' }); }
    if (!recipe || recipe.format !== 'sdap-cleaning-recipe' || typeof recipe.actions !== 'object') throw new ApiError({ key: 'recipe.not_recipe' });
    const cols = recipe.source && recipe.source.columns;
    if (cols && JSON.stringify(cols) !== JSON.stringify(App.data.original.columns) && !confirm(t('recipe.confirm_columns'))) return;
    const data = await runCleaningRequest(`/datasets/${App.data.dataset_id}/clean`, { actions: recipe.actions, format: recipe.format }, 'loading.recipe');
    const warnings = data ? data.cleaning.report.warnings.map(w => msg(w)) : [];
    box.textContent = t(recipe.source ? 'recipe.applied_from' : 'recipe.applied', { filename: recipe.source ? recipe.source.filename : '' }) + (warnings.length ? ' ' + warnings.join(' ') : '');
  } catch (err) {
    box.textContent = t('recipe.failed', { reason: errMsg(err) });
  }
});
