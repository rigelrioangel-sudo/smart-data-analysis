/* ============================================================
   DATASENSE PRO — Analyze workspace
   Pick a mode (Per Profile / Per City / Profile × City / Custom),
   a metric and optional filters; the main chart, KPIs and the exact
   table appear on this page and refresh on every change. Every
   number is computed by the backend (POST /datasets/{id}/explore);
   this file only collects the configuration and draws the results.
   Uses the shared helpers of script.js (t, msg, el, api, PALETTE …).
   ============================================================ */

'use strict';

const Explore = (() => {
  const MISSING = '\u0000missing';        // checkbox value standing for "missing value"
  const TABLE_PREVIEW_ROWS = 100;
  const MODES = ['profile', 'city', 'cross', 'custom'];
  // Notes that must stay visible (correctness / privacy); all others are folded away
  const IMPORTANT_NOTES = new Set(['ex.note.no_rows', 'ex.main.none', 'ex.note.measure_override', 'ex.note.sensitive',
    'ex.note.date_filter_missing']);

  const S = {
    datasetId: null, catalog: null, by: {}, mode: 'custom', showSetup: false,
    selected: new Set(), roles: {}, primary: '', breakdown: '', date: '', split: '', secondary: null,
    analyses: new Set(), topN: 10, freq: 'auto', metric: { type: 'records' },
    filters: {}, focus: {}, dateFrom: '', dateTo: '', extraFilters: [],
    result: null, lastConfig: null, seq: 0, ran: false, busy: false, search: '',
    chartTypes: {}, heatBasis: {}, showData: new Set(), openFilter: null, error: null, allFindings: false,
    modeStates: {},
  };
  let charts = [];

  // ─── Helpers ───────────────────────────────────────────────

  function roleOf(name) { return S.roles[name] || (S.by[name] ? S.by[name].role : 'ignore'); }
  function selectedOrdered() { return S.catalog.columns.map(c => c.name).filter(n => S.selected.has(n)); }
  function selectedWithRole(role) { return selectedOrdered().filter(n => roleOf(n) === role); }
  function resolved() { return S.result ? S.result.config : null; }
  function metricKey(m) { return m.type === 'records' ? 'records' : `${m.type}:${m.column}`; }
  function metricLabel(m) { return t('ex.metric_name.' + m.type, { column: m.column || '' }); }

  // A label from the backend: data value (string), message ({key}), missing (null) or a numeric bin ([lo, hi])
  function labelText(v, freq) {
    if (v === null || v === undefined) return t('common.missing_value');
    if (Array.isArray(v)) return `${fmtShort(v[0])} – ${fmtShort(v[1])}`;
    if (typeof v === 'object') return msg(v);
    return freq ? I18N.period(v, freq) : String(v);
  }
  function labelHtml(v, freq) {
    if (v !== null && typeof v === 'object' && !Array.isArray(v)) return msgHtml(v);
    return v === null || v === undefined ? `<span class="missing-cell">${escapeHtml(t('common.missing_value'))}</span>` : raw(labelText(v, freq));
  }
  function truncate(s, n = 22) { s = String(s); return s.length > n ? s.slice(0, n - 1) + '…' : s; }
  function pctText(v) { return v === null || v === undefined ? '—' : I18N.pct(v); }
  function infoTip(text) { return text ? `<span class="info-tip" tabindex="0" role="note" title="${escapeHtml(text)}" aria-label="${escapeHtml(text)}">i</span>` : ''; }

  function destroy() { charts.forEach(c => { try { c.destroy(); } catch (_) {} }); charts = []; }

  // ─── Lifecycle (called from script.js) ─────────────────────

  function reset(data) {
    S.seq += 1;
    S.datasetId = data.dataset_id;
    setCatalog(data.explore_catalog);
    Object.assign(S, { filters: {}, focus: {}, dateFrom: '', dateTo: '', extraFilters: [], result: null, lastConfig: null,
      ran: false, busy: false, search: '', chartTypes: {}, heatBasis: {}, showData: new Set(), openFilter: null, error: null,
      metric: { type: 'records' }, showSetup: false, allFindings: false, modeStates: {} });
    $('ex-col-search').value = '';
    applyMode(MODES.find(m => S.catalog.presets[m]) || 'custom');
  }

  function setCatalog(cat) {
    S.catalog = cat;
    S.by = Object.fromEntries(cat.columns.map(c => [c.name, c]));
  }

  // Each mode remembers its own column setup, so Back / Forward (or switching modes)
  // returns to what the user had there; filters stay shared across modes.
  const SNAPSHOT_KEYS = ['roles', 'primary', 'breakdown', 'date', 'split', 'topN', 'freq', 'focus', 'showSetup'];

  function snapshot() {
    const snap = Object.fromEntries(SNAPSHOT_KEYS.map(k => [k, JSON.parse(JSON.stringify(S[k]))]));
    return { ...snap, selected: [...S.selected], analyses: [...S.analyses], secondary: S.secondary ? [...S.secondary] : null };
  }

  function switchMode(mode) {
    if (!S.catalog || mode === S.mode && S.modeStates[mode]) return false;
    S.modeStates[S.mode] = snapshot();
    const saved = S.modeStates[mode];
    if (saved) {
      SNAPSHOT_KEYS.forEach(k => { S[k] = JSON.parse(JSON.stringify(saved[k])); });
      Object.assign(S, { mode, selected: new Set(saved.selected), analyses: new Set(saved.analyses), secondary: saved.secondary ? [...saved.secondary] : null });
      if (S.metric.column) S.selected.add(S.metric.column);
    } else {
      applyMode(mode);
    }
    return true;
  }

  // Restore a mode from browser history (no new history entry)
  function setMode(mode) {
    if (!S.catalog || !MODES.includes(mode) || !S.catalog.presets[mode] || mode === S.mode) return;
    switchMode(mode);
    renderBar(); renderColumnList(); renderAssignments();
    run();
  }

  // Analyze edits silently turn the mode into Custom: keep the URL accurate without a new entry
  function becameCustom() {
    if (typeof Nav !== 'undefined') Nav.record(true);
  }

  // A mode is a ready-made configuration computed by the backend from the column roles
  function applyMode(mode) {
    const p = S.catalog.presets[mode] || S.catalog.presets.custom;
    S.mode = mode;
    S.selected = new Set(p.columns);
    S.roles = {};
    S.primary = p.primary || ''; S.breakdown = p.breakdown || ''; S.date = p.date || ''; S.split = p.split || '';
    S.secondary = p.secondary ? [...p.secondary] : null;
    S.analyses = new Set(p.analyses || S.catalog.analyses);
    S.topN = S.catalog.limits.top_n[2];
    S.freq = 'auto';
    S.focus = {};
    if (S.metric.column) S.selected.add(S.metric.column);   // keep the chosen metric
    S.showSetup = mode === 'custom';   // presets keep the chart at the top; Custom opens the column chooser
  }

  function sync(data) {
    if (!data.explore_catalog) return;
    if (data.dataset_id !== S.datasetId) reset(data);
    else if (data.explore_catalog !== S.catalog) {
      // The cleaned data changed: keep the setup where the columns still exist, then recompute
      setCatalog(data.explore_catalog);
      S.selected = new Set([...S.selected].filter(n => S.by[n]));
      Object.keys(S.roles).forEach(n => { if (!S.by[n] || !S.by[n].allowed_roles.includes(S.roles[n])) delete S.roles[n]; });
      ['primary', 'breakdown', 'date', 'split'].forEach(k => { if (S[k] && S[k] !== '__none__' && !S.by[S[k]]) S[k] = ''; });
      if (S.secondary) S.secondary = S.secondary.filter(n => S.by[n]);
      if (S.metric.column && !S.by[S.metric.column]) S.metric = { type: 'records' };
      Object.keys(S.filters).forEach(n => { if (!S.by[n] || !S.by[n].options) delete S.filters[n]; });
      Object.keys(S.focus).forEach(n => { if (!S.by[n]) delete S.focus[n]; });
      S.extraFilters = S.extraFilters.filter(n => S.by[n] && S.by[n].options);
      if (S.ran) { renderAll(); run(); return; }
    }
    renderAll();
  }

  function onShow() {
    if (!S.ran && !S.busy && S.selected.size) run();
  }

  function renderAll() {
    renderBar();
    renderColumnList();
    renderAssignments();
    renderFilters();
    renderResults();
    renderRunState();
  }

  // ─── Configuration sent to the server ──────────────────────

  function config() {
    const columns = selectedOrdered();
    const roles = {};
    Object.entries(S.roles).forEach(([k, v]) => { if (S.selected.has(k)) roles[k] = v; });
    const clean = obj => Object.fromEntries(Object.entries(obj).filter(([, v]) => v && v.length));
    return {
      columns, roles, primary: S.primary || null, breakdown: S.breakdown || null, date: S.date || null,
      split: S.split && S.selected.has(S.split) ? S.split : null,
      secondary: S.secondary ? S.secondary.filter(n => S.selected.has(n)) : null,
      analyses: S.catalog.analyses.filter(a => S.analyses.has(a)),
      metric: S.metric.type === 'records' ? { type: 'records' } : { type: S.metric.type, column: S.metric.column },
      filters: clean(S.filters), focus: clean(S.focus),
      date_from: S.dateFrom || null, date_to: S.dateTo || null, top_n: S.topN, freq: S.freq,
    };
  }

  async function run() {
    if (!S.datasetId) return;
    if (!S.selected.size) { showError({ key: 'ex.ui.need_columns' }); return; }
    const seq = ++S.seq, gen = App.gen;
    const cfg = config();
    S.busy = true; renderRunState();
    try {
      const res = await postJson(`/datasets/${S.datasetId}/explore`, cfg);
      if (seq !== S.seq || gen !== App.gen) return;
      Object.assign(S, { result: res, lastConfig: cfg, ran: true, error: null });
      $('ex-error').classList.add('hidden');
      renderAssignments();
      renderFilters();
      renderResults();
    } catch (err) {
      if (seq === S.seq) showError({ key: 'ex.ui.error', params: { reason: errMsg(err) } });
    } finally {
      if (seq === S.seq) { S.busy = false; renderRunState(); }
    }
  }
  const runSoon = debounce(run, 400);

  // Every change refreshes the results on this page (no reload, no re-upload)
  function changed() {
    renderRunState();
    runSoon();
    becameCustom();
  }

  function showError(m) {
    S.error = m;
    const box = $('ex-error');
    box.textContent = msg(m);
    box.classList.remove('hidden');
  }

  function renderRunState() {
    const btn = $('ex-run');
    btn.disabled = S.busy;
    btn.textContent = t(S.busy ? 'ex.ui.running' : 'ex.ui.run');
    $('ex-dirty').textContent = S.busy ? t('ex.ui.running') : '';
    $('ex-results').classList.toggle('ex-busy', S.busy);
    if (S.error) { $('ex-error').textContent = msg(S.error); }
  }

  // ─── Top bar: modes, metric, customize ─────────────────────

  function renderBar() {
    $('ex-modes').innerHTML = MODES.map(m => {
      const ok = !!S.catalog.presets[m];
      return `<button type="button" class="seg-btn${S.mode === m ? ' active' : ''}" data-mode="${m}" ${ok ? '' : 'disabled'}`
        + ` title="${escapeHtml(t(ok ? 'ex.mode_help.' + m : 'ex.mode_unavailable'))}" aria-pressed="${S.mode === m}">${escapeHtml(t('ex.mode.' + m))}</button>`;
    }).join('');
    const sel = $('ex-metric');
    const opts = S.catalog.metrics;
    sel.innerHTML = opts.map(m => `<option value="${escapeHtml(metricKey(m))}">${escapeHtml(metricLabel(m))}${S.by[m.column] && S.by[m.column].sensitive ? ' 🔒' : ''}</option>`).join('');
    sel.value = metricKey(S.metric);
    const cz = $('ex-customize');
    cz.setAttribute('aria-expanded', String(S.showSetup));
    cz.classList.toggle('active', S.showSetup);
    $('ex-setup').classList.toggle('hidden', !S.showSetup);
  }

  // ─── Column chooser ────────────────────────────────────────

  function renderColumnList() {
    const list = $('ex-column-list');
    const q = S.search.trim().toLowerCase();
    const cols = S.catalog.columns.filter(c => !q || c.name.toLowerCase().includes(q));
    list.innerHTML = cols.length ? '' : emptyNote(t('ex.ui.no_match'));
    cols.forEach(c => {
      const role = roleOf(c.name);
      const on = S.selected.has(c.name);
      const row = el('div', { class: `ex-col-row${on ? ' selected' : ''}`, 'data-col': c.name });
      const id = 'ex-col-' + S.catalog.columns.indexOf(c);
      const help = msg((c.help_by_role || {})[role] || c.help);
      row.innerHTML = `<input type="checkbox" id="${id}" ${on ? 'checked' : ''} data-col="${escapeHtml(c.name)}" />`
        + `<label for="${id}" class="ex-col-name" title="${escapeHtml(help)}">${raw(c.name)}</label>`
        + (c.sensitive ? `<span class="badge medium" title="${escapeHtml(t('ex.ui.sensitive_help'))}">🔒 ${escapeHtml(t('ex.ui.sensitive'))}</span>` : '')
        + `<span class="ex-col-meta" title="${escapeHtml(help)}">${escapeHtml(t('kind.' + c.kind))} · ${escapeHtml(fmtInt(c.unique))}</span>`
        + `<select class="search-input small ex-role" data-col="${escapeHtml(c.name)}" aria-label="${escapeHtml(t('ex.ui.role_label') + ': ' + c.name)}">`
        + c.allowed_roles.map(r => `<option value="${r}" ${r === role ? 'selected' : ''}>${escapeHtml(t('ex.role.' + r))}</option>`).join('')
        + `</select>`;
      list.appendChild(row);
    });
    renderCount();
  }

  function renderCount() {
    $('ex-col-count').textContent = t('ex.ui.selected_count', { count: S.selected.size, total: S.catalog.columns.length });
  }

  function dropReferences(name) {
    ['primary', 'breakdown', 'date', 'split'].forEach(k => { if (S[k] === name) S[k] = ''; });
    if (S.secondary) S.secondary = S.secondary.filter(n => n !== name);
    if (S.metric.column === name) S.metric = { type: 'records' };
  }

  function setSelected(name, on) {
    if (on) S.selected.add(name); else { S.selected.delete(name); dropReferences(name); }
    const row = $('ex-column-list').querySelector(`.ex-col-row[data-col="${CSS.escape(name)}"]`);
    if (row) row.classList.toggle('selected', on);
  }

  // ─── Assignments (groupings, date, analyses) ───────────────

  function autoOption(field) {
    const r = resolved();
    if (!r || !S.lastConfig || S.lastConfig[field] !== null) return t('ex.ui.auto');
    const v = r[field];
    return v ? t('ex.ui.auto_resolved', { column: v }) : t('ex.ui.auto_none');
  }

  function fillSelect(select, options, value) {
    select.innerHTML = options.map(([v, label]) => `<option value="${escapeHtml(v)}" ${v === value ? 'selected' : ''}>${escapeHtml(label)}</option>`).join('');
    if (![...select.options].some(o => o.selected)) select.value = '';
  }

  function renderAssignments() {
    const dims = selectedWithRole('dimension');
    const dates = selectedWithRole('date');
    const none = ['__none__', t('ex.ui.none')];
    const r = resolved();
    const prim = S.primary || (r && r.primary);
    fillSelect($('ex-primary'), [['', autoOption('primary')], ...dims.map(n => [n, n])], S.primary);
    fillSelect($('ex-split'), [['', t('ex.ui.none')], ...dims.filter(n => n !== prim).map(n => [n, n])], S.split);
    fillSelect($('ex-breakdown'), [['', autoOption('breakdown')], none, ...dims.filter(n => n !== S.primary).map(n => [n, n])], S.breakdown);
    fillSelect($('ex-date'), [['', autoOption('date')], none, ...dates.map(n => [n, n])], S.date);
    fillSelect($('ex-freq'), S.catalog.freqs.map(f => [f, t('ex.freq.' + f)]), S.freq);
    const lim = S.catalog.limits.top_n;
    const topn = $('ex-topn');
    topn.min = lim[0]; topn.max = lim[1]; topn.value = S.topN;

    const brk = S.breakdown === '__none__' ? null : (S.breakdown || (r && r.breakdown));
    const cands = dims.filter(n => n !== prim && n !== brk);
    const box = $('ex-secondary');
    if (!cands.length) {
      box.innerHTML = `<span class="muted">${escapeHtml(t(S.selected.size ? 'ex.ui.none' : 'ex.ui.select_columns_first'))}</span>`;
    } else {
      const checked = n => S.secondary ? S.secondary.includes(n) : (r && S.lastConfig && S.lastConfig.secondary === null ? r.secondary.includes(n) : true);
      box.innerHTML = cands.map(n => `<label class="check-label ex-check"><input type="checkbox" data-sec="${escapeHtml(n)}" ${checked(n) ? 'checked' : ''} /> ${raw(n)}</label>`).join('');
    }
    $('ex-analyses').innerHTML = S.catalog.analyses.map(a =>
      `<label class="check-label ex-check" title="${escapeHtml(t('ex.analysis_help.' + a))}"><input type="checkbox" data-analysis="${a}" ${S.analyses.has(a) ? 'checked' : ''} /> ${escapeHtml(t('ex.analysis.' + a))}</label>`).join('');
    renderCount();
  }

  // ─── Filters (applied on the server to everything shown) ───

  function filterColumns() {
    const r = resolved();
    const base = r ? [r.primary, r.split, ...r.secondary, r.breakdown] : [];
    const cols = [...base, ...S.extraFilters, ...Object.keys(S.filters)].filter(n => n && S.by[n] && S.by[n].options);
    return [...new Set(cols)];
  }

  function dateFilterColumn() {
    const r = resolved();
    if (r && r.date) return r.date;
    const d = S.catalog.columns.find(c => c.role === 'date' && c.date_range);
    return d ? d.name : null;
  }

  function multiSelect(key, label, column, values, onChange) {
    const c = S.by[column];
    const sel = new Set((values || []).map(v => v === null ? MISSING : v));
    const wrap = el('details', { class: 'ex-multi', 'data-key': key });
    if (S.openFilter === key) wrap.open = true;
    const summaryValue = sel.size ? t('ex.ui.n_selected', { count: sel.size }) : t('ex.ui.all_values');
    const opts = c.options.map(o => [o.value, o.count]);
    if (c.missing) opts.push([MISSING, c.missing]);
    wrap.innerHTML = `<summary class="search-input"><span class="ex-multi-label">${label}</span> <span class="ex-multi-value${sel.size ? ' on' : ''}">${escapeHtml(summaryValue)}</span></summary>`
      + `<div class="ex-multi-pop"><input type="search" class="search-input ex-multi-search" placeholder="${escapeHtml(t('ex.ui.search_values'))}" aria-label="${escapeHtml(t('ex.ui.search_values'))}" />`
      + `<div class="ex-multi-actions"><button type="button" class="btn-link" data-act="shown">${escapeHtml(t('ex.ui.select_shown'))}</button><button type="button" class="btn-link" data-act="none">${escapeHtml(t('ex.ui.select_none'))}</button></div>`
      + `<div class="ex-multi-list" role="group" aria-label="${escapeHtml(column)}">${opts.map(([v, n]) =>
        `<label class="check-label"><input type="checkbox" value="${escapeHtml(v)}" ${sel.has(v) ? 'checked' : ''} /> ${v === MISSING ? escapeHtml(t('ex.ui.missing_option')) : raw(v)} <span class="muted">${fmtInt(n)}</span></label>`).join('')}</div>`
      + (c.options_total > c.options.length ? `<p class="muted ex-multi-note">${escapeHtml(t('ex.ui.options_limited', { shown: c.options.length, total: c.options_total }))}</p>` : '')
      + `</div>`;
    const search = wrap.querySelector('.ex-multi-search');
    const items = [...wrap.querySelectorAll('.ex-multi-list label')];
    search.addEventListener('input', () => {
      const q = search.value.trim().toLowerCase();
      items.forEach(l => { l.classList.toggle('hidden', !!q && !l.textContent.toLowerCase().includes(q)); });
    });
    const emit = () => {
      const vals = [...wrap.querySelectorAll('.ex-multi-list input:checked')].map(i => i.value === MISSING ? null : i.value);
      onChange(vals);
      wrap.querySelector('.ex-multi-value').textContent = vals.length ? t('ex.ui.n_selected', { count: vals.length }) : t('ex.ui.all_values');
      wrap.querySelector('.ex-multi-value').classList.toggle('on', vals.length > 0);
    };
    wrap.querySelector('.ex-multi-list').addEventListener('change', emit);
    wrap.querySelector('.ex-multi-actions').addEventListener('click', e => {
      const b = e.target.closest('button');
      if (!b) return;
      items.forEach(l => { const i = l.querySelector('input'); if (b.dataset.act === 'none') i.checked = false; else if (!l.classList.contains('hidden')) i.checked = true; });
      emit();
    });
    wrap.addEventListener('toggle', () => {
      if (wrap.open) { S.openFilter = key; document.querySelectorAll('.ex-multi[open]').forEach(d => { if (d !== wrap) d.open = false; }); }
      else if (S.openFilter === key) S.openFilter = null;
    });
    return wrap;
  }

  function renderFilters() {
    const card = $('ex-filters-card');
    card.classList.toggle('hidden', !S.ran);
    if (!S.ran) return;
    const row = $('ex-filters');
    row.innerHTML = `<span class="ex-filter-title">${escapeHtml(t('ex.ui.filters'))}</span>`;
    filterColumns().forEach(n => {
      row.appendChild(multiSelect('f:' + n, raw(n), n, S.filters[n], vals => { S.filters[n] = vals; runSoon(); }));
    });
    const dcol = dateFilterColumn();
    if (dcol) {
      const range = (S.by[dcol] && S.by[dcol].date_range) || [];
      const d = el('div', { class: 'ex-date-range' });
      d.innerHTML = `<span class="ex-multi-label">${raw(dcol)}</span>`
        + `<input type="date" class="search-input small" id="ex-date-from" aria-label="${escapeHtml(t('ex.ui.date_from'))}" title="${escapeHtml(t('ex.ui.date_from'))}" value="${escapeHtml(S.dateFrom)}" min="${range[0] || ''}" max="${range[1] || ''}" />`
        + `<span aria-hidden="true">–</span><input type="date" class="search-input small" id="ex-date-to" aria-label="${escapeHtml(t('ex.ui.date_to'))}" title="${escapeHtml(t('ex.ui.date_to'))}" value="${escapeHtml(S.dateTo)}" min="${range[0] || ''}" max="${range[1] || ''}" />`;
      d.querySelector('#ex-date-from').addEventListener('change', e => { S.dateFrom = e.target.value; runSoon(); });
      d.querySelector('#ex-date-to').addEventListener('change', e => { S.dateTo = e.target.value; runSoon(); });
      row.appendChild(d);
    }
    const r = resolved();
    if (r && r.primary && S.by[r.primary] && S.by[r.primary].options) {
      const lab = `${escapeHtml(t('ex.ui.compare'))}: ${raw(r.primary)}`;
      const ms = multiSelect('c:' + r.primary, lab, r.primary, S.focus[r.primary], vals => {
        S.focus = { [r.primary]: vals.filter(v => v !== null).slice(0, S.catalog.limits.max_focus) };
        if (!S.analyses.has('compare')) S.analyses.add('compare');
        runSoon();
      });
      ms.classList.add('ex-compare');
      ms.title = t('ex.ui.compare_help', { column: r.primary });
      row.appendChild(ms);
    }
    const others = S.catalog.columns.filter(c => c.options && !filterColumns().includes(c.name));
    if (others.length) {
      const add = el('select', { class: 'search-input small', 'aria-label': t('ex.ui.add_filter') });
      add.innerHTML = `<option value="">${escapeHtml(t('ex.ui.add_filter'))}</option>` + others.map(c => `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)}</option>`).join('');
      add.addEventListener('change', () => {
        if (!add.value) return;
        S.extraFilters.push(add.value);
        S.openFilter = 'f:' + add.value;
        renderFilters();
      });
      row.appendChild(add);
    }
    const rec = S.result.records;
    $('ex-filter-status').textContent = rec.is_filtered ? t('ex.ui.filter_status', { count: rec.filtered, total: rec.total }) : t('ex.ui.filter_status_all', { total: rec.total });
    $('ex-reset-filters').classList.toggle('hidden', !rec.is_filtered && !Object.keys(S.focus).length);
  }

  // ─── Results ───────────────────────────────────────────────

  function renderResults() {
    destroy();
    const box = $('ex-results');
    box.innerHTML = '';
    const r = S.result;
    if (!r) return;
    const important = r.notes.filter(n => IMPORTANT_NOTES.has(n.key));
    const other = r.notes.filter(n => !IMPORTANT_NOTES.has(n.key));
    important.forEach(n => box.appendChild(el('div', { class: 'ex-alert' }, msgHtml(n))));

    if (r.kpis.length) {
      const grid = el('div', { class: 'kpi-grid ex-kpis' });
      r.kpis.forEach(k => {
        const isMsg = k.value && typeof k.value === 'object';
        const value = isMsg ? msgHtml(k.value) : typeof k.value === 'number' ? escapeHtml(fmt(k.value)) : raw(k.value);
        grid.appendChild(el('div', { class: 'kpi-card' }, `<div class="kpi-label">${msgHtml(k.label)}</div><div class="kpi-value${isMsg || typeof k.value === 'string' ? ' small' : ''}">${value}</div><div class="kpi-sub">${msgHtml(k.sub)}</div>`));
      });
      box.appendChild(grid);
    }

    if (r.main) box.appendChild(mainBlock(r.main));

    // Index insights so highlights can jump to their section
    const insights = {};
    let domN = 0;
    r.sections.forEach(s => s.insights.forEach(i => { insights[i.id] = { i, dom: 'ex-ins-' + (domN++) }; }));

    const top = el('div', { class: 'grid-2col ex-top' });
    const hl = el('div', { class: 'card' }, `<h3 class="card-title">${escapeHtml(t('ex.ui.key_findings'))}</h3>`);
    const list = el('div', { class: 'mini-list' });
    const ids = r.highlights.filter(id => insights[id]);
    ids.slice(0, S.allFindings ? ids.length : 3).forEach(id => {
      const x = insights[id];
      const item = el('div', { class: `mini-item ${x.i.priority}` }, `<div class="mini-title">${msgHtml(x.i.title)} ${badge(x.i.priority, t('priority.' + x.i.priority))}</div><div class="mini-text">${msgHtml(x.i.finding)}</div>`);
      const b = el('button', { class: 'btn-link', type: 'button', text: t('ex.ui.see_evidence') });
      b.addEventListener('click', () => { const n = document.getElementById(x.dom); if (n) { const d = n.closest('details'); if (d) d.open = true; flash(n); } });
      item.appendChild(b);
      list.appendChild(item);
    });
    if (!list.children.length) list.innerHTML = emptyNote(t('ex.ui.no_findings'));
    hl.appendChild(list);
    if (ids.length > 3) {
      const more = el('button', { class: 'btn-link', type: 'button', text: t(S.allFindings ? 'ex.ui.fewer' : 'ex.ui.more_findings', { count: ids.length - 3 }) });
      more.addEventListener('click', () => { S.allFindings = !S.allFindings; renderResults(); });
      hl.appendChild(more);
    }
    top.appendChild(hl);

    const exp = el('div', { class: 'card ex-export' }, `<h3 class="card-title">${escapeHtml(t('ex.ui.export'))} ${infoTip(t('ex.ui.export_help'))}</h3>`
      + `<div class="export-btns">${['html', 'xlsx', 'json'].map(f => `<button type="button" class="btn-secondary small" data-fmt="${f}">${escapeHtml(t('format.' + f))}</button>`).join('')}</div>`
      + `<p class="export-lang">${escapeHtml(t('export.in_language', { language: { key: 'language.' + reportLang() } }))}</p><p class="muted ex-export-msg" role="status"></p>`
      + (other.length ? `<details class="ex-notes"><summary>${escapeHtml(t('ex.ui.notes', { count: other.length }))}</summary><ul>${other.map(n => `<li>${msgHtml(n)}</li>`).join('')}</ul></details>` : ''));
    exp.querySelectorAll('button[data-fmt]').forEach(b => b.addEventListener('click', () => exportReport(b.dataset.fmt, exp.querySelector('.ex-export-msg'))));
    top.appendChild(exp);
    box.appendChild(top);

    // Detail sections, folded; the section repeating the main chart is left out
    const m = r.main, c = r.config;
    const dup = m && m.shape === 'single' && c.metric.type === 'records' ? 'dist-' + c.primary : null;
    const sections = r.sections.filter(s => s.id !== dup);
    if (sections.length) box.appendChild(el('h3', { class: 'ex-details-title', text: t('ex.ui.details') }));
    sections.forEach((s, idx) => {
      const sec = el('details', { class: 'card full-card ex-section', id: 'ex-sec-' + idx });
      if (idx < 1) sec.open = true;
      sec.innerHTML = `<summary class="card-title">${msgHtml(s.title)} ${infoTip(msg(s.intro))}</summary>`;
      const body = el('div', { class: 'ex-section-body' });
      const grid = el('div', { class: 'charts-grid-big ex-charts' });
      body.appendChild(grid);
      const drawn = new Set();
      const draw = () => { if (!drawn.size) { s.charts.forEach(ch => renderChart(ch, grid)); drawn.add(1); } };
      s.tables.forEach(tb => body.appendChild(tableBlock(tb)));
      if (s.insights.length) {
        const ins = el('div', { class: 'ex-insights' });
        s.insights.forEach(i => ins.appendChild(el('article', { class: `insight-card ${i.priority}`, id: insights[i.id].dom }, insightHtml(i))));
        body.appendChild(ins);
      }
      (s.notes || []).forEach(n => body.appendChild(el('p', { class: 'chart-note' }, msgHtml(n))));
      sec.appendChild(body);
      box.appendChild(sec);
      // Charts of folded sections are drawn when first opened (canvas needs a visible size)
      if (sec.open) draw(); else sec.addEventListener('toggle', () => { if (sec.open) draw(); });
    });
  }

  // The chart answering the current question, with its exact table beside it
  function mainBlock(m) {
    const wrap = el('section', { class: 'ex-main', 'aria-label': msg(m.chart.title) });
    const chartBox = el('div', { class: 'ex-main-chart' });
    renderChart(m.chart, chartBox, null, true);
    wrap.appendChild(chartBox);
    if (m.table) {
      const tb = el('div', { class: 'card ex-main-table' });
      tb.innerHTML = `<h3 class="card-title">${escapeHtml(t('ex.ui.table'))}</h3><div class="table-scroll">${tableHtml(m.table, TABLE_PREVIEW_ROWS)}</div>`
        + (m.table.rows.length > TABLE_PREVIEW_ROWS || m.table.truncated ? `<p class="chart-note">${escapeHtml(t('ex.ui.more_rows', { shown: Math.min(TABLE_PREVIEW_ROWS, m.table.rows.length), total: m.table.row_count }))}</p>` : '');
      wrap.appendChild(tb);
    } else {
      wrap.classList.add('single');
    }
    return wrap;
  }

  function flash(node) {
    if (!node) return;
    node.scrollIntoView({ behavior: 'smooth', block: 'center' });
    node.classList.add('flash');
    setTimeout(() => node.classList.remove('flash'), 1600);
  }

  function insightHtml(i) {
    const ev = i.evidence.map(e => `<li>${msgHtml(e)}</li>`).join('');
    const lim = i.limitations.map(x => `<li>${msgHtml(x)}</li>`).join('');
    const cols = i.columns.map(c => `<span class="col">${raw(c)}</span>`).join('');
    return `<div class="insight-head"><div class="insight-title">${msgHtml(i.title)}</div><div class="insight-badges">${badge(i.priority, t('priority.' + i.priority))}</div></div>
      <p class="insight-finding">${msgHtml(i.finding)}</p>
      <details class="more"><summary>${escapeHtml(t('card.details'))}</summary>
      <dl class="insight-body">
        <dt>${escapeHtml(t('card.why'))}</dt><dd>${msgHtml(i.why)}</dd>
        <dt>${escapeHtml(t('card.action'))}</dt><dd class="action-text">${msgHtml(i.action)}</dd>
        <dt>${escapeHtml(t('card.evidence'))}</dt><dd><ul>${ev}</ul><div class="evidence">${cols}</div></dd>
        ${lim ? `<dt>${escapeHtml(t('card.limitations'))}</dt><dd><ul>${lim}</ul></dd>` : ''}
      </dl></details>`;
  }

  // ─── Tables ────────────────────────────────────────────────

  function cellHtml(v, type, freq) {
    if (type === 'count_pct') return v ? `${escapeHtml(fmtInt(v.n))} <span class="muted">(${escapeHtml(pctText(v.pct))})</span>` : '—';
    if (v !== null && typeof v === 'object' && !Array.isArray(v)) return msgHtml(v);
    if (type === 'label') return labelHtml(v);
    if (v === null || v === undefined) return '—';
    if (type === 'period') return escapeHtml(I18N.period(v, freq));
    if (type === 'int') return escapeHtml(fmtInt(v));
    if (type === 'pct') return escapeHtml(pctText(v));
    if (type === 'num') return escapeHtml(fmt(v));
    return raw(v);
  }

  function tableHtml(tb, limit) {
    const head = tb.columns.map(c => `<th class="${['int', 'pct', 'num', 'count_pct'].includes(c.type) ? 'num' : ''}">${typeof c.label === 'object' ? msgHtml(c.label) : raw(c.label)}</th>`).join('');
    const partial = new Set(tb.partial || []);
    const td = (v, j) => `<td class="${['int', 'pct', 'num', 'count_pct'].includes(tb.columns[j].type) ? 'num' : ''}">${cellHtml(v, tb.columns[j].type, tb.freq)}</td>`;
    const rows = tb.rows.slice(0, limit).map((r, i) => `<tr${partial.has(i) ? ` class="ex-partial" title="${escapeHtml(t('ex.ui.partial_period'))}"` : ''}>${r.map(td).join('')}</tr>`).join('');
    const total = tb.total ? `<tfoot><tr>${tb.total.map(td).join('')}</tr></tfoot>` : '';
    return `<table class="ex-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody>${total}</table>`
      + (tb.note ? `<p class="chart-note">${msgHtml(tb.note)}</p>` : '');
  }

  function tableBlock(tb) {
    const d = el('details', { class: 'ex-table-block' });
    const more = tb.rows.length > TABLE_PREVIEW_ROWS || tb.truncated;
    d.innerHTML = `<summary>${escapeHtml(t('ex.ui.table'))}: ${msgHtml(tb.title)}</summary><div class="table-scroll">${tableHtml(tb, TABLE_PREVIEW_ROWS)}</div>`
      + (more ? `<p class="chart-note">${escapeHtml(t('ex.ui.more_rows', { shown: Math.min(TABLE_PREVIEW_ROWS, tb.rows.length), total: tb.row_count }))}</p>` : '');
    return d;
  }

  // ─── Charts ────────────────────────────────────────────────

  function defaultType(ch) {
    if (ch.default) return ch.default;
    if (ch.type === 'bar') {
      const long = Math.max(0, ...ch.labels.map(l => labelText(l).length));
      if (ch.labels.length > 8 || long > 14) return 'hbar';
    }
    return ch.type;
  }

  function renderChart(ch, grid, replace = null, main = false) {
    const type = S.chartTypes[ch.id] && ch.alt_types.includes(S.chartTypes[ch.id]) ? S.chartTypes[ch.id] : defaultType(ch);
    const card = el('div', { class: 'chart-card ex-chart' + (type === 'heatmap' ? ' wide' : '') + (main ? ' main' : '') });
    const rec = ch.default || ch.type;
    const types = ch.alt_types.length > 1
      ? `<div class="seg-toggle ex-type" role="group" aria-label="${escapeHtml(t('ex.ui.chart_type'))}">${ch.alt_types.map(x =>
        `<button type="button" class="seg-btn${x === type ? ' active' : ''}" data-type="${x}" aria-pressed="${x === type}" title="${escapeHtml(x === rec ? t('ex.ui.recommended_type') : t('ex.ui.type.' + x))}">${escapeHtml(t('ex.ui.type.' + x))}${x === rec ? ' ★' : ''}</button>`).join('')}</div>` : '';
    const canvasChart = !['heatmap', 'box'].includes(type);
    const tools = `<div class="ex-chart-tools">${types}`
      + (main || ['heatmap', 'scatter'].includes(type) ? '' : `<button type="button" class="btn-link ex-data-btn" aria-expanded="${S.showData.has(ch.id)}">${escapeHtml(t(S.showData.has(ch.id) ? 'ex.ui.hide_data' : 'ex.ui.show_data'))}</button>`)
      + (canvasChart ? `<button type="button" class="btn-link ex-png-btn" title="${escapeHtml(t('ex.ui.download_png_title'))}">${escapeHtml(t('ex.ui.download_png'))}</button>` : '') + '</div>';
    const Tag = main ? 'h3' : 'div';
    card.innerHTML = `<div class="ex-chart-head"><${Tag} class="chart-card-title ex-chart-title" title="${escapeHtml(msg(ch.title))}">${msgHtml(ch.title)} ${infoTip(msg(ch.explanation))}</${Tag}>${tools}</div><div class="ex-chart-body"></div>`
      + (main ? `<p class="chart-note ex-denominator">${msgHtml(ch.explanation)}</p>` : '')
      + (ch.note ? `<p class="chart-note">${msgHtml(ch.note)}</p>` : '')
      + `<div class="ex-chart-data table-scroll${S.showData.has(ch.id) ? '' : ' hidden'}"></div>`;
    if (replace) grid.replaceChild(card, replace); else grid.appendChild(card);
    const body = card.querySelector('.ex-chart-body');
    let chart = null;
    if (type === 'heatmap') {
      body.appendChild(heatmapBlock(ch));
    } else if (type === 'box') {
      const st = ch.stats;
      body.innerHTML = boxPlotSvg(labelText(ch.x_label), { min: st.min, max: st.max, box: st.box, outliers_mild: st.outliers_mild });
    } else {
      const hbar = type === 'hbar';
      const wrap = el('div', { class: main ? 'chart-wrap-main' : 'chart-wrap-big' });
      if (hbar) wrap.style.height = Math.max(main ? 320 : 280, Math.min(900, ch.labels.length * 28 + 60)) + 'px';
      wrap.appendChild(el('canvas', { role: 'img', 'aria-label': msg(ch.title) }));
      body.appendChild(wrap);
      chart = new Chart(wrap.querySelector('canvas'), withDirection(chartConfig(ch, type)));
      charts.push(chart);
    }
    const dataBox = card.querySelector('.ex-chart-data');
    if (S.showData.has(ch.id)) dataBox.innerHTML = chartDataHtml(ch, type);
    card.querySelectorAll('.ex-type .seg-btn').forEach(b => b.addEventListener('click', () => {
      S.chartTypes[ch.id] = b.dataset.type;
      const idx = charts.indexOf(chart);
      if (chart) { chart.destroy(); if (idx >= 0) charts.splice(idx, 1); }
      renderChart(ch, card.parentNode, card, main);
    }));
    const dataBtn = card.querySelector('.ex-data-btn');
    if (dataBtn) dataBtn.addEventListener('click', () => {
      const open = !S.showData.has(ch.id);
      if (open) S.showData.add(ch.id); else S.showData.delete(ch.id);
      dataBox.classList.toggle('hidden', !open);
      if (open) dataBox.innerHTML = chartDataHtml(ch, type);
      dataBtn.textContent = t(open ? 'ex.ui.hide_data' : 'ex.ui.show_data');
      dataBtn.setAttribute('aria-expanded', open);
    });
    const png = card.querySelector('.ex-png-btn');
    if (png && chart) png.addEventListener('click', () => {
      const a = document.createElement('a');
      a.href = chart.toBase64Image('image/png', 1);
      a.download = (msg(ch.title).replace(/[\\/:*?"<>|]+/g, '_').slice(0, 80) || 'chart') + '.png';
      a.click();
    });
  }

  // Paint the card background into the canvas so downloaded PNGs are readable
  const background = { id: 'exBackground', beforeDraw(chart) {
    const { ctx, width, height } = chart;
    ctx.save(); ctx.globalCompositeOperation = 'destination-over'; ctx.fillStyle = cssVar('--surface'); ctx.fillRect(0, 0, width, height); ctx.restore();
  } };

  function seriesPct(ch, s) {
    if (s.pct) return s.pct;
    return s.values.map((v, j) => {
      const tot = ch.series.reduce((a, x) => a + (x.values[j] || 0), 0);
      return tot ? Math.round(v / tot * 10000) / 100 : null;
    });
  }

  function chartConfig(ch, type) {
    const ax = axisColors();
    const labels = ch.labels.map(l => labelText(l, ch.freq));
    const color = i => PALETTE[i % PALETTE.length];
    const xTitle = ch.x_label ? labelText(ch.x_label) : '';
    const yTitle = ch.y_label ? labelText(ch.y_label) : '';
    const tick = { color: ax.tick, font: { size: 11 } };
    const catTicks = { ...tick, autoSkip: true, maxRotation: 40, callback(v) { return truncate(this.getLabelForValue(v)); } };
    const numTicks = { ...tick, callback: v => fmtShort(v) };
    const title = text => ({ display: !!text, text, color: ax.tick, font: { size: 11, weight: '600' } });
    const base = { responsive: true, maintainAspectRatio: false, interaction: { mode: 'nearest', intersect: true },
      plugins: { legend: { display: ch.series.length > 1, position: 'bottom', labels: { color: ax.tick, boxWidth: 12, padding: 12 } } } };

    if (type === 'scatter') {
      const s = ch.series[0];
      return { type: 'scatter', plugins: [background],
        data: { datasets: [{ data: s.points.map(([x, y]) => ({ x, y })), backgroundColor: color(0) + '99', pointRadius: 3 }] },
        options: { ...base, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => `${xTitle}: ${fmt(c.raw.x)} · ${yTitle}: ${fmt(c.raw.y)}` } } },
          scales: { x: { title: title(xTitle), ticks: numTicks, grid: { color: ax.grid } }, y: { position: I18N.rtl ? 'right' : 'left', title: title(yTitle), ticks: numTicks, grid: { color: ax.grid } } } } };
    }

    if (type === 'doughnut') {
      const s = ch.series[0];
      return { type: 'doughnut', plugins: [background],
        data: { labels, datasets: [{ data: s.values, backgroundColor: s.values.map((_, i) => color(i)), borderColor: cssVar('--surface'), borderWidth: 2 }] },
        options: { responsive: true, maintainAspectRatio: false, cutout: '58%',
          plugins: { legend: { display: true, position: 'right', labels: { color: ax.tick, boxWidth: 12, padding: 10,
            generateLabels: chart => Chart.overrides.doughnut.plugins.legend.labels.generateLabels(chart).map((l, i) => ({ ...l, text: `${truncate(labels[i], 26)} · ${pctText(s.pct ? s.pct[i] : null)}` })) } },
            tooltip: { callbacks: { label: c => `${labels[c.dataIndex]}: ${t('ex.ui.tooltip_count_pct', { value: fmtInt(c.raw), pct: pctText(s.pct ? s.pct[c.dataIndex] : null) })}` } } } } };
    }

    if (type === 'line') {
      const partial = new Set(ch.partial || []);
      return { type: 'line', plugins: [background],
        data: { labels, datasets: ch.series.map((s, i) => ({
          label: labelText(s.name), data: s.values, borderColor: color(i), backgroundColor: color(i) + '22',
          fill: ch.series.length === 1, tension: 0.25, borderWidth: 2,
          pointRadius: s.values.map((_, j) => partial.has(j) ? 4 : 2),
          pointBackgroundColor: s.values.map((_, j) => partial.has(j) ? 'transparent' : color(i)),
          pointBorderColor: color(i),
          segment: { borderDash: c => (partial.has(c.p1DataIndex) || partial.has(c.p0DataIndex)) ? [4, 4] : undefined },
        })) },
        options: { ...base, interaction: { mode: 'index', intersect: false },
          plugins: { ...base.plugins, tooltip: { callbacks: {
            label: c => `${c.dataset.label}: ${fmt(c.raw)}`,
            footer: items => items.length && partial.has(items[0].dataIndex) ? t('ex.ui.partial_period') : '' } } },
          scales: { x: { title: title(xTitle), ticks: { ...tick, maxTicksLimit: 14 }, grid: { color: ax.grid } },
                    y: { beginAtZero: true, title: title(yTitle), ticks: { ...numTicks, precision: 0 }, grid: { color: ax.grid } } } } };
    }

    const stacked = type === 'stacked' || type === 'stacked100';
    const usePct = type === 'stacked100' || (type === 'grouped' && ch.percent);
    const hbar = type === 'hbar';
    const single = ch.series.length === 1;
    const datasets = ch.series.map((s, i) => {
      const pct = s.pct || (stacked || usePct ? seriesPct(ch, s) : null);
      return { label: labelText(s.name), data: usePct ? pct : s.values, _counts: s.values, _pct: pct, _n: s.counts,
        backgroundColor: color(single ? (type === 'histogram' ? 1 : 0) : i) + (type === 'histogram' ? '88' : 'dd'),
        borderColor: color(single ? 1 : i), borderWidth: type === 'histogram' ? 1 : 0, borderRadius: stacked ? 0 : 4,
        maxBarThickness: 56, ...(type === 'histogram' ? { barPercentage: 1, categoryPercentage: 1 } : {}) };
    });
    const valueAxis = { beginAtZero: true, stacked, ticks: usePct ? { ...tick, callback: v => I18N.pct(v, 0) } : numTicks, grid: { color: ax.grid },
      title: title(usePct ? t('ex.axis.share') : yTitle), ...(type === 'stacked100' ? { max: 100 } : {}) };
    const catAxis = { stacked, ticks: catTicks, grid: { display: false }, title: title(xTitle) };
    return { type: 'bar', plugins: [background],
      data: { labels, datasets },
      options: { ...base, indexAxis: hbar ? 'y' : 'x',
        plugins: { ...base.plugins, tooltip: { callbacks: {
          title: items => items.length ? labels[items[0].dataIndex] : '',
          label: c => {
            const d = c.dataset, j = c.dataIndex, n = d._counts[j], p = d._pct ? d._pct[j] : null;
            const prefix = single ? '' : d.label + ': ';
            if (d._n) return prefix + `${fmt(n)} · ${t('charts.n_label', { count: d._n[j] })}`;
            if (usePct) return prefix + t('ex.ui.tooltip_pct_count', { pct: pctText(p), value: fmt(n) });
            return prefix + (p !== null && p !== undefined ? t('ex.ui.tooltip_count_pct', { value: fmt(n), pct: pctText(p) }) : fmt(n));
          } } } },
        scales: hbar ? { x: valueAxis, y: { ...catAxis, ticks: { ...catTicks, autoSkip: false } } } : { x: catAxis, y: valueAxis } } };
  }

  function chartDataHtml(ch, type) {
    if (type === 'box') {
      const st = ch.stats;
      return `<table class="ex-table"><tbody>${['count', 'mean', 'min', 'q1', 'median', 'q3', 'max', 'outliers_mild'].map(k =>
        `<tr><th>${escapeHtml(t(k === 'outliers_mild' ? 'ex.col.outliers' : 'ex.col.' + k))}</th><td class="num">${escapeHtml(fmt(st[k]))}</td></tr>`).join('')}</tbody></table>`;
    }
    const head = `<th>${raw(ch.x_label ? labelText(ch.x_label) : '')}</th>` + ch.series.map(s => `<th class="num">${raw(labelText(s.name))}</th>`).join('');
    const rows = ch.labels.map((l, j) => `<tr><td>${labelHtml(Array.isArray(l) ? labelText(l) : l, ch.freq)}</td>${ch.series.map(s => {
      const p = s.pct ? s.pct[j] : null;
      return `<td class="num">${escapeHtml(fmt(s.values[j]))}${p !== null && p !== undefined ? ` <span class="muted">(${escapeHtml(pctText(p))})</span>` : ''}</td>`;
    }).join('')}</tr>`).join('');
    return `<table class="ex-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`;
  }

  // Heatmap data: a cross section (rows / labels / matrix) or the split main chart (heat)
  function heatData(ch) {
    if (ch.heat) return { rows: ch.heat.rows, cols: ch.heat.cols, m: ch.heat.matrix, a: ch.heat.row_label, b: ch.heat.col_label };
    return { rows: ch.rows, cols: ch.labels, m: ch.matrix, a: labelText(ch.y_label), b: labelText(ch.x_label) };
  }

  function heatmapBlock(ch) {
    const h = heatData(ch);
    const percent = h.m.percent !== false;
    const basis = percent ? (S.heatBasis[ch.id] || 'counts') : 'counts';
    const wrap = el('div', { class: 'ex-heat' });
    const bases = [['counts', t('ex.ui.basis.counts')], ['pct_row', t('ex.ui.basis.pct_row', { column: h.a })],
                   ['pct_col', t('ex.ui.basis.pct_col', { column: h.b })], ['pct_total', t('ex.ui.basis.pct_total')]];
    wrap.innerHTML = (percent ? `<div class="ex-heat-tools"><div class="seg-toggle" role="group" aria-label="${escapeHtml(t('ex.ui.basis'))}">${bases.map(([k, l]) =>
      `<button type="button" class="seg-btn${k === basis ? ' active' : ''}" data-basis="${k}">${escapeHtml(l)}</button>`).join('')}</div></div>` : '')
      + `<div class="table-scroll">${heatmapTable(h, basis)}</div><p class="chart-note">${escapeHtml(t('ex.ui.basis_note.' + basis, { column: basis === 'pct_row' ? h.a : h.b }))}</p>`;
    wrap.querySelectorAll('[data-basis]').forEach(btn => btn.addEventListener('click', () => {
      S.heatBasis[ch.id] = btn.dataset.basis;
      wrap.replaceWith(heatmapBlock(ch));
    }));
    return wrap;
  }

  function heatmapTable(h, basis) {
    const m = h.m;
    const vals = basis === 'counts' ? m.counts : m[basis];
    const max = Math.max(1, ...vals.flat().filter(v => v !== null && v !== undefined));
    const show = v => v === null || v === undefined ? '—' : basis === 'counts' ? fmt(v) : pctText(v);
    const head = `<th class="ex-heat-corner">${raw(h.a)} \\ ${raw(h.b)}</th>` + h.cols.map(c => `<th>${labelHtml(c)}</th>`).join('') + `<th>${escapeHtml(t('ex.total'))}</th>`;
    const body = h.rows.map((r, i) => `<tr><th>${labelHtml(r)}</th>${vals[i].map((v, j) => {
      const alpha = v ? (0.08 + 0.72 * v / max).toFixed(2) : 0;
      const tip = m.pct_row && m.pct_row[i][j] !== null && m.pct_row[i][j] !== undefined
        ? t('ex.ui.tooltip_cell', { a: labelText(r), b: labelText(h.cols[j]), count: fmt(m.counts[i][j]), pct_col: pctText(m.pct_col[i][j]), pct_row: pctText(m.pct_row[i][j]), a_col: h.a, b_col: h.b })
        : `${labelText(r)} × ${labelText(h.cols[j])}: ${fmt(m.counts[i][j])}`;
      return `<td style="background:rgba(108,99,255,${alpha})" title="${escapeHtml(tip)}">${escapeHtml(show(v))}</td>`;
    }).join('')}<td class="ex-heat-total">${escapeHtml(fmt(m.row_totals[i]))}</td></tr>`).join('');
    const foot = `<tr><th>${escapeHtml(t('ex.total'))}</th>${m.col_totals.map(v => `<td class="ex-heat-total">${escapeHtml(fmt(v))}</td>`).join('')}<td class="ex-heat-total">${escapeHtml(fmt(m.total))}</td></tr>`;
    return `<table class="ex-table ex-heat-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody><tfoot>${foot}</tfoot></table>`;
  }

  // ─── Export (uses the configuration of the results on screen) ─

  async function exportReport(format, status) {
    if (!S.lastConfig) return;
    status.textContent = '';
    const lang = reportLang();
    try {
      let resp;
      try {
        resp = await fetch(`${API_BASE}/datasets/${S.datasetId}/explore/export?format=${format}&lang=${lang}`,
          { method: 'POST', credentials: FETCH_CREDENTIALS, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(S.lastConfig) });
      } catch (_) { throw new ApiError({ key: 'err.network', params: { url: API_BASE || location.origin } }); }
      if (!resp.ok) {
        let data = null;
        try { data = await resp.json(); } catch (_) { /* not JSON */ }
        throw new ApiError(data && data.error && data.error.key ? data.error : { key: 'err.server', params: { status: resp.status } });
      }
      const blob = await resp.blob();
      const cd = resp.headers.get('Content-Disposition') || '';
      const m = /filename\*=UTF-8''([^;]+)/i.exec(cd);
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = m ? decodeURIComponent(m[1]) : `analysis.${format}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
      App.exported = true;
      renderStepper();
    } catch (err) {
      status.textContent = t('ex.ui.export_failed', { reason: errMsg(err) });
    }
  }

  // ─── Events ────────────────────────────────────────────────

  function init() {
    $('ex-modes').addEventListener('click', e => {
      const b = e.target.closest('[data-mode]');
      if (!b || b.disabled || !S.catalog || b.dataset.mode === S.mode) return;
      switchMode(b.dataset.mode);
      renderBar(); renderColumnList(); renderAssignments();
      run();
      if (typeof Nav !== 'undefined') Nav.record();   // one history entry per mode change
    });
    $('ex-metric').addEventListener('change', e => {
      const m = S.catalog.metrics.find(x => metricKey(x) === e.target.value) || { type: 'records' };
      S.metric = { ...m };
      if (m.column && !S.selected.has(m.column)) { S.selected.add(m.column); renderColumnList(); renderAssignments(); }
      changed();
    });
    $('ex-customize').addEventListener('click', () => { S.showSetup = !S.showSetup; renderBar(); });
    $('ex-column-list').addEventListener('change', e => {
      const box = e.target.closest('input[type=checkbox][data-col]');
      if (box) {
        setSelected(box.dataset.col, box.checked);
        S.mode = 'custom'; renderBar(); renderAssignments(); changed();
        return;
      }
      const sel = e.target.closest('select.ex-role');
      if (sel) {
        const n = sel.dataset.col;
        if (sel.value === S.by[n].role) delete S.roles[n]; else S.roles[n] = sel.value;
        dropReferences(n);
        if (!S.selected.has(n) && !['ignore', 'identifier', 'text'].includes(sel.value)) S.selected.add(n);
        S.mode = 'custom';
        renderBar(); renderColumnList(); renderAssignments(); changed();
      }
    });
    $('ex-col-search').addEventListener('input', debounce(e => { S.search = e.target.value; renderColumnList(); }, 150));
    $('ex-select-all').addEventListener('click', () => {
      if (!S.catalog) return;
      S.catalog.columns.slice(0, S.catalog.limits.max_columns).forEach(c => S.selected.add(c.name));
      S.mode = 'custom'; renderBar(); renderColumnList(); renderAssignments(); changed();
    });
    $('ex-clear').addEventListener('click', () => {
      if (!S.catalog) return;
      S.selected = new Set();
      S.primary = ''; S.breakdown = ''; S.date = ''; S.split = ''; S.secondary = null; S.metric = { type: 'records' };
      S.mode = 'custom'; renderBar(); renderColumnList(); renderAssignments();
      becameCustom();
    });
    $('ex-recommended').addEventListener('click', () => {
      if (!S.catalog) return;
      applyMode('custom');
      renderBar(); renderColumnList(); renderAssignments(); changed();
    });
    const onSelect = (id, key) => $(id).addEventListener('change', e => {
      S[key] = e.target.value;
      if (key === 'primary' && S.breakdown === S.primary) S.breakdown = '';
      if (key === 'primary' && S.split === S.primary) S.split = '';
      if (key !== 'freq') S.mode = 'custom';
      renderBar(); renderAssignments(); changed();
    });
    onSelect('ex-primary', 'primary');
    onSelect('ex-split', 'split');
    onSelect('ex-breakdown', 'breakdown');
    onSelect('ex-date', 'date');
    onSelect('ex-freq', 'freq');
    $('ex-topn').addEventListener('change', e => {
      const lim = S.catalog ? S.catalog.limits.top_n : [3, 50];
      S.topN = Math.max(lim[0], Math.min(lim[1], parseInt(e.target.value, 10) || lim[2]));
      e.target.value = S.topN;
      changed();
    });
    $('ex-secondary').addEventListener('change', () => {
      S.secondary = [...$('ex-secondary').querySelectorAll('input:checked')].map(i => i.dataset.sec);
      changed();
    });
    $('ex-analyses').addEventListener('change', e => {
      const a = e.target.dataset.analysis;
      if (!a) return;
      if (e.target.checked) S.analyses.add(a); else S.analyses.delete(a);
      changed();
    });
    $('ex-run').addEventListener('click', run);
    $('ex-reset-filters').addEventListener('click', () => {
      S.filters = {}; S.focus = {}; S.dateFrom = ''; S.dateTo = ''; S.extraFilters = []; S.openFilter = null;
      run();
    });
    document.addEventListener('click', e => {
      if (!e.target.closest('.ex-multi')) document.querySelectorAll('.ex-multi[open]').forEach(d => { d.open = false; });
    });
  }
  init();

  return {
    reset, sync, onShow, run, config, setMode,
    get mode() { return S.mode; },
    redraw: () => { if (S.result) renderResults(); },
    get state() { return S; },
  };
})();
