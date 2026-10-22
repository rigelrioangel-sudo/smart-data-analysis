/* ============================================================
   DATASENSE PRO — Centralized translation (English / Arabic)

   - One dictionary per language: locales/<lang>.json (shared with the
     backend report generator, so UI and exports always agree).
   - t(key, params)      → plain text
   - th(key, params)     → safe HTML; data values isolated in <bdi>
   - msg(m) / msgHtml(m) → render a backend message {key, params}
   - Parameter rules (same as backend/core/i18n.py):
       string  → data value, inserted unchanged (never translated)
       number  → localized number
       array   → items rendered and joined with the locale separator
       {key}   → nested message (translated)
       {date}  → localized date      {period, freq} → localized period
   - Plural templates: {"one": "...", "other": "..."} chosen by `count`
     using Intl.PluralRules (Arabic: zero/one/two/few/many/other).
   - Language choice is stored in localStorage and restored on load.
   ============================================================ */

'use strict';

const I18N = (() => {
  const LANGS = ['en', 'ar'];
  const STORAGE_KEY = 'sdap.lang';
  const dicts = {};
  const missing = new Set();
  const listeners = [];
  let lang = 'en';

  function initialLang() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (LANGS.includes(saved)) return saved;
    return (navigator.language || '').toLowerCase().startsWith('ar') ? 'ar' : 'en';
  }

  function base() {
    // Locales live next to index.html; when the page is opened from file:// they are fetched from the API server.
    return (location.protocol === 'file:' && window.API_BASE_RESOLVED) ? window.API_BASE_RESOLVED : '';
  }

  async function load(l) {
    if (dicts[l]) return dicts[l];
    const resp = await fetch(`${base()}/locales/${l}.json`, { cache: 'no-cache' });
    if (!resp.ok) throw new Error(`Locale ${l} could not be loaded (${resp.status})`);
    dicts[l] = await resp.json();
    return dicts[l];
  }

  const escapeHtml = s => String(s ?? '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));

  function locale() { return lang === 'ar' ? 'ar' : 'en-US'; }

  function lookup(key) {
    const d = dicts[lang] || {};
    if (key in d) return d[key];
    missing.add(`${lang}:${key}`);
    const en = dicts.en || {};
    return key in en ? en[key] : key;
  }

  function num(v, opts = {}) {
    if (v === null || v === undefined || (typeof v === 'number' && !Number.isFinite(v))) return '—';
    if (typeof v !== 'number') return String(v);
    let digits = opts.maximumFractionDigits;
    if (digits === undefined) {
      const a = Math.abs(v);
      digits = Number.isInteger(v) ? 0 : a >= 1000 ? 0 : a >= 100 ? 1 : a >= 1 ? 2 : 3;
    }
    return new Intl.NumberFormat(locale(), { maximumFractionDigits: digits, ...opts }).format(v);
  }

  function pct(v, digits = 1) {
    if (v === null || v === undefined) return '—';
    return new Intl.NumberFormat(locale(), { style: 'percent', maximumFractionDigits: digits }).format(v / 100);
  }

  function parseISODate(iso) {
    const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso));
    return m ? new Date(Date.UTC(+m[1], +m[2] - 1, +m[3])) : null;
  }

  function date(iso) {
    const d = parseISODate(iso);
    if (!d) return String(iso);
    return new Intl.DateTimeFormat(locale(), { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' }).format(d);
  }

  function period(label, freq) {
    label = String(label);
    let m;
    if (freq === 'month' && (m = /^(\d{4})-(\d{2})$/.exec(label))) {
      return new Intl.DateTimeFormat(locale(), { year: 'numeric', month: 'short', timeZone: 'UTC' }).format(new Date(Date.UTC(+m[1], +m[2] - 1, 1)));
    }
    if (freq === 'quarter' && (m = /^(\d{4})Q(\d)$/.exec(label))) {
      return t('period.quarter', { q: +m[2], year: lang === 'ar' ? num(+m[1], { useGrouping: false }) : m[1] });
    }
    if (freq === 'week' && label.includes('/')) return t('period.week_of', { date: { date: label.split('/')[0] } });
    if (freq === 'year' && /^\d{4}$/.test(label)) return num(+label, { useGrouping: false });
    if (/^\d{4}-\d{2}-\d{2}$/.test(label)) return date(label);
    return label;
  }

  function list(items, html) {
    return items.map(x => param(x, html)).join(lookupSep());
  }
  function lookupSep() {
    const d = dicts[lang] || {};
    return d['list.separator'] || ', ';
  }

  function param(v, html) {
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      if ('key' in v) return html ? msgHtml(v) : msg(v);
      if ('date' in v) return date(v.date);
      if ('period' in v) return period(v.period, v.freq);
    }
    if (Array.isArray(v)) return list(v, html);
    if (typeof v === 'number') return num(v);
    if (v === null || v === undefined) return '—';
    return html ? `<bdi>${escapeHtml(v)}</bdi>` : String(v);
  }

  function format(key, params = {}, html = false) {
    let tpl = lookup(key);
    if (tpl && typeof tpl === 'object' && !Array.isArray(tpl)) {
      const cat = new Intl.PluralRules(locale()).select(Number(params.count ?? 0));
      tpl = tpl[cat] ?? tpl.other ?? Object.values(tpl)[0];
    }
    tpl = String(tpl);
    if (html) tpl = escapeHtml(tpl);
    return tpl.replace(/\{(\w+)\}/g, (all, name) => (name in params ? param(params[name], html) : all));
  }

  const t = (key, params) => format(key, params, false);
  const th = (key, params) => format(key, params, true);

  function msg(m) {
    if (m === null || m === undefined) return '';
    if (Array.isArray(m)) return m.map(msg).join(' ');
    if (typeof m === 'string') return m;
    return format(m.key, m.params || {}, false);
  }
  function msgHtml(m) {
    if (m === null || m === undefined) return '';
    if (Array.isArray(m)) return m.map(msgHtml).join(' ');
    if (typeof m === 'string') return escapeHtml(m);
    return format(m.key, m.params || {}, true);
  }

  function apply(root = document) {
    root.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
    root.querySelectorAll('[data-i18n-html]').forEach(el => { el.innerHTML = th(el.dataset.i18nHtml); });
    root.querySelectorAll('[data-i18n-placeholder]').forEach(el => { el.placeholder = t(el.dataset.i18nPlaceholder); });
    root.querySelectorAll('[data-i18n-title]').forEach(el => { el.title = t(el.dataset.i18nTitle); });
    root.querySelectorAll('[data-i18n-aria]').forEach(el => { el.setAttribute('aria-label', t(el.dataset.i18nAria)); });
    document.title = t('app.title');
  }

  function applyDirection() {
    const html = document.documentElement;
    html.lang = lang;
    html.dir = lang === 'ar' ? 'rtl' : 'ltr';
    document.querySelectorAll('.lang-btn').forEach(b => {
      const on = b.dataset.lang === lang;
      b.classList.toggle('active', on);
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
  }

  async function setLang(l) {
    if (!LANGS.includes(l)) l = 'en';
    await Promise.all([load('en'), load(l)]);
    lang = l;
    localStorage.setItem(STORAGE_KEY, l);
    applyDirection();
    apply();
    listeners.forEach(fn => { try { fn(l); } catch (e) { console.error(e); } });
  }

  const ready = setLang(initialLang())
    .catch(err => { console.error(err); })
    .finally(() => document.documentElement.classList.remove('i18n-loading'));

  return {
    t, th, msg, msgHtml, num, pct, date, period, list: items => list(items, false), apply, setLang, ready,
    onChange: fn => listeners.push(fn), get lang() { return lang; }, get rtl() { return lang === 'ar'; },
    get locale() { return locale(); }, missing, escapeHtml, dicts,
  };
})();
