/* ============================================================
   DATASENSE PRO — static-hosting bridge
   When the site is published without a server (window.SDAP_STATIC),
   API requests (/upload, /datasets/…, /health) are answered by the
   Python engine running in a Web Worker (py-worker.js) instead of the
   network. Everything else in the app stays unchanged.
   ============================================================ */
'use strict';

(function () {
  if (!window.SDAP_STATIC) return;

  const API = /^\/(upload$|datasets\/|health$)/;
  const worker = new Worker('py-worker.js', { type: 'module' });
  const pending = new Map();
  let seq = 0;
  let stage = 'download';

  // While the engine loads, the loading overlay explains the (one-time) wait
  function showStage() {
    const text = document.querySelector('#loading-overlay .spinner-text');
    const overlay = document.getElementById('loading-overlay');
    if (!text || !overlay || overlay.classList.contains('hidden') || typeof I18N === 'undefined') return;
    if (stage === 'download' || stage === 'code') text.textContent = I18N.t('loading.engine');
  }
  const observer = new MutationObserver(showStage);
  document.addEventListener('DOMContentLoaded', () => {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) observer.observe(overlay, { attributes: true, attributeFilter: ['class'] });
  });

  worker.onmessage = e => {
    const m = e.data;
    if (m.type === 'status') {
      stage = m.stage;
      if (m.stage === 'failed') console.error('Analysis engine failed to start:', m.message);
      showStage();
      return;
    }
    const p = pending.get(m.id);
    if (!p) return;
    pending.delete(m.id);
    p(new Response(m.status === 204 ? null : m.body, { status: m.status, headers: m.headers }));
  };

  function apiPath(url) {
    if (url.origin !== location.origin) return null;
    return API.test(url.pathname) ? url.pathname : null;
  }

  const realFetch = window.fetch.bind(window);
  window.fetch = async function (input, init) {
    const req = new Request(input, init);
    const url = new URL(req.url, location.href);
    const path = apiPath(url);
    if (!path) return realFetch(input, init);
    if (path === '/health') return new Response(JSON.stringify({ status: 'ok', engine: stage }), { headers: { 'content-type': 'application/json' } });
    const body = ['GET', 'HEAD'].includes(req.method) ? null : await req.arrayBuffer();
    return new Promise(resolve => {
      const id = ++seq;
      pending.set(id, resolve);
      worker.postMessage({ id, method: req.method, path, query: url.search.replace(/^\?/, ''),
        headers: [...req.headers], body }, body ? [body] : []);
    });
  };

  // Report downloads are links; without a server they are fetched from the engine and saved locally
  document.addEventListener('click', async e => {
    const a = e.target.closest && e.target.closest('a[href]');
    if (!a || e.defaultPrevented) return;
    const url = new URL(a.href, location.href);
    if (!apiPath(url)) return;
    e.preventDefault();
    const resp = await window.fetch(url.href);
    if (!resp.ok) return;
    const blob = await resp.blob();
    const cd = resp.headers.get('Content-Disposition') || '';
    const m = /filename\*=UTF-8''([^;]+)/i.exec(cd);
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = m ? decodeURIComponent(m[1]) : 'download';
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 2000);
  });
})();
