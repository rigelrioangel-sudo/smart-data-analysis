/* ============================================================
   DATASENSE PRO — in-browser analysis engine (static hosting)

   Runs the unchanged Python backend (backend/main.py + core/) inside
   the visitor's browser with Pyodide. API requests from the page are
   passed here and answered by the FastAPI app through ASGI, so uploaded
   files never leave the visitor's device.
   ============================================================ */
// Module worker: Pyodide 314 ships its loader as an ES module only
import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs';

const PYODIDE_URL = 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/';
// Bundled with Pyodide (dependencies are resolved automatically)
const PACKAGES = ['micropip', 'numpy', 'pandas', 'scipy', 'pydantic', 'fastapi', 'xlrd'];
// Pure-Python wheels from PyPI, pinned to the versions the test-suite passes with
const WHEELS = ['openpyxl==3.1.5', 'xlsxwriter==3.2.9', 'python-multipart==0.0.32'];

const status = (stage, extra = {}) => self.postMessage({ type: 'status', stage, ...extra });

async function boot() {
  status('download');
  const py = await loadPyodide({ indexURL: PYODIDE_URL });
  await py.loadPackage(PACKAGES);
  const micropip = py.pyimport('micropip');
  await micropip.install(WHEELS);
  status('code');

  // Backend sources and translations, laid out as in the repository
  const manifest = await (await fetch('py/manifest.json', { cache: 'no-cache' })).json();
  for (const rel of manifest.files) {
    const text = await (await fetch('py/' + rel, { cache: 'no-cache' })).text();
    const dest = '/app/' + rel;
    py.FS.mkdirTree(dest.slice(0, dest.lastIndexOf('/')));
    py.FS.writeFile(dest, text);
  }
  for (const lang of ['en', 'ar']) {
    py.FS.mkdirTree('/app/frontend/locales');
    py.FS.writeFile(`/app/frontend/locales/${lang}.json`, await (await fetch(`locales/${lang}.json`, { cache: 'no-cache' })).text());
  }

  await py.runPythonAsync(`
import os, sys, secrets
import anyio.to_thread

# The browser has no threads: run "threadpool" work inline (FastAPI sync endpoints, file I/O)
async def _run_sync(func, *args, abandon_on_cancel=False, cancellable=None, limiter=None):
    return func(*args)
anyio.to_thread.run_sync = _run_sync

sys.path.insert(0, "/app/backend")
os.chdir("/app/backend")
import main

_COOKIE = "sdap_sid=" + secrets.token_urlsafe(32)   # one private session for this browser tab

async def handle(method, path, query, headers, body):
    hdrs = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers if k.lower() != "cookie"]
    hdrs.append((b"cookie", _COOKIE.encode()))
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
             "scheme": "https", "path": path, "raw_path": path.encode(), "query_string": query.encode(),
             "headers": hdrs, "client": ("127.0.0.1", 0), "server": ("localhost", 443), "root_path": ""}
    payload = body.to_bytes() if hasattr(body, "to_bytes") else bytes(body or b"")
    state = {"sent": False}

    async def receive():
        if not state["sent"]:
            state["sent"] = True
            return {"type": "http.request", "body": payload, "more_body": False}
        return {"type": "http.disconnect"}

    out = {"status": 500, "headers": [], "body": bytearray()}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
            out["headers"] = [[k.decode("latin-1"), v.decode("latin-1")] for k, v in message.get("headers", [])]
        elif message["type"] == "http.response.body":
            out["body"] += message.get("body", b"")

    await main.app(scope, receive, send)
    return out["status"], out["headers"], bytes(out["body"])
`);
  status('ready');
  return py.globals.get('handle');
}

const engine = boot().catch(err => {
  status('failed', { message: String(err && err.message || err) });
  throw err;
});

self.onmessage = async e => {
  const { id, method, path, query, headers, body } = e.data;
  try {
    const handle = await engine;
    // An empty body is passed as an empty byte array (JS null would arrive as a JsNull object, not None)
    const result = await handle(method, path, query, headers, body ? new Uint8Array(body) : new Uint8Array(0));
    const [code, hdrs, data] = result.toJs();
    result.destroy();
    const bytes = data.slice().buffer;
    self.postMessage({ id, status: code, headers: hdrs, body: bytes }, [bytes]);
  } catch (err) {
    const msg = JSON.stringify({ detail: String(err && err.message || err), error: { key: 'err.engine_failed' } });
    self.postMessage({ id, status: 503, headers: [['content-type', 'application/json']], body: new TextEncoder().encode(msg).buffer });
  }
};
