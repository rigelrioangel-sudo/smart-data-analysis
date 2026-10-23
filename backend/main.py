import uvicorn
import contextvars
import os
import json
import math
import re
import secrets
from typing import Dict, List, Literal, Optional, Union

from urllib.parse import quote

from fastapi import Body, FastAPI, File, HTTPException, Path, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd

from core import explore, service
from core.cleaning import DecisionError, RECIPE_FORMAT
from core.i18n import LANGS, K, M, render
from core.loader import MAX_UPLOAD_BYTES, LoadError
from core.reporting import (MEDIA, ExportError, export_analysis_report, export_cleaned, export_cleaning_log,
                            export_executive_summary, export_quality_report)
from core.store import DatasetStore
from core.utils import jsonable

app = FastAPI(title="Smart Data Analysis Platform", version="3.0.0")


class ApiError(Exception):
    """An API error with a translatable message. Responses carry both the English `detail`
    (backward compatible) and `error: {key, params}` so the interface can show it in any language."""

    def __init__(self, status_code, msg):
        self.status_code, self.msg = status_code, msg


@app.exception_handler(ApiError)
async def _api_error(request: Request, exc: ApiError):
    return JSONResponse(status_code=exc.status_code, content={"detail": render(exc.msg, "en"), "error": exc.msg})


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors() and [
        {k: v for k, v in e.items() if k in ("loc", "msg", "type")} for e in exc.errors()], "error": K("err.invalid_request")})

store = DatasetStore()
DATASET_ID = Path(..., pattern=r"^[0-9a-f]{32}$", description="Dataset id returned by /upload")

# Serve frontend static files
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app.mount("/static", StaticFiles(directory=FRONTEND_DIR, html=False), name="static")

# ─── Visitor sessions: each browser only ever sees its own uploads ───
# A random token in an HttpOnly cookie identifies the browser; every dataset is stored with
# the token that uploaded it and looked up only with the caller's token (see core/store.py).
SESSION_COOKIE = "sdap_sid"
SESSION_MAX_AGE = 12 * 3600
# Hosts that show the site inside their own page (e.g. Hugging Face Spaces) set these two:
# SDAP_FRAME_ANCESTORS="https://huggingface.co" and SDAP_COOKIE_SAMESITE="none" (HTTPS only).
FRAME_ANCESTORS = os.environ.get("SDAP_FRAME_ANCESTORS", "").strip()
COOKIE_SAMESITE = os.environ.get("SDAP_COOKIE_SAMESITE", "lax").strip().lower()
if COOKIE_SAMESITE not in ("lax", "strict", "none"):
    COOKIE_SAMESITE = "lax"
_SID_RE = re.compile(r"^[A-Za-z0-9_\-]{43}$")
_current_sid: contextvars.ContextVar = contextvars.ContextVar("sdap_sid", default=None)


def current_owner():
    return _current_sid.get()


@app.middleware("http")
async def _session_and_security(request: Request, call_next):
    sid = request.cookies.get(SESSION_COOKIE)
    fresh = not (sid and _SID_RE.match(sid))
    if fresh:
        sid = secrets.token_urlsafe(32)
    token = _current_sid.set(sid)
    try:
        response = await call_next(request)
    finally:
        _current_sid.reset(token)
    if fresh:
        https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https"
        samesite = COOKIE_SAMESITE if (COOKIE_SAMESITE != "none" or https) else "lax"   # "none" requires HTTPS
        response.set_cookie(SESSION_COOKIE, sid, max_age=SESSION_MAX_AGE, httponly=True, samesite=samesite,
                            secure=https or samesite == "none", path="/")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    if FRAME_ANCESTORS:
        response.headers.setdefault("Content-Security-Policy", f"frame-ancestors 'self' {FRAME_ANCESTORS}")
    else:
        response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.url.path.startswith("/datasets/") or request.url.path == "/upload":
        response.headers.setdefault("Cache-Control", "no-store")   # never cache a visitor's data in shared caches
    return response


# CORS: only the local development servers (or the origins in SDAP_CORS_ORIGINS) may call the API
# from another origin. The public site itself is same-origin and needs no CORS.
_DEV_PORTS = ("5500", "5501", "5173", "3000")
CORS_ORIGINS = [o.strip() for o in os.environ.get("SDAP_CORS_ORIGINS", "").split(",") if o.strip()] or \
    [f"http://{h}:{p}" for h in ("127.0.0.1", "localhost") for p in _DEV_PORTS]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.get("/")
async def serve_index():
    """Serve the main HTML page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(index_path)


def safe_val(v):
    """Convert numpy/pandas scalars to safe JSON-serializable types."""
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def generate_analysis(df: pd.DataFrame) -> dict:
    """Legacy summary of the cleaned dataset (kept for backward compatibility of /upload)."""
    analysis = {}
    analysis["row_count"] = int(df.shape[0])
    analysis["column_count"] = int(df.shape[1])

    # Column types
    analysis["column_types"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Missing value counts
    analysis["missing_counts"] = {col: int(v) for col, v in df.isnull().sum().items()}

    # Missing percentage per column
    analysis["missing_pct"] = {
        col: round(int(v) / df.shape[0] * 100, 2) if df.shape[0] else 0.0
        for col, v in df.isnull().sum().items()
    }

    # Duplicate rows (unhashable cells such as JSON lists are compared as text)
    try:
        analysis["duplicate_rows"] = int(df.duplicated().sum())
    except TypeError:
        analysis["duplicate_rows"] = int(df.astype(str).duplicated().sum())

    # Numeric summary
    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        desc = numeric_cols.describe()
        numeric_summary = {}
        for col in desc.columns:
            numeric_summary[col] = {stat: safe_val(val) for stat, val in desc[col].items()}
        analysis["numeric_summary"] = numeric_summary

        # Correlation matrix
        corr = numeric_cols.corr()
        analysis["correlation"] = {
            col: {c2: safe_val(v2) for c2, v2 in row.items()}
            for col, row in corr.items()
        }

        # Distribution data (histogram bins) for each numeric column
        dist_data = {}
        for col in numeric_cols.columns:
            series = numeric_cols[col].replace([float("inf"), float("-inf")], float("nan")).dropna()
            if len(series) > 0:
                counts, bin_edges = pd.cut(series, bins=min(20, len(series.unique())), retbins=True)
                hist = counts.value_counts(sort=False)
                dist_data[col] = {
                    "labels": [f"{e:.2f}" for e in bin_edges[1:]],
                    "values": [int(v) for v in hist.values],
                }
        analysis["distributions"] = dist_data
    else:
        analysis["numeric_summary"] = {}
        analysis["correlation"] = {}
        analysis["distributions"] = {}

    # Categorical summary (top 10 values)
    cat_cols = df.select_dtypes(include=["object", "string", "category"])
    cat_summary = {}
    for col in cat_cols.columns:
        try:
            vc = df[col].value_counts().head(10)
        except TypeError:
            vc = df[col].dropna().astype(str).value_counts().head(10)
        cat_summary[col] = {str(k): int(v) for k, v in vc.items()}
    analysis["categorical_summary"] = cat_summary

    # Boolean summary
    bool_cols = df.select_dtypes(include="bool")
    bool_summary = {}
    for col in bool_cols.columns:
        vc = df[col].value_counts()
        bool_summary[col] = {str(k): int(v) for k, v in vc.items()}
    analysis["boolean_summary"] = bool_summary

    # Datetime columns
    dt_cols = df.select_dtypes(include=["datetime64", "datetimetz"])
    dt_summary = {}
    for col in dt_cols.columns:
        if df[col].isnull().all():
            continue
        dt_summary[col] = {
            "min": str(df[col].min()),
            "max": str(df[col].max()),
            "range_days": int((df[col].max() - df[col].min()).days),
        }
    analysis["datetime_summary"] = dt_summary

    return analysis


def _legacy_analysis(df: pd.DataFrame) -> dict:
    try:
        return generate_analysis(df)
    except Exception:  # the legacy summary must never block the main workflow
        return {}


def _get(dataset_id: str) -> service.Dataset:
    # Another visitor's dataset is reported exactly like a missing one
    ds = store.get(dataset_id, owner=current_owner())
    if ds is None:
        raise ApiError(404, K("err.dataset_not_found"))
    return ds


def _read_limited(upload: UploadFile) -> bytes:
    """Read the upload in chunks, refusing anything over the size limit."""
    chunks, size = [], 0
    while True:
        chunk = upload.file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise ApiError(413, M("err.too_large", limit=MAX_UPLOAD_BYTES // (1024 * 1024)))
        chunks.append(chunk)
    return b"".join(chunks)


def _safe_filename(name: str) -> str:
    base = os.path.basename((name or "dataset").replace("\\", "/"))
    base = re.sub(r"[^\w.\- ]+", "_", base).strip() or "dataset"
    return base[:120]


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    """Upload a CSV/Excel/JSON file: profile it, propose a cleaning plan, apply the safe fixes and analyze."""
    filename = _safe_filename(file.filename)
    content = _read_limited(file)
    try:
        ds = service.create_dataset(store.new_id(), filename, content, _legacy_analysis)
    except LoadError as e:
        raise ApiError(e.status_code, e.msg)
    except Exception as e:
        raise ApiError(422, M("err.analyze_failed", detail=str(e)))
    store.put(ds, owner=current_owner())
    return JSONResponse(content=service.state_payload(ds))


@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str = DATASET_ID):
    return JSONResponse(content=service.state_payload(_get(dataset_id)))


class ActionDecision(BaseModel):
    enabled: bool
    options: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)


class CleanRequest(BaseModel):
    actions: Dict[str, ActionDecision] = Field(default_factory=dict, max_length=5000)
    format: Optional[str] = None  # present when a saved recipe is re-applied


@app.post("/datasets/{dataset_id}/clean")
def clean_dataset(dataset_id: str = DATASET_ID, body: CleanRequest = Body(...)):
    """Apply cleaning decisions (or a saved recipe) to the original upload."""
    ds = _get(dataset_id)
    if body.format is not None and body.format != RECIPE_FORMAT:
        raise ApiError(400, K("err.not_recipe"))
    decisions = {k: v.model_dump() for k, v in body.actions.items()}
    try:
        service.apply_decisions(ds, decisions, _legacy_analysis)
    except DecisionError as e:
        raise ApiError(400, e.msg)
    return JSONResponse(content=service.state_payload(ds))


@app.post("/datasets/{dataset_id}/undo")
def undo_cleaning(dataset_id: str = DATASET_ID):
    ds = _get(dataset_id)
    if not service.undo(ds, _legacy_analysis):
        raise ApiError(409, K("err.nothing_to_undo"))
    return JSONResponse(content=service.state_payload(ds))


@app.post("/datasets/{dataset_id}/reset")
def reset_cleaning(dataset_id: str = DATASET_ID):
    """Turn every cleaning action off: the working data becomes the original upload again."""
    ds = _get(dataset_id)
    service.reset_to_original(ds, _legacy_analysis)
    return JSONResponse(content=service.state_payload(ds))


@app.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str = DATASET_ID):
    if not store.delete(dataset_id, owner=current_owner()):
        raise ApiError(404, K("err.dataset_not_found"))
    return {"deleted": dataset_id}


@app.get("/datasets/{dataset_id}/rows")
def get_rows(
    dataset_id: str = DATASET_ID,
    version: Literal["original", "cleaned"] = "cleaned",
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=service.MAX_PAGE_SIZE),
    search: str = Query("", max_length=200),
    sort: Optional[str] = Query(None, max_length=500),
    order: Literal["asc", "desc"] = "asc",
    filters: str = Query("", max_length=5000, description="JSON list of {column, op, value}"),
    changed_only: bool = False,
):
    ds = _get(dataset_id)
    try:
        parsed = json.loads(filters) if filters else []
        if not isinstance(parsed, list) or not all(isinstance(f, dict) for f in parsed):
            raise ValueError
    except ValueError:
        raise ApiError(400, K("err.filters_format"))
    try:
        return JSONResponse(content=service.query_rows(ds, version, page, page_size, search, sort, order, parsed, changed_only))
    except service.QueryError as e:
        raise ApiError(400, e.msg)


@app.get("/datasets/{dataset_id}/changes")
def get_changes(
    dataset_id: str = DATASET_ID,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=service.MAX_PAGE_SIZE),
    column: Optional[str] = Query(None, max_length=500),
    action: Optional[str] = Query(None, max_length=600),
):
    return JSONResponse(content=service.query_changes(_get(dataset_id), page, page_size, column, action))


ACTION_DECISIONS = {"pending", "accepted", "deferred", "dismissed"}


def _parse_decisions(raw: str) -> dict:
    """Manager decisions on recommended actions, sent by the browser; strictly validated."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        raise ApiError(400, K("err.decisions_format"))
    if not isinstance(data, dict) or len(data) > 50:
        raise ApiError(400, K("err.decisions_format"))
    out = {}
    for k, v in data.items():
        if not str(k).isdigit() or v not in ACTION_DECISIONS:
            raise ApiError(400, K("err.decisions_format"))
        out[int(k)] = v
    return out


EXPORTS = {
    "cleaned": (export_cleaned, ("csv", "xlsx", "json"), "cleaned"),
    "cleaning-log": (export_cleaning_log, ("json", "csv", "xlsx"), "cleaning_log"),
    "quality-report": (export_quality_report, ("html", "json", "xlsx"), "quality_report"),
    "analysis-report": (export_analysis_report, ("html", "json", "xlsx"), "analysis_report"),
    "executive-summary": (export_executive_summary, ("html", "json"), "executive_summary"),
    "recipe": (None, ("json",), "cleaning_recipe"),
}


@app.get("/datasets/{dataset_id}/export/{artifact}")
def export_artifact(
    dataset_id: str = DATASET_ID,
    artifact: Literal["cleaned", "cleaning-log", "quality-report", "analysis-report", "executive-summary", "recipe"] = Path(...),
    format: str = Query("csv", pattern=r"^(csv|xlsx|json|html)$"),
    lang: str = Query("en", pattern=r"^(en|ar)$", description="Report language (never applied to data values)"),
    decisions: str = Query("", max_length=4000, description="Executive summary only: JSON {action_index: decision}"),
):
    ds = _get(dataset_id)
    fn, formats, suffix = EXPORTS[artifact]
    if format not in formats:
        raise ApiError(400, M("err.export_format", formats=list(formats)))
    try:
        with ds.lock:
            if fn is None:
                content = json.dumps(ds.recipe(), indent=1).encode("utf-8")
            elif artifact == "cleaned":
                content = fn(ds, format)  # data exports never depend on the interface language
            elif artifact == "executive-summary":
                content = fn(ds, format, lang, _parse_decisions(decisions))
            else:
                content = fn(ds, format, lang)
    except ExportError as e:
        raise ApiError(400, e.msg)
    stem = re.sub(r"[^\w\-]+", "_", os.path.splitext(ds.filename)[0])[:80] or "dataset"
    lang_suffix = f"_{lang}" if artifact not in ("cleaned", "recipe") else ""
    download = f"{stem}_{suffix}{lang_suffix}.{format}"
    # ASCII fallback plus RFC 5987 UTF-8 name, so non-Latin (e.g. Arabic) file names work
    ascii_name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", download).strip("_") or f"dataset_{suffix}.{format}"
    return Response(content=content, media_type=MEDIA[format],
                    headers={"Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(download)}",
                             "X-Content-Type-Options": "nosniff"})


# ─── Analyze workspace (user-chosen columns, groupings and filters) ──

FilterValues = List[Optional[str]]


class ExploreConfig(BaseModel):
    """Which columns to analyze and how. Empty / null fields are filled automatically
    from the selected columns; "__none__" switches a role off explicitly."""
    columns: List[str] = Field(default_factory=list, max_length=200)
    roles: Dict[str, str] = Field(default_factory=dict, max_length=500)
    primary: Optional[str] = None
    secondary: Optional[List[str]] = Field(None, max_length=50)
    breakdown: Optional[str] = None
    date: Optional[str] = None
    analyses: Optional[List[str]] = Field(None, max_length=20)
    filters: Dict[str, FilterValues] = Field(default_factory=dict, max_length=200)
    focus: Dict[str, FilterValues] = Field(default_factory=dict, max_length=50)
    date_from: Optional[str] = Field(None, max_length=40)
    date_to: Optional[str] = Field(None, max_length=40)
    date_filter_column: Optional[str] = None
    top_n: int = Field(10, ge=1, le=500)
    freq: str = Field("auto", max_length=20)
    metric: Optional[Dict[str, Optional[str]]] = Field(None, max_length=4)  # {"type": records|distinct|sum|mean|median, "column": …}
    split: Optional[str] = None                                             # second grouping of the main chart


def _explore(fn, *args):
    try:
        return fn(*args)
    except explore.ExploreError as e:
        raise ApiError(400, e.msg)


@app.get("/datasets/{dataset_id}/explore/columns")
def explore_columns(dataset_id: str = DATASET_ID):
    """Columns with detected types, suggested roles, uses, filter choices and a recommended setup."""
    ds = _get(dataset_id)
    with ds.lock:
        return JSONResponse(content=jsonable(explore.catalog(ds)))


@app.post("/datasets/{dataset_id}/explore")
def explore_run(dataset_id: str = DATASET_ID, body: ExploreConfig = Body(...)):
    """Run the analysis for the chosen columns / roles / filters on the cleaned data."""
    return JSONResponse(content=_explore(explore.run_analysis, _get(dataset_id), body.model_dump()))


@app.post("/datasets/{dataset_id}/explore/export")
def explore_export(
    dataset_id: str = DATASET_ID,
    body: ExploreConfig = Body(...),
    format: str = Query("html", pattern=r"^(html|xlsx|json)$"),
    lang: str = Query("en", pattern=r"^(en|ar)$"),
):
    ds = _get(dataset_id)
    try:
        content = _explore(explore.export_report, ds, body.model_dump(), format, lang)
    except ExportError as e:
        raise ApiError(400, e.msg)
    stem = re.sub(r"[^\w\-]+", "_", os.path.splitext(ds.filename)[0])[:80] or "dataset"
    download = f"{stem}_analysis_{lang}.{format}"
    ascii_name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", download).strip("_") or f"analysis.{format}"
    return Response(content=content, media_type=MEDIA[format],
                    headers={"Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(download)}",
                             "X-Content-Type-Options": "nosniff"})


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


# index.html references style.css / script.js relatively, so also serve the
# frontend from the root. Mounted last so API routes above take precedence.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    # Local by default; hosting platforms set PORT (and SDAP_HOST=0.0.0.0) to serve the public site
    uvicorn.run("main:app", host=os.environ.get("SDAP_HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")),
                reload=os.environ.get("SDAP_RELOAD", "1") == "1", proxy_headers=True,
                app_dir=os.path.dirname(os.path.abspath(__file__)))
