"""Orchestration: upload → profile → plan → clean → analyze, plus data queries."""
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from . import cleaning, explore
from .eda import analysis_frame, run_eda
from .inference import infer_frame
from .insights import generate_insights
from .summary import build_summary
from .loader import load_dataframe
from .inference import parse_number
from .i18n import K, M, render
from .profiling import profile_dataframe
from .utils import display_series, display_value, jsonable, mask_changes, mask_value, sensitive_columns

MAX_HISTORY = 50
MAX_PAGE_SIZE = 200
FILTER_OPS = {"contains", "not_contains", "equals", "not_equals", "gt", "gte", "lt", "lte", "is_missing", "not_missing"}


@dataclass
class CleanState:
    result: cleaning.CleanResult
    profile: dict
    report: dict
    eda: dict
    insights: list
    legacy_analysis: dict
    summary: dict = None
    frame: pd.DataFrame = None   # typed view of the cleaned data (used by the Analyze workspace)


@dataclass
class Dataset:
    id: str
    filename: str
    meta: dict
    raw: pd.DataFrame
    profile: dict
    plan: list
    history: list = field(default_factory=list)   # list of resolved decision sets; last is current
    state: CleanState = None
    created: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock)
    _search_cache: dict = field(default_factory=dict)

    @property
    def decisions(self):
        return self.history[-1] if self.history else cleaning.default_decisions(self.plan)

    def recipe(self):
        return cleaning.make_recipe(self.filename, self.raw, self.decisions)


def create_dataset(dataset_id, filename, content, legacy_analysis_fn=None):
    raw, meta = load_dataframe(filename, content)
    profile = profile_dataframe(raw, infer_frame(raw))
    plan = cleaning.build_plan(raw, profile)
    ds = Dataset(id=dataset_id, filename=filename, meta=meta, raw=raw, profile=profile, plan=plan)
    apply_decisions(ds, cleaning.default_decisions(plan), legacy_analysis_fn)
    return ds


def apply_decisions(ds: Dataset, decisions, legacy_analysis_fn=None, push=True):
    with ds.lock:
        result = cleaning.apply_cleaning(ds.raw, ds.plan, decisions)
        inferences = infer_frame(result.df)
        profile_after = profile_dataframe(result.df, inferences)
        report = build_quality_report(ds, result, profile_after)
        frame, provisional = analysis_frame(result.df, inferences)
        eda = run_eda(frame, profile_after, provisional)
        insights = generate_insights(ds.profile, profile_after, report, eda, ds.plan, result.decisions)
        legacy = legacy_analysis_fn(result.df) if legacy_analysis_fn else {}
        for col in sensitive_columns(profile_after) & set(legacy.get("categorical_summary") or {}):
            legacy["categorical_summary"][col] = {mask_value(k): v for k, v in legacy["categorical_summary"][col].items()}
        # The legacy summary works on dtypes; identifiers stored as numbers must not get averages or correlations
        ids = {c["name"] for c in profile_after["columns"] if c["role"] == "identifier"}
        for key in ("numeric_summary", "distributions"):
            for col in ids & set(legacy.get(key) or {}):
                del legacy[key][col]
        if legacy.get("correlation"):
            legacy["correlation"] = {a: {b: v for b, v in row.items() if b not in ids}
                                     for a, row in legacy["correlation"].items() if a not in ids}
        summary = build_summary(profile_after, report, eda, insights)
        ds.state = CleanState(result, profile_after, report, eda, insights, legacy, summary, frame)
        if push:
            ds.history.append(result.decisions)
            del ds.history[:-MAX_HISTORY]
        ds._search_cache.clear()
        return ds


def undo(ds: Dataset, legacy_analysis_fn=None):
    with ds.lock:
        if len(ds.history) <= 1:
            return False
        ds.history.pop()
        apply_decisions(ds, ds.history[-1], legacy_analysis_fn, push=False)
        return True


def reset_to_original(ds: Dataset, legacy_analysis_fn=None):
    """Disable every action: the working data equals the original upload."""
    off = {a["id"]: {"enabled": False} for a in ds.plan}
    return apply_decisions(ds, off, legacy_analysis_fn)


# ─── Quality report ──────────────────────────────────────────

def build_quality_report(ds: Dataset, result: cleaning.CleanResult, after: dict) -> dict:
    before = ds.profile
    raw = ds.raw
    alive = result.df.index
    removed_cols = [c for c, f in result.column_map.items() if f is None]
    kept_raw_cols = [c for c, f in result.column_map.items() if f is not None]
    after_cols = {c["name"]: c for c in after["columns"]}
    ops = result.operations

    def op_sum(key, types=None):
        return int(sum(o.get(key, 0) or 0 for o in ops if o["status"] == "applied" and (types is None or o["type"] in types)))

    total_cells = int(raw.shape[0] * raw.shape[1])
    excluded_cells = int((raw.shape[0] - len(alive)) * raw.shape[1] + len(alive) * len(removed_cols))
    changed = result.change_count
    converted = int(result.converted.loc[alive, kept_raw_cols].to_numpy().sum()) if kept_raw_cols and len(alive) else 0

    per_column = []
    for c in before["columns"]:
        final = result.column_map.get(c["name"])
        a = after_cols.get(final) if final else None
        per_column.append({
            "column": c["name"], "final_name": final, "removed": final is None,
            "type_before": c["kind"], "type_after": a["kind"] if a else None,
            "dtype_after": a["dtype"] if a else None,
            "missing_before": c["missing_total"], "missing_after": a["missing_total"] if a else None,
            "invalid_before": c["invalid_count"], "invalid_after": a["invalid_count"] if a else None,
            "changed": int(result.changed.loc[alive, c["name"]].sum()) if final and len(alive) else 0,
            "issues_after": len(a["issues"]) if a else 0,
        })

    applied_ids = {k for k, v in result.decisions.items() if v["enabled"]}
    pending = [{"id": a["id"], "title": a["title"], "column": a["column"], "affected": a["affected"], "group": a["group"]}
               for a in ds.plan if a["category"] == "review" and a["id"] not in applied_ids and a["affected"]]
    rejected_safe = [{"id": a["id"], "title": a["title"], "column": a["column"]}
                     for a in ds.plan if a["category"] == "safe" and a["id"] not in applied_ids]

    return {
        "rows": {"before": int(raw.shape[0]), "after": int(len(alive)), "removed": int(raw.shape[0] - len(alive))},
        "columns": {"before": int(raw.shape[1]), "after": int(result.df.shape[1]), "removed": removed_cols,
                    "added": result.added_columns,
                    "renamed": {k: v for k, v in result.column_map.items() if v is not None and v != k}},
        "missing": {
            "before": {"null": sum(c["null"] for c in before["columns"]), "empty": sum(c["empty"] for c in before["columns"]),
                       "markers": before["marker_cells"],
                       "total": sum(c["missing_total"] for c in before["columns"])},
            "after": {"total": sum(c["missing_total"] for c in after["columns"]),
                      "markers": after["marker_cells"]},
            "imputed": op_sum("values_imputed"),
        },
        "duplicates": {
            "exact_before": before["exact_duplicate_rows"],
            "after_normalization": before["normalized_duplicate_rows"],
            "removed_exact": op_sum("rows_removed", {"drop_exact_duplicates"}),
            "removed_by_key": op_sum("rows_removed", {"drop_key_duplicates"}),
            "remaining_exact": after["exact_duplicate_rows"],
            "key_duplicates_remaining": sum(k["extra_rows"] for k in after["key_duplicates"]),
        },
        "invalid": {
            "detected": before["invalid_cells"],
            "set_to_missing": op_sum("values_set_missing", {"convert_type"}),
            "impossible_set_to_missing": op_sum("values_set_missing", {"impossible_values"}),
            "reformatted": op_sum("values_changed", {"convert_type"}),
            "unresolved": after["invalid_cells"],
        },
        # retained = cells kept with their meaning intact (includes pure type conversions)
        "values": {"total_before": total_cells, "changed": changed, "type_converted": converted,
                   "excluded": excluded_cells, "retained": max(0, total_cells - changed - excluded_cells),
                   "logged_changes": len(result.cell_changes), "log_truncated": len(result.cell_changes) >= cleaning.MAX_LOGGED_CHANGES},
        "rows_excluded": len(result.excluded_rows),
        "per_column": per_column,
        "issues_before": before["issue_counts"], "issues_after": after["issue_counts"],
        "unresolved_issues": [i for i in after["issues"] if i["severity"] in ("high", "medium")],
        "pending_review": pending, "disabled_safe_fixes": rejected_safe,
        "warnings": result.warnings,
    }


# ─── API payload ─────────────────────────────────────────────

def state_payload(ds: Dataset) -> dict:
    st = ds.state
    df = st.result.df
    preview = [{c: (display_value(v) or "") for c, v in zip(df.columns, row)} for row in df.head(10).itertuples(index=False)]
    return jsonable({
        "dataset_id": ds.id, "filename": ds.filename, "file_info": ds.meta,
        "columns": list(df.columns), "total_rows": int(len(df)), "preview": preview,
        "analysis": st.legacy_analysis,
        "original": {"columns": list(ds.raw.columns), "rows": int(len(ds.raw))},
        "profile_original": ds.profile, "profile": st.profile,
        "plan": ds.plan, "decisions": st.result.decisions,
        "history_depth": len(ds.history), "can_undo": len(ds.history) > 1,
        "cleaning": {"operations": st.result.operations, "report": st.report,
                     "column_map": st.result.column_map, "excluded_rows": st.result.excluded_rows[:500],
                     "excluded_rows_total": len(st.result.excluded_rows),
                     "cell_changes_total": int(st.result.changed.to_numpy().sum())},
        "eda": st.eda, "insights": st.insights, "summary": st.summary,
        "explore_catalog": explore.catalog(ds),
    })


# ─── Row queries (search / filter / sort / paginate) ─────────

class QueryError(ValueError):
    def __init__(self, msg):
        super().__init__(render(msg, "en"))
        self.msg = msg


def _frame(ds, version):
    if version == "original":
        return ds.raw
    if version == "cleaned":
        return ds.state.result.df
    raise QueryError(K("err.invalid_version"))


def _display_frame(ds, version):
    key = version
    if key not in ds._search_cache:
        df = _frame(ds, version)
        ds._search_cache[key] = pd.DataFrame(
            {c: display_series(df[c]).fillna("") for c in df.columns}, index=df.index).astype(object)
    return ds._search_cache[key]


def _numeric(s: pd.Series):
    if ptypes.is_numeric_dtype(s) and not ptypes.is_bool_dtype(s):
        return s.astype(float)
    return s.map(lambda v: parse_number(v)[0]).astype(float)


def _filter_mask(df, disp, f):
    col, op, value = f.get("column"), f.get("op"), f.get("value")
    if col not in df.columns:
        raise QueryError(M("err.unknown_column", column=str(col)))
    if op not in FILTER_OPS:
        raise QueryError(M("err.unknown_filter_op", op=str(op)))
    s, d = df[col], disp[col]
    if op == "is_missing":
        return d.eq("")
    if op == "not_missing":
        return d.ne("")
    value = "" if value is None else str(value)
    if op in ("contains", "not_contains"):
        m = d.str.lower().str.contains(value.lower(), regex=False)
        return m if op == "contains" else ~m
    if op in ("equals", "not_equals"):
        m = d.str.lower() == value.strip().lower()
        return m if op == "equals" else ~m
    if ptypes.is_datetime64_any_dtype(s):
        try:
            target = pd.Timestamp(value)
        except (ValueError, TypeError):
            raise QueryError(M("err.not_a_date", value=value))
        left = s
    else:
        try:
            target = float(value)
        except ValueError:
            raise QueryError(M("err.not_a_number", value=value))
        left = _numeric(s)
    m = {"gt": left > target, "gte": left >= target, "lt": left < target, "lte": left <= target}[op]
    return m.fillna(False).astype(bool)


def query_rows(ds: Dataset, version="cleaned", page=1, page_size=50, search="", sort=None, order="asc",
               filters=None, changed_only=False):
    df = _frame(ds, version)
    disp = _display_frame(ds, version)
    mask = pd.Series(True, index=df.index)
    if search:
        q = search.lower()
        hit = pd.Series(False, index=df.index)
        for col in disp.columns:
            hit |= disp[col].str.lower().str.contains(q, regex=False)
        mask &= hit
    for f in (filters or [])[:10]:
        mask &= _filter_mask(df, disp, f)

    result = ds.state.result
    raw_cols_for = {final: raw for raw, final in result.column_map.items() if final is not None}
    if changed_only:
        changed_rows = result.changed.any(axis=1)
        mask &= changed_rows.reindex(df.index, fill_value=False)
    idx = df.index[mask.to_numpy()]

    if sort:
        if sort not in df.columns:
            raise QueryError(M("err.unknown_column", column=str(sort)))
        s = df.loc[idx, sort]
        if ptypes.is_numeric_dtype(s) or ptypes.is_datetime64_any_dtype(s):
            key = s
        else:
            # Raw/text columns: sort numerically when most present values are numbers
            text = disp.loc[idx, sort]
            num = _numeric(s)
            present = int((text != "").sum())
            key = num if present and num.notna().sum() >= 0.8 * present else text.str.lower().replace("", np.nan)
        idx = key.sort_values(ascending=(order != "desc"), na_position="last", kind="stable").index

    page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    total_pages = max(1, -(-len(idx) // page_size))
    page = max(1, min(int(page), total_pages))
    page_idx = idx[(page - 1) * page_size: page * page_size]

    rows = []
    changed_frame = result.changed
    for i in page_idx:
        values = [disp.at[i, c] for c in df.columns]
        row = {"row": int(i) + 1, "values": values}
        if version == "cleaned":
            changes = {}
            for j, c in enumerate(df.columns):
                rc = raw_cols_for.get(c)
                if rc is not None and i in changed_frame.index and changed_frame.at[i, rc]:
                    changes[j] = display_value(ds.raw.at[i, rc])
            row["original"] = changes
        rows.append(row)
    return jsonable({"version": version, "columns": list(df.columns), "total_rows": int(len(df)),
                     "matched_rows": int(len(idx)), "page": page, "page_size": page_size,
                     "total_pages": total_pages, "rows": rows})


def query_changes(ds: Dataset, page=1, page_size=50, column=None, action=None):
    changes = ds.state.result.cell_changes
    if column:
        changes = [c for c in changes if c["column"] == column or c.get("final_column") == column]
    if action:
        changes = [c for c in changes if c["action"] == action]
    page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    total_pages = max(1, -(-len(changes) // page_size))
    page = max(1, min(int(page), total_pages))
    return jsonable({"total": len(changes), "page": page, "page_size": page_size, "total_pages": total_pages,
                     "truncated": len(ds.state.result.cell_changes) >= cleaning.MAX_LOGGED_CHANGES,
                     "changes": mask_changes(changes[(page - 1) * page_size: page * page_size], sensitive_columns(ds.profile))})
