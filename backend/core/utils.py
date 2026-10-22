import datetime as dt
import math
import re

import numpy as np
import pandas as pd


def is_missing(v) -> bool:
    """Scalar-safe missing check (None, NaN, NaT, pd.NA)."""
    if v is None or v is pd.NA or v is pd.NaT:
        return True
    if isinstance(v, float):
        return math.isnan(v)
    if isinstance(v, (np.floating,)):
        return bool(np.isnan(v))
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return pd.isna(v)
    return False


def jsonable(obj):
    """Recursively convert numpy / pandas values into JSON-safe Python types."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date)):
        return display_value(obj)
    return str(obj)


def display_value(v):
    """Human-readable string for a single cell, or None when missing."""
    if is_missing(v):
        return None
    if isinstance(v, (pd.Timestamp, dt.datetime)):
        ts = pd.Timestamp(v)
        if ts.hour == ts.minute == ts.second == 0 and ts.microsecond == 0 and ts.nanosecond == 0:
            return ts.strftime("%Y-%m-%d")
        return ts.isoformat(sep=" ")
    if isinstance(v, dt.date):
        return v.isoformat()
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (np.integer,)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        if f.is_integer() and abs(f) < 1e15:
            return str(int(f))
        return format(f, ".10g")
    return str(v)


def display_series(s: pd.Series) -> pd.Series:
    """display_value applied per unique value (fast on large, repetitive columns); missing → NaN."""
    present = s.dropna()
    if present.empty:
        return pd.Series(np.nan, index=s.index, dtype=object)
    try:
        uniques = pd.unique(present.to_numpy(dtype=object))
        mapping = {u: display_value(u) for u in uniques}
        return s.map(mapping).astype(object)
    except TypeError:  # unhashable cells
        return s.map(lambda v: display_value(v)).astype(object)


def rnd(v, digits=4):
    """Round floats for presentation, keeping None for NaN/inf."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, digits)


MASK = "•"


def mask_value(v):
    """Privacy-safe display of a personal identifier: only the last 3 characters stay visible
    (9647701234567 → ••••567, "Client 00012" → ••••012). None stays None."""
    if v is None:
        return None
    s = display_value(v) if not isinstance(v, str) else v
    if s is None:
        return None
    s = s.strip()
    return MASK * 4 + (s[-3:] if len(s) > 6 else "")


def sensitive_columns(profile) -> set:
    return {c["name"] for c in profile["columns"] if c.get("sensitive")}


def mask_changes(changes, sensitive: set):
    """Cleaning-log entries with before/after values of personal identifiers masked."""
    if not sensitive:
        return changes
    return [dict(c, before=mask_value(c.get("before")), after=mask_value(c.get("after")))
            if c.get("column") in sensitive or c.get("final_column") in sensitive else c for c in changes]


_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def name_tokens(name: str) -> list:
    """Split a column name into lowercase word tokens ("orderTotal_USD" -> [order, total, usd])."""
    spaced = _CAMEL_RE.sub(" ", str(name))
    return [t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def benjamini_hochberg(p_values):
    """Return BH-adjusted p-values (q-values) in the original order; None stays None."""
    idx = [i for i, p in enumerate(p_values) if p is not None and not math.isnan(p)]
    q = [None] * len(p_values)
    if not idx:
        return q
    ordered = sorted(idx, key=lambda i: p_values[i])
    m = len(ordered)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = ordered[rank - 1]
        val = min(prev, p_values[i] * m / rank)
        prev = val
        q[i] = min(1.0, val)
    return q
