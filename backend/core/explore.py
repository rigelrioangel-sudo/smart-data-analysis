"""User-configured analysis: the "Analyze" workspace.

The user chooses which columns to analyze and the role each one plays:

- dimension:  a category to group records by (profile, city, status, an OLT code …)
- date:       a date/time column for trends
- measure:    a numeric quantity (distributions, medians, relationships)
- identifier: identifies a record (id, mobile, serial number) — only counted
- text / ignore

Only the selected columns are computed. The default metric is the number of
records (for a registration dataset: registrations), because such data rarely
holds a genuine numeric business measure. Averages and correlations are only
produced for columns acting as measures — never for identifiers, phone
numbers, serial numbers or technical position codes.

Every chart, table, KPI and insight in one response is computed from the same
filtered rows, so they always agree; the browser only draws them. All text is
emitted as translatable messages (see i18n.py); data values stay unchanged.
"""
import math
import warnings

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from scipy import stats

from .eda import _choose_freq, strength_label
from .i18n import D, K, M, P, render
from .inference import coerce_series, parse_number
from .profiling import numeric_stats
from .utils import benjamini_hochberg, display_series, jsonable, name_tokens, rnd

ROLES = ("dimension", "date", "measure", "identifier", "text", "ignore")
ANALYSES = ("distribution", "breakdown", "trend", "cross", "compare", "numeric")
FREQS = ("auto", "day", "week", "month", "quarter", "year")
FREQ_CODE = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}
COARSER = {"day": "week", "week": "month", "month": "quarter", "quarter": "year"}

AUTO, NONE = None, "__none__"
DEFAULT_TOP_N, MIN_TOP_N, MAX_TOP_N = 10, 3, 50
MAX_COLUMNS = 40
MAX_SECONDARY = 4
MAX_DIMENSION_LEVELS = 5000   # columns with more distinct values cannot be grouped by
MAX_OPTIONS = 300             # filter choices sent to the browser per column
MAX_FILTER_VALUES = 500
MAX_FOCUS = 12
MAX_TABLE_ROWS = 500
MAX_TREND_SERIES = 6
MAX_BREAKDOWN_LEVELS = 12
MAX_CROSS_LEVELS = 200
MAX_PERIODS = 1000
MAX_MEASURES = 6
MIN_GROUP_N = 20              # smallest group whose percentages are compared in insights
RESIDUAL_FLAG = 3.0           # |adjusted residual| that marks an unusual combination
SCATTER_POINTS = 500

STATUS_TOKENS = {"status", "state", "stage", "result", "outcome", "decision", "approval"}
# Grouping columns recommended first: offer/profile-like, then location-like, then generic types
PRIORITY_TOKENS = [
    {"profile", "plan", "package", "product", "service", "tariff", "offer", "bundle", "subscription", "speed"},
    {"city", "region", "governorate", "province", "district", "area", "zone", "branch", "location", "country", "site"},
    {"type", "category", "segment", "channel", "source", "class", "group", "department"},
]
DATE_TOKENS = {"create", "created", "creation", "registration", "registered", "signup", "date", "time", "timestamp"}

OTHER = K("ex.other")          # label of the "all other values" bucket
_OTHER_KEY = "\x00other"       # internal sentinel while computing


class ExploreError(ValueError):
    def __init__(self, msg):
        super().__init__(render(msg, "en"))
        self.msg = msg


def _pct(part, whole, digits=2):
    return rnd(part / whole * 100, digits) if whole else None


def _q_msg(q):
    if q is None:
        return K("ex.q_na")
    return K("ex.q_small") if q < 0.001 else M("ex.q", q=rnd(q, 3))


def _counts(labels: pd.Series) -> pd.Series:
    """Value counts, largest first; ties in alphabetical order (deterministic)."""
    vc = labels.value_counts(dropna=True)
    if vc.empty:
        return vc
    vc.index = vc.index.astype(str)
    return vc.sort_index(kind="stable").sort_values(ascending=False, kind="stable")


def _out_label(v):
    return OTHER if v == _OTHER_KEY else v


# ─── Column catalog ──────────────────────────────────────────

def _status_like(name):
    return bool(set(name_tokens(name)) & STATUS_TOKENS)


def _priority(name):
    toks = set(name_tokens(name))
    for i, group in enumerate(PRIORITY_TOKENS):
        if toks & group:
            return i
    return len(PRIORITY_TOKENS)


def _default_role(c):
    kind, role = c["kind"], c["role"]
    n = c["non_missing"] or 0
    ratio = c["unique"] / n if n else 0
    if kind == "empty" or role in ("constant", "empty"):
        return "ignore"
    if c.get("sensitive"):
        return "identifier"
    if kind == "datetime":
        return "date"
    if kind == "numeric" and role == "measure":
        return "measure"
    if kind == "boolean" or role in ("category", "flag"):
        return "dimension"
    groupable = ratio <= 0.5 and c["unique"] <= MAX_DIMENSION_LEVELS
    if role == "identifier":
        return "dimension" if groupable else "identifier"
    return "dimension" if groupable else "text"


def _allowed_roles(c):
    cand = c.get("candidate_kinds") or {}
    allowed = ["identifier", "ignore"]
    if c.get("sensitive"):
        return allowed  # personal identifiers: counted (records / distinct), never grouped, filtered by value or averaged
    if c["kind"] != "empty" and c["unique"] <= MAX_DIMENSION_LEVELS:
        allowed.insert(0, "dimension")
    if c["kind"] == "datetime" or cand.get("datetime", 0) >= 0.5:
        allowed.append("date")
    if c["kind"] == "numeric" or cand.get("numeric", 0) >= 0.5:
        allowed.append("measure")
    if c["kind"] in ("text", "categorical"):
        allowed.append("text")
    return allowed


def _use_msg(c, role):
    if role == "dimension":
        code = c["role"] == "identifier" or (c["kind"] == "numeric")
        return M("ex.use.code" if code else "ex.use.dimension", count=c["unique"])
    if role == "date":
        dt_ = c.get("datetime") or {}
        if dt_.get("min") and dt_.get("max"):
            return M("ex.use.date", start=D(dt_["min"][:10]), end=D(dt_["max"][:10]))
        return K("ex.use.date_plain")
    if role == "measure":
        return K("ex.use.measure")
    if role == "identifier":
        return M("ex.use.sensitive" if c.get("sensitive") else "ex.use.identifier", count=c["unique"])
    if role == "text":
        return M("ex.use.text", count=c["unique"])
    return K("ex.use.empty" if c["kind"] == "empty" or c["role"] == "empty" else "ex.use.constant")


def _label_series(ds, col):
    """Display labels for one column of the full cleaned data (cached); blanks are missing."""
    key = ("explore:labels", col)
    cache = ds._search_cache
    if key not in cache:
        s = display_series(ds.state.frame[col])
        blank = s.isna() | (s.astype(str).str.strip() == "")
        cache[key] = s.where(~blank, None).astype(object)
    return cache[key]


def catalog(ds) -> dict:
    """Every column with its detected type, suggested role, uses and filter choices."""
    cache = ds._search_cache
    if "explore:catalog" in cache:
        return cache["explore:catalog"]
    profile = ds.state.profile
    columns = []
    for i, c in enumerate(profile["columns"]):
        role = _default_role(c)
        entry = {
            "name": c["name"], "kind": c["kind"], "detected_role": c["role"], "role": role,
            "allowed_roles": _allowed_roles(c), "unique": c["unique"], "missing": c["missing_total"],
            "missing_pct": c["missing_pct"], "status_like": _status_like(c["name"]), "sensitive": bool(c.get("sensitive")),
            "priority": _priority(c["name"]), "help": _use_msg(c, role),
            "help_by_role": {r: _use_msg(c, r) for r in _allowed_roles(c)},
            "options": None, "options_total": None, "date_range": None,
        }
        if "dimension" in entry["allowed_roles"] and c["kind"] != "empty":
            vc = _counts(_label_series(ds, c["name"]))
            entry["options"] = [{"value": v, "count": int(n)} for v, n in vc.head(MAX_OPTIONS).items()]
            entry["options_total"] = int(len(vc))
        if c["kind"] == "datetime" and c.get("datetime"):
            entry["date_range"] = [c["datetime"]["min"][:10], c["datetime"]["max"][:10]]
        columns.append(entry)
    out = {"columns": columns, "row_count": int(profile["row_count"]), "recommended": recommend(columns),
           "limits": {"top_n": [MIN_TOP_N, MAX_TOP_N, DEFAULT_TOP_N], "max_columns": MAX_COLUMNS,
                      "max_secondary": MAX_SECONDARY, "max_focus": MAX_FOCUS, "max_options": MAX_OPTIONS},
           "analyses": list(ANALYSES), "freqs": list(FREQS), "metrics": _metric_options(columns),
           "presets": presets(columns)}
    cache["explore:catalog"] = out
    return out


def recommend(columns) -> dict:
    """Sensible default selection: an offer/profile grouping, a location, a status and a date."""
    order = {c["name"]: i for i, c in enumerate(columns)}
    dims = [c for c in columns if c["role"] == "dimension" and 2 <= c["unique"] <= 200]
    plain = sorted([c for c in dims if not c["status_like"]], key=lambda c: (c["priority"], c["unique"] > 50, order[c["name"]]))
    ranked = [c for c in plain if c["priority"] < len(PRIORITY_TOKENS)]
    primary = (ranked or [c for c in plain if c["unique"] <= 50] or [None])[0]
    secondary = None
    if primary is not None:
        others = [c for c in ranked if c["name"] != primary["name"] and c["priority"] != primary["priority"]]
        secondary = others[0] if others else None
    status = next((c for c in dims if c["status_like"] and c["unique"] <= MAX_BREAKDOWN_LEVELS * 2), None)
    dates = [c for c in columns if c["role"] == "date"]
    dates.sort(key=lambda c: (not (set(name_tokens(c["name"])) & DATE_TOKENS), c["missing"], order[c["name"]]))
    date = dates[0] if dates else None
    if primary is None and status is not None:
        primary, status = status, None
    chosen = [c["name"] for c in (primary, secondary, status, date) if c is not None]
    return {"columns": chosen, "primary": primary and primary["name"], "secondary": [secondary["name"]] if secondary else [],
            "breakdown": status and status["name"], "date": date and date["name"], "analyses": list(ANALYSES)}


METRICS = ("records", "distinct", "sum", "mean", "median")


def _metric_options(columns):
    """Metrics that make sense for this dataset: record count, distinct identifiers, numeric aggregates."""
    out = [{"type": "records"}]
    out += [{"type": "distinct", "column": c["name"]} for c in columns if c["role"] == "identifier" and c["unique"]]
    out += [{"type": t, "column": c["name"]} for c in columns if c["role"] == "measure" for t in ("sum", "mean", "median")]
    return out


def presets(columns) -> dict:
    """Ready-made analysis modes (Per Profile / Per City / Profile × City / Custom) as configurations."""
    rec = recommend(columns)
    dims = [c for c in columns if c["role"] == "dimension" and 2 <= c["unique"] <= 200 and not c["status_like"]]
    prof = next((c["name"] for c in dims if c["priority"] == 0), None)
    city = next((c["name"] for c in dims if c["priority"] == 1), None)
    status, date = rec["breakdown"], rec["date"]
    extras = [x for x in (status, date) if x]
    out = {}
    if prof:
        out["profile"] = {"columns": [prof, *extras], "primary": prof, "secondary": [], "breakdown": status, "date": date,
                          "split": None, "analyses": ["distribution", "breakdown", "trend", "compare"]}
    if city:
        out["city"] = {"columns": [city, *([prof] if prof else []), *extras], "primary": city,
                       "secondary": [prof] if prof else [], "breakdown": status, "date": date, "split": None,
                       "analyses": ["distribution", "breakdown", "trend", "cross", "compare"]}
    if prof and city:
        out["cross"] = {"columns": [prof, city], "primary": prof, "secondary": [city], "breakdown": NONE, "date": NONE,
                        "split": city, "analyses": ["cross", "distribution"]}
    out["custom"] = {**{k: rec[k] for k in ("columns", "primary", "secondary", "breakdown", "date", "analyses")}, "split": None}
    return out


# ─── Configuration ───────────────────────────────────────────

def _str_list(v, field, limit):
    if v is None:
        return []
    if not isinstance(v, (list, tuple)) or len(v) > limit:
        raise ExploreError(M("ex.err.bad_list", field=K(f"ex.field.{field}"), limit=limit))
    return list(v)


def resolve(cat: dict, cfg: dict):
    """Validate a configuration against the catalog and fill automatic choices.

    Returns (resolved_config, notes). Unknown columns, impossible roles and
    malformed values raise ExploreError; analyses that the selection cannot
    support are skipped with a note that explains why and suggests a fix.
    """
    cfg = cfg or {}
    by = {c["name"]: c for c in cat["columns"]}
    notes = []

    def col(name, field):
        if not isinstance(name, str) or name not in by:
            raise ExploreError(M("ex.err.unknown_column", column=str(name), field=K(f"ex.field.{field}")))
        return name

    roles = {}
    raw_roles = cfg.get("roles") or {}
    if not isinstance(raw_roles, dict):
        raise ExploreError(M("ex.err.bad_list", field=K("ex.field.roles"), limit=len(by)))
    for name, r in raw_roles.items():
        col(name, "roles")
        if r not in ROLES:
            raise ExploreError(M("ex.err.bad_role", column=name, role=str(r)))
        if r not in by[name]["allowed_roles"]:
            raise ExploreError(M("ex.err.role_not_possible", column=name, role=K(f"ex.role.{r}"),
                                 allowed=[K(f"ex.role.{a}") for a in by[name]["allowed_roles"]]))
        roles[name] = r

    def role(n):
        return roles.get(n, by[n]["role"])

    columns = []
    for n in _str_list(cfg.get("columns"), "columns", MAX_COLUMNS):
        if col(n, "columns") not in columns:
            columns.append(n)
    if not columns:
        columns = list(cat["recommended"]["columns"])
        if columns:
            notes.append(M("ex.note.recommended_used", columns=columns))
    selected = set(columns)

    def pick(field, value, need):
        if value in (None, ""):
            return AUTO
        if value == NONE:
            return NONE
        col(value, field)
        if value not in selected:
            raise ExploreError(M("ex.err.not_selected", column=value, field=K(f"ex.field.{field}")))
        if role(value) != need:
            raise ExploreError(M("ex.err.wrong_role", column=value, field=K(f"ex.field.{field}"),
                                 role=K(f"ex.role.{role(value)}"), need=K(f"ex.role.{need}")))
        return value

    primary = pick("primary", cfg.get("primary"), "dimension")
    breakdown = pick("breakdown", cfg.get("breakdown"), "dimension")
    date = pick("date", cfg.get("date"), "date")
    if primary is not AUTO and primary == breakdown:
        raise ExploreError(M("ex.err.same_column", column=primary))

    dims = [n for n in columns if role(n) == "dimension"]
    dates = [n for n in columns if role(n) == "date"]
    measures = [n for n in columns if role(n) == "measure"]
    identifiers = [n for n in columns if role(n) in ("identifier", "text")]

    if date is AUTO:
        date = dates[0] if dates else None
        if len(dates) > 1:
            notes.append(M("ex.note.date_auto", column=date, others=dates[1:]))
    elif date == NONE:
        date = None

    explicit_breakdown = breakdown not in (AUTO, NONE)
    if primary in (AUTO, NONE):
        cands = [n for n in dims if not (explicit_breakdown and n == breakdown)]
        cands.sort(key=lambda n: (by[n]["status_like"], by[n]["priority"], columns.index(n)))
        primary = cands[0] if cands else None
    if breakdown is AUTO:
        cands = [n for n in dims if n != primary and by[n]["status_like"]]
        breakdown = cands[0] if cands else None
    elif breakdown == NONE:
        breakdown = None

    raw_secondary = cfg.get("secondary")
    if raw_secondary is None:
        secondary = [n for n in dims if n not in (primary, breakdown)]
    else:
        secondary = []
        for n in _str_list(raw_secondary, "secondary", MAX_SECONDARY * 3):
            pick("secondary", n, "dimension")
            if n not in (primary, breakdown) and n not in secondary:
                secondary.append(n)
    if len(secondary) > MAX_SECONDARY:
        notes.append(M("ex.note.too_many_groupings", limit=MAX_SECONDARY, columns=secondary[MAX_SECONDARY:]))
        secondary = secondary[:MAX_SECONDARY]
    unused = [n for n in dims if n not in [primary, breakdown] + secondary]
    if unused and raw_secondary is not None:
        notes.append(M("ex.note.dims_unused", columns=unused))
    metric_col = (cfg.get("metric") or {}).get("column") if isinstance(cfg.get("metric"), dict) else None
    for n in identifiers:
        if by[n].get("sensitive"):
            notes.append(M("ex.note.sensitive", column=n))
        elif n != metric_col:
            notes.append(M("ex.note.identifier_selected", column=n, role=K(f"ex.role.{role(n)}")))
    for n in measures:
        if by[n]["detected_role"] == "identifier":
            notes.append(M("ex.note.measure_override", column=n))
    for n in dims:
        c = by[n]
        if c["unique"] and c["unique"] > 0.9 * max(1, cat["row_count"] - c["missing"]) and c["unique"] > 50:
            notes.append(M("ex.note.dimension_unique", column=n, count=c["unique"]))

    requested = cfg.get("analyses")
    requested = list(ANALYSES) if requested is None else _str_list(requested, "analyses", len(ANALYSES))
    for a in requested:
        if a not in ANALYSES:
            raise ExploreError(M("ex.err.bad_analysis", analysis=str(a)))
    explicit_analyses = cfg.get("analyses") is not None

    def suggest(pred, exclude=()):
        order = {c["name"]: i for i, c in enumerate(cat["columns"])}
        found = [c for c in cat["columns"] if pred(c) and c["name"] not in exclude and c["name"] not in selected]
        found.sort(key=lambda c: (c["priority"], c["role"] != "dimension" or c["unique"] > 50, order[c["name"]]))
        return [c["name"] for c in found][:4]

    feasible = []
    for a in requested:
        reason = None
        if a in ("distribution", "compare") and not primary:
            reason = M("ex.na.grouping", analysis=K(f"ex.analysis.{a}"),
                       suggest=suggest(lambda c: c["role"] == "dimension" and 2 <= c["unique"] <= 200))
        elif a == "breakdown" and not (primary and breakdown):
            reason = M("ex.na.breakdown", suggest=suggest(lambda c: c["role"] == "dimension" and c["status_like"], (primary,)))
        elif a == "trend" and not date:
            reason = M("ex.na.trend", suggest=suggest(lambda c: "date" in c["allowed_roles"]))
        elif a == "cross" and not (primary and secondary):
            reason = M("ex.na.cross", suggest=suggest(lambda c: c["role"] == "dimension" and 2 <= c["unique"] <= 200
                                                      and not c["status_like"], (primary,)))
        elif a == "numeric" and not measures:
            avail = suggest(lambda c: c["role"] == "measure")
            reason = M("ex.na.numeric", suggest=avail) if avail else K("ex.na.numeric_none")
        if reason is not None and reason.get("params", {}).get("suggest") == []:
            reason = M("ex.na.unavailable", analysis=K(f"ex.analysis.{a}"))  # nothing to suggest: keep it short
        if reason is None:
            feasible.append(a)
        elif explicit_analyses or a != "numeric" or measures:
            notes.append(reason)
        elif a == "numeric":
            notes.append(K("ex.na.numeric_default"))

    try:
        top_n = int(cfg.get("top_n") or DEFAULT_TOP_N)
    except (TypeError, ValueError):
        raise ExploreError(M("ex.err.bad_value", field=K("ex.field.top_n")))
    top_n = max(MIN_TOP_N, min(MAX_TOP_N, top_n))
    freq = cfg.get("freq") or "auto"
    if freq not in FREQS:
        raise ExploreError(M("ex.err.bad_value", field=K("ex.field.freq")))

    filters = {}
    raw_filters = cfg.get("filters") or {}
    if not isinstance(raw_filters, dict) or len(raw_filters) > MAX_COLUMNS:
        raise ExploreError(M("ex.err.bad_list", field=K("ex.field.filters"), limit=MAX_COLUMNS))
    for n, values in raw_filters.items():
        col(n, "filters")
        if by[n]["options"] is None:
            raise ExploreError(M("ex.err.not_filterable", column=n))
        values = _str_list(values, "filters", MAX_FILTER_VALUES)
        if any(v is not None and not isinstance(v, str) for v in values):
            raise ExploreError(M("ex.err.bad_value", field=K("ex.field.filters")))
        if values:
            filters[n] = values

    focus = {}
    raw_focus = cfg.get("focus") or {}
    if not isinstance(raw_focus, dict):
        raise ExploreError(M("ex.err.bad_list", field=K("ex.field.focus"), limit=MAX_FOCUS))
    for n, values in raw_focus.items():
        col(n, "focus")
        values = [v for v in _str_list(values, "focus", MAX_FOCUS) if isinstance(v, str)]
        if values and n in [primary, breakdown] + secondary:
            focus[n] = list(dict.fromkeys(values))

    date_from, date_to = cfg.get("date_from") or None, cfg.get("date_to") or None
    date_filter = cfg.get("date_filter_column") or date or next((c["name"] for c in cat["columns"] if c["role"] == "date"), None)
    if date_from or date_to:
        col(date_filter, "date_filter_column")
        if by[date_filter]["role"] != "date" and roles.get(date_filter) != "date":
            raise ExploreError(M("ex.err.wrong_role", column=date_filter, field=K("ex.field.date_filter_column"),
                                 role=K(f"ex.role.{role(date_filter)}"), need=K("ex.role.date")))
        parsed = []
        for v in (date_from, date_to):
            if v is None:
                parsed.append(None)
                continue
            try:
                parsed.append(pd.Timestamp(str(v)))
            except (ValueError, TypeError):
                raise ExploreError(M("ex.err.not_a_date", value=str(v)))
        if parsed[0] is not None and parsed[1] is not None and parsed[0] > parsed[1]:
            raise ExploreError(M("ex.err.date_order", start=str(date_from), end=str(date_to)))

    metric = cfg.get("metric") or {"type": "records"}
    if not isinstance(metric, dict) or metric.get("type") not in METRICS:
        raise ExploreError(M("ex.err.bad_value", field=K("ex.field.metric")))
    if metric["type"] == "records":
        metric = {"type": "records"}
    else:
        mcol = col(metric.get("column"), "metric")
        if mcol not in selected:
            raise ExploreError(M("ex.err.not_selected", column=mcol, field=K("ex.field.metric")))
        if metric["type"] == "distinct" and role(mcol) not in ("identifier", "text", "dimension"):
            raise ExploreError(M("ex.err.metric_distinct", column=mcol))
        if metric["type"] in ("sum", "mean", "median") and role(mcol) != "measure":
            raise ExploreError(M("ex.err.metric_measure", column=mcol, role=K(f"ex.role.{role(mcol)}")))
        metric = {"type": metric["type"], "column": mcol}

    split = cfg.get("split") or None
    if split is not None:
        col(split, "split")
        if split not in selected or role(split) != "dimension" or split == primary:
            raise ExploreError(M("ex.err.bad_split", column=split))
        if not primary:
            split = None

    rc = {"columns": columns, "roles": {n: role(n) for n in columns}, "role_overrides": roles,
          "metric": metric, "split": split,
          "primary": primary, "secondary": secondary, "breakdown": breakdown, "date": date,
          "measures": measures[:MAX_MEASURES], "identifiers": identifiers,
          "analyses": feasible, "requested_analyses": requested, "top_n": top_n, "freq": freq,
          "filters": filters, "focus": focus, "date_from": date_from, "date_to": date_to,
          "date_filter_column": date_filter if (date_from or date_to) else None}
    if len(measures) > MAX_MEASURES:
        notes.append(M("ex.note.too_many_measures", limit=MAX_MEASURES, columns=measures[MAX_MEASURES:]))
    return rc, notes


# ─── Analysis context ────────────────────────────────────────

class _Ctx:
    def __init__(self, ds, cat, rc):
        self.ds, self.cat, self.rc = ds, cat, rc
        self.by = {c["name"]: c for c in cat["columns"]}
        self.frame = ds.state.frame
        self.top_n = rc["top_n"]
        self.focus = rc["focus"]
        self.tests = []       # (p_value, message params to receive the BH-adjusted q)
        self.notes = []
        self._dates = {}
        mask = pd.Series(True, index=self.frame.index)
        for n, values in rc["filters"].items():
            labs = _label_series(ds, n)
            wanted = [v for v in values if v is not None]
            m = labs.isin(wanted)
            if any(v is None for v in values):
                m |= labs.isna()
            mask &= m.to_numpy()
        if rc["date_filter_column"]:
            d = self._full_dates(rc["date_filter_column"])
            if rc["date_from"]:
                mask &= (d >= pd.Timestamp(rc["date_from"])).to_numpy()
            if rc["date_to"]:
                end = pd.Timestamp(rc["date_to"])
                if len(str(rc["date_to"])) <= 10:
                    end = end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                mask &= (d <= end).to_numpy()
            missing = int(d.isna().sum())
            if missing:
                self.notes.append(M("ex.note.date_filter_missing", column=rc["date_filter_column"], count=missing))
        self.mask = mask.to_numpy(dtype=bool)
        self.n_total = int(len(self.frame))
        self.n = int(self.mask.sum())

    def labels(self, col):
        return _label_series(self.ds, col)[self.mask]

    def _full_dates(self, col):
        if col not in self._dates:
            s = self.frame[col]
            if not ptypes.is_datetime64_any_dtype(s):
                s = coerce_series(self.ds.state.result.df[col].astype(object), "datetime")[0]
            self._dates[col] = s
        return self._dates[col]

    def dates(self, col):
        return self._full_dates(col)[self.mask]

    def numbers(self, col):
        s = self.frame[col][self.mask]
        if ptypes.is_numeric_dtype(s) and not ptypes.is_bool_dtype(s):
            out = s.astype(float)
        else:
            out = s.map(lambda v: parse_number(v)[0]).astype(float)
        return out.replace([np.inf, -np.inf], np.nan)

    def levels(self, col, vc, limit=None):
        focus = self.focus.get(col)
        if focus:
            return list(focus)
        return list(vc.index[: (limit or self.top_n)])

    def register(self, p, params):
        if p is not None and not (isinstance(p, float) and math.isnan(p)):
            self.tests.append((float(p), params))
        else:
            params["q"] = _q_msg(None)

    def finish_tests(self):
        qs = benjamini_hochberg([p for p, _ in self.tests])
        for (_, params), q in zip(self.tests, qs):
            params["q"] = _q_msg(q)


def _chart(cid, ctype, title, labels, series, explanation, x_label=None, y_label=None, alt=None, **extra):
    return {"id": cid, "type": ctype, "title": title, "labels": labels, "series": series,
            "explanation": explanation, "x_label": x_label, "y_label": y_label,
            "alt_types": alt or [ctype], **extra}


def _table(tid, title, columns, rows, note=None, total=None, truncated=False, row_count=None):
    return {"id": tid, "title": title, "columns": columns, "rows": rows, "note": note, "total": total,
            "truncated": truncated, "row_count": len(rows) if row_count is None else row_count}


def _insight(ctx, section, title, finding, why, action, evidence=None, columns=None, priority="medium", limitations=None):
    return {"id": section, "section": section, "title": title, "finding": finding,
            "why": why, "action": action, "evidence": evidence or [], "columns": columns or [],
            "priority": priority, "limitations": limitations or []}


def _chi2(ct: pd.DataFrame):
    """(chi2, p, cramers_v, share of cells with expected count < 5) or None."""
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None
    obs = ct.to_numpy(dtype=float)
    obs = obs[obs.sum(axis=1) > 0][:, obs.sum(axis=0) > 0]
    if obs.shape[0] < 2 or obs.shape[1] < 2:
        return None
    chi2, p, _, expected = stats.chi2_contingency(obs, correction=False)
    n = obs.sum()
    k = min(obs.shape) - 1
    v = math.sqrt(chi2 / (n * k)) if k > 0 and n else 0.0
    return chi2, float(p), v, float((expected < 5).mean())


def _collapse(ct: pd.DataFrame, rows_keep, cols_keep):
    """Crosstab limited to the kept levels, with everything else summed into an "other" row/column."""
    out = ct.reindex(index=[r for r in rows_keep if r in ct.index], columns=[c for c in cols_keep if c in ct.columns], fill_value=0)
    out = out.reindex(index=list(rows_keep), columns=list(cols_keep), fill_value=0)
    rest_rows = ct.index.difference(pd.Index(rows_keep))
    rest_cols = ct.columns.difference(pd.Index(cols_keep))
    if len(rest_cols):
        out[_OTHER_KEY] = ct.loc[:, rest_cols].sum(axis=1).reindex(out.index, fill_value=0)
    if len(rest_rows):
        other_row = ct.loc[rest_rows].sum(axis=0)
        row = [int(other_row.reindex(pd.Index(cols_keep), fill_value=0).get(c, 0)) for c in cols_keep]
        if len(rest_cols):
            row.append(int(ct.loc[rest_rows, rest_cols].to_numpy().sum()))
        out.loc[_OTHER_KEY] = row
    return out.astype(int)


# ─── Sections ────────────────────────────────────────────────

def sec_distribution(ctx, col):
    labs = ctx.labels(col)
    n = ctx.n
    vc = _counts(labs)
    if vc.empty:
        ctx.notes.append(M("ex.note.no_values", column=col))
        return None
    missing = n - int(vc.sum())
    lv = ctx.levels(col, vc)
    counts = [int(vc.get(v, 0)) for v in lv]
    other = int(vc.sum()) - sum(counts)
    labels = list(lv) + ([OTHER] if other else [])
    values = counts + ([other] if other else [])
    chart = _chart(f"dist-{col}", "bar", M("ex.chart.dist_title", column=col), labels,
                   [{"name": K("ex.records"), "values": values, "pct": [_pct(v, n) for v in values]}],
                   M("ex.chart.dist_explain", column=col, total=n, shown=len(lv), levels=len(vc)),
                   x_label=col, y_label=K("ex.axis.records"), alt=["bar", "hbar"],
                   note=M("ex.note.missing_values", column=col, count=missing) if missing else None)
    cum, rows = 0, []
    for rank, (v, c) in enumerate(vc.items(), start=1):
        cum += int(c)
        if rank <= MAX_TABLE_ROWS:
            rows.append([rank, v, int(c), _pct(c, n), _pct(cum, n)])
    if missing:
        rows.append([None, None, missing, _pct(missing, n), None])
    table = _table(f"dist-{col}-table", M("ex.table.dist_title", column=col),
                   [{"label": K("ex.col.rank"), "type": "int"}, {"label": col, "type": "label"},
                    {"label": K("ex.col.records"), "type": "int"}, {"label": K("ex.col.share"), "type": "pct"},
                    {"label": K("ex.col.cum_share"), "type": "pct"}], rows,
                   note=K("ex.table.share_of_filtered"), total=[None, K("ex.total"), n, 100.0 if n else None, None],
                   truncated=len(vc) > MAX_TABLE_ROWS, row_count=len(vc) + (1 if missing else 0))

    insights = []
    sec = f"dist-{col}"
    top_v, top_c = vc.index[0], int(vc.iloc[0])
    finding = [M("ex.ins.top", column=col, value=top_v, count=top_c, total=n, pct=_pct(top_c, n))]
    if len(vc) >= 2:
        v2, c2 = vc.index[1], int(vc.iloc[1])
        finding.append(M("ex.ins.top_next", value=v2, count=c2, pct=_pct(c2, n), ratio=rnd(top_c / c2, 2) if c2 else None))
    insights.append(_insight(ctx, sec, M("ex.ins.top_title", column=col, value=top_v), finding, K("ex.why.volume"),
                             M("ex.act.volume", value=top_v, column=col),
                             [M("ex.ev.levels", column=col, count=len(vc)), M("ex.ev.records", count=n)], [col]))
    if len(vc) >= 4:
        k = 3
        share = _pct(int(vc.iloc[:k].sum()), n)
        n80 = int(np.searchsorted(np.cumsum(vc.to_numpy()) / vc.sum(), 0.8) + 1)
        insights.append(_insight(ctx, sec, M("ex.ins.concentration_title", column=col),
                                 M("ex.ins.concentration", column=col, k=k, pct=share, n80=n80, levels=len(vc),
                                   values=list(vc.index[:k])),
                                 K("ex.why.concentration"), K("ex.act.concentration"),
                                 [M("ex.ev.records", count=n)], [col], priority="low"))
    if len(vc) >= 5:
        small = vc[vc < 0.01 * n]
        if len(small):
            insights.append(_insight(ctx, sec, M("ex.ins.small_title", column=col),
                                     M("ex.ins.small", column=col, count=len(small), records=int(small.sum()),
                                       pct=_pct(int(small.sum()), n), examples=list(small.index[-3:])),
                                     K("ex.why.small"), K("ex.act.small"), [M("ex.ev.records", count=n)], [col], priority="low"))
    return {"id": sec, "kind": "distribution", "title": M("ex.sec.distribution", column=col),
            "intro": M("ex.sec.distribution_intro", column=col), "columns": [col],
            "charts": [chart], "tables": [table], "insights": insights}


def sec_breakdown(ctx, group, status):
    g, s = ctx.labels(group), ctx.labels(status)
    ok = (g.notna() & s.notna()).to_numpy()
    n_ok = int(ok.sum())
    if n_ok == 0:
        ctx.notes.append(M("ex.note.no_pairs", a=group, b=status))
        return None
    g, s = g[ok], s[ok]
    ct = pd.crosstab(g, s)
    ct.index, ct.columns = ct.index.astype(str), ct.columns.astype(str)
    sc = _counts(s)
    statuses = list(sc.index)
    shown_st = statuses if len(statuses) <= MAX_BREAKDOWN_LEVELS else statuses[:MAX_BREAKDOWN_LEVELS - 1]
    if ctx.focus.get(status):
        shown_st = [v for v in ctx.focus[status]]
    gc = _counts(g)
    lv = ctx.levels(group, gc)
    disp = _collapse(ct, lv, shown_st)
    row_labels = [_out_label(r) for r in disp.index]
    st_labels = [_out_label(c) for c in disp.columns]
    totals = disp.sum(axis=1)
    series = []
    for j, st in enumerate(disp.columns):
        vals = [int(x) for x in disp[st].to_numpy()]
        series.append({"name": st_labels[j], "values": vals, "pct": [_pct(v, t) for v, t in zip(vals, totals)]})
    chart = _chart(f"bd-{group}-{status}", "stacked", M("ex.chart.breakdown_title", status=status, group=group), row_labels, series,
                   M("ex.chart.breakdown_explain", status=status, group=group, count=n_ok),
                   x_label=group, y_label=K("ex.axis.records"), alt=["stacked100", "stacked", "grouped"], default="stacked100")
    if n_ok < ctx.n:
        chart["note"] = M("ex.note.pairs_missing", a=group, b=status, count=ctx.n - n_ok)
    rows = []
    for i, r in enumerate(disp.index):
        t = int(totals.iloc[i])
        rows.append([row_labels[i], t] + [{"n": int(disp.iat[i, j]), "pct": _pct(disp.iat[i, j], t)} for j in range(disp.shape[1])])
    st_tot = disp.sum(axis=0)
    total_row = [K("ex.total"), n_ok] + [{"n": int(st_tot.iloc[j]), "pct": _pct(st_tot.iloc[j], n_ok)} for j in range(disp.shape[1])]
    table = _table(f"bd-{group}-{status}-table", M("ex.table.breakdown_title", status=status, group=group),
                   [{"label": group, "type": "label"}, {"label": K("ex.col.records"), "type": "int"}]
                   + [{"label": lab, "type": "count_pct"} for lab in st_labels], rows,
                   note=M("ex.table.pct_within", column=group), total=total_row)

    insights, sec = [], f"bd-{group}-{status}"
    test = _chi2(ct)
    ev = []
    v = None
    if test:
        _, p, v, low_exp = test
        tmsg = M("ex.ev.chi2", v=rnd(v, 3), strength=K(f"strength.{strength_label(v)}"), n=n_ok, q=None)
        ctx.register(p, tmsg["params"])
        ev.append(tmsg)
        if low_exp > 0.2:
            ev.append(K("ex.ev.sparse"))
    full_tot = ct.sum(axis=1)
    min_n = MIN_GROUP_N if (full_tot >= MIN_GROUP_N).sum() >= 2 else 5
    elig = full_tot[full_tot >= min_n].index
    spreads = []
    if len(statuses) >= 2 and len(elig) >= 2:
        for st in statuses[:4]:
            share = ct.loc[elig, st] / full_tot[elig]
            hi, lo = share.idxmax(), share.idxmin()
            spreads.append((float(share[hi] - share[lo]) * 100, st, hi, lo))
        spreads.sort(key=lambda x: -x[0])

    def spread_params(gap, st, hi, lo):
        return dict(status=st, group_col=group,
                    hi=hi, hi_pct=_pct(ct.at[hi, st], full_tot[hi], 1), hi_n=int(ct.at[hi, st]), hi_total=int(full_tot[hi]),
                    lo=lo, lo_pct=_pct(ct.at[lo, st], full_tot[lo], 1), lo_n=int(ct.at[lo, st]), lo_total=int(full_tot[lo]),
                    overall_pct=_pct(sc[st], n_ok, 1), gap=rnd(gap, 1))

    ev_n = ev + [M("ex.ev.min_group", count=min_n)]
    if spreads and v is not None and v < 0.1:
        # Negligible association overall: report the largest gap once, as a minor observation
        gap, st, hi, lo = spreads[0]
        insights.append(_insight(ctx, sec, M("ex.ins.status_slight_title", status_col=status, group_col=group),
                                 M("ex.ins.status_slight", status_col=status, v=rnd(v, 3), **spread_params(gap, st, hi, lo)),
                                 K("ex.why.status_similar"), K("ex.act.status_similar"), ev_n, [group, status], priority="low",
                                 limitations=[K("ex.note.association")]))
    else:
        # At most two statuses; with few statuses their gaps mirror each other
        for gap, st, hi, lo in [s for s in spreads if s[0] >= 5][:2]:
            insights.append(_insight(
                ctx, sec, M("ex.ins.status_spread_title", status=st, group_col=group),
                M("ex.ins.status_spread", **spread_params(gap, st, hi, lo)),
                K("ex.why.status"), M("ex.act.status", status=st, group=hi), ev_n,
                [group, status], priority="high" if gap >= 15 and (v or 0) >= 0.2 else "medium",
                limitations=[K("ex.note.association")]))
    best_gap = spreads[0][0] if spreads else 0.0
    if not insights and test:
        insights.append(_insight(ctx, sec, M("ex.ins.status_similar_title", group_col=group, status_col=status),
                                 M("ex.ins.status_similar", group_col=group, status_col=status, gap=rnd(best_gap, 1)),
                                 K("ex.why.status_similar"), K("ex.act.status_similar"), ev, [group, status], priority="low"))
    return {"id": sec, "kind": "breakdown", "title": M("ex.sec.breakdown", status=status, group=group),
            "intro": M("ex.sec.breakdown_intro", status=status, group=group), "columns": [group, status],
            "charts": [chart], "tables": [table], "insights": insights}


def _auto_freq(days):
    """Readable interval for a manager view: days up to two months, weeks up to ~9 months, then months."""
    if days <= 62:
        return "day"
    if days <= 270:
        return "week"
    if days <= 366 * 6:
        return "month"
    return _choose_freq(days)[1]


def _periods(ctx, dcol):
    d = ctx.dates(dcol)
    ok = d.notna().to_numpy()
    if ok.sum() < 2:
        return None
    dd = d[ok]
    dmin, dmax = dd.min(), dd.max()
    days = (dmax - dmin).days
    if ctx.rc["freq"] == "auto":
        label = _auto_freq(days)
    else:
        label = ctx.rc["freq"]
    while True:
        per = dd.dt.to_period(FREQ_CODE[label])
        full = pd.period_range(per.min(), per.max(), freq=FREQ_CODE[label])
        if len(full) <= MAX_PERIODS or label not in COARSER:
            break
        ctx.notes.append(M("ex.note.freq_coarser", requested=K(f"freq.{label}"), used=K(f"freq.{COARSER[label]}")))
        label = COARSER[label]
    partial = set()
    if label != "day" and len(full) > 1:
        if dmin.normalize() > full[0].start_time:
            partial.add(0)
        if dmax.normalize() < full[-1].end_time.normalize():
            partial.add(len(full) - 1)
    return {"ok": ok, "per": per, "full": full, "freq": label, "partial": sorted(partial),
            "missing": int((~ok).sum()), "min": dmin, "max": dmax}


def _complete(info, series: pd.Series):
    keep = [i for i in range(len(info["full"])) if i not in info["partial"]]
    return series.iloc[keep] if len(keep) >= 2 else series


def _trend_by(ctx, info, col, limit=MAX_TREND_SERIES):
    labs = ctx.labels(col)[info["ok"]]
    has = labs.notna().to_numpy()
    vc = _counts(labs)
    lv = ctx.levels(col, vc, min(ctx.top_n, limit))
    ct = pd.crosstab(info["per"][has], labs[has].astype(str))
    ct = ct.reindex(index=info["full"], columns=lv, fill_value=0)
    return lv, ct, int(vc.sum()) - int(vc.reindex(lv).fillna(0).sum())


def sec_trend(ctx, dcol):
    info = _periods(ctx, dcol)
    if info is None:
        ctx.notes.append(M("ex.note.trend_insufficient", column=dcol))
        return None
    full, freq = info["full"], info["freq"]
    labels = [str(p) for p in full]
    total = info["per"].value_counts().reindex(full, fill_value=0)
    fk = K(f"freq.{freq}")
    sec = f"trend-{dcol}"
    charts = [_chart(f"{sec}-total", "line", M("ex.chart.trend_total_title", freq=fk), labels,
                     [{"name": K("ex.records"), "values": [int(v) for v in total.to_numpy()]}],
                     M("ex.chart.trend_explain", column=dcol, freq=fk), x_label=dcol, y_label=K("ex.axis.records"),
                     alt=["line", "bar"], freq=freq, partial=info["partial"],
                     note=M("ex.note.dates_missing", column=dcol, count=info["missing"]) if info["missing"] else None)]
    table_cols = [{"label": K("ex.col.period"), "type": "period"}, {"label": K("ex.col.records"), "type": "int"}]
    table_data = [[int(v)] for v in total.to_numpy()]
    insights = []
    rc = ctx.rc

    groups_done = []
    for col, kind in ([(rc["primary"], "group")] if rc["primary"] else []) + \
                     ([(rc["breakdown"], "status")] if rc["breakdown"] else []) + \
                     ([(rc["secondary"][0], "group")] if rc["secondary"] else []):
        if col in groups_done:
            continue
        groups_done.append(col)
        lv, ct, other = _trend_by(ctx, info, col)
        if not lv:
            continue
        series = [{"name": v, "values": [int(x) for x in ct[v].to_numpy()]} for v in lv]
        if kind == "status":
            charts.append(_chart(f"{sec}-{col}", "stacked", M("ex.chart.trend_status_title", status=col, freq=fk), labels, series,
                                 M("ex.chart.trend_status_explain", status=col), x_label=dcol, y_label=K("ex.axis.records"),
                                 alt=["stacked", "stacked100", "line"], freq=freq, partial=info["partial"],
                                 note=M("ex.note.other_not_shown", count=other) if other else None))
        else:
            charts.append(_chart(f"{sec}-{col}", "line", M("ex.chart.trend_group_title", group=col, freq=fk), labels, series,
                                 M("ex.chart.trend_group_explain", group=col, count=len(lv)), x_label=dcol,
                                 y_label=K("ex.axis.records"), alt=["line", "stacked"], freq=freq, partial=info["partial"],
                                 note=M("ex.note.other_not_shown", count=other) if other else None))
        if col == rc["primary"]:
            for v in lv:
                table_cols.append({"label": v, "type": "int"})
            for i, row in enumerate(table_data):
                row.extend(int(ct[v].iloc[i]) for v in lv)
            insights.extend(_group_trend_insights(ctx, info, sec, dcol, col, lv, ct))

    rows = [[labels[i], *table_data[i]] for i in range(len(labels))]
    table = _table(f"{sec}-table", M("ex.table.trend_title", freq=fk), table_cols, rows,
                   note=M("ex.table.trend_note", count=len(info["partial"])) if info["partial"] else None)
    table["freq"] = freq
    table["partial"] = info["partial"]

    comp = _complete(info, total)
    if len(comp) >= 2:
        first, last = int(comp.iloc[0]), int(comp.iloc[-1])
        ev = [M("ex.ev.periods", count=len(comp), freq=fk)]
        if info["partial"]:
            ev.append(M("ex.ev.partial_excluded", count=len(info["partial"])))
        direction = "up" if last > first else "down" if last < first else "flat"
        if len(comp) >= 4 and comp.nunique() > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, p = stats.spearmanr(np.arange(len(comp)), comp.to_numpy())
            tmsg = M("ex.ev.spearman_trend", rho=rnd(rho, 3), q=None)
            ctx.register(p, tmsg["params"])
            ev.append(tmsg)
            if p >= 0.05:
                direction = "flat" if abs(last - first) <= 0.1 * max(first, 1) else direction + "_uncertain"
        else:
            ev.append(K("ex.ev.few_periods"))
        pct = _pct(last - first, first, 1) if first else None
        insights.insert(0, _insight(ctx, sec, M(f"ex.ins.trend_title_{direction}", column=dcol),
                                    M("ex.ins.trend_total", first=first, first_period=P(str(comp.index[0]), freq),
                                      last=last, last_period=P(str(comp.index[-1]), freq),
                                      pct=pct, periods=len(comp), freq=fk),
                                    K("ex.why.trend"), K("ex.act.trend"), ev, [dcol],
                                    priority="high" if pct is not None and abs(pct) >= 30 and direction in ("up", "down") else "medium"))
        diffs = comp.diff().iloc[1:]
        if len(diffs) and (diffs.max() > 0 or diffs.min() < 0):
            parts = []
            big = False
            for kind_, idx in (("rise", diffs.idxmax()), ("drop", diffs.idxmin())):
                change = float(diffs[idx])
                if (kind_ == "rise" and change <= 0) or (kind_ == "drop" and change >= 0):
                    continue
                pos = comp.index.get_loc(idx)
                prev = int(comp.iloc[pos - 1])
                cur = int(comp.iloc[pos])
                pc = _pct(cur - prev, prev, 1) if prev else None
                big |= pc is not None and abs(pc) >= 30 and abs(cur - prev) >= 10
                parts.append(M(f"ex.ins.trend_{kind_}", period=P(str(idx), freq), prev_period=P(str(comp.index[pos - 1]), freq),
                               prev=prev, cur=cur, pct=pc))
            if parts:
                peak = comp.idxmax()
                insights.insert(1, _insight(ctx, sec, K("ex.ins.trend_changes_title"), parts, K("ex.why.trend_change"),
                                            M("ex.act.trend_change", period=P(str(diffs.idxmax() if diffs.max() > 0 else diffs.idxmin()), freq)),
                                            [M("ex.ev.peak", period=P(str(peak), freq), count=int(comp.max())), *ev[:1]],
                                            [dcol], priority="high" if big else "medium"))
    return {"id": sec, "kind": "trend", "title": M("ex.sec.trend", column=dcol), "columns": [dcol],
            "intro": M("ex.sec.trend_intro", column=dcol, freq=fk, start=D(str(info["min"].date())), end=D(str(info["max"].date()))),
            "charts": charts, "tables": [table], "insights": insights}


def _group_trend_insights(ctx, info, sec, dcol, col, lv, ct):
    comp = _complete(info, ct)
    if len(comp) < 2:
        return []
    freq = info["freq"]
    moves = []
    for v in lv:
        first, last = int(comp[v].iloc[0]), int(comp[v].iloc[-1])
        if comp[v].sum() < MIN_GROUP_N or first < 5:
            continue
        moves.append((v, first, last, (last - first) / first * 100))
    if len(moves) < 2:
        return []
    moves.sort(key=lambda m: -m[3])
    up, down = moves[0], moves[-1]
    if up[3] - down[3] < 10:
        return []
    return [_insight(ctx, sec, M("ex.ins.group_trend_title", group_col=col),
                     M("ex.ins.group_trend", group_col=col, first_period=P(str(comp.index[0]), freq), last_period=P(str(comp.index[-1]), freq),
                       up=up[0], up_first=up[1], up_last=up[2], up_pct=rnd(up[3], 1),
                       down=down[0], down_first=down[1], down_last=down[2], down_pct=rnd(down[3], 1)),
                     K("ex.why.group_trend"), M("ex.act.group_trend", up=up[0], down=down[0]),
                     [M("ex.ev.periods", count=len(comp), freq=K(f"freq.{freq}")), M("ex.ev.min_first", count=5)],
                     [dcol, col], priority="medium")]


def sec_cross(ctx, a, b):
    ga, gb = ctx.labels(a), ctx.labels(b)
    ok = (ga.notna() & gb.notna()).to_numpy()
    n_ok = int(ok.sum())
    if n_ok == 0:
        ctx.notes.append(M("ex.note.no_pairs", a=a, b=b))
        return None
    ga, gb = ga[ok], gb[ok]
    ca, cb = _counts(ga), _counts(gb)
    ct = pd.crosstab(ga, gb)
    ct.index, ct.columns = ct.index.astype(str), ct.columns.astype(str)
    stat_ct = _collapse(ct, list(ca.index[:MAX_CROSS_LEVELS - 1]) if len(ca) > MAX_CROSS_LEVELS else list(ca.index),
                        list(cb.index[:MAX_CROSS_LEVELS - 1]) if len(cb) > MAX_CROSS_LEVELS else list(cb.index))
    la, lb = ctx.levels(a, ca), ctx.levels(b, cb)
    disp = _collapse(ct, la, lb)
    rows_l = [_out_label(r) for r in disp.index]
    cols_l = [_out_label(c) for c in disp.columns]
    counts = disp.to_numpy(dtype=int)
    # Totals of displayed rows/columns count every record of that value (all columns), so
    # percentages within a row or column are exact even when the table shows top values only.
    row_full = [int(ct.loc[r].sum()) if r in ct.index else int(counts[i].sum()) for i, r in enumerate(disp.index)]
    col_full = [int(ct[c].sum()) if c in ct.columns else int(counts[:, j].sum()) for j, c in enumerate(disp.columns)]
    matrix = {
        "counts": counts.tolist(),
        "pct_col": [[_pct(counts[i][j], col_full[j]) for j in range(len(cols_l))] for i in range(len(rows_l))],
        "pct_row": [[_pct(counts[i][j], row_full[i]) for j in range(len(cols_l))] for i in range(len(rows_l))],
        "pct_total": [[_pct(counts[i][j], n_ok) for j in range(len(cols_l))] for i in range(len(rows_l))],
        "row_totals": row_full, "col_totals": col_full, "total": n_ok,
    }
    sec = f"cross-{a}-{b}"
    heat = _chart(f"{sec}-heat", "heatmap", M("ex.chart.heatmap_title", a=a, b=b), cols_l, [],
                  M("ex.chart.heatmap_explain", a=a, b=b, count=n_ok), x_label=b, y_label=a,
                  alt=["heatmap"], rows=rows_l, matrix=matrix,
                  note=M("ex.note.pairs_missing", a=a, b=b, count=ctx.n - n_ok) if n_ok < ctx.n else None)
    comp_series = [{"name": rows_l[i], "values": [int(x) for x in counts[i]], "pct": matrix["pct_col"][i]} for i in range(len(rows_l))]
    stacked = _chart(f"{sec}-mix", "stacked", M("ex.chart.mix_title", a=a, b=b), cols_l, comp_series,
                     M("ex.chart.mix_explain", a=a, b=b), x_label=b, y_label=K("ex.axis.share"),
                     alt=["stacked100", "stacked", "grouped"], default="stacked100")
    ga_n, gb_n = min(len(la), 6), min(len(lb), 8)
    grouped = _chart(f"{sec}-grouped", "grouped", M("ex.chart.grouped_title", a=a, b=b), cols_l[:gb_n],
                     [{"name": rows_l[i], "values": [int(x) for x in counts[i][:gb_n]], "pct": matrix["pct_col"][i][:gb_n]}
                      for i in range(ga_n)],
                     M("ex.chart.grouped_explain", a=a, b=b, rows=ga_n, cols=gb_n), x_label=b, y_label=K("ex.axis.records"),
                     alt=["grouped", "stacked"])
    cross_table = _table(f"{sec}-table", M("ex.table.cross_title", a=a, b=b),
                         [{"label": a, "type": "label"}] + [{"label": c, "type": "int"} for c in cols_l] + [{"label": K("ex.total"), "type": "int"}],
                         [[rows_l[i]] + [int(x) for x in counts[i]] + [row_full[i]] for i in range(len(rows_l))],
                         note=K("ex.table.cross_note"), total=[K("ex.total")] + col_full + [n_ok])
    cross_table["matrix"] = matrix
    cross_table["kind"] = "crosstab"

    insights, tables = [], [cross_table]
    test = _chi2(stat_ct)
    ev = []
    if test:
        _, p, v, low_exp = test
        tmsg = M("ex.ev.chi2", v=rnd(v, 3), strength=K(f"strength.{strength_label(v)}"), n=n_ok, q=None)
        ctx.register(p, tmsg["params"])
        ev.append(tmsg)
        if low_exp > 0.2:
            ev.append(K("ex.ev.sparse"))
        insights.append(_insight(ctx, sec, M("ex.ins.cross_assoc_title", a=a, b=b),
                                 M("ex.ins.cross_assoc", a=a, b=b, v=rnd(v, 3), strength=K(f"strength.{strength_label(v)}")),
                                 K("ex.why.cross"), M("ex.act.cross", a=a, b=b), ev, [a, b],
                                 priority="medium" if v >= 0.1 else "low", limitations=[K("ex.note.association")]))
        unusual = _unusual(stat_ct)
        if unusual:
            tables.append(_table(f"{sec}-unusual", M("ex.table.unusual_title", a=a, b=b),
                                 [{"label": a, "type": "label"}, {"label": b, "type": "label"},
                                  {"label": K("ex.col.observed"), "type": "int"}, {"label": K("ex.col.expected"), "type": "num"},
                                  {"label": K("ex.col.ratio"), "type": "num"}, {"label": K("ex.col.share_in_b"), "type": "pct"},
                                  {"label": K("ex.col.share_overall"), "type": "pct"}, {"label": K("ex.col.residual"), "type": "num"}],
                                 [[u["a"], u["b"], u["observed"], u["expected"], u["ratio"], u["pct_in_b"], u["pct_overall"], u["residual"]]
                                  for u in unusual], note=M("ex.table.unusual_note", threshold=RESIDUAL_FLAG)))
            for u in [x for x in unusual if x["residual"] > 0][:2] + [x for x in unusual if x["residual"] < 0][:2]:
                hi = u["residual"] > 0
                insights.append(_insight(
                    ctx, sec, M("ex.ins.combo_high_title" if hi else "ex.ins.combo_low_title", a=u["a"], b=u["b"]),
                    M("ex.ins.combo", a_col=a, a=u["a"], b_col=b, b=u["b"], observed=u["observed"], expected=u["expected"],
                      pct_in=u["pct_in_b"], pct_all=u["pct_overall"]),
                    K("ex.why.combo"), M("ex.act.combo", a=u["a"], b=u["b"]),
                    [M("ex.ev.residual", residual=u["residual"], threshold=RESIDUAL_FLAG)] + ev[:1], [a, b],
                    priority="high" if abs(u["residual"]) >= 5 else "medium", limitations=[K("ex.note.association")]))
    return {"id": sec, "kind": "cross", "title": M("ex.sec.cross", a=a, b=b), "columns": [a, b],
            "intro": M("ex.sec.cross_intro", a=a, b=b), "charts": [heat, stacked, grouped], "tables": tables,
            "insights": insights, "notes": [K("ex.note.pct_basis"), K("ex.note.association")]}


def _unusual(ct: pd.DataFrame, limit=10):
    obs = ct.to_numpy(dtype=float)
    n = obs.sum()
    if n == 0:
        return []
    r = obs.sum(axis=1, keepdims=True)
    c = obs.sum(axis=0, keepdims=True)
    exp = r @ c / n
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sqrt(exp * (1 - r / n) * (1 - c / n))
        adj = np.where(denom > 0, (obs - exp) / denom, 0.0)
    out = []
    for i, a in enumerate(ct.index):
        for j, b in enumerate(ct.columns):
            if a == _OTHER_KEY or b == _OTHER_KEY:
                continue
            if abs(adj[i, j]) >= RESIDUAL_FLAG and max(obs[i, j], exp[i, j]) >= 5:
                out.append({"a": a, "b": b, "observed": int(obs[i, j]), "expected": rnd(exp[i, j], 1),
                            "ratio": rnd(obs[i, j] / exp[i, j], 2) if exp[i, j] else None,
                            "pct_in_b": _pct(obs[i, j], c[0, j]), "pct_overall": _pct(r[i, 0], n),
                            "residual": rnd(adj[i, j], 2)})
    out.sort(key=lambda u: -abs(u["residual"]))
    return out[:limit]


def sec_compare(ctx, col):
    labs = ctx.labels(col)
    vc = _counts(labs)
    focus = ctx.focus.get(col)
    if focus and len(focus) >= 2:
        values = focus
    else:
        if len(vc) < 2:
            return None
        values = list(vc.index[:3])
        ctx.notes.append(M("ex.note.compare_default", column=col, values=values))
    n = ctx.n
    sec = f"compare-{col}"
    cnt = [int(vc.get(v, 0)) for v in values]
    rank = {v: i + 1 for i, v in enumerate(vc.index)}
    charts = [_chart(f"{sec}-volume", "bar", M("ex.chart.compare_volume_title", column=col), list(values),
                     [{"name": K("ex.records"), "values": cnt, "pct": [_pct(c, n) for c in cnt]}],
                     M("ex.chart.compare_volume_explain", column=col, count=len(values)), x_label=col,
                     y_label=K("ex.axis.records"), alt=["bar", "hbar"])]
    cols = [{"label": col, "type": "label"}, {"label": K("ex.col.records"), "type": "int"},
            {"label": K("ex.col.share"), "type": "pct"}, {"label": K("ex.col.rank"), "type": "int"}]
    rows = [[v, cnt[i], _pct(cnt[i], n), rank.get(v)] for i, v in enumerate(values)]
    insights = []
    sub = labs.isin(values).to_numpy()
    order = sorted(range(len(values)), key=lambda i: -cnt[i])
    if len(order) >= 2 and cnt[order[1]]:
        a_, b_ = order[0], order[1]
        insights.append(_insight(ctx, sec, M("ex.ins.compare_volume_title", column=col, values=list(values)),
                                 M("ex.ins.compare_volume", a=values[a_], na=cnt[a_], pa=_pct(cnt[a_], n), b=values[b_], nb=cnt[b_],
                                   pb=_pct(cnt[b_], n), ratio=rnd(cnt[a_] / cnt[b_], 2)),
                                 K("ex.why.volume"), M("ex.act.compare", column=col), [M("ex.ev.records", count=n)], [col], priority="low"))

    status = ctx.rc["breakdown"]
    if status and status != col:
        s = ctx.labels(status)
        both = sub & s.notna().to_numpy()
        if both.any():
            ct = pd.crosstab(labs[both].astype(str), s[both].astype(str)).reindex(index=values, fill_value=0)
            stc = _counts(s[both])
            shown = list(stc.index[:MAX_BREAKDOWN_LEVELS])
            ct = ct.reindex(columns=shown, fill_value=0)
            tot = ct.sum(axis=1)
            charts.append(_chart(f"{sec}-status", "grouped", M("ex.chart.compare_status_title", status=status, column=col), shown,
                                 [{"name": v, "values": [int(x) for x in ct.loc[v]], "pct": [_pct(x, tot[v]) for x in ct.loc[v]]} for v in values],
                                 M("ex.chart.compare_status_explain", status=status, column=col), x_label=status,
                                 y_label=K("ex.axis.share"), alt=["grouped", "stacked100"], default="grouped", percent=True))
            for st in shown:
                cols.append({"label": st, "type": "count_pct"})
            for i, v in enumerate(values):
                rows[i].extend({"n": int(ct.at[v, st]), "pct": _pct(ct.at[v, st], tot[v])} for st in shown)
            if len(order) >= 2 and tot[values[order[0]]] and tot[values[order[1]]]:
                va, vb = values[order[0]], values[order[1]]
                diffs = [(st, ct.at[va, st] / tot[va] * 100 - ct.at[vb, st] / tot[vb] * 100) for st in shown]
                st, d = max(diffs, key=lambda x: abs(x[1]))
                if abs(d) >= 3:
                    insights.append(_insight(ctx, sec, M("ex.ins.compare_status_title", status=st, a=va, b=vb),
                                             M("ex.ins.compare_status", status=st, a=va, pa=_pct(ct.at[va, st], tot[va], 1),
                                               b=vb, pb=_pct(ct.at[vb, st], tot[vb], 1), diff=rnd(abs(d), 1)),
                                             K("ex.why.status"), M("ex.act.status", status=st, group=va if d > 0 else vb),
                                             [M("ex.ev.records", count=int(both.sum()))], [col, status],
                                             priority="medium" if abs(d) >= 10 else "low", limitations=[K("ex.note.association")]))

    dcol = ctx.rc["date"]
    if dcol:
        info = _periods(ctx, dcol)
        if info:
            lab_ok = labs[info["ok"]]
            has = lab_ok.isin(values).to_numpy()
            ct = pd.crosstab(info["per"][has], lab_ok[has].astype(str)).reindex(index=info["full"], columns=values, fill_value=0)
            charts.append(_chart(f"{sec}-trend", "line", M("ex.chart.compare_trend_title", column=col, freq=K(f"freq.{info['freq']}")),
                                 [str(p) for p in info["full"]],
                                 [{"name": v, "values": [int(x) for x in ct[v].to_numpy()]} for v in values],
                                 M("ex.chart.trend_group_explain", group=col, count=len(values)), x_label=dcol,
                                 y_label=K("ex.axis.records"), alt=["line", "stacked"], freq=info["freq"], partial=info["partial"]))
            comp = _complete(info, ct)
            if len(comp) >= 2:
                cols += [{"label": M("ex.col.first_period", period=P(str(comp.index[0]), info["freq"])), "type": "int"},
                         {"label": M("ex.col.last_period", period=P(str(comp.index[-1]), info["freq"])), "type": "int"},
                         {"label": K("ex.col.change"), "type": "pct"}]
                for i, v in enumerate(values):
                    f_, l_ = int(comp[v].iloc[0]), int(comp[v].iloc[-1])
                    rows[i] += [f_, l_, _pct(l_ - f_, f_, 1) if f_ else None]

    sec_ = ctx.rc["secondary"][0] if ctx.rc["secondary"] else None
    if sec_ and sec_ != col:
        s = ctx.labels(sec_)
        both = sub & s.notna().to_numpy()
        if both.any():
            stc = _counts(s[both])
            shown = list(stc.index[:8])
            ct = pd.crosstab(labs[both].astype(str), s[both].astype(str)).reindex(index=values, fill_value=0)
            tot = ct.sum(axis=1)
            ct = ct.reindex(columns=shown, fill_value=0)
            charts.append(_chart(f"{sec}-{sec_}", "grouped", M("ex.chart.compare_mix_title", other=sec_, column=col), shown,
                                 [{"name": v, "values": [int(x) for x in ct.loc[v]], "pct": [_pct(x, tot[v]) for x in ct.loc[v]]} for v in values],
                                 M("ex.chart.compare_mix_explain", other=sec_, column=col), x_label=sec_,
                                 y_label=K("ex.axis.share"), alt=["grouped", "stacked100"], default="grouped", percent=True))
    table = _table(f"{sec}-table", M("ex.table.compare_title", column=col), cols, rows, note=K("ex.table.compare_note"))
    return {"id": sec, "kind": "compare", "title": M("ex.sec.compare", column=col, values=list(values)), "columns": [col],
            "intro": M("ex.sec.compare_intro", column=col), "charts": charts, "tables": [table], "insights": insights}


def sec_numeric(ctx, measures):
    charts, tables, insights = [], [], []
    sec = "numeric"
    data = {m: ctx.numbers(m) for m in measures}
    primary = ctx.rc["primary"]
    rows = []
    for m in measures:
        y = data[m].dropna()
        st = numeric_stats(y)
        if st is None:
            ctx.notes.append(M("ex.note.no_values", column=m))
            continue
        rows.append([m, st["count"], st["mean"], st["median"], st["std"], st["min"], st["q1"], st["q3"], st["max"],
                     st["outliers_mild"]])
        h = st["histogram"]
        if h:
            charts.append(_chart(f"num-{m}-hist", "histogram", M("ex.chart.hist_title", column=m),
                                 [[h["edges"][i], h["edges"][i + 1]] for i in range(len(h["counts"]))],
                                 [{"name": K("ex.records"), "values": h["counts"]}],
                                 M("ex.chart.hist_explain", column=m, count=st["count"], median=st["median"]),
                                 x_label=m, y_label=K("ex.axis.records"), alt=["histogram"]))
        charts.append(_chart(f"num-{m}-box", "box", M("ex.chart.box_title", column=m), [m], [],
                             M("ex.chart.box_explain", column=m, count=st["outliers_mild"]), x_label=m, alt=["box"],
                             stats={k: st[k] for k in ("min", "max", "q1", "median", "q3", "mean", "count", "outliers_mild")} | {"box": st["box"]}))
        insights.append(_insight(ctx, sec, M("ex.ins.measure_title", column=m),
                                 M("ex.ins.measure", column=m, median=st["median"], mean=st["mean"], count=st["count"],
                                   min=st["min"], max=st["max"], outliers=st["outliers_mild"]),
                                 K("ex.why.measure"), K("ex.act.measure"), [M("ex.ev.records", count=st["count"])], [m], priority="low"))
        if primary:
            g = ctx.labels(primary)
            ok = g.notna().to_numpy() & data[m].notna().to_numpy()
            if ok.sum() >= 8:
                gv, yv = g[ok].astype(str), data[m][ok]
                vc = _counts(gv)
                lv = ctx.levels(primary, vc)
                grp = {v: yv[(gv == v).to_numpy()] for v in lv}
                grp = {v: s for v, s in grp.items() if len(s) >= 2}
                if len(grp) >= 2:
                    med = [rnd(s.median(), 4) for s in grp.values()]
                    charts.append(_chart(f"num-{m}-by-{primary}", "bar", M("ex.chart.median_by_title", column=m, group=primary),
                                         list(grp), [{"name": K("ex.median"), "values": med, "counts": [int(len(s)) for s in grp.values()]}],
                                         M("ex.chart.median_by_explain", column=m, group=primary), x_label=primary, y_label=m,
                                         alt=["bar", "hbar"]))
                    groups_all = [yv[(gv == v).to_numpy()] for v in vc.index]
                    groups_all = [s for s in groups_all if len(s) >= 2]
                    if len(groups_all) >= 2 and pd.concat(groups_all).nunique() > 1:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            h_, p = stats.kruskal(*groups_all)
                        nn, kk = sum(len(s) for s in groups_all), len(groups_all)
                        eps2 = max(0.0, (h_ - kk + 1) / (nn - kk)) if nn > kk else None
                        hi = max(grp, key=lambda v: grp[v].median())
                        lo = min(grp, key=lambda v: grp[v].median())
                        tmsg = M("ex.ev.kruskal", effect=rnd(eps2, 3), q=None, count=nn)
                        ctx.register(p, tmsg["params"])
                        insights.append(_insight(ctx, sec, M("ex.ins.measure_group_title", column=m, group=primary),
                                                 M("ex.ins.measure_group", column=m, group=primary, hi=hi, hi_med=rnd(grp[hi].median(), 4),
                                                   lo=lo, lo_med=rnd(grp[lo].median(), 4)),
                                                 K("ex.why.measure_group"), M("ex.act.measure_group", column=m, group=primary),
                                                 [tmsg], [m, primary], priority="medium" if (eps2 or 0) >= 0.06 else "low",
                                                 limitations=[K("ex.note.association")]))
    if rows:
        tables.append(_table("num-table", K("ex.table.numeric_title"),
                             [{"label": K("ex.col.column"), "type": "label"}] +
                             [{"label": K(f"ex.col.{k}"), "type": "int" if k in ("count", "outliers") else "num"}
                              for k in ("count", "mean", "median", "std", "min", "q1", "q3", "max", "outliers")], rows))
    pairs = [(a, b) for i, a in enumerate(measures) for b in measures[i + 1:]][:3]
    for a, b in pairs:
        x, y = data[a], data[b]
        ok = (x.notna() & y.notna()).to_numpy()
        if ok.sum() < 8 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, p = stats.spearmanr(x[ok], y[ok])
        pts = pd.DataFrame({"x": x[ok], "y": y[ok]})
        sampled = len(pts) > SCATTER_POINTS
        if sampled:
            pts = pts.sample(SCATTER_POINTS, random_state=0)
        tmsg = M("ex.ev.spearman", rho=rnd(rho, 3), strength=K(f"strength.{strength_label(rho)}"), n=int(ok.sum()), q=None)
        ctx.register(p, tmsg["params"])
        charts.append(_chart(f"num-{a}-{b}", "scatter", M("ex.chart.scatter_title", x=a, y=b), [],
                             [{"name": K("ex.records"), "points": [[rnd(u, 6), rnd(v, 6)] for u, v in pts.to_numpy()]}],
                             M("ex.chart.scatter_sampled" if sampled else "ex.chart.scatter_explain", x=a, y=b, count=int(ok.sum()), sample=len(pts)),
                             x_label=a, y_label=b, alt=["scatter"]))
        insights.append(_insight(ctx, sec, M("ex.ins.pair_title", x=a, y=b),
                                 M("ex.ins.pair", x=a, y=b, rho=rnd(rho, 3), strength=K(f"strength.{strength_label(rho)}")),
                                 K("ex.why.pair"), K("ex.act.pair"), [tmsg], [a, b],
                                 priority="medium" if abs(rho) >= 0.3 else "low", limitations=[K("ex.note.association")]))
    if not charts:
        return None
    return {"id": sec, "kind": "numeric", "title": K("ex.sec.numeric"), "intro": M("ex.sec.numeric_intro", columns=list(measures)),
            "columns": list(measures), "charts": charts, "tables": tables, "insights": insights}


# ─── Main result: the chosen metric for the chosen grouping, shown first ─

def _metric_label(metric):
    if metric["type"] == "records":
        return K("ex.metric_name.records")
    return M(f"ex.metric_name.{metric['type']}", column=metric["column"])


def _agg_by(ctx, groups: pd.Series, metric) -> pd.Series:
    """Metric per group value (index = label), largest first, ties alphabetical."""
    t = metric["type"]
    if t == "records":
        return _counts(groups)
    v = ctx.labels(metric["column"]) if t == "distinct" else ctx.numbers(metric["column"])
    df = pd.DataFrame({"g": groups.to_numpy(), "v": v.to_numpy()}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    out = df.groupby("g")["v"].nunique() if t == "distinct" else df.groupby("g")["v"].agg(t).astype(float)
    out.index = out.index.astype(str)
    return out.sort_index(kind="stable").sort_values(ascending=False, kind="stable")


def _agg_rows(ctx, metric, rows=None):
    """Metric over the filtered rows (optionally only `rows`, a boolean array)."""
    t = metric["type"]
    if t == "records":
        return int(rows.sum()) if rows is not None else ctx.n
    v = ctx.labels(metric["column"]) if t == "distinct" else ctx.numbers(metric["column"])
    if rows is not None:
        v = v[rows]
    if t == "distinct":
        return int(v.nunique())
    v = v.dropna()
    return rnd(getattr(v, t)(), 4) if len(v) else None


def _num(v, metric):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return int(v) if metric["type"] in ("records", "distinct") else rnd(v, 4)


def recommend_chart(shape, n_levels=0, metric_type="records", long_labels=False, n_split=0):
    """(default chart type, allowed chart types) for a result shape. Pie/doughnut only for a few
    categories whose shares add up to 100% (record counts); lines only for time; stacking only for counts."""
    if shape == "single":
        allowed = ["bar", "hbar"]
        if metric_type == "records" and n_levels <= 6:
            allowed.append("doughnut")
        if metric_type == "records" and n_levels <= 4:
            return "doughnut", allowed
        return ("hbar" if n_levels > 8 or long_labels else "bar"), allowed
    if shape == "split":
        allowed = (["stacked100", "stacked", "grouped", "heatmap"] if metric_type == "records"
                   else ["grouped", "heatmap"])
        if n_levels > 8 or n_split > 8:
            return "heatmap", allowed
        return ("stacked100" if metric_type == "records" else "grouped"), allowed
    if shape == "time":
        return "line", ["line", "bar"]
    if shape == "histogram":
        return "histogram", ["histogram"]
    return "scatter", ["scatter"]


def _denominator(ctx, metric, total):
    t = metric["type"]
    if t == "records":
        return M("ex.denom.records", total=total)
    if t == "distinct":
        return M("ex.denom.distinct", total=total, column=metric["column"])
    return M("ex.denom.numeric", agg=K(f"ex.agg.{t}"), column=metric["column"])


def _main_single(ctx, col, metric):
    labs = ctx.labels(col)
    vals = _agg_by(ctx, labs, metric)
    if vals.empty:
        return None
    lv = ctx.levels(col, vals)
    pct_ok = metric["type"] in ("records", "distinct")
    total = _agg_rows(ctx, metric)
    shown = [_num(vals.get(v), metric) if v in vals.index else (0 if pct_ok else None) for v in lv]
    rest = (labs.notna() & ~labs.isin(lv)).to_numpy()
    labels, values = list(lv), list(shown)
    if rest.any():
        labels.append(OTHER)
        values.append(_num(_agg_rows(ctx, metric, rest), metric))
    pct = [_pct(v, total) if pct_ok and v is not None else None for v in values]
    default, allowed = recommend_chart("single", len(labels), metric["type"], max(len(str(x)) for x in lv) > 14)
    chart = _chart("main", default, M("ex.main.title_single", metric=_metric_label(metric), column=col), labels,
                   [{"name": _metric_label(metric), "values": values, "pct": pct if pct_ok else None}],
                   _denominator(ctx, metric, total), x_label=col, y_label=_metric_label(metric), alt=allowed,
                   default=default, sorted=True)
    rows = []
    for rank, (v, x) in enumerate(vals.items(), start=1):
        if rank > MAX_TABLE_ROWS:
            break
        val = _num(x, metric)
        rows.append([rank, v, val, _pct(val, total) if pct_ok else None])
    missing = labs.isna().to_numpy()
    if missing.any():
        val = _num(_agg_rows(ctx, metric, missing), metric)
        rows.append([None, None, val, _pct(val, total) if pct_ok else None])
    cols = [{"label": K("ex.col.rank"), "type": "int"}, {"label": col, "type": "label"},
            {"label": _metric_label(metric), "type": "int" if pct_ok else "num"}]
    if pct_ok:
        cols.append({"label": K("ex.col.share"), "type": "pct"})
    else:
        rows = [r[:3] for r in rows]
    total_row = [None, K("ex.total"), _num(total, metric)] + ([100.0 if metric["type"] == "records" and total else None] if pct_ok else [])
    table = _table("main-table", M("ex.main.title_single", metric=_metric_label(metric), column=col), cols, rows,
                   note=_denominator(ctx, metric, total), total=total_row, truncated=len(vals) > MAX_TABLE_ROWS,
                   row_count=len(vals) + int(missing.any()))
    return {"shape": "single", "chart": chart, "table": table, "total": _num(total, metric)}


def _main_split(ctx, a, b, metric):
    ga, gb = ctx.labels(a), ctx.labels(b)
    ok = (ga.notna() & gb.notna()).to_numpy()
    if not ok.any():
        return None
    ga_ok, gb_ok = ga[ok].astype(str), gb[ok].astype(str)
    t = metric["type"]
    if t == "records":
        mat = pd.crosstab(ga_ok, gb_ok)
    else:
        v = (ctx.labels(metric["column"]) if t == "distinct" else ctx.numbers(metric["column"]))[ok]
        df = pd.DataFrame({"a": ga_ok.to_numpy(), "b": gb_ok.to_numpy(), "v": v.to_numpy()}).dropna()
        grp = df.groupby(["a", "b"])["v"]
        mat = (grp.nunique() if t == "distinct" else grp.agg(t)).unstack()
    mat.index, mat.columns = mat.index.astype(str), mat.columns.astype(str)
    la = ctx.levels(a, _agg_by(ctx, ga[ok], metric))
    lb = ctx.levels(b, _agg_by(ctx, gb[ok], metric))
    m = mat.reindex(index=la, columns=lb)
    if t in ("records", "distinct"):
        m = m.fillna(0)
    values = [[_num(x, metric) for x in row] for row in m.to_numpy()]
    a_arr, b_arr = ga.to_numpy(dtype=object), gb.to_numpy(dtype=object)
    row_tot = [_num(_agg_rows(ctx, metric, ok & (a_arr == r)), metric) for r in la]
    col_tot = [_num(_agg_rows(ctx, metric, ok & (b_arr == c)), metric) for c in lb]
    grand = _num(_agg_rows(ctx, metric, ok), metric)
    pct_ok = t == "records"   # shares of distinct ids overlap across cells; averages have no shares
    pct_row = [[_pct(x, row_tot[i]) if pct_ok else None for x in row] for i, row in enumerate(values)]
    matrix = {"counts": values, "row_totals": row_tot, "col_totals": col_tot, "total": grand, "percent": pct_ok,
              "pct_row": pct_row,
              "pct_col": [[_pct(values[i][j], col_tot[j]) if pct_ok else None for j in range(len(lb))] for i in range(len(la))],
              "pct_total": [[_pct(x, grand) if pct_ok else None for x in row] for row in values]}
    default, allowed = recommend_chart("split", len(la), t, n_split=len(lb))
    hidden = (int(mat.shape[0]) - len(la), int(mat.shape[1]) - len(lb))
    chart = _chart("main", default, M("ex.main.title_split", metric=_metric_label(metric), a=a, b=b), la,
                   [{"name": c, "values": [values[i][j] for i in range(len(la))],
                     "pct": [pct_row[i][j] for i in range(len(la))] if pct_ok else None} for j, c in enumerate(lb)],
                   _denominator(ctx, metric, grand), x_label=a, y_label=_metric_label(metric), alt=allowed, default=default,
                   heat={"rows": la, "cols": lb, "matrix": matrix, "row_label": a, "col_label": b},
                   note=M("ex.note.top_shown", a=a, b=b, hidden_a=hidden[0], hidden_b=hidden[1]) if any(hidden) else None)
    table = _table("main-table", M("ex.main.title_split", metric=_metric_label(metric), a=a, b=b),
                   [{"label": a, "type": "label"}] + [{"label": c, "type": "int" if t != "mean" and t != "median" and t != "sum" else "num"} for c in lb]
                   + [{"label": K("ex.total"), "type": "int" if t in ("records", "distinct") else "num"}],
                   [[la[i]] + values[i] + [row_tot[i]] for i in range(len(la))],
                   note=_denominator(ctx, metric, grand), total=[K("ex.total")] + col_tot + [grand])
    return {"shape": "split", "chart": chart, "table": table, "total": grand}


def _main_time(ctx, dcol, metric):
    info = _periods(ctx, dcol)
    if info is None:
        return None
    per = info["per"].astype(str)
    t = metric["type"]
    if t == "records":
        vals = per.value_counts()
    else:
        v = (ctx.labels(metric["column"]) if t == "distinct" else ctx.numbers(metric["column"]))[info["ok"]]
        df = pd.DataFrame({"p": per.to_numpy(), "v": v.to_numpy()}).dropna()
        vals = df.groupby("p")["v"].nunique() if t == "distinct" else df.groupby("p")["v"].agg(t)
    labels = [str(p) for p in info["full"]]  # chronological, including empty periods
    values = [_num(vals.get(p), metric) if p in vals.index else (0 if t in ("records", "distinct") else None) for p in labels]
    default, allowed = recommend_chart("time")
    fk = K(f"freq.{info['freq']}")
    chart = _chart("main", default, M("ex.main.title_time", metric=_metric_label(metric), freq=fk), labels,
                   [{"name": _metric_label(metric), "values": values}], _denominator(ctx, metric, _agg_rows(ctx, metric)),
                   x_label=dcol, y_label=_metric_label(metric), alt=allowed, default=default, freq=info["freq"],
                   partial=info["partial"])
    table = _table("main-table", M("ex.main.title_time", metric=_metric_label(metric), freq=fk),
                   [{"label": K("ex.col.period"), "type": "period"},
                    {"label": _metric_label(metric), "type": "int" if t in ("records", "distinct") else "num"}],
                   [[p, v] for p, v in zip(labels, values)])
    table["freq"], table["partial"] = info["freq"], info["partial"]
    return {"shape": "time", "chart": chart, "table": table, "total": _num(_agg_rows(ctx, metric), metric)}


def _main_numeric(ctx, measures):
    if len(measures) >= 2:
        a, b = measures[:2]
        x, y = ctx.numbers(a), ctx.numbers(b)
        ok = (x.notna() & y.notna()).to_numpy()
        if ok.sum() < 3:
            return None
        pts = pd.DataFrame({"x": x[ok], "y": y[ok]})
        if len(pts) > SCATTER_POINTS:
            pts = pts.sample(SCATTER_POINTS, random_state=0)
        chart = _chart("main", "scatter", M("ex.chart.scatter_title", x=a, y=b), [],
                       [{"name": K("ex.records"), "points": [[rnd(u, 6), rnd(v, 6)] for u, v in pts.to_numpy()]}],
                       M("ex.chart.scatter_explain", x=a, y=b, count=int(ok.sum())), x_label=a, y_label=b, alt=["scatter"])
        return {"shape": "scatter", "chart": chart, "table": None, "total": int(ok.sum())}
    m = measures[0]
    st = numeric_stats(ctx.numbers(m).dropna())
    if not st or not st["histogram"]:
        return None
    h = st["histogram"]
    labels = [[h["edges"][i], h["edges"][i + 1]] for i in range(len(h["counts"]))]
    chart = _chart("main", "histogram", M("ex.chart.hist_title", column=m), labels,
                   [{"name": K("ex.records"), "values": h["counts"], "pct": [_pct(c, st["count"]) for c in h["counts"]]}],
                   M("ex.chart.hist_explain", column=m, count=st["count"], median=st["median"]),
                   x_label=m, y_label=K("ex.axis.records"), alt=["histogram"])
    table = _table("main-table", M("ex.chart.hist_title", column=m),
                   [{"label": m, "type": "label"}, {"label": K("ex.col.records"), "type": "int"}, {"label": K("ex.col.share"), "type": "pct"}],
                   [[lab, c, p] for lab, c, p in zip(labels, h["counts"], chart["series"][0]["pct"])])
    return {"shape": "histogram", "chart": chart, "table": table, "total": st["count"]}


def main_result(ctx):
    rc, metric = ctx.rc, ctx.rc["metric"]
    if rc["primary"]:
        out = _main_split(ctx, rc["primary"], rc["split"], metric) if rc["split"] else _main_single(ctx, rc["primary"], metric)
    elif rc["date"]:
        out = _main_time(ctx, rc["date"], metric)
    elif rc["measures"]:
        out = _main_numeric(ctx, rc["measures"])
    else:
        out = None
    if out:
        out["metric"] = _metric_label(metric)
    return out


# ─── KPIs and entry point ────────────────────────────────────

def _kpis(ctx, main=None):
    rc, n = ctx.rc, ctx.n
    out = [{"id": "records", "label": K("ex.kpi.records"), "value": n,
            "sub": M("ex.kpi.records_filtered", total=ctx.n_total, pct=_pct(n, ctx.n_total, 1)) if n != ctx.n_total else K("ex.kpi.records_all")}]
    metric = rc["metric"]
    if metric["type"] != "records":
        out.append({"id": "metric", "label": _metric_label(metric), "value": _num(_agg_rows(ctx, metric), metric),
                    "sub": K(f"ex.kpi.metric_sub.{metric['type']}")})
    for col in [rc["primary"]] + rc["secondary"][:2]:
        if not col:
            continue
        vc = _counts(ctx.labels(col))
        if vc.empty:
            continue
        out.append({"id": f"distinct-{col}", "label": M("ex.kpi.distinct", column=col), "value": int(len(vc)),
                    "sub": M("ex.kpi.top_value", value=vc.index[0], pct=_pct(int(vc.iloc[0]), n, 1))})
    if rc["breakdown"]:
        vc = _counts(ctx.labels(rc["breakdown"]))
        if not vc.empty:
            out.append({"id": "status", "label": M("ex.kpi.top_status", column=rc["breakdown"]), "value": vc.index[0],
                        "sub": M("ex.kpi.share_count", pct=_pct(int(vc.iloc[0]), n, 1), count=int(vc.iloc[0]))})
    if rc["date"]:
        d = ctx.dates(rc["date"]).dropna()
        if len(d):
            out.append({"id": "period", "label": M("ex.kpi.period", column=rc["date"]),
                        "value": M("ex.kpi.range", start=D(str(d.min().date())), end=D(str(d.max().date()))),
                        "sub": M("ex.kpi.days", count=int((d.max() - d.min()).days) + 1)})
    for col in rc["identifiers"][:2]:
        if metric["type"] == "distinct" and metric["column"] == col:
            continue
        labs = ctx.labels(col)
        vc = labs.value_counts()
        repeated = int(vc[vc > 1].sum())
        out.append({"id": f"ident-{col}", "label": M("ex.kpi.distinct", column=col), "value": int(len(vc)),
                    "sub": M("ex.kpi.repeated", count=repeated)})
    return out


PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def run_analysis(ds, cfg) -> dict:
    """Run the configured analysis on the cleaned data (filtered as requested)."""
    with ds.lock:
        cat = catalog(ds)
        rc, notes = resolve(cat, cfg)
        ctx = _Ctx(ds, cat, rc)
        sections = []
        if ctx.n == 0:
            notes.append(K("ex.note.no_rows"))
        else:
            a = set(rc["analyses"])
            dims = [c for c in [rc["primary"]] + rc["secondary"] + [rc["breakdown"]] if c]
            if "distribution" in a:
                sections += [sec_distribution(ctx, c) for c in dims]
            if "breakdown" in a and rc["breakdown"]:
                sections += [sec_breakdown(ctx, g, rc["breakdown"]) for g in [rc["primary"]] + rc["secondary"]]
            if "trend" in a and rc["date"]:
                sections.append(sec_trend(ctx, rc["date"]))
            if "cross" in a:
                sections += [sec_cross(ctx, rc["primary"], s) for s in rc["secondary"]]
            if "compare" in a and rc["primary"]:
                sections.append(sec_compare(ctx, rc["primary"]))
            if "numeric" in a and rc["measures"]:
                sections.append(sec_numeric(ctx, rc["measures"]))
        sections = [s for s in sections if s]
        main = main_result(ctx) if ctx.n else None
        if ctx.n and main is None:
            notes.insert(0, M("ex.main.none", suggest=[c["name"] for c in cat["columns"]
                                                       if c["role"] in ("dimension", "date") and c["name"] not in rc["columns"]][:3]))
        ctx.finish_tests()
        # Insight ids are unique per response
        all_insights = []
        for s in sections:
            for i, ins in enumerate(s["insights"]):
                ins["id"] = f"{s['id']}-{i}"
                all_insights.append(ins)
        highlights = sorted(all_insights, key=lambda x: PRIORITY_ORDER[x["priority"]])[:8]
        filtered = ctx.n != ctx.n_total
        return jsonable({
            "config": rc, "notes": notes + ctx.notes,
            "records": {"total": ctx.n_total, "filtered": ctx.n, "is_filtered": filtered},
            "kpis": _kpis(ctx, main) if ctx.n else [], "main": main,
            "sections": sections, "highlights": [h["id"] for h in highlights],
            "metric": K("ex.metric.records"),
        })


# ─── Export (same configuration → same numbers as on screen) ─

def _txt(v, lang):
    """Plain text for a label / message / data value in a report."""
    if v is None:
        return render(K("common.missing_value"), lang)
    if isinstance(v, dict) and "key" in v:
        return render(v, lang)
    return v


def _table_grid(tbl, lang, excel):
    from .i18n import format_number, format_period
    header = []
    for c in tbl["columns"]:
        lab = _txt(c["label"], lang)
        if c["type"] == "count_pct" and excel:
            header += [lab, f"{lab} %"]
        else:
            header.append(f"{lab} %" if c["type"] == "pct" and excel else lab)
    rows = []
    for r in tbl["rows"] + ([tbl["total"]] if tbl.get("total") else []):
        out = []
        for c, v in zip(tbl["columns"], r):
            t = c["type"]
            if t == "count_pct":
                if excel:
                    out += [v["n"] if v else None, v["pct"] if v else None]
                else:
                    out.append("—" if not v else f"{format_number(v['n'], lang)} ({format_number(v['pct'], lang)}%)"
                               if v.get("pct") is not None else format_number(v["n"], lang))
            elif t == "period" and isinstance(v, str):
                out.append(format_period(v, tbl.get("freq"), lang))
            elif t == "label" or isinstance(v, dict):
                out.append(_txt(v, lang) if (v is not None or t == "label") else None)
            elif t == "pct" and v is not None and not excel:
                out.append(f"{format_number(v, lang)}%")
            else:
                out.append(v)
        rows.append(out)
    return header, rows


def _settings_rows(result, lang):
    rc = result["config"]
    t = lambda key, **p: render(M(key, **p), lang)  # noqa: E731
    none = t("ex.none")
    rows = [[t("ex.field.columns"), ", ".join(rc["columns"]) or none],
            [t("ex.field.primary"), rc["primary"] or none],
            [t("ex.field.secondary"), ", ".join(rc["secondary"]) or none],
            [t("ex.field.breakdown"), rc["breakdown"] or none],
            [t("ex.field.date"), rc["date"] or none],
            [t("ex.field.analyses"), ", ".join(t(f"ex.analysis.{a}") for a in rc["analyses"]) or none],
            [t("ex.field.top_n"), rc["top_n"]], [t("ex.field.freq"), t(f"ex.freq.{rc['freq']}")]]
    for col, values in rc["filters"].items():
        rows.append([t("ex.field.filter_on", column=col), ", ".join(_txt(v, lang) for v in values)])
    if rc["date_from"] or rc["date_to"]:
        rows.append([t("ex.field.date_range", column=rc["date_filter_column"]), f"{rc['date_from'] or '…'} → {rc['date_to'] or '…'}"])
    for col, values in rc["focus"].items():
        rows.append([t("ex.field.focus_on", column=col), ", ".join(values)])
    rows.append([t("ex.kpi.records"), t("ex.kpi.records_of", count=result["records"]["filtered"], total=result["records"]["total"])])
    return rows


def _chart_rows(ch, lang):
    """Long-format data of one chart: (series, label, value, percent)."""
    if ch["type"] == "heatmap" or ch.get("heat"):
        h = _heat(ch)
        m = h["matrix"]
        return [[_txt(r, lang), _txt(c, lang), m["counts"][i][j], m["pct_col"][i][j]]
                for i, r in enumerate(h["rows"]) for j, c in enumerate(h["cols"])]
    if ch["type"] == "box":
        return [[_txt(ch["x_label"], lang), k, v, None] for k, v in ch["stats"].items() if k != "box"]
    if ch["type"] == "scatter":
        return []
    out = []
    for s in ch["series"]:
        for j, lab in enumerate(ch["labels"]):
            if ch["type"] == "histogram":
                lab = f"{lab[0]} – {lab[1]}"
            out.append([_txt(s["name"], lang), _txt(lab, lang), s["values"][j], (s.get("pct") or [None] * len(s["values"]))[j]])
    return out


def export_report(ds, cfg, fmt, lang="en") -> bytes:
    import datetime as dt
    import json
    from .reporting import ExportError, _doc, _table, workbook_bytes
    result = run_analysis(ds, cfg)
    if result.get("main"):  # the main chart and its table come first in every report
        mn = result["main"]
        result["sections"].insert(0, {"id": "main", "kind": "main", "title": mn["chart"]["title"], "intro": None, "columns": [],
                                      "charts": [mn["chart"]], "tables": [mn["table"]] if mn["table"] else [], "insights": []})
    generated = dt.datetime.now().isoformat(timespec="seconds")
    t = lambda key, **p: render(M(key, **p), lang)  # noqa: E731
    insights = [i for s in result["sections"] for i in s["insights"]]

    def ins_text(i):
        return {"section": i["section"], "priority": i["priority"], "title": render(i["title"], lang),
                "finding": render(i["finding"], lang), "why": render(i["why"], lang), "action": render(i["action"], lang),
                "evidence": [render(e, lang) for e in i["evidence"]], "limitations": [render(x, lang) for x in i["limitations"]],
                "columns": i["columns"]}

    if fmt == "json":
        body = {"dataset": ds.filename, "lang": lang, "generated_at": generated, "result": result,
                "findings": [ins_text(i) for i in insights], "notes": [render(n, lang) for n in result["notes"]]}
        return json.dumps(body, ensure_ascii=False, indent=1).encode("utf-8")

    if fmt == "xlsx":
        sheets = [(t("ex.xl.settings"), [t("ex.xl.setting"), t("ex.xl.value")], _settings_rows(result, lang))]
        sheets.append((t("ex.xl.findings"), [t(f"ex.xl.{k}") for k in ("section", "priority", "title", "finding", "why", "action", "evidence")],
                       [[render(s["title"], lang), t(f"priority.{i['priority']}"), *(ins_text(i)[k] for k in ("title", "finding", "why", "action")),
                         " | ".join(ins_text(i)["evidence"])] for s in result["sections"] for i in s["insights"]]))
        n = 0
        for s in result["sections"]:
            for tbl in s["tables"]:
                n += 1
                header, rows = _table_grid(tbl, lang, excel=True)
                sheets.append((f"{n} {render(tbl['title'], lang)}", header, rows))
                if tbl.get("kind") == "crosstab":
                    m = tbl["matrix"]
                    pct_rows = [[r[0]] + m["pct_col"][i] for i, r in enumerate(rows[:len(m["pct_col"])])]
                    n += 1
                    sheets.append((f"{n} {t('ex.xl.pct_within', column=s['columns'][1])}", header[:-1], pct_rows))
        chart_rows = []
        for s in result["sections"]:
            for ch in s["charts"]:
                chart_rows += [[render(ch["title"], lang), *r] for r in _chart_rows(ch, lang)]
        sheets.append((t("ex.xl.chart_data"), [t(f"ex.xl.{k}") for k in ("chart", "series", "label", "value", "pct")], chart_rows))
        if result["notes"]:
            sheets.append((t("ex.xl.notes"), [t("ex.xl.note")], [[render(x, lang)] for x in result["notes"]]))
        return workbook_bytes(sheets, rtl=(lang == "ar"))

    if fmt == "html":
        import html as _html
        E = _html.escape

        def h(m):
            return render(m, lang, html=True)

        parts = [f"<h1>{E(t('ex.report.title'))}</h1>",
                 f'<p class="muted"><bdi>{E(ds.filename)}</bdi> · {E(t("ex.report.generated", date=generated))} · '
                 f'{E(t("ex.kpi.records_of", count=result["records"]["filtered"], total=result["records"]["total"]))}</p>',
                 f"<h2>{E(t('ex.report.settings'))}</h2>", _table([t("ex.xl.setting"), t("ex.xl.value")], _settings_rows(result, lang), lang)]
        if result["kpis"]:
            parts.append('<div class="kpis">' + "".join(
                f'<div class="kpi">{h(k["label"])}<b>{h(k["value"]) if isinstance(k["value"], dict) else E(str(_kpi_num(k["value"], lang)))}</b>'
                f'<span class="muted">{h(k.get("sub"))}</span></div>' for k in result["kpis"]) + "</div>")
        by_id = {i["id"]: i for i in insights}
        if result["highlights"]:
            parts.append(f"<h2>{E(t('ex.report.highlights'))}</h2>" + "".join(_ins_html(by_id[x], lang) for x in result["highlights"] if x in by_id))
        if result["notes"]:
            parts.append(f"<h2>{E(t('ex.report.notes'))}</h2><ul>" + "".join(f"<li>{h(x)}</li>" for x in result["notes"]) + "</ul>")
        for s in result["sections"]:
            parts.append(f"<h2>{h(s['title'])}</h2><p class=\"explain\">{h(s.get('intro'))}</p>")
            charts = "".join(f'<div class="card"><h3>{h(ch["title"])}</h3>{_chart_html(ch, lang)}'
                             f'<p class="explain">{h(ch["explanation"])}</p>'
                             + (f'<p class="muted">{h(ch["note"])}</p>' if ch.get("note") else "") + "</div>" for ch in s["charts"])
            parts.append(f'<div class="grid">{charts}</div>')
            for tbl in s["tables"]:
                header, rows = _table_grid(tbl, lang, excel=False)
                parts.append(f"<h3>{h(tbl['title'])}</h3>" + _table(header, rows, lang)
                             + (f'<p class="muted">{h(tbl["note"])}</p>' if tbl.get("note") else ""))
            parts += [_ins_html(i, lang) for i in s["insights"]]
            parts += [f'<p class="muted">{h(x)}</p>' for x in s.get("notes", [])]
        parts.append(f'<div class="footer">{E(t("ex.report.footer"))}</div>')
        return _doc(t("ex.report.title"), "".join(parts), lang).encode("utf-8")
    raise ExportError(M("err.export_format", formats=["html", "xlsx", "json"]))


def _heat(ch):
    """Heatmap data of a chart: the cross-section heatmap or the split main chart."""
    if ch.get("heat"):
        return ch["heat"]
    return {"rows": ch["rows"], "cols": ch["labels"], "matrix": ch["matrix"]}


def _kpi_num(v, lang):
    from .i18n import format_number
    return format_number(v, lang) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def _ins_html(i, lang):
    import html as _html
    E = _html.escape

    def h(m):
        return render(m, lang, html=True)

    ev = "".join(f"<li>{h(e)}</li>" for e in i["evidence"])
    lim = "".join(f"<li>{h(x)}</li>" for x in i["limitations"])
    cols = "".join(f'<span class="tag"><bdi>{E(c)}</bdi></span>' for c in i["columns"])
    t = lambda key: render(K(key), lang)  # noqa: E731
    return (f'<div class="ins {i["priority"]}"><h4>{h(i["title"])}<span class="badge {i["priority"]}">{E(t("priority." + i["priority"]))}</span></h4><dl>'
            f'<dt>{E(t("card.finding"))}</dt><dd>{h(i["finding"])}</dd><dt>{E(t("card.why"))}</dt><dd>{h(i["why"])}</dd>'
            f'<dt>{E(t("card.action"))}</dt><dd>{h(i["action"])}</dd><dt>{E(t("card.evidence"))}</dt><dd><ul>{ev}</ul>{cols}</dd>'
            + (f'<dt>{E(t("card.limitations"))}</dt><dd><ul>{lim}</ul></dd>' if lim else "") + "</dl></div>")


def _chart_html(ch, lang):
    """Static rendering of a chart for the HTML report (SVG for single series, tables otherwise)."""
    import html as _html
    from .i18n import format_number, format_period
    from .reporting import _table, svg_bars, svg_box, svg_line
    E = _html.escape
    labels = ch["labels"]
    if ch["type"] in ("line",) and ch.get("freq"):
        labels = [format_period(p, ch["freq"], lang) for p in labels]
    labels = [f"{format_number(x[0], lang)} – {format_number(x[1], lang)}" if isinstance(x, list) else _txt(x, lang) for x in labels]
    title = render(ch["title"], lang)
    if ch["type"] == "heatmap" or (ch.get("heat") and len(ch["series"]) > 1):
        h = _heat(ch)
        m = h["matrix"]
        peak = max([max((v or 0) for v in r) for r in m["counts"]] + [1])
        head = "".join(f"<th><bdi>{E(str(_txt(c, lang)))}</bdi></th>" for c in h["cols"])
        body = "".join("<tr><th><bdi>" + E(str(_txt(r, lang))) + "</bdi></th>" + "".join(
            f'<td style="background:rgba(108,99,255,{0.08 + 0.7 * (v or 0) / peak:.2f})">'
            f"{E(format_number(v, lang)) if v is not None else '—'}</td>" for j, v in enumerate(row)) + "</tr>"
            for i, (r, row) in enumerate(zip(h["rows"], m["counts"])))
        return f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>"
    if ch["type"] == "box":
        st = ch["stats"]
        return svg_box(ch["x_label"], st["box"], st["min"], st["max"])
    if ch["type"] == "scatter":
        return f'<p class="muted">{E(render(ch["explanation"], lang))}</p>'
    if len(ch["series"]) == 1:
        values = ch["series"][0]["values"]
        return (svg_line if ch["type"] == "line" else svg_bars)(labels, values, title, lang=lang)
    header = [_txt(ch["x_label"], lang) or ""] + [str(_txt(s["name"], lang)) for s in ch["series"]]
    rows = [[labels[j]] + [(f"{format_number(s['values'][j], lang)} ({format_number(s['pct'][j], lang)}%)"
                            if s.get("pct") and s["pct"][j] is not None else s["values"][j]) for s in ch["series"]]
            for j in range(len(labels))]
    return _table(header, rows, lang)
