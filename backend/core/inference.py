"""Column type and role inference.

Works on both raw columns (object dtype, original strings / native values)
and already-typed columns from the cleaned dataset, so the same profiling
code describes the data before and after cleaning.

Inference runs on unique values (weighted by their counts) to stay fast on
large files with repeated values.
"""
import datetime as dt
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .utils import name_tokens

MISSING_MARKERS = {
    "na", "n/a", "n.a.", "nan", "null", "none", "nil", "-", "--", "?",
    "missing", "#n/a", "#na", "undefined", "not available", "#null!",
}
BOOL_TRUE = {"true", "yes", "y", "t"}
BOOL_FALSE = {"false", "no", "n", "f"}

_PLAIN_NUM = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_CURRENCY = re.compile(r"^([+-]?)\s*[$€£¥]\s*(.+)$|^(.+?)\s*[$€£¥]$")
_PERCENT = re.compile(r"^(.+?)\s*%$")
_LEADING_ZERO = re.compile(r"^[+-]?0\d+$")
_DATE_LIKE = re.compile(
    r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:[AaPp][Mm])?(?:Z|[+-]\d{2}:?\d{2})?)?$"
    r"|^\d{1,2}[ -][A-Za-z]{3,9}[ -,]+\d{2,4}$|^[A-Za-z]{3,9}[ .-]\d{1,2},?[ -]\d{2,4}$"
)

# Candidate date formats; "ISO8601" also covers timestamps with time zones
DATE_FORMATS = [
    "ISO8601", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y",
    "%Y/%m/%d", "%d/%m/%y", "%m/%d/%y", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M",
    "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%d %b %Y", "%d-%b-%Y", "%d %B %Y",
    "%b %d %Y", "%b %d, %Y", "%B %d, %Y", "%B %d %Y", "%d-%b-%y",
]
# Formats that read the same string differently (day/month order)
_AMBIGUOUS_PAIRS = {
    "%d/%m/%Y": "%m/%d/%Y", "%m/%d/%Y": "%d/%m/%Y", "%d-%m-%Y": "%m-%d-%Y", "%m-%d-%Y": "%d-%m-%Y",
    "%d/%m/%y": "%m/%d/%y", "%m/%d/%y": "%d/%m/%y", "%d/%m/%Y %H:%M": "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M": "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S": "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S": "%d/%m/%Y %H:%M:%S",
}

_ID_TOKENS = {"id", "uuid", "guid", "key", "code", "sku", "no", "nr", "number", "zip", "zipcode",
              "postcode", "postal", "phone", "tel", "isbn", "ean", "upc", "iban", "ssn",
              "mobile", "msisdn", "imei", "imsi", "serial", "sn", "mac",
              "user", "username", "userid", "subscriber", "login", "account"}
# Long digit strings (phone numbers, national/subscriber ids): identifiers whatever the column is called
_LONG_DIGITS = re.compile(r"^\+?\d{9,}$")
# Network-position fields (slot, PON port, FDT/FAT box, OLT, ring …). Their numbers are
# positions, not quantities; they are treated as codes when several appear together.
_NETWORK_CODE_TOKENS = {"slot", "pon", "port", "fdt", "fat", "olt", "ring", "onu", "ont", "vlan",
                        "splitter", "rack", "shelf"}


_SENSITIVE_TOKENS = {"user", "username", "userid", "msisdn", "mobile", "phone", "tel", "telephone", "email", "mail",
                     "subscriber", "customer", "client", "passport", "iban", "imsi", "imei", "address", "national", "nid"}
_SENSITIVE_JOINED = {"clientname", "customername", "fullname", "firstname", "lastname", "surname", "username",
                     "subscribername", "contactname", "name", "personname"}


def is_sensitive(inf) -> bool:
    """Personal identifiers whose values must not appear in charts, findings or reports:
    phone-like numbers, or identifier / free-text columns named like a person or contact."""
    if inf.phone_like:
        return True
    if inf.role not in ("identifier", "text"):
        return False
    tokens = name_tokens(inf.name)
    return bool(set(tokens) & _SENSITIVE_TOKENS) or "".join(tokens) in _SENSITIVE_JOINED


def is_id_like_name(name: str) -> bool:
    tokens = name_tokens(name)
    if not tokens:
        return False
    return tokens[-1] in _ID_TOKENS or (tokens[0] in {"id", "uuid", "guid"})


def parse_number(v):
    """Return (float_value, format) or (None, None) when v is not numeric."""
    if isinstance(v, (bool, np.bool_)):
        return None, None
    if isinstance(v, (int, np.integer)):
        return float(v), "native"
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None, None
        return f, "native"
    if not isinstance(v, str):
        return None, None
    s = v.strip()
    if _PLAIN_NUM.match(s):
        return float(s), "plain"
    if _THOUSANDS.match(s):
        return float(s.replace(",", "")), "thousands"
    m = _PERCENT.match(s)
    if m:
        inner, fmt = parse_number(m.group(1))
        if inner is not None and fmt in ("plain", "thousands"):
            return inner, "percent"
        return None, None
    m = _CURRENCY.match(s)
    if m:
        sign = m.group(1) or ""
        body = (m.group(2) if m.group(2) is not None else m.group(3)).strip()
        inner, fmt = parse_number(body)
        if inner is not None and fmt in ("plain", "thousands"):
            return (-inner if sign == "-" else inner), "currency"
    return None, None


def parse_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in BOOL_TRUE:
            return True
        if s in BOOL_FALSE:
            return False
    return None


def _to_naive(ts_series: pd.Series) -> pd.Series:
    if getattr(ts_series.dt, "tz", None) is not None:
        return ts_series.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts_series


def parse_dates(values: pd.Series, fmt: str) -> pd.Series:
    """Parse a Series of strings with one format; failures become NaT."""
    try:
        parsed = pd.to_datetime(values, format=fmt, errors="coerce", utc=(fmt == "ISO8601"))
    except (ValueError, TypeError):
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if fmt == "ISO8601":
        parsed = parsed.dt.tz_localize(None)
    return _to_naive(parsed)


@dataclass
class ColumnInference:
    name: str
    kind: str = "empty"            # numeric | boolean | datetime | categorical | text | empty
    role: str = "empty"            # measure | identifier | category | flag | date | text | constant | empty
    typed_source: bool = False     # True when the column already had a real dtype
    n_total: int = 0
    n_null: int = 0
    n_empty: int = 0
    n_marker: int = 0
    marker_values: dict = field(default_factory=dict)
    n_valid: int = 0               # non-missing, non-empty, non-marker values
    n_parsed: int = 0              # valid values that parse as `kind`
    n_invalid: int = 0
    invalid_values: dict = field(default_factory=dict)
    n_whitespace: int = 0
    formats: dict = field(default_factory=dict)   # numeric format -> count
    date_format: str = None
    date_ambiguous: bool = False
    date_alternatives: list = field(default_factory=list)
    leading_zeros: int = 0
    phone_like: bool = False       # values are long digit strings (phone / subscriber / national ids)
    n_unique: int = 0
    integer_valued: bool = False
    candidate_kinds: dict = field(default_factory=dict)  # share of valid values parseable per kind
    parsed: pd.Series = None       # typed values aligned to the column index (NaN/NaT/NA for missing+invalid)
    normalized: pd.Series = None   # original values trimmed, with blanks/markers as missing

    @property
    def parse_rate(self):
        return self.n_parsed / self.n_valid if self.n_valid else None

    @property
    def missing_total(self):
        return self.n_null + self.n_empty + self.n_marker


def _top(counter: dict, n=10):
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[:n])


_AMOUNT_TOKENS = {"amount", "price", "cost", "revenue", "sales", "total", "balance", "salary", "income", "value",
                  "fee", "fees", "payment", "paid", "budget", "profit", "spend", "tax", "debt", "credit", "debit", "sum"}


def _phone_like(name: str, values: pd.Series) -> bool:
    """Long digit strings of (mostly) one length, e.g. 9647701234567 — phone / subscriber / national ids.
    Columns named like money amounts are never treated this way."""
    if set(name_tokens(name)) & _AMOUNT_TOKENS:
        return False
    u = pd.Series(pd.unique(values.dropna().to_numpy(dtype=object))[:2000], dtype=object)
    if u.empty:
        return False
    text = u.map(lambda v: str(int(v)) if isinstance(v, (int, float, np.integer, np.floating)) and float(v).is_integer()
                 else str(v).strip())
    if text.str.match(_LONG_DIGITS).mean() < 0.9:
        return False
    lengths = text.str.lstrip("+").str.len()
    return float(lengths.value_counts(normalize=True).iloc[0]) >= 0.8


def _assign_role(inf: ColumnInference, values: pd.Series):
    n = inf.n_parsed if inf.kind in ("numeric", "boolean", "datetime") else inf.n_valid
    if inf.kind == "empty" or n == 0:
        inf.role = "empty"
        return
    if inf.n_unique == 1:
        inf.role = "constant"
        return
    id_name = is_id_like_name(inf.name)
    ratio = inf.n_unique / n
    if inf.kind in ("numeric", "text", "categorical") and _phone_like(inf.name, values):
        # Phone-like / subscriber numbers: never a quantity, even when every value parses as a number
        inf.phone_like = True
        inf.role = "identifier"
        if inf.kind != "numeric":
            inf.kind = "text" if ratio > 0.5 else "categorical"
        return
    if inf.kind == "numeric":
        if id_name and inf.integer_valued:
            inf.role = "identifier"
        elif inf.integer_valued and ratio == 1 and n >= 20:
            diffs = np.diff(values.dropna().to_numpy(dtype=float))
            inf.role = "identifier" if len(diffs) and np.all(diffs == 1) else "measure"
        else:
            inf.role = "measure"
    elif inf.kind == "boolean":
        inf.role = "flag"
    elif inf.kind == "datetime":
        inf.role = "date"
    else:
        if id_name or inf.leading_zeros:
            inf.role = "identifier"
            inf.kind = "text" if ratio > 0.5 else "categorical"
        elif inf.n_unique <= 50 and (ratio <= 0.5 or n < 20):
            inf.role, inf.kind = "category", "categorical"
        elif ratio > 0.9 and n >= 20:
            sample = values.dropna().astype(str).head(500)
            short = sample.str.len().median() <= 24 and (sample.str.contains(r"\s").mean() < 0.2)
            inf.role, inf.kind = ("identifier" if short else "text"), "text"
        else:
            inf.role, inf.kind = "text", "text"


def _infer_typed(series: pd.Series, name: str) -> ColumnInference:
    inf = ColumnInference(name=name, typed_source=True, n_total=len(series))
    inf.normalized = series
    inf.n_null = int(series.isna().sum())
    inf.n_valid = inf.n_parsed = inf.n_total - inf.n_null
    if ptypes.is_bool_dtype(series):
        inf.kind = "boolean"
        inf.parsed = series.astype("boolean")
    elif ptypes.is_datetime64_any_dtype(series):
        inf.kind = "datetime"
        inf.parsed = _to_naive(series)
    else:
        inf.kind = "numeric"
        f = series.astype(float)
        inf.n_invalid = int(np.isinf(f).sum())
        if inf.n_invalid:
            inf.invalid_values = _top({str(v): int(c) for v, c in f[np.isinf(f)].value_counts().items()})
            inf.n_parsed -= inf.n_invalid
            f = f.replace([np.inf, -np.inf], np.nan)
        inf.parsed = f
        nonnull = f.dropna()
        inf.integer_valued = bool(len(nonnull) and np.all(np.mod(nonnull, 1) == 0))
    if inf.n_valid == 0:
        inf.kind = "empty"
    inf.n_unique = int(inf.parsed.nunique(dropna=True))
    _assign_role(inf, inf.parsed)
    return inf


def infer_column(series: pd.Series, name: str = None) -> ColumnInference:
    name = str(name if name is not None else series.name)
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype(object)
    if not ptypes.is_object_dtype(series) and not ptypes.is_string_dtype(series):
        return _infer_typed(series, name)

    inf = ColumnInference(name=name, n_total=len(series))
    is_null = series.isna()
    is_str = series.map(type).eq(str)
    stripped = series.where(is_str).str.strip()
    is_empty = is_str & (stripped == "")
    lowered = stripped.str.lower()
    is_marker = is_str & ~is_empty & lowered.isin(MISSING_MARKERS)
    inf.n_null = int(is_null.sum())
    inf.n_empty = int(is_empty.sum())
    inf.n_marker = int(is_marker.sum())
    if inf.n_marker:
        inf.marker_values = _top({str(k): int(v) for k, v in stripped[is_marker].value_counts().items()})
    inf.n_whitespace = int((is_str & ~is_empty & (stripped != series)).sum())

    valid_mask = ~is_null & ~is_empty & ~is_marker
    trimmed = series.where(~is_str, stripped)
    # Trimmed values with blanks/markers unified as missing (used for duplicate estimates)
    inf.normalized = trimmed.where(valid_mask, None)
    values = trimmed[valid_mask]
    inf.n_valid = int(valid_mask.sum())
    if inf.n_valid == 0:
        inf.kind = inf.role = "empty"
        inf.parsed = pd.Series([None] * len(series), index=series.index, dtype=object)
        return inf

    vc = values.value_counts(dropna=False)
    uniques = list(vc.index)
    counts = vc.to_numpy()
    inf.n_unique = len(uniques)
    total = counts.sum()

    # Numeric / boolean candidates
    num_map, fmt_counts, n_num, n_lead = {}, {}, 0, 0
    bool_map, n_bool = {}, 0
    date_like_idx, native_dates = [], {}
    for u, c in zip(uniques, counts):
        val, fmt = parse_number(u)
        if val is not None:
            num_map[u] = val
            n_num += c
            fmt_counts[fmt] = fmt_counts.get(fmt, 0) + int(c)
            if isinstance(u, str) and _LEADING_ZERO.match(u):
                n_lead += c
        b = parse_bool(u)
        if b is not None:
            bool_map[u] = b
            n_bool += c
        if isinstance(u, (pd.Timestamp, dt.datetime, dt.date)) and not isinstance(u, dt.time):
            native_dates[u] = pd.Timestamp(u)
        elif isinstance(u, str) and _DATE_LIKE.match(u):
            date_like_idx.append(u)

    # Date candidates: choose the format that parses the most values
    date_map, n_date, best_fmt, ambiguous, alternatives = dict(native_dates), 0, None, False, []
    n_date = sum(c for u, c in zip(uniques, counts) if u in native_dates)
    if date_like_idx:
        weights = dict(zip(uniques, counts))
        cand = pd.Series(date_like_idx, dtype=object)
        results = []
        for fmt in DATE_FORMATS:
            parsed = parse_dates(cand, fmt)
            ok = parsed.notna()
            score = int(sum(weights[u] for u in cand[ok]))
            if score:
                results.append((score, fmt, parsed))
        if results:
            results.sort(key=lambda r: -r[0])
            best_score, best_fmt, best_parsed = results[0]
            alternatives = [fmt for score, fmt, _ in results if score == best_score]
            alt = _AMBIGUOUS_PAIRS.get(best_fmt)
            ambiguous = alt in alternatives
            for u, p in zip(cand, best_parsed):
                if not pd.isna(p):
                    date_map[u] = p
            n_date += best_score

    inf.candidate_kinds = {k: round(v / total, 4) for k, v in
                           (("numeric", n_num), ("boolean", n_bool), ("datetime", n_date)) if v}
    inf.leading_zeros = int(n_lead)

    if n_bool == total and n_num < total:
        inf.kind, mapping, inf.n_parsed = "boolean", bool_map, int(n_bool)
    elif n_num / total >= 0.5 and n_lead == 0 and n_num >= n_date:
        inf.kind, mapping, inf.n_parsed = "numeric", num_map, int(n_num)
        inf.formats = fmt_counts
    elif n_date / total >= 0.5:
        inf.kind, mapping, inf.n_parsed = "datetime", date_map, int(n_date)
        inf.date_format = best_fmt or "native"
        inf.date_ambiguous = bool(ambiguous)
        inf.date_alternatives = alternatives if ambiguous else []
    else:
        inf.kind, mapping, inf.n_parsed = "text", None, int(total)

    if mapping is not None:
        invalid = {str(u): int(c) for u, c in zip(uniques, counts) if u not in mapping}
        inf.n_invalid = int(sum(invalid.values()))
        inf.invalid_values = _top(invalid)
        mapped = values.map(mapping)
        if inf.kind == "numeric":
            parsed = pd.Series(np.nan, index=series.index, dtype=float)
            parsed[valid_mask] = mapped.astype(float)
            nonnull = parsed.dropna()
            inf.integer_valued = bool(len(nonnull) and np.all(np.mod(nonnull, 1) == 0))
        elif inf.kind == "datetime":
            parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
            parsed[valid_mask] = pd.to_datetime(mapped, errors="coerce")
        else:
            parsed = pd.Series(pd.NA, index=series.index, dtype="boolean")
            parsed[valid_mask] = mapped.astype("boolean")
        inf.parsed = parsed
        inf.n_unique = int(parsed.nunique(dropna=True))
    else:
        parsed = pd.Series([None] * len(series), index=series.index, dtype=object)
        parsed[valid_mask] = values.astype(str)
        inf.parsed = parsed

    _assign_role(inf, inf.parsed)
    return inf


def _blank(v) -> bool:
    if v is None or v is pd.NA or v is pd.NaT:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return isinstance(v, str) and v.strip() == ""


def coerce_series(series: pd.Series, kind: str, date_format: str = None):
    """Convert a column to `kind`.

    Returns (typed_series, invalid_mask, formatted_mask): invalid values become
    missing in the typed series; formatted_mask marks values whose text had to
    be reinterpreted (currency, thousands separators, percent signs).
    """
    false = pd.Series(False, index=series.index)
    if kind == "numeric" and ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series):
        return series, false, false
    if kind == "datetime" and ptypes.is_datetime64_any_dtype(series):
        return series, false, false
    if kind == "boolean" and ptypes.is_bool_dtype(series):
        return series.astype("boolean"), false, false

    blank = series.map(_blank)
    values = series[~blank].map(lambda v: v.strip() if isinstance(v, str) else v)
    uniques = pd.unique(values.to_numpy(dtype=object)) if len(values) else []
    mapping, formatted = {}, set()
    if kind == "numeric":
        for u in uniques:
            val, fmt = parse_number(u)
            if val is not None:
                mapping[u] = val
                if fmt in ("thousands", "currency", "percent"):
                    formatted.add(u)
    elif kind == "boolean":
        for u in uniques:
            b = parse_bool(u)
            if b is not None:
                mapping[u] = b
    elif kind == "datetime":
        str_uniques = [u for u in uniques if isinstance(u, str)]
        for u in uniques:
            if isinstance(u, (pd.Timestamp, dt.datetime, dt.date)) and not isinstance(u, dt.time):
                mapping[u] = pd.Timestamp(u)
        if str_uniques:
            fmt = date_format if date_format and date_format != "native" else None
            if fmt is None:
                fmt = infer_column(pd.Series(str_uniques, dtype=object)).date_format or "ISO8601"
            if fmt != "native":
                parsed = parse_dates(pd.Series(str_uniques, dtype=object), fmt)
                for u, p in zip(str_uniques, parsed):
                    if not pd.isna(p):
                        mapping[u] = p
    else:
        raise ValueError(f"Unknown target type {kind!r}")

    mapped = values.map(lambda v: mapping.get(v))
    invalid = false.copy()
    invalid[values.index] = mapped.isna().to_numpy()
    fmt_mask = false.copy()
    fmt_mask[values.index] = values.isin(formatted).to_numpy() if formatted else False

    if kind == "numeric":
        out = pd.Series(np.nan, index=series.index, dtype=float)
        out[values.index] = mapped.astype(float).to_numpy()
        nonnull = out.dropna()
        if len(nonnull) and np.all(np.mod(nonnull, 1) == 0) and nonnull.abs().max() < 2 ** 53:
            out = out.astype("Int64")
    elif kind == "datetime":
        out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        out[values.index] = pd.to_datetime(mapped, errors="coerce").to_numpy()
    else:
        out = pd.Series(pd.NA, index=series.index, dtype="boolean")
        out[values.index] = mapped.astype("boolean").to_numpy()
    return out, invalid, fmt_mask


def infer_frame(df: pd.DataFrame) -> dict:
    out = {col: infer_column(df[col], col) for col in df.columns}
    # Several network-position columns together mark network inventory data: their
    # integer values are positions (slot 3, port 7), never quantities to average.
    codes = [c for c in df.columns if set(name_tokens(c)) & _NETWORK_CODE_TOKENS]
    if len(codes) >= 3:
        for c in codes:
            if out[c].kind == "numeric" and out[c].role == "measure" and out[c].integer_valued:
                out[c].role = "identifier"
    return out


def case_variants(series: pd.Series, limit=20):
    """Groups of labels that differ only by case / internal whitespace."""
    vals = series.dropna()
    vals = vals[vals.map(lambda v: isinstance(v, str))].str.strip()
    vals = vals[vals != ""]
    if vals.empty:
        return []
    vc = vals.value_counts()
    groups = {}
    for label, count in vc.items():
        key = re.sub(r"\s+", " ", label).casefold()
        groups.setdefault(key, []).append((label, int(count)))
    out = []
    for key, members in groups.items():
        if len(members) > 1:
            members.sort(key=lambda m: (-m[1], not m[0].istitle(), m[0]))
            out.append({"canonical": members[0][0], "variants": [{"value": v, "count": c} for v, c in members]})
    out.sort(key=lambda g: -sum(v["count"] for v in g["variants"]))
    return out[:limit]
