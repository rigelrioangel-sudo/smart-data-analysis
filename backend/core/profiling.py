"""Data profiling: per-column statistics and data-quality issues."""
import re

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .inference import case_variants, infer_frame, is_sensitive
from .i18n import K, M, render
from .utils import display_series, display_value, mask_value, name_tokens, rnd

NON_NEGATIVE_TOKENS = {
    "age", "price", "cost", "amount", "qty", "quantity", "count", "units", "salary", "revenue",
    "sales", "weight", "height", "distance", "duration", "income", "population", "stock",
    "inventory", "fee", "fees", "tax", "length", "width", "area", "volume", "visits", "views",
}
PERCENT_TOKENS = {"pct", "percent", "percentage"}
BIRTH_TOKENS = {"birth", "dob", "born", "birthdate", "birthday"}


def cardinality_label(n_unique, n):
    if n == 0:
        return "empty"
    if n_unique <= 1:
        return "constant"
    if n_unique == n:
        return "unique"
    ratio = n_unique / n
    if n_unique <= 10:
        return "low"
    if ratio < 0.5:
        return "medium"
    return "high"


def histogram(values: np.ndarray, max_bins=20):
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    n_unique = len(np.unique(values))
    bins = int(min(max_bins, max(1, n_unique), max(1, int(np.ceil(np.log2(len(values)) + 1)) * 2)))
    if values.min() == values.max():
        edges = np.array([values.min() - 0.5, values.max() + 0.5])
        counts = np.array([len(values)])
    else:
        counts, edges = np.histogram(values, bins=bins)
    return {"edges": [rnd(e, 6) for e in edges], "counts": [int(c) for c in counts]}


def numeric_stats(series: pd.Series):
    s = series.dropna().astype(float)
    s = s[np.isfinite(s)]
    if s.empty:
        return None
    q1, med, q3 = (float(x) for x in s.quantile([0.25, 0.5, 0.75]))
    iqr = q3 - q1
    mild_lo, mild_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    ext_lo, ext_hi = q1 - 3 * iqr, q3 + 3 * iqr
    inside = s[(s >= mild_lo) & (s <= mild_hi)]
    mild = s[(s < mild_lo) | (s > mild_hi)] if iqr > 0 else s.iloc[0:0]
    extreme = s[(s < ext_lo) | (s > ext_hi)] if iqr > 0 else s.iloc[0:0]
    arr = s.to_numpy()
    return {
        "count": int(len(s)),
        "mean": rnd(s.mean(), 6), "median": rnd(med, 6), "std": rnd(s.std(), 6) if len(s) > 1 else None,
        "min": rnd(s.min(), 6), "max": rnd(s.max(), 6), "q1": rnd(q1, 6), "q3": rnd(q3, 6),
        "iqr": rnd(iqr, 6), "sum": rnd(s.sum(), 6),
        "skew": rnd(s.skew(), 4) if len(s) > 2 else None,
        "zeros": int((s == 0).sum()), "negatives": int((s < 0).sum()),
        "outliers_mild": int(len(mild)), "outliers_extreme": int(len(extreme)),
        "outlier_bounds": {"mild": [rnd(mild_lo, 6), rnd(mild_hi, 6)], "extreme": [rnd(ext_lo, 6), rnd(ext_hi, 6)]},
        "box": {
            "whisker_low": rnd(inside.min() if len(inside) else s.min(), 6),
            "q1": rnd(q1, 6), "median": rnd(med, 6), "q3": rnd(q3, 6),
            "whisker_high": rnd(inside.max() if len(inside) else s.max(), 6),
            "outliers": [rnd(v, 6) for v in np.sort(mild.to_numpy())[:: max(1, len(mild) // 40)][:40]],
        },
        "histogram": histogram(arr),
    }


def impossible_masks(name: str, series: pd.Series) -> list:
    """Domain-plausibility rules as (rule, description, mask) for a typed numeric or datetime column."""
    tokens = set(name_tokens(name))
    rules = []
    if ptypes.is_datetime64_any_dtype(series):
        s = series
        if tokens & BIRTH_TOKENS:
            rules.append(("future_birth_date", "Birth dates in the future", s > pd.Timestamp.now()))
        rules.append(("implausible_year", "Years before 1900 or after 2100",
                      (s.dt.year < 1900) | (s.dt.year > 2100)))
    elif ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series):
        s = series.astype(float)
        if tokens & NON_NEGATIVE_TOKENS:
            rules.append(("negative", "Negative values in a column named like a non-negative quantity", s < 0))
        if "age" in tokens:
            rules.append(("age_over_120", "Ages above 120", s > 120))
        if tokens & PERCENT_TOKENS:
            rules.append(("percent_out_of_range", "Percentages outside 0–100", (s < 0) | (s > 100)))
        rules.append(("infinite", "Infinite values", pd.Series(np.isinf(s), index=s.index)))
    return [(r, d, m.fillna(False).astype(bool)) for r, d, m in rules if m.fillna(False).any()]


def impossible_summary(name: str, series: pd.Series) -> list:
    return [{"rule": r, "description": d, "label": K(f"rule.{r}"), "count": int(m.sum()),
             "examples": [display_value(v) for v in series[m].head(5)]}
            for r, d, m in impossible_masks(name, series)]


def _issue(severity, code, column, msg):
    """An issue with a translatable message plus its English text (for backward compatibility)."""
    return {"severity": severity, "code": code, "column": column, "msg": msg, "message": render(msg, "en")}


def _top_values(series: pd.Series, n_valid: int, limit=10):
    vals = series.dropna()
    if vals.empty:
        return []
    vc = vals.value_counts().head(limit)
    return [{"value": display_value(k), "count": int(v), "pct": rnd(v / n_valid * 100, 2) if n_valid else None}
            for k, v in vc.items()]


def profile_column(name: str, raw: pd.Series, inf) -> dict:
    n = inf.n_total
    parsed = inf.parsed
    col = {
        "name": name, "kind": inf.kind, "role": inf.role, "dtype": str(raw.dtype),
        "count": n, "non_missing": inf.n_valid, "null": inf.n_null, "empty": inf.n_empty,
        "markers": inf.marker_values, "marker_count": inf.n_marker,
        "missing_total": inf.missing_total,
        "missing_pct": rnd(inf.missing_total / n * 100, 2) if n else 0.0,
        "unique": inf.n_unique,
        "unique_pct": rnd(inf.n_unique / inf.n_valid * 100, 2) if inf.n_valid else None,
        "cardinality": cardinality_label(inf.n_unique, inf.n_parsed if inf.kind in ("numeric", "datetime", "boolean") else inf.n_valid),
        "parse_rate": rnd(inf.parse_rate, 4),
        "invalid_count": inf.n_invalid, "invalid_examples": [{"value": k, "count": v} for k, v in inf.invalid_values.items()],
        "candidate_kinds": inf.candidate_kinds,
        "formats": inf.formats, "date_format": inf.date_format, "date_ambiguous": inf.date_ambiguous,
        "date_alternatives": inf.date_alternatives,
        "whitespace_values": inf.n_whitespace, "leading_zeros": inf.leading_zeros,
        "case_variants": [], "top_values": [], "dominant": None,
        "numeric": None, "datetime": None, "impossible": [], "issues": [],
    }
    if inf.kind in ("categorical", "text", "boolean") or (inf.kind == "numeric" and inf.n_unique <= 20):
        col["top_values"] = _top_values(parsed, inf.n_parsed or inf.n_valid)
        if col["top_values"]:
            col["dominant"] = col["top_values"][0]
    if inf.kind == "categorical":
        col["case_variants"] = case_variants(raw)
    if inf.kind == "numeric" and inf.role != "identifier":
        # Identifiers and codes that happen to be numeric (ids, phone numbers, slots) get no
        # mean / sum / histogram: those numbers are labels, not quantities
        col["numeric"] = numeric_stats(parsed)
        col["impossible"] = impossible_summary(name, raw if inf.typed_source else parsed)
    if inf.kind == "datetime":
        s = parsed.dropna()
        if len(s):
            col["datetime"] = {
                "min": display_value(s.min()), "max": display_value(s.max()),
                "range_days": int((s.max() - s.min()).days), "invalid_count": inf.n_invalid,
                "has_time": bool(((s.dt.hour != 0) | (s.dt.minute != 0) | (s.dt.second != 0)).any()),
            }
        col["impossible"] = impossible_summary(name, parsed)
    col["sensitive"] = is_sensitive(inf)
    if col["sensitive"]:
        _mask_column_values(col)
    col["issues"] = column_issues(col)
    return col


def _mask_column_values(col):
    """Personal identifiers: every value shown in the profile, issues, plan and reports is masked."""
    for tv in col["top_values"]:
        tv["value"] = mask_value(tv["value"])
    col["dominant"] = col["top_values"][0] if col["top_values"] else None
    col["invalid_examples"] = [{"value": mask_value(e["value"]), "count": e["count"]} for e in col["invalid_examples"]]
    col["case_variants"] = [{"canonical": mask_value(g["canonical"]),
                             "variants": [{"value": mask_value(v["value"]), "count": v["count"]} for v in g["variants"]]}
                            for g in col["case_variants"]]
    for rule in col["impossible"]:
        rule["examples"] = [mask_value(e) for e in rule["examples"]]


def column_issues(c) -> list:
    issues = []

    def add(severity, code, msg):
        issues.append(_issue(severity, code, c["name"], msg))

    if c["kind"] == "empty":
        add("high", "empty_column", M("issue.empty_column"))
        return issues
    if c["role"] == "constant":
        add("medium", "constant", M("issue.constant", value=c["dominant"]["value"] if c["dominant"] else "—"))
    if c["missing_pct"] and c["missing_pct"] >= 20:
        add("high" if c["missing_pct"] >= 50 else "medium", "missing",
            M("issue.missing_high", pct=c["missing_pct"], count=c["missing_total"], total=c["count"]))
    elif c["missing_total"]:
        add("low", "missing", M("issue.missing_low", count=c["missing_total"], pct=c["missing_pct"]))
    if c["marker_count"]:
        add("medium", "missing_markers", M("issue.missing_markers", count=c["marker_count"], markers=list(c["markers"])))
    if c["empty"]:
        add("low", "empty_strings", M("issue.empty_strings", count=c["empty"]))
    if c["invalid_count"]:
        pct = round(c["invalid_count"] / max(1, c["non_missing"]) * 100, 1)
        add("high", "mixed_types", M("issue.mixed_types", count=c["invalid_count"], pct=pct, kind=K(f"kind_value.{c['kind']}"),
                                     examples=[e["value"] for e in c["invalid_examples"][:3]]))
    if c["whitespace_values"]:
        add("low", "whitespace", M("issue.whitespace", count=c["whitespace_values"]))
    if c["case_variants"]:
        ex = c["case_variants"][0]
        add("medium", "inconsistent_labels", M("issue.inconsistent_labels", count=len(c["case_variants"]),
                                               examples=[v["value"] for v in ex["variants"][:3]]))
    if c["date_ambiguous"]:
        add("medium", "ambiguous_dates", M("issue.ambiguous_dates"))
    fmts = sorted(k for k in c["formats"] if k in ("currency", "percent", "thousands"))
    if fmts:
        add("low", "formatted_numbers", M("issue.formatted_numbers", formats=[K(f"fmt.{f}") for f in fmts]))
    if c["leading_zeros"]:
        add("info", "leading_zeros", M("issue.leading_zeros"))
    for rule in c["impossible"]:
        add("high", "impossible_values", M("issue.impossible_values", count=rule["count"], rule=rule["label"],
                                           examples=[str(e) for e in rule["examples"][:3]]))
    num = c["numeric"]
    if num and c["role"] == "measure":
        if num["outliers_extreme"]:
            add("medium", "outliers", M("issue.outliers_extreme", count=num["outliers_extreme"],
                                        low=num["outlier_bounds"]["extreme"][0], high=num["outlier_bounds"]["extreme"][1]))
        elif num["outliers_mild"]:
            add("info", "outliers", M("issue.outliers_mild", count=num["outliers_mild"]))
        if num["skew"] is not None and abs(num["skew"]) > 1 and num["count"] >= 20:
            add("info", "skewed", M("issue.skewed", skew=num["skew"], median=num["median"]))
    if c["kind"] == "text" and c["role"] == "text":
        add("info", "high_cardinality", M("issue.high_cardinality", count=c["unique"]))
    return issues


def _name_issue(column, code, msg):
    return {"column": column, "code": code, "msg": msg, "message": render(msg, "en")}


def column_name_issues(columns) -> list:
    issues = []
    names = list(columns)
    base_names = set(names)
    styles = set()
    for name in names:
        if name != name.strip():
            issues.append(_name_issue(name, "name_whitespace", M("issue.name_whitespace")))
        if re.match(r"^Unnamed: \d+$", name) or not name.strip():
            issues.append(_name_issue(name, "name_missing", M("issue.name_missing")))
        m = re.match(r"^(.*)\.(\d+)$", name)
        if m and m.group(1) in base_names:
            issues.append(_name_issue(name, "name_duplicate", M("issue.name_duplicate", base=m.group(1), name=name)))
        stripped = name.strip()
        if re.search(r"[^\w\s.\-]", stripped):
            issues.append(_name_issue(name, "name_special_chars", M("issue.name_special_chars")))
        if " " in stripped:
            styles.add("spaces")
        elif "_" in stripped:
            styles.add("snake_case")
        elif re.search(r"[a-z][A-Z]", stripped):
            styles.add("camelCase")
        elif stripped.isupper():
            styles.add("UPPER")
        else:
            styles.add("lower")
    if len(styles - {"lower"}) > 1:
        issues.append(_name_issue(None, "name_inconsistent_style",
                                  M("issue.name_inconsistent_style", styles=[K(f"style.{s}") for s in sorted(styles)])))
    return issues


def _normalized_frame(df: pd.DataFrame, inferences: dict) -> pd.DataFrame:
    """Values with whitespace trimmed and missing markers unified (used to estimate duplicates)."""
    return pd.DataFrame({col: inferences[col].normalized for col in df.columns}, index=df.index)


def row_hashes(df: pd.DataFrame) -> pd.Series:
    """Stable per-row hash; missing values of any kind hash identically."""
    cols = {}
    for c in df.columns:
        s = df[c]
        cols[c] = s.astype(object).where(s.notna(), "\x00NA").astype(str)
    return pd.util.hash_pandas_object(pd.DataFrame(cols, index=df.index), index=False)


def duplicate_info(df: pd.DataFrame):
    """(number of exact duplicate rows beyond the first occurrence, example groups of 1-based row numbers)."""
    if df.empty or df.shape[1] == 0:
        return 0, []
    keys = row_hashes(df)
    n = int(keys.duplicated(keep="first").sum())
    examples = []
    if n:
        dup_keys = keys[keys.duplicated(keep=False)]
        for _, idx in dup_keys.groupby(dup_keys, sort=False).groups.items():
            examples.append([int(i) + 1 for i in list(idx)[:10]])
            if len(examples) >= 5:
                break
    return n, examples


def key_duplicates(df: pd.DataFrame, columns: list):
    out = []
    for c in columns:
        # Personal identifiers (a user, a phone number) legitimately repeat across records,
        # so repeats are not treated as duplicate keys
        if c["role"] != "identifier" or c.get("sensitive"):
            continue
        s = df[c["name"]].dropna()
        if s.empty:
            continue
        vc = display_series(s).value_counts()
        dups = vc[vc > 1]
        ratio = len(vc) / len(s)
        # Mostly-unique identifiers that repeat look like primary keys with duplicate records;
        # heavily repeated ones (ratio < 0.5) are more likely foreign keys and are not flagged.
        if len(dups) and ratio >= 0.5:
            out.append({
                "column": c["name"], "duplicated_values": int(len(dups)), "affected_rows": int(dups.sum()),
                "extra_rows": int(dups.sum() - len(dups)), "unique_ratio": rnd(ratio, 4),
                "examples": [{"value": k, "count": int(v)} for k, v in dups.head(5).items()],
            })
    return out


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


def profile_dataframe(df: pd.DataFrame, inferences: dict = None) -> dict:
    inferences = inferences or infer_frame(df)
    columns = [profile_column(col, df[col], inferences[col]) for col in df.columns]
    n_rows, n_cols = df.shape
    raw_dups, raw_examples = duplicate_info(df)
    norm = _normalized_frame(df, inferences)
    norm_dups, norm_examples = duplicate_info(norm)
    keys = key_duplicates(norm, columns)

    issues = []
    for c in columns:
        issues.extend(c["issues"])
    name_issues = column_name_issues(df.columns)
    for ni in name_issues:
        issues.append(_issue("low", ni["code"], ni["column"], ni["msg"]))
    if raw_dups:
        issues.append(_issue("high", "duplicate_rows", None, M("issue.duplicate_rows", count=raw_dups)))
    if norm_dups > raw_dups:
        issues.append(_issue("medium", "near_duplicate_rows", None, M("issue.near_duplicate_rows", count=norm_dups - raw_dups)))
    empty_rows = int(norm.isna().all(axis=1).sum()) if n_rows and n_cols else 0
    if empty_rows:
        issues.append(_issue("medium", "empty_rows", None, M("issue.empty_rows", count=empty_rows)))
    for k in keys:
        issues.append(_issue("high", "key_duplicates", k["column"], M(
            "issue.key_duplicates", column=k["column"], count=k["duplicated_values"], extra=k["extra_rows"])))
    if 0 < n_rows < 30:
        issues.append(_issue("info", "small_sample", None, M("issue.small_sample", count=int(n_rows))))
    issues.sort(key=lambda i: (SEVERITY_ORDER[i["severity"]], str(i["column"])))

    missing_cells = sum(c["null"] + c["empty"] for c in columns)
    marker_cells = sum(c["marker_count"] for c in columns)
    type_counts, role_counts = {}, {}
    for c in columns:
        type_counts[c["kind"]] = type_counts.get(c["kind"], 0) + 1
        role_counts[c["role"]] = role_counts.get(c["role"], 0) + 1
    cells = n_rows * n_cols
    return {
        "row_count": int(n_rows), "column_count": int(n_cols), "cell_count": int(cells),
        "missing_cells": int(missing_cells), "marker_cells": int(marker_cells),
        "missing_pct": rnd((missing_cells + marker_cells) / cells * 100, 2) if cells else 0.0,
        "invalid_cells": int(sum(c["invalid_count"] for c in columns)),
        "empty_rows": empty_rows,
        "exact_duplicate_rows": raw_dups, "duplicate_examples": raw_examples,
        "normalized_duplicate_rows": norm_dups, "normalized_duplicate_examples": norm_examples,
        "key_duplicates": keys, "column_name_issues": name_issues,
        "constant_columns": [c["name"] for c in columns if c["role"] == "constant"],
        "empty_columns": [c["name"] for c in columns if c["kind"] == "empty"],
        "type_counts": type_counts, "role_counts": role_counts,
        "issue_counts": {s: sum(1 for i in issues if i["severity"] == s) for s in SEVERITY_ORDER},
        "columns": columns, "issues": issues,
    }
