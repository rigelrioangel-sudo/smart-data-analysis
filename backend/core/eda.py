"""Exploratory data analysis on the cleaned dataset.

Method choice follows the variable types (from the cleaned profile):
- numeric × numeric:          Pearson r and Spearman ρ
- categorical × categorical:  Cramér's V (chi-square test)
- categorical × numeric:      correlation ratio η (Kruskal–Wallis test)

Identifiers, constants and free-text columns are excluded from relationships
because statistics on them are meaningless. All p-values within the analysis
are adjusted with Benjamini–Hochberg (q-values) to limit false discoveries.
"""
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .i18n import K, M, render
from .utils import benjamini_hochberg, display_series, name_tokens, rnd

MAX_NUMERIC = 25
MAX_CATEGORICAL = 15
MAX_LEVELS = 20
MIN_PAIR_N = 8
ADDITIVE_TOKENS = {"sales", "revenue", "amount", "units", "qty", "quantity", "count", "cost", "costs", "profit",
                   "total", "spend", "spent", "income", "orders", "visits", "views", "clicks", "volume"}


def strength_label(v):
    a = abs(v)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "weak"
    if a < 0.5:
        return "moderate"
    return "strong"


def is_additive(name):
    return bool(set(name_tokens(name)) & ADDITIVE_TOKENS)


def _excl(column, reason):
    return {"column": column, "reason_msg": reason, "reason": render(reason, "en")}


def _select_columns(profile):
    measures, categories, dates, excluded = [], [], [], []
    for c in profile["columns"]:
        if c["role"] == "identifier":
            excluded.append(_excl(c["name"], K("excl.identifier")))
        elif c["role"] in ("constant", "empty"):
            excluded.append(_excl(c["name"], K(f"excl.{c['role']}")))
        elif c["kind"] == "numeric" and c["role"] == "measure":
            measures.append(c)
        elif c["kind"] in ("categorical", "boolean") and 2 <= c["unique"] <= MAX_LEVELS:
            categories.append(c)
        elif c["kind"] == "datetime" and c["unique"] >= 2:
            dates.append(c)
        elif c["kind"] == "text" or (c["kind"] == "categorical" and c["unique"] > MAX_LEVELS):
            excluded.append(_excl(c["name"], M("excl.free_text" if c["kind"] == "text" else "excl.too_many_levels", count=c["unique"])))
        elif c["kind"] == "numeric":
            excluded.append(_excl(c["name"], K("excl.not_measure")))
    for c in profile["columns"]:
        if c["kind"] == "numeric" and c["role"] == "measure" and c["name"] not in [m["name"] for m in measures]:
            excluded.append(_excl(c["name"], K("excl.limit")))
    return measures[:MAX_NUMERIC], categories[:MAX_CATEGORICAL], dates[:3], excluded


def _cat_series(df, name):
    return display_series(df[name])


def numeric_pairs(df, measures):
    pairs = []
    names = [m["name"] for m in measures]
    data = {n: pd.to_numeric(df[n], errors="coerce").astype(float) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = data[a], data[b]
            ok = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            n = int(ok.sum())
            if n < MIN_PAIR_N or x[ok].nunique() < 2 or y[ok].nunique() < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pr, pp = stats.pearsonr(x[ok], y[ok])
                sr, sp = stats.spearmanr(x[ok], y[ok])
            pairs.append({"a": a, "b": b, "kind": "numeric-numeric", "method": "spearman",
                          "value": rnd(sr, 4), "p_value": float(sp), "pearson": rnd(pr, 4), "pearson_p": float(pp),
                          "n": n})
    return pairs


def categorical_pairs(df, categories):
    pairs = []
    names = [c["name"] for c in categories]
    data = {n: _cat_series(df, n) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ok = data[a].notna() & data[b].notna()
            n = int(ok.sum())
            if n < MIN_PAIR_N:
                continue
            table = pd.crosstab(data[a][ok], data[b][ok])
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue
            chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
            k = min(table.shape) - 1
            v = math.sqrt(chi2 / (n * k)) if k > 0 else 0.0
            low_expected = float((expected < 5).mean())
            pairs.append({"a": a, "b": b, "kind": "categorical-categorical", "method": "cramers_v",
                          "value": rnd(v, 4), "p_value": float(p), "n": n,
                          "caution": K("caution.low_expected") if low_expected > 0.2 else None})
    return pairs


def group_comparisons(df, categories, measures):
    results = []
    for c in categories:
        g = _cat_series(df, c["name"])
        for m in measures:
            y = pd.to_numeric(df[m["name"]], errors="coerce").astype(float)
            ok = g.notna() & y.notna() & np.isfinite(y)
            if ok.sum() < MIN_PAIR_N:
                continue
            groups = {lvl: y[ok & (g == lvl)] for lvl in g[ok].unique()}
            groups = {k: v for k, v in groups.items() if len(v) >= 2}
            if len(groups) < 2:
                continue
            all_vals = pd.concat(groups.values())
            if all_vals.nunique() < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                h, p = stats.kruskal(*groups.values())
            n, k = len(all_vals), len(groups)
            grand = all_vals.mean()
            ss_between = sum(len(v) * (v.mean() - grand) ** 2 for v in groups.values())
            ss_total = ((all_vals - grand) ** 2).sum()
            eta = math.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0
            eps2 = max(0.0, (h - k + 1) / (n - k)) if n > k else None
            summary = sorted(
                ({"level": lvl, "count": int(len(v)), "mean": rnd(v.mean(), 4), "median": rnd(v.median(), 4),
                  "std": rnd(v.std(), 4) if len(v) > 1 else None, "sum": rnd(v.sum(), 4)} for lvl, v in groups.items()),
                key=lambda r: -(r["median"] if r["median"] is not None else -math.inf))
            results.append({"group": c["name"], "measure": m["name"], "kind": "categorical-numeric", "method": "eta",
                            "value": rnd(eta, 4), "p_value": float(p), "n": int(n), "groups": summary,
                            "test": "Kruskal–Wallis", "statistic": rnd(h, 4), "effect_size": rnd(eps2, 4),
                            "additive": is_additive(m["name"])})
    return results


def _choose_freq(days):
    if days <= 60:
        return "D", "day"
    if days <= 366:
        return "W", "week"
    if days <= 366 * 5:
        return "M", "month"
    if days <= 366 * 20:
        return "Q", "quarter"
    return "Y", "year"


def trends(df, dates, measures):
    out = []
    for d in dates:
        s = pd.to_datetime(df[d["name"]], errors="coerce")
        ok = s.notna()
        if ok.sum() < MIN_PAIR_N:
            continue
        days = (s[ok].max() - s[ok].min()).days
        if days < 1:
            continue
        freq, label = _choose_freq(days)
        periods = s[ok].dt.to_period(freq)
        full_index = pd.period_range(periods.min(), periods.max(), freq=freq)
        counts = periods.value_counts().reindex(full_index, fill_value=0)
        entry = {"date_column": d["name"], "freq": label, "periods": [str(p) for p in full_index],
                 "counts": [int(v) for v in counts.values], "measures": {}, "tests": [],
                 "note": K("note.partial_periods")}
        series_to_test = [("record count", counts.astype(float))]
        for m in measures[:4]:
            y = pd.to_numeric(df[m["name"]], errors="coerce").astype(float)[ok]
            grp = y.groupby(periods)
            mean = grp.mean().reindex(full_index)
            total = grp.sum(min_count=1).reindex(full_index)
            agg = "sum" if is_additive(m["name"]) else "mean"
            entry["measures"][m["name"]] = {"agg": agg, "mean": [rnd(v, 4) for v in mean.values],
                                            "sum": [rnd(v, 4) for v in total.values]}
            series_to_test.append((m["name"], (total if agg == "sum" else mean)))
        for name, values in series_to_test:
            v = values.dropna()
            if len(v) < 4 or v.nunique() < 2:
                continue
            pos = np.array([full_index.get_loc(p) for p in v.index])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, p = stats.spearmanr(pos, v.to_numpy())
            first, last = float(v.iloc[0]), float(v.iloc[-1])
            entry["tests"].append({"series": name, "agg": "count" if name == "record count" else entry["measures"][name]["agg"],
                                   "periods": int(len(v)), "spearman": rnd(rho, 4), "p_value": float(p),
                                   "first": rnd(first, 4), "last": rnd(last, 4),
                                   "first_period": str(v.index[0]), "last_period": str(v.index[-1]),
                                   "pct_change": rnd((last - first) / abs(first) * 100, 2) if first else None})
        out.append(entry)
    return out


def scatter_samples(df, pairs, limit=4, points=400):
    out = []
    for pr in [p for p in pairs if p["kind"] == "numeric-numeric"][:limit]:
        x = pd.to_numeric(df[pr["a"]], errors="coerce").astype(float)
        y = pd.to_numeric(df[pr["b"]], errors="coerce").astype(float)
        ok = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        sub = pd.DataFrame({"x": x[ok], "y": y[ok]})
        if len(sub) > points:
            sub = sub.sample(points, random_state=0)
        out.append({"x": pr["a"], "y": pr["b"], "points": [[rnd(a, 6), rnd(b, 6)] for a, b in sub.to_numpy()],
                    "sampled": bool(ok.sum() > points), "n": int(ok.sum()), "spearman": pr["value"], "pearson": pr.get("pearson")})
    return out


def correlation_matrix(df, measures, method):
    names = [m["name"] for m in measures]
    if len(names) < 2:
        return {}
    data = pd.DataFrame({n: pd.to_numeric(df[n], errors="coerce").astype(float) for n in names})
    data = data.replace([np.inf, -np.inf], np.nan)
    corr = data.corr(method=method, min_periods=MIN_PAIR_N)
    return {a: {b: rnd(corr.loc[a, b], 4) for b in names} for a in names}


def analysis_frame(df: pd.DataFrame, inferences: dict):
    """Typed view of the cleaned data for analysis.

    Columns whose type conversion is still pending review are analyzed with the
    same provisional parsing the profile uses; values that cannot be parsed are
    left out and reported. Returns (frame, provisional_columns).
    """
    out, provisional = {}, []
    for col in df.columns:
        inf = inferences[col]
        if inf.kind in ("numeric", "datetime", "boolean") and not inf.typed_source and inf.role != "identifier":
            out[col] = inf.parsed
            provisional.append({"column": col, "kind": inf.kind, "excluded_values": inf.n_invalid})
        else:
            out[col] = df[col]
    return pd.DataFrame(out, index=df.index), provisional


def run_eda(df: pd.DataFrame, profile: dict, provisional=None) -> dict:
    provisional = provisional or []
    measures, categories, dates, excluded = _select_columns(profile)
    num_pairs = numeric_pairs(df, measures)
    cat_pairs = categorical_pairs(df, categories)
    groups = group_comparisons(df, categories, measures)
    trend = trends(df, dates, measures)

    # One family of tests for false-discovery control
    family = num_pairs + cat_pairs + groups
    trend_tests = [t for e in trend for t in e["tests"]]
    qs = benjamini_hochberg([x["p_value"] for x in family] + [t["p_value"] for t in trend_tests])
    for x, q in zip(family + trend_tests, qs):
        x["q_value"] = rnd(q, 6)
        x["p_value"] = rnd(x["p_value"], 6)
    for x in family:
        x["strength"] = strength_label(x["value"] or 0)
        x["significant"] = bool(x["q_value"] is not None and x["q_value"] < 0.05)

    relationships = sorted(
        [{k: v for k, v in x.items() if k != "groups"} | ({"a": x["group"], "b": x["measure"]} if "group" in x else {})
         for x in family],
        key=lambda r: -abs(r["value"] or 0))
    groups.sort(key=lambda g: -(g["effect_size"] or 0))

    n = profile["row_count"]
    # Limitations are translatable messages; English text is available via render(msg, "en")
    limitations = []
    if provisional:
        limitations.append(M("lim.provisional", columns=[
            M("lim.provisional_col_excluded" if p["excluded_values"] else "lim.provisional_col",
              column=p["column"], kind=K(f"kind.{p['kind']}"), count=p["excluded_values"]) for p in provisional]))
    if n < 30:
        limitations.append(M("lim.small_sample", count=int(n)))
    if excluded:
        limitations.append(M("lim.excluded", items=[M("lim.excluded_item", column=e["column"], reason=e["reason_msg"])
                                                    for e in excluded[:8]]))
    if len(family) > 1:
        limitations.append(M("lim.multiple_tests", count=len(family) + len(trend_tests)))
    limitations.append(K("lim.causation"))
    high_missing = [c["name"] for c in profile["columns"] if c["missing_pct"] and c["missing_pct"] >= 20]
    if high_missing:
        limitations.append(M("lim.high_missing", columns=high_missing[:6]))
    if trend:
        limitations.append(K("lim.partial_periods"))
    if any(p.get("caution") for p in cat_pairs):
        limitations.append(K("lim.sparse_tables"))

    return {
        "columns_used": {"measures": [m["name"] for m in measures], "categories": [c["name"] for c in categories],
                         "dates": [d["name"] for d in dates]},
        "excluded": excluded,
        "correlation": {"pearson": correlation_matrix(df, measures, "pearson"),
                        "spearman": correlation_matrix(df, measures, "spearman")},
        "relationships": relationships[:60],
        "groups": groups[:20],
        "trends": trend,
        "scatter": scatter_samples(df, sorted(num_pairs, key=lambda r: -abs(r["value"] or 0))),
        "missing": [{"column": c["name"], "missing": c["missing_total"], "pct": c["missing_pct"]} for c in profile["columns"]],
        "rows_with_missing": int(df.isna().any(axis=1).sum()) if len(df) else 0,
        "outliers": [{"column": m["name"], "mild": m["numeric"]["outliers_mild"], "extreme": m["numeric"]["outliers_extreme"],
                      "bounds": m["numeric"]["outlier_bounds"]["mild"]} for m in measures if m["numeric"]],
        "limitations": limitations,
        "provisional": provisional,
    }
