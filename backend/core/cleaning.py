"""Cleaning plan and deterministic cleaning pipeline.

`build_plan` proposes actions from the profile of the original upload. Each
action is either:

- "safe":   lossless normalisation (trimming, unifying blank cells, type
            conversion where every value parses). Enabled by default.
- "review": anything that removes rows/columns, sets values to missing,
            imputes, merges labels or reinterprets formatted values.
            Never applied unless the user approves it.

`apply_cleaning` is a pure function of (original data, plan, decisions):
the original upload is never modified, every changed cell is tracked, and the
same decisions always produce the same output (the "recipe").
"""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .inference import MISSING_MARKERS, coerce_series
from .profiling import impossible_masks, row_hashes
from .i18n import K, M, render
from .utils import display_series, display_value, mask_value, rnd

MAX_LOGGED_CHANGES = 50_000
RECIPE_FORMAT = "sdap-cleaning-recipe"

GROUP_ORDER = ["structure", "formatting", "missing", "types", "labels", "validity", "duplicates", "outliers", "imputation"]


class DecisionError(ValueError):
    def __init__(self, msg):
        super().__init__(render(msg, "en"))
        self.msg = msg


def _ex(before, after, count=None):
    e = {"before": display_value(before) if before is not None else None,
         "after": display_value(after) if after is not None else None}
    if count is not None:
        e["count"] = int(count)
    return e


def snake_case(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name.strip())
    s = re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_").lower()
    return s or "column"


def _clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip())


# ─── Plan ────────────────────────────────────────────────────

def _option(name, default, choices):
    """A configurable option; labels are translatable messages."""
    return {"name": name, "label_msg": K(f"option.{name}"), "label": render(K(f"option.{name}"), "en"), "default": default,
            "choices": [{"value": v, "label_msg": K(lk), "label": render(K(lk), "en")} for v, lk in choices]}


def build_plan(raw: pd.DataFrame, profile: dict) -> list:
    plan = []
    cols = {c["name"]: c for c in profile["columns"]}

    def add(title, desc, **a):
        a.setdefault("options", [])
        a.setdefault("examples", [])
        a.setdefault("column", None)
        a["default_enabled"] = a["category"] == "safe"
        a["title_msg"] = title
        a["desc"] = desc if isinstance(desc, list) else [desc]
        # English text kept for API consumers that don't render messages
        a["title"] = render(title, "en")
        a["description"] = render(a["desc"], "en")
        plan.append(a)

    # Column names
    renames = {c: _clean_name(c) for c in raw.columns if _clean_name(c) != c}
    if renames:
        add(K("action.trim_names"), K("desc.trim_names"), id="trim_names", type="trim_names", category="safe",
            group="structure", unit="columns", affected=len(renames),
            examples=[{"before": k, "after": v} for k, v in list(renames.items())[:5]])

    for name in raw.columns:
        c = cols[name]
        s = raw[name]
        is_str = s.map(lambda v: isinstance(v, str))
        if c["whitespace_values"]:
            sample = s[is_str & (s.str.strip() != s) & (s.str.strip() != "")].value_counts().head(5)
            add(K("action.trim_whitespace"), K("desc.trim_whitespace"), id=f"trim_whitespace:{name}",
                type="trim_whitespace", column=name, category="safe", group="formatting", unit="values",
                affected=c["whitespace_values"], examples=[_ex(v, v.strip(), n) for v, n in sample.items()])
        if c["empty"]:
            add(K("action.empty_to_missing"), K("desc.empty_to_missing"), id=f"empty_to_missing:{name}",
                type="empty_to_missing", column=name, category="safe", group="missing", unit="values",
                affected=c["empty"], examples=[{"before": "", "after": None, "count": c["empty"]}])
        if c["marker_count"]:
            typed = c["kind"] in ("numeric", "datetime", "boolean")
            add(K("action.missing_markers"),
                M("desc.markers_typed", kind=K(f"kind.{c['kind']}")) if typed else K("desc.markers_text"),
                id=f"missing_markers:{name}", type="missing_markers", column=name,
                category="safe" if typed else "review", group="missing", unit="values", affected=c["marker_count"],
                examples=[_ex(k, None, v) for k, v in c["markers"].items()])
        if c["case_variants"]:
            n_aff = sum(v["count"] for g in c["case_variants"] for v in g["variants"] if v["value"] != g["canonical"])
            add(K("action.normalize_labels"), K("desc.normalize_labels"), id=f"normalize_labels:{name}",
                type="normalize_labels", column=name, category="review", group="labels", unit="values", affected=n_aff,
                examples=[_ex(v["value"], g["canonical"], v["count"]) for g in c["case_variants"][:3]
                          for v in g["variants"] if v["value"] != g["canonical"]][:6])

        _plan_type_conversion(add, name, c)

        if c["impossible"]:
            n_aff = sum(r["count"] for r in c["impossible"])
            add(K("action.impossible_values"),
                [M("desc.impossible", rules=[M("desc.rule_count", rule=r["label"], count=r["count"]) for r in c["impossible"]]),
                 K("desc.originals_kept")],
                id=f"impossible_values:{name}", type="impossible_values", column=name, category="review",
                group="validity", unit="values", affected=n_aff,
                examples=[_ex(e, None) for r in c["impossible"] for e in r["examples"][:3]][:6])

        num = c["numeric"]
        if num and c["role"] == "measure" and num["outliers_mild"]:
            add(K("action.outliers"),
                M("desc.outliers", count=num["outliers_mild"], low=num["outlier_bounds"]["mild"][0],
                  high=num["outlier_bounds"]["mild"][1], extreme=num["outliers_extreme"]),
                id=f"outliers:{name}", type="outliers", column=name, category="review", group="outliers",
                unit="values", affected=num["outliers_mild"],
                examples=[{"before": display_value(v), "after": None, "after_msg": K("ex.flagged")}
                          for v in num["box"]["outliers"][:5]],
                options=[
                    _option("method", "iqr3" if num["outliers_extreme"] else "iqr1.5",
                            [("iqr1.5", "choice.iqr_mild"), ("iqr3", "choice.iqr_extreme")]),
                    _option("handling", "flag", [("flag", "choice.flag"), ("exclude_rows", "choice.exclude_rows"),
                                                 ("set_missing", "choice.set_missing")]),
                ])

        if 0 < c["missing_pct"] <= 50 and c["role"] in ("measure", "category", "flag") and c["kind"] != "empty":
            if c["kind"] == "numeric":
                choices = [("median", "choice.median"), ("mean", "choice.mean"), ("zero", "choice.zero")]
            else:
                choices = [("mode", "choice.mode")]
                if c["kind"] != "boolean":
                    choices.append(("unknown", "choice.unknown"))
            add(K("action.fill_missing"), M("desc.fill_missing", count=c["missing_total"]),
                id=f"fill_missing:{name}", type="fill_missing", column=name, category="review", group="imputation",
                unit="values", affected=c["missing_total"], options=[_option("strategy", choices[0][0], choices)])

        if c["role"] in ("constant", "empty"):
            add(K("action.drop_column"),
                K("desc.drop_column_empty") if c["role"] == "empty" else
                M("desc.drop_column_constant", value=c["dominant"]["value"] if c["dominant"] else "—"),
                id=f"drop_column:{name}", type="drop_column", column=name, category="review", group="structure",
                unit="columns", affected=1)

    if profile.get("empty_rows"):
        add(K("action.drop_empty_rows"), M("desc.drop_empty_rows", count=profile["empty_rows"]),
            id="drop_empty_rows", type="drop_empty_rows", category="review", group="structure", unit="rows",
            affected=profile["empty_rows"])

    dup_n = profile["normalized_duplicate_rows"]
    if dup_n:
        desc = [M("desc.drop_exact_duplicates", count=profile["exact_duplicate_rows"])]
        if dup_n != profile["exact_duplicate_rows"]:
            desc.append(M("desc.dup_after_normalization", count=dup_n))
        desc.append(K("desc.keep_first"))
        add(K("action.drop_exact_duplicates"), desc, id="drop_exact_duplicates", type="drop_exact_duplicates",
            category="review", group="duplicates", unit="rows", affected=dup_n,
            examples=[{"before": None, "after": None, "before_msg": M("ex.rows", rows=list(g)),
                       "after_msg": M("ex.keep_row", row=g[0])}
                      for g in (profile["normalized_duplicate_examples"] or profile["duplicate_examples"])[:5]])
    for k in profile["key_duplicates"]:
        add(K("action.drop_key_duplicates"),
            M("desc.drop_key_duplicates", count=k["duplicated_values"], column=k["column"], extra=k["extra_rows"]),
            id=f"drop_key_duplicates:{k['column']}", type="drop_key_duplicates", column=k["column"], category="review",
            group="duplicates", unit="rows", affected=k["extra_rows"],
            examples=[{"before": None, "after": None, "before_msg": M("ex.value_times", value=e["value"], count=e["count"]),
                       "after_msg": K("ex.keep_one")} for e in k["examples"]],
            options=[_option("keep", "first", [("first", "choice.first"), ("last", "choice.last")])])

    if any(snake_case(_clean_name(c)) != _clean_name(c) for c in raw.columns):
        examples = [{"before": c, "after": snake_case(c)} for c in raw.columns if snake_case(_clean_name(c)) != _clean_name(c)]
        add(K("action.standardize_names"), K("desc.standardize_names"), id="standardize_names",
            type="standardize_names", category="review", group="structure", unit="columns", affected=len(examples),
            examples=examples[:6])

    for a in plan:  # personal identifiers: examples show masked values only
        if a["column"] is not None and cols.get(a["column"], {}).get("sensitive"):
            for e in a["examples"]:
                for side in ("before", "after"):
                    if isinstance(e.get(side), str) and e[side]:
                        e[side] = mask_value(e[side])
    plan.sort(key=lambda a: (0 if a["category"] == "safe" else 1, GROUP_ORDER.index(a["group"])))
    return plan


def _plan_type_conversion(add, name, c):
    kind = c["kind"]
    if kind not in ("numeric", "datetime", "boolean") or c["dtype"] != "object":
        return
    if kind == "numeric" and c["role"] == "identifier":
        return  # identifiers and codes stay text: no leading zeros or digits lost, never treated as quantities
    if c["parse_rate"] is None:
        return
    invalid = c["invalid_count"]
    fmts = {k: v for k, v in c["formats"].items() if k in ("currency", "percent", "thousands")}
    risky_fmt = sorted(k for k in fmts if k in ("currency", "percent"))
    kind_label = K(f"kind_value.{kind}")
    reasons = []
    if invalid:
        reasons.append(M("reason.invalid", count=invalid, kind=kind_label,
                         examples=[e["value"] for e in c["invalid_examples"][:3]]))
    if risky_fmt:
        reasons.append(M("reason.symbols", formats=[M("desc.format_count", format=K(f"fmt.{k}"), count=fmts[k]) for k in risky_fmt]))
    if c["date_ambiguous"]:
        reasons.append(K("reason.ambiguous_dates"))
    if c["parse_rate"] < 0.8:
        reasons.append(M("reason.mixed", pct=round(c["parse_rate"] * 100, 1), kind=kind_label))
    category = "review" if reasons else "safe"
    desc = [M("desc.convert_parsed", count=c["non_missing"] - invalid, kind=kind_label)]
    if fmts:
        desc.append(M("desc.convert_formats", formats=[M("desc.format_count", format=K(f"fmt.{k}"), count=v)
                                                         for k, v in fmts.items()]))
    desc.append(M("desc.convert_review", reasons=reasons) if reasons else K("desc.convert_safe"))
    options = []
    if kind == "datetime" and c["date_ambiguous"]:
        options.append(_option("date_format", c["date_format"],
                               [(f, f"datefmt.{f}" if f in DATE_FORMAT_LABELS else "datefmt.other") for f in c["date_alternatives"]]))
    add(K(f"action.convert_type.{kind}"), desc, id=f"convert_type:{name}", type="convert_type", column=name,
        category=category, group="types", unit="values", affected=c["non_missing"], target_kind=kind,
        options=options, examples=[_ex(e["value"], None, e["count"]) for e in c["invalid_examples"][:4]])


DATE_FORMAT_LABELS = {"%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%d/%m/%y", "%m/%d/%y",
                      "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S"}


# ─── Decisions ───────────────────────────────────────────────

def default_decisions(plan):
    return {a["id"]: {"enabled": a["default_enabled"], "options": {o["name"]: o["default"] for o in a["options"]}}
            for a in plan}


def resolve_decisions(plan, decisions):
    """Merge user decisions with plan defaults and validate options. Returns (resolved, warnings)."""
    decisions = decisions or {}
    by_id = {a["id"]: a for a in plan}
    resolved, warnings = {}, []
    for a in plan:
        d = decisions.get(a["id"])
        enabled = a["default_enabled"] if d is None else bool(d.get("enabled", False))
        opts = {o["name"]: o["default"] for o in a["options"]}
        for k, v in ((d or {}).get("options") or {}).items():
            spec = next((o for o in a["options"] if o["name"] == k), None)
            if spec is None:
                raise DecisionError(M("err.unknown_option", option=str(k), action=a["id"]))
            if v not in [ch["value"] for ch in spec["choices"]]:
                raise DecisionError(M("err.invalid_option_value", value=str(v), option=str(k), action=a["id"]))
            opts[k] = v
        resolved[a["id"]] = {"enabled": enabled, "options": opts}
    unknown = sorted(set(decisions) - set(by_id))
    if unknown:
        warnings.append(M("warn.unknown_actions", count=len(unknown), ids=unknown[:5]))
    return resolved, warnings


def make_recipe(filename, raw, decisions):
    return {"format": RECIPE_FORMAT, "version": 1,
            "source": {"filename": filename, "rows": int(raw.shape[0]), "columns": list(raw.columns)},
            "actions": decisions}


# ─── Apply ───────────────────────────────────────────────────

@dataclass
class CleanResult:
    df: pd.DataFrame
    operations: list
    cell_changes: list
    changed: pd.DataFrame          # bool, raw index × raw columns: content changed
    converted: pd.DataFrame        # bool: type converted (same meaning)
    excluded_rows: list
    column_map: dict               # raw name -> final name (None when removed)
    added_columns: list
    decisions: dict
    warnings: list = field(default_factory=list)

    @property
    def change_count(self):
        alive = self.df.index
        kept = [c for c, f in self.column_map.items() if f is not None]
        return int(self.changed.loc[alive, kept].to_numpy().sum()) if kept and len(alive) else 0


class _Tracker:
    def __init__(self, raw):
        self.changed = pd.DataFrame(False, index=raw.index, columns=raw.columns)
        self.converted = pd.DataFrame(False, index=raw.index, columns=raw.columns)
        self.changes = []
        self.ops = []

    def record(self, action_id, col, mask, before, after):
        mask = mask.fillna(False).astype(bool) if mask.dtype != bool else mask
        idx = mask.index[mask.to_numpy()]
        if len(idx) == 0:
            return 0
        self.changed.loc[idx, col] = True
        room = MAX_LOGGED_CHANGES - len(self.changes)
        for i in idx[:max(0, room)]:
            self.changes.append({"row": int(i) + 1, "column": col, "before": display_value(before[i]),
                                 "after": display_value(after[i]), "action": action_id})
        return int(len(idx))

    def op(self, action, status, **info):
        entry = {"action_id": action["id"], "type": action["type"], "title": action["title"],
                 "title_msg": action["title_msg"], "column": action.get("column"), "category": action["category"],
                 "status": status}
        if isinstance(info.get("message"), dict):  # translatable note; English text kept alongside
            info["msg"] = info["message"]
            info["message"] = render(info["msg"], "en")
        entry.update(info)
        self.ops.append(entry)
        return entry


def _str_mask(s):
    return s.map(lambda v: isinstance(v, str))


def apply_cleaning(raw: pd.DataFrame, plan: list, decisions: dict) -> CleanResult:
    resolved, warnings = resolve_decisions(plan, decisions)
    work = raw.copy()
    t = _Tracker(raw)
    alive = pd.Series(True, index=raw.index)
    excluded = []
    by_type = {}
    for a in plan:
        by_type.setdefault(a["type"], []).append(a)

    def enabled(a):
        return resolved[a["id"]]["enabled"]

    def opts(a):
        return resolved[a["id"]]["options"]

    def column_actions(kind):
        return {a["column"]: a for a in by_type.get(kind, [])}

    # 1) Per-column value normalisation, in a fixed order
    for step in ("trim_whitespace", "empty_to_missing", "missing_markers", "normalize_labels",
                 "convert_type", "impossible_values", "fill_missing"):
        for col, a in column_actions(step).items():
            if not enabled(a):
                t.op(a, "rejected", message=K("op.not_approved") if a["category"] == "review" else K("op.disabled"))
                continue
            before = work[col]
            info = _apply_column_step(step, a, opts(a), col, before, work, t)
            t.op(a, info.pop("status", "applied"), options=opts(a), **info)

    # 2) Row-level removal (after normalisation so near-duplicates match)
    for a in by_type.get("drop_empty_rows", []):
        if not enabled(a):
            t.op(a, "rejected", message=K("op.empty_rows_kept"))
            continue
        sub = work.loc[alive].map(lambda v: None if isinstance(v, str) and v.strip() == "" else v)
        rows = sub.index[sub.isna().all(axis=1).to_numpy()]
        alive[rows] = False
        excluded.extend({"row": int(i) + 1, "action": a["id"], "reason": "Empty row", "reason_msg": K("excl_row.empty")} for i in rows)
        t.op(a, "applied", rows_removed=int(len(rows)))
    for a in by_type.get("drop_exact_duplicates", []):
        if not enabled(a):
            t.op(a, "rejected", message=K("op.duplicates_kept"))
            continue
        sub = work.loc[alive]
        dup = row_hashes(sub).duplicated(keep="first")
        rows = sub.index[dup.to_numpy()]
        alive[rows] = False
        excluded.extend({"row": int(i) + 1, "action": a["id"], "reason": "Exact duplicate row", "reason_msg": K("excl_row.duplicate")} for i in rows)
        t.op(a, "applied", rows_removed=int(len(rows)), options=opts(a))
    for a in by_type.get("drop_key_duplicates", []):
        col = a["column"]
        if not enabled(a):
            t.op(a, "rejected", message=K("op.key_duplicates_kept"))
            continue
        sub = work.loc[alive, col]
        keys = display_series(sub)
        dup = keys.notna() & keys.duplicated(keep=opts(a)["keep"])
        rows = sub.index[dup.to_numpy()]
        alive[rows] = False
        excluded.extend({"row": int(i) + 1, "action": a["id"], "reason": f"Duplicate {col}", "reason_msg": M("excl_row.key_duplicate", column=col)} for i in rows)
        t.op(a, "applied", rows_removed=int(len(rows)), options=opts(a))

    # 3) Outliers (computed on the rows that remain)
    added = []
    for col, a in column_actions("outliers").items():
        if not enabled(a):
            t.op(a, "rejected", message=K("op.outliers_kept"))
            continue
        s = work[col]
        if not ptypes.is_numeric_dtype(s):
            t.op(a, "skipped", message=K("op.not_numeric"))
            continue
        vals = s[alive].astype(float)
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        k = 3.0 if opts(a)["method"] == "iqr3" else 1.5
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        mask = alive & s.astype(float).notna() & ((s.astype(float) < lo) | (s.astype(float) > hi))
        mask = mask.fillna(False).astype(bool)
        handling = opts(a)["handling"]
        info = {"options": opts(a), "bounds": [rnd(lo, 6), rnd(hi, 6)], "outliers": int(mask.sum())}
        if handling == "flag":
            flag_name = f"{col}_outlier_flag"
            while flag_name in work.columns:
                flag_name += "_"
            work[flag_name] = mask.astype("boolean")
            added.append(flag_name)
            info["columns_added"] = [flag_name]
        elif handling == "exclude_rows":
            rows = mask.index[mask.to_numpy()]
            alive[rows] = False
            excluded.extend({"row": int(i) + 1, "action": a["id"], "reason": f"Outlier in {col}", "reason_msg": M("excl_row.outlier", column=col)} for i in rows)
            info["rows_removed"] = int(len(rows))
        else:
            after = s.astype(object).where(~mask, None)
            info["values_set_missing"] = t.record(a["id"], col, mask, s, after)
            work[col] = s.mask(mask)
        t.op(a, "applied", **info)

    # 4) Column removal and renaming
    column_map = {c: c for c in raw.columns}
    for col, a in column_actions("drop_column").items():
        if not enabled(a):
            t.op(a, "rejected", message=K("op.column_kept"))
            continue
        column_map[col] = None
        t.op(a, "applied", columns_removed=[col])

    final = {c: c for c in list(raw.columns) + added}
    for a in by_type.get("trim_names", []):
        if enabled(a):
            for c in final:
                final[c] = _clean_name(final[c])
            t.op(a, "applied", columns_renamed=sum(1 for c in raw.columns if _clean_name(c) != c))
        else:
            t.op(a, "rejected", message=K("op.names_kept"))
    for a in by_type.get("standardize_names", []):
        if enabled(a):
            for c in final:
                final[c] = snake_case(final[c])
            t.op(a, "applied", columns_renamed=sum(1 for c in raw.columns if snake_case(c) != c))
        else:
            t.op(a, "rejected", message=K("op.names_kept"))

    keep_cols = [c for c in list(raw.columns) + added if column_map.get(c, c) is not None]
    out = work.loc[alive, keep_cols].copy()
    seen, new_names = {}, []
    for c in keep_cols:
        n = final[c]
        base, k = n, 2
        while n in seen:
            n = f"{base}_{k}"
            k += 1
        seen[n] = True
        new_names.append(n)
    rename = dict(zip(keep_cols, new_names))
    out.columns = new_names
    for c in raw.columns:
        column_map[c] = rename.get(c) if column_map[c] is not None else None
    for ch in t.changes:
        ch["final_column"] = column_map.get(ch["column"])

    return CleanResult(df=out, operations=t.ops, cell_changes=t.changes, changed=t.changed, converted=t.converted,
                       excluded_rows=excluded, column_map=column_map, added_columns=[rename[c] for c in added],
                       decisions=resolved, warnings=warnings)


def _apply_column_step(step, a, options, col, s, work, t):
    aid = a["id"]
    if step == "trim_whitespace":
        is_str = _str_mask(s)
        after = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        mask = is_str & (after != s)
        n = t.record(aid, col, mask, s, after)
        work[col] = after
        return {"values_changed": n}

    if step in ("empty_to_missing", "missing_markers"):
        stripped = s.map(lambda v: v.strip().lower() if isinstance(v, str) else None)
        mask = stripped.eq("") if step == "empty_to_missing" else stripped.isin(MISSING_MARKERS)
        mask = mask & _str_mask(s)
        after = s.where(~mask, None)
        n = t.record(aid, col, mask, s, after)
        work[col] = after
        return {"values_set_missing": n}

    if step == "normalize_labels":
        is_str = _str_mask(s)
        if not is_str.any():
            return {"status": "skipped", "message": K("op.no_text")}
        keys = s.where(is_str).map(lambda v: re.sub(r"\s+", " ", v.strip()).casefold() if isinstance(v, str) else None)
        counts = s[is_str].map(lambda v: v.strip()).value_counts()
        canonical = {}
        # Most frequent spelling wins; ties prefer Title Case, then alphabetical (deterministic)
        for label, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], not kv[0].istitle(), kv[0])):
            canonical.setdefault(re.sub(r"\s+", " ", label).casefold(), label)
        after = s.copy()
        after[is_str] = keys[is_str].map(canonical)
        mask = is_str & (after != s)
        n = t.record(aid, col, mask, s, after)
        work[col] = after
        return {"values_changed": n}

    if step == "convert_type":
        kind = a["target_kind"]
        typed, invalid, formatted = coerce_series(s, kind, options.get("date_format"))
        n_missing = t.record(aid, col, invalid, s, pd.Series(None, index=s.index, dtype=object))
        n_fmt = t.record(aid, col, formatted & ~invalid, s, typed)
        conv = typed.notna() & ~formatted
        t.converted.loc[conv[conv].index, col] = True
        work[col] = typed
        return {"values_converted": int(conv.sum()), "values_changed": n_fmt, "values_set_missing": n_missing,
                "new_type": str(typed.dtype)}

    if step == "impossible_values":
        if not (ptypes.is_numeric_dtype(s) or ptypes.is_datetime64_any_dtype(s)) or ptypes.is_bool_dtype(s):
            return {"status": "skipped", "message": K("op.not_converted")}
        rules = impossible_masks(col, s)
        mask = pd.Series(False, index=s.index)
        for _, _, m in rules:
            mask |= m
        n = t.record(aid, col, mask, s, pd.Series(None, index=s.index, dtype=object))
        work[col] = s.mask(mask)
        return {"values_set_missing": n, "rules": [r for r, _, _ in rules]}

    if step == "fill_missing":
        strategy = options["strategy"]
        missing = s.isna() | s.map(lambda v: isinstance(v, str) and v.strip() == "")
        if not missing.any():
            return {"status": "skipped", "message": K("op.no_missing")}
        if strategy in ("median", "mean", "zero"):
            if not ptypes.is_numeric_dtype(s):
                return {"status": "skipped", "message": K("op.not_numeric")}
            vals = s.astype(float)
            fill = {"median": vals.median(), "mean": vals.mean(), "zero": 0.0}[strategy]
            if ptypes.is_integer_dtype(s) and float(fill).is_integer():
                fill = int(fill)
            elif ptypes.is_integer_dtype(s):
                s = s.astype(float)
        elif strategy == "mode":
            mode = s[~missing].mode()
            if mode.empty:
                return {"status": "skipped", "message": K("op.no_mode")}
            fill = sorted(mode.tolist(), key=str)[0]
        else:
            fill = "Unknown"
        after = s.copy()
        after[missing] = fill
        n = t.record(aid, col, missing, s, after)
        work[col] = after
        return {"values_imputed": n, "fill_value": display_value(fill)}

    raise ValueError(step)
