"""Exports: cleaned data, cleaning log, quality report, analysis report, recipe.

Spreadsheet safety: in .xlsx exports (XlsxWriter, strings_to_formulas=False)
every text cell is written as a string, never as a formula or link, so values
like "=HYPERLINK(...)" cannot execute. In CSV
exports, text cells that a spreadsheet would interpret as a formula are
prefixed with an apostrophe (OWASP CSV-injection guidance); numbers such as
-3 are never altered.
"""
import datetime as dt
import html
import io
import json
import re

import pandas as pd
import xlsxwriter

from .i18n import K, M, Q, format_number, format_period, render

# Control characters that are not allowed in XLSX XML
ILLEGAL_CHARACTERS_RE = re.compile(r"[\000-\010]|[\013-\014]|[\016-\037]")

from .utils import display_value, jsonable, mask_changes, sensitive_columns

EXCEL_MAX_ROWS = 1_048_575
_FORMULA_START = re.compile(r"^[=@\t\r]|^[+-](?![\d.])")

MEDIA = {
    "csv": "text/csv; charset=utf-8",
    "json": "application/json",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "html": "text/html; charset=utf-8",
}


class ExportError(ValueError):
    def __init__(self, msg):
        super().__init__(render(msg, "en"))
        self.msg = msg


# ─── Generic writers ─────────────────────────────────────────

def _csv_safe(v):
    if isinstance(v, str) and _FORMULA_START.match(v):
        return "'" + v
    return v


def df_to_csv(df: pd.DataFrame, formula_protection=True) -> bytes:
    out = df.copy()
    if formula_protection:
        for c in out.columns:
            if out[c].dtype == object or pd.api.types.is_string_dtype(out[c]):
                out[c] = out[c].map(_csv_safe)
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].map(display_value)
    return out.to_csv(index=False).encode("utf-8-sig")


def df_to_json(df: pd.DataFrame) -> bytes:
    records = [{c: _json_cell(v) for c, v in zip(df.columns, row)} for row in df.itertuples(index=False)]
    return json.dumps(records, ensure_ascii=False, indent=1).encode("utf-8")


def _json_cell(v):
    if isinstance(v, (pd.Timestamp, dt.datetime, dt.date)):
        return display_value(v)
    return jsonable(v)


def _xl_value(v):
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if hasattr(v, "item") and not isinstance(v, str):
        try:
            v = v.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(v, float) and v in (float("inf"), float("-inf")):
        return str(v)
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(jsonable(v), ensure_ascii=False)
    if isinstance(v, str):
        return ILLEGAL_CHARACTERS_RE.sub("", v)
    return v


def _write_rows(wb, ws, header, rows, prepared=False):
    bold = wb.add_format({"bold": True})
    date_fmt = wb.add_format({"num_format": "yyyy-mm-dd hh:mm:ss"})
    for j, h in enumerate(header):
        ws.write_string(0, j, _xl_value(str(h)) or "", bold)
    for i, row in enumerate(rows, start=1):
        cells = row if prepared else [_xl_value(v) for v in row]
        for j, v in enumerate(cells):
            if v is None:
                continue
            if isinstance(v, str):
                ws.write_string(i, j, v)  # always text, never a formula or link
            elif isinstance(v, bool):
                ws.write_boolean(i, j, v)
            elif isinstance(v, (int, float)):
                ws.write_number(i, j, v)
            elif isinstance(v, (dt.datetime, dt.date)):
                ws.write_datetime(i, j, v, date_fmt)
            else:
                ws.write_string(i, j, str(v))


def _xl_column(s: pd.Series) -> list:
    """Vectorized conversion of one column to Excel-ready Python values."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return [None if pd.isna(v) else v.to_pydatetime() for v in s]
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_extension_array_dtype(s):
        return [None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in s.astype(object)]
    if pd.api.types.is_float_dtype(s):
        return [None if v != v else (str(v) if v in (float("inf"), float("-inf")) else v) for v in s.tolist()]
    if pd.api.types.is_integer_dtype(s):
        return s.tolist()
    return [_xl_value(v) for v in s.tolist()]


def workbook_bytes(sheets, rtl=False) -> bytes:
    """sheets: list of (title, header, rows) or (title, header, rows, prepared)."""
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True, "strings_to_formulas": False, "strings_to_urls": False,
                                   "strings_to_numbers": False, "nan_inf_to_errors": True})
    for sheet in sheets:
        title, header, rows = sheet[:3]
        ws = wb.add_worksheet(re.sub(r"[\[\]:*?/\\]", "_", title)[:31])
        if rtl:
            ws.right_to_left()
        _write_rows(wb, ws, header, rows, prepared=len(sheet) > 3 and sheet[3])
    wb.close()
    return buf.getvalue()


def df_to_xlsx(df: pd.DataFrame, sheet="Data") -> bytes:
    if len(df) > EXCEL_MAX_ROWS:
        raise ExportError(M("err.excel_row_limit", count=len(df)))
    columns = [_xl_column(df[c]) for c in df.columns]
    return workbook_bytes([(sheet, list(df.columns), zip(*columns) if columns else [], True)])


def _records_sheet(title, records, columns=None):
    columns = columns or (list(records[0].keys()) if records else ["(none)"])
    rows = [[_flat(r.get(c)) for c in columns] for r in records]
    return (title, columns, rows)


def _flat(v):
    if isinstance(v, (dict, list)):
        return json.dumps(jsonable(v), ensure_ascii=False)
    return v


# ─── Artifacts ───────────────────────────────────────────────
# Report artifacts take a `lang` ("en" | "ar"). The language only affects
# headings, labels and generated prose — never column names or cell values.

def _tr(lang):
    def t(key, **params):
        return render(M(key, **params), lang)
    return t


def _num(v, lang):
    return format_number(v, lang) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def export_cleaned(ds, fmt, formula_protection=True):
    df = ds.state.result.df
    if fmt == "csv":
        return df_to_csv(df, formula_protection)
    if fmt == "xlsx":
        return df_to_xlsx(df, "Cleaned data")
    if fmt == "json":
        return df_to_json(df)
    raise ExportError(M("err.export_format", formats=["csv", "xlsx", "json"]))


def _localized_ops(ops, lang):
    out = []
    for o in ops:
        o = dict(o)
        o["title_localized"] = render(o.get("title_msg") or o["title"], lang)
        o["message_localized"] = render(o["msg"], lang) if o.get("msg") else ""
        o["status_localized"] = render(K(f"status.{o['status']}"), lang)
        out.append(o)
    return out


def cleaning_log_dict(ds, lang="en"):
    r = ds.state.result
    return jsonable({
        "dataset": ds.filename, "lang": lang, "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "recipe": ds.recipe(), "operations": _localized_ops(r.operations, lang), "excluded_rows": r.excluded_rows,
        "cell_changes": mask_changes(r.cell_changes, sensitive_columns(ds.profile)), "cell_changes_logged": len(r.cell_changes),
        "column_map": r.column_map, "warnings": [render(w, lang) for w in r.warnings],
    })


OP_COLUMNS = ["action_id", "title_localized", "column", "category", "status_localized", "values_changed",
              "values_converted", "values_set_missing", "values_imputed", "rows_removed", "columns_added",
              "columns_removed", "columns_renamed", "options", "message_localized"]


def export_cleaning_log(ds, fmt, lang="en"):
    t = _tr(lang)
    log = cleaning_log_dict(ds, lang)
    op_headers = [t(f"xl.col.{c}") for c in OP_COLUMNS]
    op_rows = [[_flat(o.get(c)) for c in OP_COLUMNS] for o in log["operations"]]
    if fmt == "json":
        return json.dumps(log, ensure_ascii=False, indent=1).encode("utf-8")
    if fmt == "csv":
        return df_to_csv(pd.DataFrame(op_rows, columns=op_headers))
    if fmt == "xlsx":
        change_cols = ["row", "column", "final_column", "before", "after", "action"]
        return workbook_bytes([
            (t("xl.sheet.operations"), op_headers, op_rows),
            (t("xl.sheet.cell_changes"), [t(f"xl.col.{c}") for c in change_cols],
             [[c.get(k) for k in change_cols] for c in log["cell_changes"]]),
            (t("xl.sheet.excluded_rows"), [t("xl.col.row"), t("xl.col.action"), t("xl.col.reason")],
             [[e["row"], e["action"], render(e.get("reason_msg") or e["reason"], lang)] for e in log["excluded_rows"]]),
            (t("xl.sheet.recipe"), ["recipe_json"], [[json.dumps(log["recipe"], ensure_ascii=False)]]),
        ], rtl=lang == "ar")
    raise ExportError(M("err.export_format", formats=["json", "csv", "xlsx"]))


def _localized_issues(issues, lang):
    return [dict(i, message_localized=render(i.get("msg") or i["message"], lang)) for i in issues]


def quality_report_dict(ds, lang="en"):
    st = ds.state
    return jsonable({
        "dataset": ds.filename, "lang": lang, "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "file_info": ds.meta, "summary": {k: v for k, v in st.report.items() if k != "per_column"},
        "per_column": st.report["per_column"],
        "issues_original": _localized_issues(ds.profile["issues"], lang),
        "issues_cleaned": _localized_issues(st.profile["issues"], lang),
        "profile_original": _slim_profile(ds.profile), "profile_cleaned": _slim_profile(st.profile),
    })


def _slim_profile(p):
    cols = []
    for c in p["columns"]:
        d = {k: v for k, v in c.items() if k not in ("issues",)}
        if d.get("numeric"):
            d["numeric"] = {k: v for k, v in d["numeric"].items() if k not in ("histogram", "box")}
        cols.append(d)
    return {k: v for k, v in p.items() if k not in ("columns", "issues")} | {"columns": cols}


def export_quality_report(ds, fmt, lang="en"):
    t = _tr(lang)
    if fmt == "json":
        return json.dumps(quality_report_dict(ds, lang), ensure_ascii=False, indent=1).encode("utf-8")
    if fmt == "xlsx":
        rep = ds.state.report
        summary = [
            [t("rep.metric.rows"), rep["rows"]["before"], rep["rows"]["after"]],
            [t("rep.metric.columns"), rep["columns"]["before"], rep["columns"]["after"]],
            [t("rep.metric.missing"), rep["missing"]["before"]["total"], rep["missing"]["after"]["total"]],
            [t("rep.metric.duplicates"), rep["duplicates"]["exact_before"], rep["duplicates"]["remaining_exact"]],
            [t("rep.metric.invalid"), rep["invalid"]["detected"], rep["invalid"]["unresolved"]],
            [t("rep.metric.changed"), None, rep["values"]["changed"]],
            [t("rep.metric.excluded"), None, rep["values"]["excluded"]],
            [t("rep.metric.retained"), None, rep["values"]["retained"]],
        ]
        pc_cols = ["column", "final_name", "type_before", "type_after", "missing_before", "missing_after",
                   "invalid_before", "invalid_after", "changed"]
        pc_rows = [[c["column"], c["final_name"], t(f"kind.{c['type_before']}"),
                    t(f"kind.{c['type_after']}") if c["type_after"] else None, c["missing_before"], c["missing_after"],
                    c["invalid_before"], c["invalid_after"], c["changed"]] for c in rep["per_column"]]

        def issue_rows(issues):
            return [[t(f"severity.{i['severity']}"), i["column"], render(i.get("msg") or i["message"], lang)] for i in issues]

        issue_head = [t("xl.col.severity"), t("xl.col.column"), t("xl.col.issue")]
        return workbook_bytes([
            (t("xl.sheet.summary"), [t("xl.col.metric"), t("xl.col.before"), t("xl.col.after")], summary),
            (t("xl.sheet.per_column"), [t(f"xl.col.{c}") for c in pc_cols], pc_rows),
            (t("xl.sheet.issues_cleaned"), issue_head, issue_rows(ds.state.profile["issues"])),
            (t("xl.sheet.issues_original"), issue_head, issue_rows(ds.profile["issues"])),
            (t("xl.sheet.pending"), [t("xl.col.suggestion"), t("xl.col.column"), t("xl.col.affected")],
             [[_plan_title(ds, p["id"], lang), p["column"], p["affected"]] for p in rep["pending_review"]]),
        ], rtl=lang == "ar")
    if fmt == "html":
        return quality_html(ds, lang).encode("utf-8")
    raise ExportError(M("err.export_format", formats=["html", "json", "xlsx"]))


def _plan_title(ds, action_id, lang):
    a = next((a for a in ds.plan if a["id"] == action_id), None)
    return render(a["title_msg"], lang) if a else action_id


def _render_card(card, lang):
    """Plain-text rendering of an insight card (for JSON/XLSX consumers)."""
    return {
        "title": render(card["title"], lang), "finding": render(card["finding"], lang), "why": render(card["why"], lang),
        "action": render(card["action"], lang), "evidence": [render(e, lang) for e in card["evidence"]["items"]],
        "limitations": [render(x, lang) for x in card["limitations"]],
        "technical": render(card["technical"], lang) if card.get("technical") else "",
    }


def _render_summary(summary, insights, lang):
    by_id = {c["id"]: c for c in insights}

    def cards(ids):
        return [dict(id=i, priority=by_id[i]["priority"], direction=by_id[i]["direction"], **_render_card(by_id[i], lang))
                for i in ids if i in by_id]

    return {
        "description": [render(m, lang) for m in summary["description"]],
        "evidence_level": summary["evidence"]["level"], "evidence": render(summary["evidence"]["msg"], lang),
        "kpis": [{"id": k["id"], "label": render(k["label"], lang),
                  "value": render(k["value"], lang) if isinstance(k["value"], dict) else k["value"],
                  "value_display": _kpi_value(k, lang), "detail": render(k.get("sub"), lang),
                  "direction": k["direction"]} for k in summary["kpis"]],
        "key_findings": cards(summary["findings"]), "risks": cards(summary["risks"]),
        "opportunities": cards(summary["opportunities"]), "data_quality_warnings": cards(summary["quality_warnings"]),
        "recommended_actions": [{"priority": a["priority"], "type": a["type"], "action": render(a["msg"], lang),
                                 "insight": a["insight"]} for a in summary["actions"]],
    }


def _kpi_value(k, lang):
    v = k["value"]
    if isinstance(v, dict):
        return render(v, lang)
    s = format_number(v, lang)
    return s + ("%" if lang == "en" else "٪") if k.get("unit") == "pct" else s


def executive_summary_dict(ds, lang="en"):
    st = ds.state
    return jsonable({
        "dataset": ds.filename, "lang": lang, "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "rows": len(st.result.df), "summary": _render_summary(st.summary, st.insights, lang),
        "limitations": [render(x, lang) for x in st.eda["limitations"]],
        "structured": {"summary": st.summary, "insights": st.insights},
    })


def export_executive_summary(ds, fmt, lang="en", decisions=None):
    decisions = decisions or {}
    if fmt == "json":
        d = executive_summary_dict(ds, lang)
        for i, a in enumerate(d["summary"]["recommended_actions"]):
            a["decision"] = decisions.get(i, "pending")
            a["decision_localized"] = render(K(f"decision.{a['decision']}"), lang)
        return json.dumps(d, ensure_ascii=False, indent=1).encode("utf-8")
    if fmt == "html":
        return executive_html(ds, lang, decisions).encode("utf-8")
    raise ExportError(M("err.export_format", formats=["html", "json"]))


def analysis_report_dict(ds, lang="en"):
    st = ds.state
    return jsonable({
        "dataset": ds.filename, "lang": lang, "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "rows": len(st.result.df), "columns": list(st.result.df.columns),
        "executive_summary": _render_summary(st.summary, st.insights, lang),
        "insights_localized": [dict(id=c["id"], category=c["category"], kind=c["kind"], priority=c["priority"],
                                    direction=c["direction"], columns=c["evidence"]["columns"], **_render_card(c, lang))
                               for c in st.insights],
        "limitations_localized": [render(x, lang) for x in st.eda["limitations"]],
        "profile": _slim_profile(st.profile), "eda": st.eda, "insights": st.insights, "summary": st.summary,
        "cleaning_summary": {k: v for k, v in st.report.items() if k != "per_column"},
        "recipe": ds.recipe(),
    })


def export_analysis_report(ds, fmt, lang="en"):
    t = _tr(lang)
    if fmt == "json":
        return json.dumps(analysis_report_dict(ds, lang), ensure_ascii=False, indent=1).encode("utf-8")
    if fmt == "html":
        return analysis_html(ds, lang).encode("utf-8")
    if fmt == "xlsx":
        st = ds.state
        num_rows, cat_rows = [], []
        for c in st.profile["columns"]:
            n = c["numeric"]
            if n:
                num_rows.append([c["name"], t(f"role.{c['role']}"), n["count"], n["mean"], n["median"], n["std"], n["min"],
                                 n["q1"], n["q3"], n["max"], n["skew"], n["outliers_mild"], n["outliers_extreme"], c["missing_total"]])
            if c["kind"] in ("categorical", "boolean"):
                for tv in c["top_values"]:
                    cat_rows.append([c["name"], tv["value"], tv["count"], tv["pct"]])
        group_rows = [[g["group"], g["measure"], lv["level"], lv["count"], lv["mean"], lv["median"], g["q_value"], g["effect_size"]]
                      for g in st.eda["groups"] for lv in g["groups"]]
        trend_rows = []
        for tr in st.eda["trends"]:
            for i, p in enumerate(tr["periods"]):
                row = [tr["date_column"], p, tr["counts"][i]]
                for m, v in tr["measures"].items():
                    row += [m, v["mean"][i], v["sum"][i]]
                trend_rows.append(row)
        insight_rows = []
        for c in st.insights:
            r = _render_card(c, lang)
            insight_rows.append([t(f"kind_card.{c['kind']}"), t(f"priority.{c['priority']}"), t(f"direction.{c['direction']}"),
                                 r["title"], r["finding"], r["why"], r["action"], " | ".join(r["evidence"]),
                                 " | ".join(r["limitations"]), ", ".join(c["evidence"]["columns"]), r["technical"]])
        rel_cols = ["a", "b", "kind", "method", "value", "pearson", "strength", "p_value", "q_value", "n"]
        return workbook_bytes([
            (t("xl.sheet.insights"), [t(f"xl.col.{c}") for c in ("type", "priority", "direction", "title", "finding",
                                                                    "why", "action", "evidence", "limitations", "columns", "technical")],
             insight_rows),
            (t("xl.sheet.numeric"), [t(f"xl.col.{c}") for c in ("column", "role", "count", "mean", "median", "std", "min",
                                                                  "q1", "q3", "max", "skew", "outliers_mild", "outliers_extreme",
                                                                  "missing")], num_rows),
            (t("xl.sheet.categorical"), [t(f"xl.col.{c}") for c in ("column", "value", "count", "pct")], cat_rows),
            (t("xl.sheet.relationships"), [t(f"xl.col.rel_{c}") for c in rel_cols],
             [[r["a"], r["b"], t(f"relkind.{r['kind']}"), t(f"method.{r['method']}"), r["value"], r.get("pearson"),
               t(f"strength.{r['strength']}"), r["p_value"], r["q_value"], r["n"]] for r in st.eda["relationships"]]),
            (t("xl.sheet.groups"), [t(f"xl.col.{c}") for c in ("group", "measure", "level", "count", "mean", "median",
                                                                 "q_value", "effect_size")], group_rows),
            (t("xl.sheet.trends"), [t(f"xl.col.{c}") for c in ("date_column", "period", "records", "measure", "mean", "sum")],
             trend_rows),
            (t("xl.sheet.limitations"), [t("xl.col.limitation")], [[render(x, lang)] for x in st.eda["limitations"]]),
        ], rtl=lang == "ar")
    raise ExportError(M("err.export_format", formats=["html", "json", "xlsx"]))


# ─── HTML (self-contained, inline SVG charts) ────────────────

PALETTE = ["#6c63ff", "#00a88a", "#e05555", "#d9a030", "#3aa7a0", "#8a80f0"]
E = html.escape


def _n(v, lang="en"):
    if v is None:
        return "—"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return E(format_number(v, lang))
    return E(str(v))


def svg_bars(labels, values, title="", color=PALETTE[0], width=520, height=220, lang="en"):
    values = [v or 0 for v in values]
    if not values:
        return ""
    vmax = max(max(values), 0) or 1
    vmin = min(min(values), 0)
    span = (vmax - vmin) or 1
    pad_l, pad_b, pad_t = 56, 52, 22
    w, h = width - pad_l - 10, height - pad_b - pad_t
    bw = w / len(values)
    zero_y = pad_t + h * (vmax / span)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{E(title)}" direction="ltr">',
             f'<line x1="{pad_l}" y1="{zero_y:.1f}" x2="{pad_l + w}" y2="{zero_y:.1f}" stroke="#bbb"/>',
             f'<text x="{pad_l - 4}" y="{pad_t + 4}" font-size="11" text-anchor="end" fill="#777">{_n(float(vmax), lang)}</text>']
    if vmin < 0:
        parts.append(f'<text x="{pad_l - 4}" y="{pad_t + h}" font-size="11" text-anchor="end" fill="#777">{_n(float(vmin), lang)}</text>')
    step = max(1, -(-len(labels) // 7))  # at most ~7 axis labels
    for i, (lab, v) in enumerate(zip(labels, values)):
        bh = abs(v) / span * h
        y = zero_y - bh if v >= 0 else zero_y
        x = pad_l + i * bw
        parts.append(f'<rect x="{x + bw * 0.1:.1f}" y="{y:.1f}" width="{bw * 0.8:.1f}" height="{bh:.1f}" fill="{color}">'
                     f'<title>{E(str(lab))}: {_n(v, lang)}</title></rect>')
        if i % step == 0:
            parts.append(f'<text x="{x + bw / 2:.1f}" y="{pad_t + h + 14}" font-size="11" text-anchor="end" fill="#555" '
                         f'transform="rotate(-30 {x + bw / 2:.1f} {pad_t + h + 14})">{E(str(lab)[:18])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_line(labels, values, title="", color=PALETTE[1], width=520, height=220, lang="en"):
    pts = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(pts) < 2:
        return ""
    vmin, vmax = min(v for _, v in pts), max(v for _, v in pts)
    span = (vmax - vmin) or 1
    pad_l, pad_b, pad_t = 56, 52, 22
    w, h = width - pad_l - 10, height - pad_b - pad_t
    n = max(1, len(values) - 1)
    coords = " ".join(f"{pad_l + i / n * w:.1f},{pad_t + h - (v - vmin) / span * h:.1f}" for i, v in pts)
    step = max(1, len(labels) // 8)
    ticks = "".join(f'<text x="{pad_l + i / n * w:.1f}" y="{pad_t + h + 14}" font-size="10" text-anchor="end" fill="#666" '
                    f'transform="rotate(-30 {pad_l + i / n * w:.1f} {pad_t + h + 14})">{E(str(lab))}</text>'
                    for i, lab in enumerate(labels) if i % step == 0)
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{E(title)}" direction="ltr">'
            f'<text x="{pad_l - 4}" y="{pad_t + 4}" font-size="11" text-anchor="end" fill="#777">{_n(float(vmax), lang)}</text>'
            f'<text x="{pad_l - 4}" y="{pad_t + h}" font-size="11" text-anchor="end" fill="#777">{_n(float(vmin), lang)}</text>'
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>{ticks}</svg>')


def svg_box(name, box, vmin, vmax, width=520, height=44):
    span = (vmax - vmin) or 1
    pad = 12
    w = width - 2 * pad

    def x(v):
        return pad + (v - vmin) / span * w

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{E(name)}" direction="ltr">',
             f'<line x1="{x(box["whisker_low"]):.1f}" y1="22" x2="{x(box["whisker_high"]):.1f}" y2="22" stroke="#888"/>',
             f'<rect x="{x(box["q1"]):.1f}" y="10" width="{max(1, x(box["q3"]) - x(box["q1"])):.1f}" height="24" '
             f'fill="{PALETTE[0]}33" stroke="{PALETTE[0]}"/>',
             f'<line x1="{x(box["median"]):.1f}" y1="10" x2="{x(box["median"]):.1f}" y2="34" stroke="{PALETTE[2]}" stroke-width="2"/>']
    for o in box["outliers"]:
        parts.append(f'<circle cx="{x(o):.1f}" cy="22" r="2.5" fill="{PALETTE[2]}"/>')
    parts.append("</svg>")
    return "".join(parts)


CSS = """body{font-family:Inter,'Segoe UI',Tahoma,Arial,sans-serif;color:#1a1d2e;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.6}
html[lang=ar] body{font-family:'IBM Plex Sans Arabic','Noto Sans Arabic','Segoe UI',Tahoma,Arial,sans-serif;line-height:1.8}
h1{font-size:1.6rem;margin-bottom:.2rem}h2{font-size:1.2rem;margin-top:2rem;border-bottom:2px solid #6c63ff;padding-bottom:.3rem}
h3{font-size:1rem;margin:1.2rem 0 .4rem}.muted{color:#666;font-size:.9rem}table{border-collapse:collapse;width:100%;font-size:.82rem;margin:.5rem 0}
th,td{border:1px solid #ddd;padding:5px 8px;text-align:start;vertical-align:top}th{background:#f2f3f8}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}.card{border:1px solid #e3e5ee;border-radius:8px;padding:.8rem}
.ins{border:1px solid #e3e5ee;border-inline-start:4px solid #6c63ff;border-radius:6px;padding:.6rem .9rem;margin:.7rem 0;background:#fafafe}
.ins.urgent,.ins.negative{border-inline-start-color:#e05555}.ins.high{border-inline-start-color:#d9a030}.ins.positive{border-inline-start-color:#00a88a}
.ins h4{margin:0 0 .3rem;font-size:.98rem}.ins dl{margin:.2rem 0;display:grid;grid-template-columns:max-content 1fr;gap:.15rem .8rem;font-size:.86rem}
.ins dt{font-weight:600;color:#444}.ins dd{margin:0}.ins ul{margin:.1rem 0;padding-inline-start:1.1rem}
.tech{font-size:.78rem;color:#777;margin-top:.3rem}.tag{display:inline-block;font-size:.72rem;padding:1px 6px;border-radius:4px;background:#eceaff;color:#4a40c8;margin-inline-end:4px}
.badge{display:inline-block;font-size:.7rem;padding:1px 7px;border-radius:10px;background:#eee;margin-inline-start:6px;vertical-align:middle}
.badge.urgent{background:#fde4e4;color:#b02a2a}.badge.high{background:#fdf0d8;color:#8a5a00}.badge.positive{background:#dcf5ee;color:#06775f}.badge.negative{background:#fde4e4;color:#b02a2a}
.sev-high{color:#c03030;font-weight:600}.sev-medium{color:#a07010}.kpis{display:flex;gap:1rem;flex-wrap:wrap}
.kpi{border:1px solid #e3e5ee;border-radius:8px;padding:.6rem 1rem;min-width:160px;flex:1}.kpi b{display:block;font-size:1.3rem}
.kpi.positive{border-top:3px solid #00a88a}.kpi.negative{border-top:3px solid #e05555}.kpi.investigate{border-top:3px solid #d9a030}
.evidence-box{padding:.6rem .9rem;border-radius:6px;background:#f2f3f8;margin:.6rem 0}.evidence-box.insufficient,.evidence-box.limited{background:#fdf0d8}
.explain{font-size:.86rem;color:#444;margin:.3rem 0}.footer{margin-top:2rem;font-size:.78rem;color:#888;border-top:1px solid #ddd;padding-top:.6rem}
bdi{unicode-bidi:isolate}@media print{h2{page-break-after:avoid}.card,.ins{page-break-inside:avoid}}"""


def _doc(title, body, lang):
    d = "rtl" if lang == "ar" else "ltr"
    fonts = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;600&display=swap">'
             if lang == "ar" else "")
    return (f'<!doctype html><html lang="{lang}" dir="{d}"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{E(title)}</title>{fonts}'
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def _table(header, rows, lang="en"):
    head = "".join(f"<th>{E(str(h))}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{E(v) if isinstance(v, str) else _n(v, lang)}</td>" for v in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _h(m, lang):
    """Render a message as safe HTML (data values isolated with <bdi>)."""
    return render(m, lang, html=True)


def _card_html(card, lang):
    t = _tr(lang)
    cols = "".join(f'<span class="tag"><bdi>{E(c)}</bdi></span>' for c in card["evidence"]["columns"])
    ev = "".join(f"<li>{_h(e, lang)}</li>" for e in card["evidence"]["items"])
    lim = "".join(f"<li>{_h(x, lang)}</li>" for x in card["limitations"])
    badges = f'<span class="badge {card["priority"]}">{E(t("priority." + card["priority"]))}</span>'
    if card["direction"] in ("positive", "negative", "investigate"):
        badges += f'<span class="badge {card["direction"]}">{E(t("direction." + card["direction"]))}</span>'
    tech = f'<div class="tech">{E(t("card.technical"))}: {_h(card["technical"], lang)}</div>' if card.get("technical") else ""
    return (f'<div class="ins {card["priority"]} {card["direction"]}"><h4>{_h(card["title"], lang)}{badges}</h4><dl>'
            f'<dt>{E(t("card.finding"))}</dt><dd>{_h(card["finding"], lang)}</dd>'
            f'<dt>{E(t("card.why"))}</dt><dd>{_h(card["why"], lang)}</dd>'
            f'<dt>{E(t("card.action"))}</dt><dd>{_h(card["action"], lang)}</dd>'
            f'<dt>{E(t("card.evidence"))}</dt><dd><ul>{ev}</ul>{cols}</dd>'
            + (f'<dt>{E(t("card.limitations"))}</dt><dd><ul>{lim}</ul></dd>' if lim else "")
            + f"</dl>{tech}</div>")


def _kpis_html(summary, lang):
    out = []
    for k in summary["kpis"]:
        out.append(f'<div class="kpi {k["direction"]}">{_h(k["label"], lang)}<b>{E(_kpi_value(k, lang))}</b>'
                   f'<span class="muted">{_h(k.get("sub"), lang)}</span></div>')
    return '<div class="kpis">' + "".join(out) + "</div>"


def _summary_charts_html(summary, lang):
    parts = []
    for ch in summary["charts"]:
        if ch["type"] == "trend":
            labels = [format_period(p, ch["freq"], lang) for p in ch["labels"]]
            svg = svg_line(labels, ch["values"], render(ch["title"], lang), lang=lang)
        else:
            svg = svg_bars(ch["labels"], ch["values"], render(ch["title"], lang), PALETTE[1], lang=lang)
        parts.append(f'<div class="card"><h3>{_h(ch["title"], lang)}</h3>{svg}<p class="explain">{_h(ch["explanation"], lang)}</p></div>')
    return f'<div class="grid">{"".join(parts)}</div>' if parts else ""


def _executive_body(ds, lang, decisions=None):
    t = _tr(lang)
    st = ds.state
    s = st.summary
    by_id = {c["id"]: c for c in st.insights}
    parts = []
    parts.append(f"<h2>{E(t('rep.dataset_overview'))}</h2>" + "".join(f"<p>{_h(m, lang)}</p>" for m in s["description"]))
    parts.append(f'<div class="evidence-box {s["evidence"]["level"]}"><b>{E(t("sum.evidence_label"))}:</b> {_h(s["evidence"]["msg"], lang)}</div>')
    parts.append(_kpis_html(s, lang))
    for title_key, ids in (("sum.key_findings", s["findings"]), ("sum.risks", s["risks"]),
                           ("sum.opportunities", s["opportunities"]), ("sum.quality_warnings", s["quality_warnings"])):
        parts.append(f"<h2>{E(t(title_key))}</h2>")
        cards = [by_id[i] for i in ids if i in by_id]
        parts.append("".join(_card_html(c, lang) for c in cards) if cards else f'<p class="muted">{E(t("sum.none"))}</p>')
    charts = _summary_charts_html(s, lang)
    if charts:
        parts.append(f"<h2>{E(t('sum.charts'))}</h2>{charts}")
    def decision(i):
        if decisions is None:
            return ""
        d = decisions.get(i, "pending")
        return f' <span class="badge decision-{d}">{E(t("decision." + d))}</span>'

    parts.append(f"<h2>{E(t('sum.next_actions'))}</h2><ol>"
                 + "".join(f'<li><span class="badge {a["priority"]}">{E(t("priority." + a["priority"]))}</span> {_h(a["msg"], lang)}{decision(i)}</li>'
                           for i, a in enumerate(s["actions"])) + "</ol>")
    parts.append(f'<p class="muted">{E(t("help.quality_vs_business"))}</p>')
    return "".join(parts)


def executive_html(ds, lang="en", decisions=None):
    t = _tr(lang)
    st = ds.state
    head = (f"<h1>{E(t('rep.executive_title'))}</h1><p class='muted'><bdi>{E(ds.filename)}</bdi> · "
            f"{E(t('rep.generated', date=dt.datetime.now().strftime('%Y-%m-%d %H:%M')))}</p>")
    lim = f"<h2>{E(t('rep.limitations'))}</h2><ul>" + "".join(f"<li>{_h(x, lang)}</li>" for x in st.eda["limitations"]) + "</ul>"
    foot = f'<div class="footer">{E(t("rep.footer"))}</div>'
    return _doc(t("rep.executive_title"), head + _executive_body(ds, lang, decisions or {}) + lim + foot, lang)


def _cleaning_section(ds, lang):
    t = _tr(lang)
    rep = ds.state.report
    f = lambda v: format_number(v, lang)  # noqa: E731
    kpis = [(t("rep.metric.rows"), f"{f(rep['rows']['before'])} → {f(rep['rows']['after'])}"),
            (t("rep.metric.columns"), f"{f(rep['columns']['before'])} → {f(rep['columns']['after'])}"),
            (t("rep.metric.missing"), f"{f(rep['missing']['before']['total'])} → {f(rep['missing']['after']['total'])}"),
            (t("rep.metric.duplicates"), f"{f(rep['duplicates']['exact_before'])} → {f(rep['duplicates']['remaining_exact'])}"),
            (t("rep.metric.invalid"), f"{f(rep['invalid']['detected'])} → {f(rep['invalid']['unresolved'])}"),
            (t("rep.metric.changed"), f(rep['values']['changed'])), (t("rep.metric.excluded"), f(rep['values']['excluded']))]
    out = ['<div class="kpis">' + "".join(f'<div class="kpi">{E(k)}<b dir="ltr">{E(v)}</b></div>' for k, v in kpis) + "</div>"]
    applied = [o for o in ds.state.result.operations if o["status"] == "applied"]
    out.append(f"<h3>{E(t('rep.applied_ops'))}</h3>")
    out.append(_table([t("xl.col.title_localized"), t("xl.col.column"), t("xl.col.category"), t("xl.col.values_changed"),
                       t("xl.col.values_set_missing"), t("xl.col.rows_removed")],
                      [[render(o["title_msg"], lang), o["column"] or t("rep.dataset"), t(f"category.{o['category']}"),
                        o.get("values_changed"), o.get("values_set_missing"), o.get("rows_removed")] for o in applied], lang)
               if applied else f'<p class="muted">{E(t("rep.no_ops"))}</p>')
    if rep["pending_review"]:
        out.append(f"<h3>{E(t('rep.pending'))}</h3>")
        out.append(_table([t("xl.col.suggestion"), t("xl.col.column"), t("xl.col.affected")],
                          [[_plan_title(ds, p["id"], lang), p["column"] or t("rep.dataset"), p["affected"]]
                           for p in rep["pending_review"]], lang))
    return "".join(out)


def _issues_table(issues, lang):
    t = _tr(lang)
    if not issues:
        return f'<p class="muted">{E(t("rep.no_issues"))}</p>'
    rows = "".join(f'<tr><td class="sev-{E(i["severity"])}">{E(t("severity." + i["severity"]))}</td>'
                   f'<td><bdi>{E(str(i["column"] or t("rep.dataset")))}</bdi></td>'
                   f'<td>{_h(i.get("msg") or i["message"], lang)}</td></tr>' for i in issues)
    return (f"<table><thead><tr><th>{E(t('xl.col.severity'))}</th><th>{E(t('xl.col.column'))}</th>"
            f"<th>{E(t('xl.col.issue'))}</th></tr></thead><tbody>{rows}</tbody></table>")


def quality_html(ds, lang="en"):
    t = _tr(lang)
    st = ds.state
    rep = st.report
    per_col = _table([t(f"xl.col.{c}") for c in ("column", "final_name", "type_before", "type_after", "missing_before",
                                                 "missing_after", "invalid_before", "invalid_after", "changed")],
                     [[c["column"], c["final_name"] or t("rep.removed"), t(f"kind.{c['type_before']}"),
                       t(f"kind.{c['type_after']}") if c["type_after"] else "—", c["missing_before"], c["missing_after"],
                       c["invalid_before"], c["invalid_after"], c["changed"]] for c in rep["per_column"]], lang)
    body = (f"<h1>{E(t('rep.quality_title'))}</h1><p class='muted'><bdi>{E(ds.filename)}</bdi> · "
            f"{E(t('rep.generated', date=dt.datetime.now().strftime('%Y-%m-%d %H:%M')))}</p>"
            f"<p class='muted'>{E(t('help.quality_vs_business'))}</p>"
            f"<h2>{E(t('rep.before_after'))}</h2>{_cleaning_section(ds, lang)}"
            f"<h2>{E(t('rep.per_column'))}</h2>{per_col}"
            f"<h2>{E(t('rep.remaining_issues'))}</h2>{_issues_table(st.profile['issues'], lang)}"
            f"<h2>{E(t('rep.original_issues'))}</h2>{_issues_table(ds.profile['issues'], lang)}"
            f'<div class="footer">{E(t("rep.footer"))}</div>')
    return _doc(t("rep.quality_title"), body, lang)


def analysis_html(ds, lang="en"):
    t = _tr(lang)
    st = ds.state
    prof, eda = st.profile, st.eda
    by = {c["name"]: c for c in prof["columns"]}
    parts = [f"<h1>{E(t('rep.analysis_title'))}</h1><p class='muted'><bdi>{E(ds.filename)}</bdi> · "
             f"{E(t('rep.size', rows=prof['row_count'], columns=prof['column_count']))} · "
             f"{E(t('rep.generated', date=dt.datetime.now().strftime('%Y-%m-%d %H:%M')))}. {E(t('rep.computed_note'))}</p>"]

    parts.append(f"<h2>{E(t('rep.executive_title'))}</h2>" + _executive_body(ds, lang))

    parts.append(f"<h2>{E(t('rep.key_insights'))}</h2>")
    parts.append("".join(_card_html(c, lang) for c in st.insights))

    parts.append(f"<h2>{E(t('rep.cleaning_summary'))}</h2>" + _cleaning_section(ds, lang))
    parts.append(f"<h2>{E(t('rep.detailed'))}</h2>")

    measures = eda["columns_used"]["measures"]
    if measures:
        parts.append(f"<h3>{E(t('rep.numeric'))}</h3>")
        parts.append(_table([t(f"xl.col.{c}") for c in ("column", "count", "mean", "median", "std", "min", "q1", "q3",
                                                        "max", "skew", "outliers_mild")],
                            [[m, by[m]["numeric"]["count"], by[m]["numeric"]["mean"], by[m]["numeric"]["median"],
                              by[m]["numeric"]["std"], by[m]["numeric"]["min"], by[m]["numeric"]["q1"], by[m]["numeric"]["q3"],
                              by[m]["numeric"]["max"], by[m]["numeric"]["skew"], by[m]["numeric"]["outliers_mild"]]
                             for m in measures if by[m]["numeric"]], lang))
        charts = []
        for m in measures[:8]:
            num = by[m]["numeric"]
            if num and num["histogram"]:
                hgm = num["histogram"]
                labels = [f"{hgm['edges'][i]:.3g}–{hgm['edges'][i + 1]:.3g}" for i in range(len(hgm["counts"]))]
                title = t("chart.distribution_of", column=m)
                charts.append(f'<div class="card"><h3>{E(title)}</h3>{svg_bars(labels, hgm["counts"], title, lang=lang)}'
                              f'{svg_box(m, num["box"], num["min"], num["max"])}</div>')
        parts.append(f'<div class="grid">{"".join(charts)}</div>')

    cats = eda["columns_used"]["categories"]
    if cats:
        parts.append(f"<h3>{E(t('rep.categorical'))}</h3><div class='grid'>")
        for c in cats[:8]:
            tv = by[c]["top_values"]
            title = t("chart.frequency_title", column=c)
            parts.append(f'<div class="card"><h3>{E(title)}</h3>{svg_bars([x["value"] for x in tv], [x["count"] for x in tv], title, PALETTE[1], lang=lang)}</div>')
        parts.append("</div>")

    if eda["trends"]:
        parts.append(f"<h3>{E(t('rep.trends'))}</h3><div class='grid'>")
        for tr in eda["trends"]:
            labels = [format_period(p, tr["freq"], lang) for p in tr["periods"]]
            title = t("chart.trend_title_count", freq=K(f"freq.{tr['freq']}"))
            parts.append(f'<div class="card"><h3>{E(title)}</h3>{svg_line(labels, tr["counts"], title, lang=lang)}</div>')
            for m, v in tr["measures"].items():
                title = t(f"chart.trend_title_{v['agg']}", column=m, freq=K(f"freq.{tr['freq']}"))
                parts.append(f'<div class="card"><h3>{E(title)}</h3>{svg_line(labels, v[v["agg"]], title, PALETTE[0], lang=lang)}</div>')
        parts.append(f"</div><p class='muted'>{_h(eda['trends'][0]['note'], lang)}</p>")

    rel = [r for r in eda["relationships"] if r["kind"] != "categorical-numeric"][:15]
    if rel:
        parts.append(f"<h3>{E(t('rep.relationships'))}</h3>")
        parts.append(_table([t(f"xl.col.rel_{c}") for c in ("a", "b", "method", "value", "strength", "q_value", "n")],
                            [[r["a"], r["b"], t(f"method.{r['method']}"), r["value"], t(f"strength.{r['strength']}"),
                              r["q_value"], r["n"]] for r in rel], lang))
    if eda["groups"]:
        parts.append(f"<h3>{E(t('rep.groups'))}</h3><div class='grid'>")
        for g in eda["groups"][:6]:
            lv = g["groups"][:15]
            title = t("chart.breakdown_title_median", measure=g["measure"], group=g["group"])
            parts.append(f'<div class="card"><h3>{E(title)}</h3>{svg_bars([str(x["level"]) for x in lv], [x["median"] for x in lv], title, PALETTE[3], lang=lang)}'
                         f'<p class="muted">{_h(M("tech.kruskal", q=Q(g["q_value"]), effect=g["effect_size"], count=g["n"]), lang)}</p></div>')
        parts.append("</div>")

    miss = [m for m in eda["missing"] if m["missing"]]
    if miss:
        title = t("chart.missing_title")
        parts.append(f"<h3>{E(t('rep.missing'))}</h3>" + svg_bars([m["column"] for m in miss], [m["pct"] for m in miss], title, PALETTE[2], lang=lang))

    parts.append(f"<h2>{E(t('rep.limitations'))}</h2><ul>" + "".join(f"<li>{_h(x, lang)}</li>" for x in eda["limitations"]) + "</ul>")
    parts.append(f'<div class="footer">{E(t("rep.footer"))}</div>')
    return _doc(t("rep.analysis_title"), "".join(parts), lang)
