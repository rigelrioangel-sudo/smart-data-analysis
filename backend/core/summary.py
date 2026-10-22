"""Executive summary for non-technical readers.

Built only from computed results (cleaned profile, cleaning report, EDA and
insight cards). Nothing here invents numbers: every KPI is a direct
computation and every statement references an insight card or a profile
value. The dataset "type" is only suggested when at least two column names
support it, and the supporting columns are listed.
"""
from .i18n import D, K, M
from .insights import PRIORITY_ORDER, main_measure, measure_polarity
from .eda import is_additive
from .utils import name_tokens, rnd

DOMAINS = {
    "sales": {"sales", "revenue", "order", "orders", "price", "units", "quantity", "qty", "customer", "product", "discount", "sku"},
    "finance": {"amount", "balance", "account", "transaction", "expense", "expenses", "cost", "budget", "invoice", "payment", "debit", "credit"},
    "hr": {"employee", "salary", "department", "hire", "hired", "tenure", "position", "manager", "gender", "absence", "absences"},
    "web": {"visits", "views", "clicks", "sessions", "page", "bounce", "users", "pageviews", "conversion", "conversions"},
    "inventory": {"stock", "inventory", "warehouse", "supplier", "reorder", "shipment", "shipments"},
    "survey": {"rating", "score", "satisfaction", "response", "respondent", "answer", "question", "feedback", "nps"},
    "health": {"patient", "diagnosis", "treatment", "hospital", "bmi", "blood", "dose", "symptom"},
    "education": {"student", "grade", "course", "exam", "school", "class", "teacher", "attendance"},
}


def infer_domain(columns):
    """Suggest what the data describes, only when ≥2 columns support the same domain."""
    best, best_cols = None, []
    for domain, vocab in DOMAINS.items():
        hits = [c for c in columns if set(name_tokens(c)) & vocab]
        if len(hits) >= 2 and len(hits) > len(best_cols):
            best, best_cols = domain, hits
    return {"domain": best, "columns": best_cols[:5]} if best else None


def _date_range(profile, name):
    c = next((c for c in profile["columns"] if c["name"] == name), None)
    return c["datetime"] if c and c.get("datetime") else None


def build_summary(profile, report, eda, insights):
    used = eda["columns_used"]
    by_name = {c["name"]: c for c in profile["columns"]}
    ids = [c["name"] for c in profile["columns"] if c["role"] == "identifier"]
    n = profile["row_count"]

    # What the dataset describes
    description = [M("sum.desc.records", count=n, columns=profile["column_count"])]
    domain = infer_domain([c["name"] for c in profile["columns"]])
    if domain:
        description.append(M("sum.desc.domain", domain=K(f"domain.{domain['domain']}"), columns=domain["columns"]))
    if used["measures"]:
        description.append(M("sum.desc.measures", columns=used["measures"][:6]))
    if used["categories"]:
        description.append(M("sum.desc.categories", columns=used["categories"][:6]))
    period = None
    if used["dates"]:
        rng = _date_range(profile, used["dates"][0])
        if rng:
            period = {"column": used["dates"][0], **rng}
            description.append(M("sum.desc.period", column=used["dates"][0], start=D(rng["min"]), end=D(rng["max"])))
    else:
        description.append(K("sum.desc.no_dates"))
    if ids:
        description.append(M("sum.desc.identifiers", columns=ids[:4]))

    # KPIs (direct computations only)
    kpis = [{"id": "records", "label": K("kpi.records"), "value": n,
             "sub": M("kpi.records_sub", count=report["rows"]["before"]) if report["rows"]["before"] != n else K("kpi.records_all"),
             "direction": "neutral"}]
    if period:
        kpis.append({"id": "period", "label": K("kpi.period"),
                     "value": M("kpi.period_value", start=D(period["min"]), end=D(period["max"])),
                     "sub": M("kpi.days", count=period["range_days"], column=period["column"]), "direction": "neutral"})
    main = main_measure(eda)
    trend_card = None
    if main:
        num = by_name[main]["numeric"]
        additive = is_additive(main) and (num["min"] or 0) >= 0
        trend_card = next((c for c in insights if c["category"] == "trend" and main in c["evidence"]["columns"]), None)
        kpis.append({"id": "main_measure", "column": main,
                     "label": M("kpi.total_of" if additive else "kpi.typical_of", column=main),
                     "value": num["sum"] if additive else num["median"],
                     "sub": (trend_card["title"] if trend_card else M("kpi.measure_sub", count=num["count"])),
                     "direction": trend_card["direction"] if trend_card else "neutral"})
    complete = rnd(100 - (profile["missing_pct"] or 0), 1)
    open_issues = profile["issue_counts"].get("high", 0) + profile["issue_counts"].get("medium", 0)
    kpis.append({"id": "completeness", "label": K("kpi.completeness"), "value": complete, "unit": "pct",
                 "sub": M("kpi.open_issues", count=open_issues),
                 "direction": "positive" if complete >= 95 and not open_issues else "negative" if complete < 80 else "investigate"})

    business = [c for c in insights if c["kind"] == "business"]
    quality = [c for c in insights if c["kind"] == "data_quality"]
    findings = sorted(business, key=lambda c: (PRIORITY_ORDER[c["priority"]], -c["score"]))[:5]
    if not findings:
        findings = [c for c in insights if c["kind"] == "observation"][:3]
    risks = [c for c in business if c["risk"] or c["direction"] == "negative"][:4]
    risks += [c for c in quality if c["priority"] in ("urgent", "high")][:4]
    opportunities = [c for c in business if c["opportunity"]][:4]
    warnings_ = [c for c in quality if c["priority"] in ("urgent", "high", "medium")][:6]

    # Recommended actions, data fixes first; deduplicated
    actions, seen = [], set()
    if report["pending_review"]:
        actions.append({"priority": "high", "type": "fix_data", "msg": M("act.review_pending", count=len(report["pending_review"])),
                        "insight": None})
        seen.add("pending")
    for c in sorted(insights, key=lambda c: (KIND_RANK[c["kind"]], PRIORITY_ORDER[c["priority"]], -c["score"])):
        sig = repr(c["action"])
        if sig in seen or c["kind"] == "observation" and c["category"] != "no_findings":
            continue
        seen.add(sig)
        actions.append({"priority": c["priority"], "insight": c["id"], "msg": c["action"],
                        "type": "fix_data" if c["kind"] == "data_quality" else "investigate"})
        if len(actions) >= 8:
            break

    # Evidence level for the overall conclusions
    significant = len(business)
    if n < 30 or not significant:
        evidence = {"level": "insufficient" if not significant else "limited",
                    "msg": M("sum.evidence.insufficient", count=n) if not significant else M("sum.evidence.limited", count=n)}
    elif any(c["priority"] == "urgent" for c in quality):
        evidence = {"level": "limited", "msg": K("sum.evidence.quality")}
    else:
        evidence = {"level": "good", "msg": M("sum.evidence.good", count=significant)}

    return {
        "description": description, "domain": domain, "kpis": kpis,
        "findings": [c["id"] for c in findings], "risks": _unique([c["id"] for c in risks]),
        "opportunities": [c["id"] for c in opportunities], "quality_warnings": [c["id"] for c in warnings_],
        "actions": actions, "evidence": evidence, "main_measure": main,
        "charts": _charts(profile, eda, insights, main),
    }


KIND_RANK = {"data_quality": 0, "business": 1, "observation": 2}


def _unique(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def _charts(profile, eda, insights, main):
    charts = []
    if eda["trends"]:
        t = eda["trends"][0]
        series, agg = None, "count"
        if main and main in t["measures"]:
            series, agg = main, t["measures"][main]["agg"]
        values = t["counts"] if series is None else t["measures"][series][agg]
        card = next((c for c in insights if c["category"] == "trend" and
                     (series in c["evidence"]["columns"] if series else len(c["evidence"]["columns"]) == 1)), None)
        charts.append({
            "type": "trend", "date_column": t["date_column"], "freq": t["freq"], "series": series, "agg": agg,
            "labels": t["periods"], "values": values,
            "title": M(f"chart.trend_title_{agg}", column=series, freq=K(f"freq.{t['freq']}")) if series
            else M("chart.trend_title_count", freq=K(f"freq.{t['freq']}")),
            "explanation": card["finding"] if card else K("chart.no_clear_trend"),
            "direction": card["direction"] if card else "neutral", "insight": card["id"] if card else None,
            "note": t["note"],
        })
    groups = [g for g in eda["groups"] if not main or g["measure"] == main] or eda["groups"]
    if groups:
        g = sorted(groups, key=lambda x: (not x["significant"], -(x["effect_size"] or 0)))[0]
        agg = "sum" if g["additive"] else "median"
        levels = sorted(g["groups"], key=lambda x: -(x[agg] or 0))[:12]
        card = next((c for c in insights if c["category"] == "difference" and
                     set(c["evidence"]["columns"]) == {g["group"], g["measure"]}), None)
        charts.append({
            "type": "breakdown", "group": g["group"], "measure": g["measure"], "agg": agg,
            "labels": [str(x["level"]) for x in levels], "values": [x[agg] for x in levels],
            "counts": [x["count"] for x in levels],
            "title": M(f"chart.breakdown_title_{agg}", measure=g["measure"], group=g["group"]),
            "explanation": card["finding"] if card else K("chart.breakdown_not_significant"),
            "direction": "investigate" if card else "neutral", "insight": card["id"] if card else None,
        })
    elif eda["columns_used"]["categories"]:
        name = eda["columns_used"]["categories"][0]
        c = next(c for c in profile["columns"] if c["name"] == name)
        charts.append({"type": "frequency", "column": name, "labels": [t["value"] for t in c["top_values"]],
                       "values": [t["count"] for t in c["top_values"]], "title": M("chart.frequency_title", column=name),
                       "explanation": K("chart.frequency_note"), "direction": "neutral", "insight": None})
    return charts


__all__ = ["build_summary", "infer_domain", "measure_polarity"]
