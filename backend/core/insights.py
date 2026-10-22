"""Evidence-based, action-oriented insight cards.

Every card is generated from a computed statistic (profile, cleaning report
or EDA result). A card explains, in plain language:

    finding      what the data shows
    why          why it may matter
    action       what a manager could do next
    evidence     the columns and computed values behind it
    limitations  uncertainty / missing information

All text is a translatable message ({"key", "params"}); data values are
passed through unchanged. Rules fire only when thresholds are met, nothing is
inferred beyond the data, and relationships are always called associations.

kind:      data_quality | business | observation
priority:  urgent | high | medium | low     (data-driven: severity / effect size)
direction: positive | negative | investigate | neutral
           (positive/negative only when the column's meaning makes it clear,
            e.g. rising revenue or rising costs; otherwise "investigate")
"""
from .eda import is_additive
from .i18n import K, M, P, Q
from .utils import name_tokens, rnd

PRIORITY_ORDER = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
KIND_ORDER = {"data_quality": 0, "business": 1, "observation": 2}

# Column-name vocabularies used only to decide whether "up" is good or bad
UP_IS_GOOD = {"revenue", "sales", "profit", "profits", "income", "orders", "visits", "views", "clicks", "customers",
              "conversions", "users", "margin", "rating", "satisfaction", "retention", "signups", "subscribers"}
UP_IS_BAD = {"cost", "costs", "expense", "expenses", "returns", "returned", "churn", "complaints", "defects", "errors",
             "delay", "delays", "refunds", "absences", "incidents", "downtime", "cancellations", "losses", "debt"}
MAIN_MEASURE_PRIORITY = ["revenue", "sales", "profit", "income", "amount", "total", "spend", "value", "units",
                         "quantity", "qty", "orders", "visits", "count"]


def measure_polarity(name):
    tokens = set(name_tokens(name))
    if tokens & UP_IS_BAD:
        return "up_bad"
    if tokens & UP_IS_GOOD:
        return "up_good"
    return None


def direction_for_change(name, increased):
    pol = measure_polarity(name)
    if pol == "up_good":
        return "positive" if increased else "negative"
    if pol == "up_bad":
        return "negative" if increased else "positive"
    return "investigate"


def _card(cards, category, kind, priority, title, finding, why, action, evidence, limitations=None,
          technical=None, direction="neutral", score=0.0, risk=False, opportunity=False, columns=None, stats=None):
    cards.append({
        "id": f"{category}-{len(cards) + 1}", "category": category, "kind": kind, "priority": priority,
        "direction": direction, "risk": risk, "opportunity": opportunity, "score": round(float(score), 4),
        "title": title, "finding": finding, "why": why, "action": action,
        "evidence": {"columns": columns or [], "items": evidence, "stats": stats or {}},
        "limitations": limitations or [], "technical": technical,
    })


def _quality_cards(cards, profile_after, report, eda):
    used = set(eda["columns_used"]["measures"] + eda["columns_used"]["categories"] + eda["columns_used"]["dates"])
    by_name = {c["name"]: c for c in profile_after["columns"]}
    n = profile_after["row_count"]

    dups = report["duplicates"]["remaining_exact"]
    if dups:
        _card(cards, "quality", "data_quality", "urgent", M("ins.dup.title", count=dups), M("ins.dup.finding", count=dups),
              K("ins.dup.why"), K("ins.dup.action"), [M("ev.duplicates", count=dups, total=n)],
              direction="negative", score=100 + dups / max(1, n) * 100, risk=True, stats={"duplicates": dups, "rows": n})

    for k in profile_after["key_duplicates"]:
        _card(cards, "quality", "data_quality", "urgent", M("ins.keydup.title", column=k["column"]),
              M("ins.keydup.finding", count=k["duplicated_values"], extra=k["extra_rows"], column=k["column"]),
              K("ins.keydup.why"), K("ins.keydup.action"),
              [M("ev.value_times", value=e["value"], count=e["count"]) for e in k["examples"][:3]],
              [K("ins.keydup.limit")], direction="negative", score=95 + k["extra_rows"] / max(1, n) * 100, risk=True,
              columns=[k["column"]], stats={"duplicated_values": k["duplicated_values"], "extra_rows": k["extra_rows"]})

    for c in profile_after["columns"]:
        if c["kind"] == "empty":
            _card(cards, "quality", "data_quality", "low", M("ins.empty_col.title", column=c["name"]),
                  M("ins.empty_col.finding", column=c["name"]), K("ins.empty_col.why"), K("ins.empty_col.action"),
                  [M("ev.missing", count=c["missing_total"], total=c["count"])], direction="neutral", score=10,
                  columns=[c["name"]], stats={"missing": c["missing_total"]})
            continue
        if c["missing_pct"] and c["missing_pct"] >= 20:
            important = c["name"] in used
            pri = ("urgent" if c["missing_pct"] >= 50 else "high") if important else "medium"
            _card(cards, "quality", "data_quality", pri, M("ins.missing.title", column=c["name"], pct=c["missing_pct"]),
                  M("ins.missing.finding", column=c["name"], count=c["missing_total"], total=c["count"], pct=c["missing_pct"]),
                  K("ins.missing.why"), K("ins.missing.action"), [M("ev.missing", count=c["missing_total"], total=c["count"])],
                  [K("ins.missing.limit")], direction="negative", score=60 + c["missing_pct"] / 2 + (20 if important else 0),
                  risk=important, columns=[c["name"]], stats={"missing": c["missing_total"], "missing_pct": c["missing_pct"]})
        if c["invalid_count"]:
            _card(cards, "quality", "data_quality", "high", M("ins.invalid.title", column=c["name"]),
                  M("ins.invalid.finding", count=c["invalid_count"], column=c["name"], kind=K(f"kind_value.{c['kind']}"),
                    examples=[e["value"] for e in c["invalid_examples"][:3]]),
                  K("ins.invalid.why"), K("ins.invalid.action"),
                  [M("ev.value_times", value=e["value"], count=e["count"]) for e in c["invalid_examples"][:3]],
                  direction="negative", score=70 + c["invalid_count"] / max(1, c["count"]) * 100, risk=True,
                  columns=[c["name"]], stats={"invalid": c["invalid_count"]})
        for r in c["impossible"]:
            _card(cards, "quality", "data_quality", "urgent", M("ins.impossible.title", column=c["name"]),
                  M("ins.impossible.finding", count=r["count"], column=c["name"], rule=r["label"]),
                  K("ins.impossible.why"), K("ins.impossible.action"),
                  [M("ev.examples", examples=[str(e) for e in r["examples"][:3]])],
                  direction="negative", score=90 + r["count"] / max(1, c["count"]) * 100, risk=True,
                  columns=[c["name"]], stats={"count": r["count"], "rule": r["rule"]})

    for o in eda["outliers"]:
        if o["extreme"]:
            num = by_name[o["column"]]["numeric"]
            _card(cards, "quality", "data_quality", "medium", M("ins.outliers.title", column=o["column"]),
                  M("ins.outliers.finding", count=o["extreme"], column=o["column"], max=num["max"], median=num["median"]),
                  K("ins.outliers.why"), K("ins.outliers.action"),
                  [M("ev.outlier_range", low=num["outlier_bounds"]["extreme"][0], high=num["outlier_bounds"]["extreme"][1]),
                   M("ev.max_median", max=num["max"], median=num["median"])],
                  [K("ins.outliers.limit")], direction="investigate", score=40 + o["extreme"] / max(1, num["count"]) * 100,
                  columns=[o["column"]], stats={"extreme": o["extreme"], "mild": o["mild"], "max": num["max"], "median": num["median"]},
                  technical=M("tech.outliers", mild=o["mild"], extreme=o["extreme"]))


def _trend_cards(cards, eda):
    for t in eda["trends"]:
        for test in t["tests"]:
            if not (test["q_value"] is not None and test["q_value"] < 0.05 and abs(test["spearman"] or 0) >= 0.5):
                continue
            up = test["spearman"] > 0
            is_count = test["series"] == "record count"
            subject = K("ins.trend.records") if is_count else M(f"ins.trend.subject_{test['agg']}", column=test["series"])
            direction = "investigate" if is_count else direction_for_change(test["series"], up)
            pct = test["pct_change"]
            finding = M("ins.trend.finding_up" if up else "ins.trend.finding_down", subject=subject,
                        first=test["first"], first_period=P(test["first_period"], t["freq"]),
                        last=test["last"], last_period=P(test["last_period"], t["freq"]))
            why = {"positive": K("ins.trend.why_positive"), "negative": K("ins.trend.why_negative")}.get(direction, K("ins.trend.why_neutral"))
            evidence = [M("ev.trend_points", first=test["first"], first_period=P(test["first_period"], t["freq"]),
                          last=test["last"], last_period=P(test["last_period"], t["freq"])),
                        M("ev.periods", count=test["periods"], freq=K(f"freq.{t['freq']}"), column=t["date_column"])]
            if pct is not None:
                evidence.append(M("ev.pct_change", pct=pct))
            strong = abs(test["spearman"]) >= 0.7
            _card(cards, "trend", "business", "high" if strong else "medium",
                  M("ins.trend.title_up" if up else "ins.trend.title_down", subject=subject), finding, why,
                  M("ins.trend.action", freq=K(f"freq.{t['freq']}")), evidence,
                  [K("ins.trend.limit_partial"), K("lim.causation_short")],
                  technical=M("tech.trend", rho=test["spearman"], q=Q(test["q_value"]), count=test["periods"]),
                  direction=direction, score=abs(test["spearman"]) * 60 + (10 if not is_count else 0),
                  risk=direction == "negative", opportunity=direction == "positive",
                  columns=[t["date_column"]] + ([] if is_count else [test["series"]]),
                  stats={k: test[k] for k in ("spearman", "q_value", "first", "last", "pct_change", "periods")})


def _relationship_cards(cards, eda):
    rels = [r for r in eda["relationships"] if r["significant"] and abs(r["value"] or 0) >= 0.3
            and r["n"] >= 10 and r["kind"] != "categorical-numeric"]
    for r in rels[:4]:
        strength = K(f"strength.{r['strength']}")
        if r["kind"] == "numeric-numeric":
            pos = r["value"] > 0
            finding = M("ins.rel.finding_pos" if pos else "ins.rel.finding_neg", a=r["a"], b=r["b"], strength=strength)
            title = M("ins.rel.title_pos" if pos else "ins.rel.title_neg", a=r["a"], b=r["b"])
            technical = M("tech.corr", rho=r["value"], r=r["pearson"], q=Q(r["q_value"]), count=r["n"])
        else:
            finding = M("ins.rel.finding_cat", a=r["a"], b=r["b"], strength=strength)
            title = M("ins.rel.title_cat", a=r["a"], b=r["b"])
            technical = M("tech.cramers", v=r["value"], q=Q(r["q_value"]), count=r["n"])
        limits = [K("lim.causation_short")]
        if r.get("caution"):
            limits.append(r["caution"])
        _card(cards, "relationship", "business", "high" if abs(r["value"]) >= 0.5 else "medium", title, finding,
              K("ins.rel.why"), K("ins.rel.action"), [M("ev.records_compared", count=r["n"]), M("ev.strength", strength=strength)],
              limits, technical=technical, direction="investigate", score=abs(r["value"]) * 50,
              columns=[r["a"], r["b"]], stats={k: r.get(k) for k in ("method", "value", "pearson", "q_value", "n")})


def _difference_cards(cards, eda):
    for g in [g for g in eda["groups"] if g["significant"] and (g["effect_size"] or 0) >= 0.06][:4]:
        hi, lo = g["groups"][0], g["groups"][-1]
        additive = g["additive"] and lo["median"] is not None and lo["median"] > 0
        evidence = [M("ev.group_value", group=str(hi["level"]), value=hi["median"], count=hi["count"]),
                    M("ev.group_value", group=str(lo["level"]), value=lo["median"], count=lo["count"]),
                    M("ev.groups_compared", count=len(g["groups"]), total=g["n"])]
        if additive:
            evidence.append(M("ev.ratio", ratio=rnd(hi["median"] / lo["median"], 2)))
        large = g["effect_size"] >= 0.14
        _card(cards, "difference", "business", "high" if large else "medium",
              M("ins.diff.title", measure=g["measure"], group=g["group"]),
              M("ins.diff.finding", measure=g["measure"], group=g["group"], high=str(hi["level"]), high_value=hi["median"],
                low=str(lo["level"]), low_value=lo["median"]),
              K("ins.diff.why"), M("ins.diff.action", high=str(hi["level"]), low=str(lo["level"])), evidence,
              [K("ins.diff.limit_median"), K("lim.causation_short")],
              technical=M("tech.kruskal", q=Q(g["q_value"]), effect=g["effect_size"], count=g["n"]),
              direction="investigate", score=g["effect_size"] * 100 + (5 if additive else 0), opportunity=additive,
              columns=[g["group"], g["measure"]],
              stats={"q_value": g["q_value"], "effect_size": g["effect_size"], "highest": hi, "lowest": lo})


def _concentration_cards(cards, eda):
    made = 0
    for g in eda["groups"]:
        if made >= 2 or not g["additive"] or len(g["groups"]) < 3:
            continue
        sums = sorted((x for x in g["groups"] if x["sum"] is not None), key=lambda x: -x["sum"])
        total = sum(x["sum"] for x in sums)
        if total <= 0 or any(x["sum"] < 0 for x in sums):
            continue
        share = sums[0]["sum"] / total * 100
        if share < 40:
            continue
        made += 1
        _card(cards, "concentration", "business", "medium",
              M("ins.conc.title", level=str(sums[0]["level"]), share=rnd(share, 1), measure=g["measure"]),
              M("ins.conc.finding", level=str(sums[0]["level"]), share=rnd(share, 1), measure=g["measure"],
                count=len(sums), group=g["group"]),
              K("ins.conc.why"), M("ins.conc.action", level=str(sums[0]["level"])),
              [M("ev.share", level=str(sums[0]["level"]), value=rnd(sums[0]["sum"], 2), total=rnd(total, 2))],
              [K("ins.conc.limit")], direction="investigate", score=share / 2, risk=True, opportunity=True,
              columns=[g["group"], g["measure"]],
              stats={"top_level": sums[0]["level"], "share_pct": rnd(share, 1), "total": rnd(total, 2)})


def _observation_cards(cards, profile_after, eda):
    by_name = {c["name"]: c for c in profile_after["columns"]}
    for name in eda["columns_used"]["measures"]:
        num = by_name[name]["numeric"]
        if num and num["skew"] is not None and num["skew"] > 1 and num["count"] >= 20:
            _card(cards, "pattern", "observation", "low", M("ins.skew.title", column=name),
                  M("ins.skew.finding", column=name, median=num["median"], mean=num["mean"]),
                  K("ins.skew.why"), M("ins.skew.action", column=name),
                  [M("ev.mean_median", mean=num["mean"], median=num["median"])],
                  technical=M("tech.skew", skew=num["skew"]), direction="neutral", score=5,
                  columns=[name], stats={"skew": num["skew"], "mean": num["mean"], "median": num["median"]})
    for name in eda["columns_used"]["categories"]:
        c = by_name[name]
        dom = c["dominant"]
        if dom and dom["pct"] and dom["pct"] >= 60 and c["unique"] >= 2 and c["non_missing"] >= 20:
            _card(cards, "pattern", "observation", "low", M("ins.dominant.title", value=dom["value"], column=name),
                  M("ins.dominant.finding", value=dom["value"], pct=dom["pct"], column=name, count=dom["count"],
                    total=c["non_missing"]),
                  K("ins.dominant.why"), K("ins.dominant.action"), [M("ev.share_count", count=dom["count"], total=c["non_missing"])],
                  direction="neutral", score=4, columns=[name], stats={"share_pct": dom["pct"]})


def _no_findings_card(cards, profile_after, eda):
    used = eda["columns_used"]
    reasons = []
    if profile_after["row_count"] < 30:
        reasons.append(M("ins.none.reason_small", count=profile_after["row_count"]))
    if not used["measures"]:
        reasons.append(K("ins.none.reason_no_measures"))
    if not used["dates"]:
        reasons.append(K("ins.none.reason_no_dates"))
    if not used["categories"]:
        reasons.append(K("ins.none.reason_no_categories"))
    _card(cards, "no_findings", "observation", "medium", K("ins.none.title"), K("ins.none.finding"), K("ins.none.why"),
          K("ins.none.action"), reasons or [K("ins.none.reason_weak")], direction="neutral", score=1)


def generate_insights(profile_before, profile_after, report, eda, plan, decisions) -> list:
    cards = []
    _quality_cards(cards, profile_after, report, eda)
    _trend_cards(cards, eda)
    _relationship_cards(cards, eda)
    _difference_cards(cards, eda)
    _concentration_cards(cards, eda)
    _observation_cards(cards, profile_after, eda)
    if not any(c["kind"] == "business" for c in cards):
        _no_findings_card(cards, profile_after, eda)
    cards.sort(key=lambda c: (KIND_ORDER[c["kind"]], PRIORITY_ORDER[c["priority"]], -c["score"]))
    for i, c in enumerate(cards, 1):
        c["rank"] = i
    return cards


def main_measure(eda):
    measures = eda["columns_used"]["measures"]
    if not measures:
        return None
    for token in MAIN_MEASURE_PRIORITY:
        for m in measures:
            if token in name_tokens(m):
                return m
    return measures[0]
