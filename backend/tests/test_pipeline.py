"""Unit tests for loading, inference, profiling, cleaning, EDA and insights."""
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from core import loader  # noqa: E402
from core.cleaning import DecisionError, apply_cleaning, build_plan, default_decisions, resolve_decisions  # noqa: E402
from core.eda import analysis_frame, run_eda  # noqa: E402
from core.i18n import render  # noqa: E402
from core.inference import coerce_series, infer_column, infer_frame, parse_number  # noqa: E402
from core.insights import generate_insights  # noqa: E402
from core.profiling import profile_dataframe  # noqa: E402
from sample_data import sample_csv_bytes  # noqa: E402


def load(text, name="t.csv"):
    df, _ = loader.load_dataframe(name, text.encode() if isinstance(text, str) else text)
    return df


def plan_for(df):
    return build_plan(df, profile_dataframe(df))


def ids(plan, category=None):
    return {a["id"] for a in plan if category is None or a["category"] == category}


@pytest.fixture(scope="module")
def sales():
    raw = load(sample_csv_bytes(), "sales.csv")
    profile = profile_dataframe(raw)
    plan = build_plan(raw, profile)
    return raw, profile, plan


# ─── Loader ──────────────────────────────────────────────────

def test_loader_preserves_original_strings():
    df = load("a,b\n 007 ,N/A\n1.50,\n")
    assert df["a"].tolist() == [" 007 ", "1.50"]
    assert df["b"].tolist() == ["N/A", ""]


@pytest.mark.parametrize("name,content,message", [
    ("x.txt", b"a\n1", "Unsupported file type"),
    ("x.csv", b"", "Empty file"),
    ("x.csv", b"   \n\n", "no data"),
    ("x.csv", b"a,b\x00\x00\x01", "binary"),
    ("x.json", b"{broken", "malformed"),
    ("x.json", b'{"a": [1, 2], "b": [1]}', "different lengths"),
    ("x.xlsx", b"not a zip file at all", "not a valid Excel"),
    ("x.xls", b"PK\x03\x04garbage", "could not be read"),
])
def test_loader_rejects_bad_files(name, content, message):
    with pytest.raises(loader.LoadError) as e:
        loader.load_dataframe(name, content)
    assert message.lower() in e.value.message.lower()


def test_loader_size_limit(monkeypatch):
    monkeypatch.setattr(loader, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(loader.LoadError) as e:
        loader.load_dataframe("x.csv", b"a,b\n1,2\n3,4\n")
    assert e.value.status_code == 413


def test_loader_malformed_csv_is_reported_not_dropped():
    with pytest.raises(loader.LoadError) as e:
        loader.load_dataframe("x.csv", b"a,b\n1,2\n3,4,5,6\n")
    assert "malformed" in e.value.message


def test_excel_multiple_sheets_noted():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame({"a": [1]}).to_excel(w, sheet_name="First", index=False)
        pd.DataFrame({"b": [2]}).to_excel(w, sheet_name="Second", index=False)
    df, meta = loader.load_dataframe("x.xlsx", buf.getvalue())
    assert list(df.columns) == ["a"] and meta["sheet"] == "First"
    assert any("only the first sheet" in render(n, "en") for n in meta["notes"])


# ─── Inference ───────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    ("12", (12.0, "plain")), ("-3.5", (-3.5, "plain")), ("1,234.5", (1234.5, "thousands")),
    ("$12.50", (12.5, "currency")), ("-$4", (-4.0, "currency")), ("45%", (45.0, "percent")),
    ("abc", (None, None)), ("inf", (None, None)), ("1.2.3", (None, None)), (True, (None, None)), (7, (7.0, "native")),
])
def test_parse_number(value, expected):
    assert parse_number(value) == expected


def test_leading_zero_codes_stay_text():
    inf = infer_column(pd.Series(["02134", "10001", "00501"], dtype=object), "zip")
    assert inf.kind != "numeric" and inf.role == "identifier" and inf.leading_zeros == 2


def test_mixed_type_column_reports_invalid_values():
    inf = infer_column(pd.Series(["1", "2", "3", "four", "5"], dtype=object), "qty")
    assert inf.kind == "numeric" and inf.n_invalid == 1 and inf.invalid_values == {"four": 1}


def test_mostly_text_column_is_not_forced_numeric():
    inf = infer_column(pd.Series(["a", "b", "c", "1"], dtype=object), "label")
    assert inf.kind in ("categorical", "text")


def test_malformed_dates_detected():
    inf = infer_column(pd.Series(["2024-01-31", "2024-02-30", "2024-13-01", "2024-03-01"], dtype=object), "d")
    assert inf.kind == "datetime"
    assert set(inf.invalid_values) == {"2024-02-30", "2024-13-01"}


def test_ambiguous_day_month_dates():
    amb = infer_column(pd.Series(["01/02/2024", "03/04/2024", "05/06/2024"], dtype=object), "d")
    assert amb.kind == "datetime" and amb.date_ambiguous
    clear = infer_column(pd.Series(["01/02/2024", "25/04/2024", "05/06/2024"], dtype=object), "d")
    assert clear.date_format == "%d/%m/%Y" and not clear.date_ambiguous


def test_boolean_and_identifier_roles():
    assert infer_column(pd.Series(["yes", "no", "Yes"], dtype=object), "active").kind == "boolean"
    ids_ = infer_column(pd.Series([str(i) for i in range(1, 31)], dtype=object), "row")
    assert ids_.role == "identifier"  # sequential unique integers
    assert infer_column(pd.Series(["5", "3", "5", "1"], dtype=object), "customer_id").role == "identifier"


def test_coerce_series_marks_invalid_and_formatted():
    typed, invalid, formatted = coerce_series(pd.Series(["1", "$2", "x", None], dtype=object), "numeric")
    assert str(typed.dtype) == "Int64"
    assert typed.tolist()[:2] == [1, 2] and pd.isna(typed[2]) and pd.isna(typed[3])
    assert invalid.tolist() == [False, False, True, False]
    assert formatted.tolist() == [False, True, False, False]


# ─── Profiling ───────────────────────────────────────────────

def test_profile_counts_on_planted_defects(sales):
    raw, p, _ = sales
    cols = {c["name"]: c for c in p["columns"]}
    assert p["row_count"] == 247 and p["column_count"] == 10
    assert cols["units"]["marker_count"] == 4 and cols["units"]["invalid_count"] == 2
    assert cols["units"]["invalid_examples"] == [{"value": "abc", "count": 2}]
    assert cols["Region"]["whitespace_values"] == 3
    assert {v["value"] for g in cols["Region"]["case_variants"] for v in g["variants"]} >= {"North", "north", "NORTH"}
    assert cols["order_date"]["invalid_count"] == 2
    assert cols["unit_price"]["formats"]["currency"] == 10
    assert cols["revenue"]["formats"]["thousands"] == 2
    assert cols["currency"]["role"] == "constant" and cols["notes"]["kind"] == "empty"
    assert cols["order_id"]["role"] == "identifier"
    assert any(r["rule"] == "negative" and r["count"] == 3 for r in cols["units"]["impossible"])
    assert p["exact_duplicate_rows"] == 5
    key = {k["column"]: k for k in p["key_duplicates"]}
    assert key["order_id"]["extra_rows"] == 7  # 5 exact duplicates + 2 conflicting rows


def test_profile_numeric_stats_match_pandas():
    df = load("v\n" + "\n".join(str(x) for x in [1, 2, 3, 4, 100]) + "\n")
    n = profile_dataframe(df)["columns"][0]["numeric"]
    s = pd.Series([1, 2, 3, 4, 100], dtype=float)
    assert n["mean"] == pytest.approx(s.mean()) and n["median"] == 3 and n["std"] == pytest.approx(s.std())
    assert n["q1"] == 2 and n["q3"] == 4 and n["outliers_mild"] == 1 and n["outliers_extreme"] == 1


def test_whitespace_near_duplicates_are_separate_from_exact():
    df = load("a,b\nx,1\nx ,1\nx,1\n")
    p = profile_dataframe(df)
    assert p["exact_duplicate_rows"] == 1 and p["normalized_duplicate_rows"] == 2


def test_empty_dataset_profiles_without_error():
    p = profile_dataframe(load("a,b\n"))
    assert p["row_count"] == 0 and p["exact_duplicate_rows"] == 0


# ─── Cleaning ────────────────────────────────────────────────

def test_plan_classifies_safe_and_review(sales):
    _, _, plan = sales
    safe, review = ids(plan, "safe"), ids(plan, "review")
    assert {"trim_whitespace:Region", "missing_markers:units", "convert_type:revenue"} <= safe
    # Identifiers are never converted to numbers: they keep their exact text (leading zeros, long digit strings)
    assert "convert_type:order_id" not in safe | review
    # Anything destructive, lossy or judgment-based must need review
    assert {"convert_type:units", "convert_type:unit_price", "convert_type:order_date", "normalize_labels:Region",
            "impossible_values:units", "drop_exact_duplicates", "drop_key_duplicates:order_id",
            "fill_missing:units", "drop_column:currency", "drop_column:notes"} <= review
    assert all(not a["default_enabled"] for a in plan if a["category"] == "review")


def test_defaults_never_remove_rows_impute_or_drop_values(sales):
    raw, _, plan = sales
    before = raw.copy()
    r = apply_cleaning(raw, plan, None)
    assert len(r.df) == len(raw) and not r.excluded_rows
    assert raw.equals(before)  # the original is never modified
    for ch in r.cell_changes:
        # Safe fixes only trim, unify blanks/markers or reformat thousands separators
        assert ch["action"].split(":")[0] in {"trim_whitespace", "empty_to_missing", "missing_markers", "convert_type"}
        if ch["after"] is None:
            assert (ch["before"] or "").strip().lower() in {"", "n/a"}
    assert not any(o.get("values_imputed") for o in r.operations)


def test_approving_everything_applies_each_action(sales):
    raw, _, plan = sales
    r = apply_cleaning(raw, plan, {a["id"]: {"enabled": True} for a in plan})
    ops = {o["action_id"]: o for o in r.operations}
    assert ops["drop_exact_duplicates"]["rows_removed"] == 5
    assert ops["drop_key_duplicates:order_id"]["rows_removed"] == 2
    assert ops["convert_type:units"]["values_set_missing"] == 2          # 'abc' ×2
    assert ops["convert_type:order_date"]["values_set_missing"] == 2     # invalid dates
    assert ops["convert_type:unit_price"]["values_changed"] == 10        # '$' removed
    assert ops["impossible_values:units"]["values_set_missing"] == 3
    assert ops["normalize_labels:Region"]["values_changed"] == 2
    assert len(r.df) == 240 and "currency" not in r.df.columns and "notes" not in r.df.columns
    assert str(r.df["units"].dtype) == "Int64" and str(r.df["order_date"].dtype).startswith("datetime64")
    assert set(r.df["region"].unique()) == {"North", "South", "East", "West"}  # snake_case rename applied
    assert r.df["order_id"].is_unique


def test_cleaning_is_deterministic(sales):
    raw, _, plan = sales
    dec = {a["id"]: {"enabled": True} for a in plan}
    a, b = apply_cleaning(raw, plan, dec), apply_cleaning(raw, plan, dec)
    pd.testing.assert_frame_equal(a.df, b.df)
    assert a.cell_changes == b.cell_changes


def test_invalid_values_keep_original_in_log(sales):
    raw, _, plan = sales
    r = apply_cleaning(raw, plan, {"convert_type:units": {"enabled": True}})
    logged = [c for c in r.cell_changes if c["action"] == "convert_type:units"]
    assert sorted(c["before"] for c in logged) == ["abc", "abc"] and all(c["after"] is None for c in logged)
    assert all(raw.at[c["row"] - 1, "units"] == "abc" for c in logged)


def test_outlier_flag_keeps_all_values():
    df = load("amount\n" + "\n".join(["10", "11", "12", "13", "9", "10", "11", "500"]) + "\n")
    plan = plan_for(df)
    r = apply_cleaning(df, plan, {"outliers:amount": {"enabled": True, "options": {"handling": "flag"}}})
    assert len(r.df) == 8 and r.df["amount_outlier_flag"].sum() == 1
    r2 = apply_cleaning(df, plan, {"outliers:amount": {"enabled": True, "options": {"handling": "exclude_rows"}}})
    assert len(r2.df) == 7 and r2.excluded_rows[0]["row"] == 8


def test_blank_lines_are_rows_not_dropped():
    df, meta = loader.load_dataframe("t.csv", b"score\n1\n\n3\n\n\n")
    assert df["score"].tolist() == ["1", "", "3"]  # inner blank = missing value; trailing blanks reported
    assert any("blank line" in render(n, "en") for n in meta["notes"])
    multi = load("a,b\n1,2\n,\n3,4\n")
    plan = plan_for(multi)
    act = next(a for a in plan if a["id"] == "drop_empty_rows")
    assert act["category"] == "review" and act["affected"] == 1
    assert len(apply_cleaning(multi, plan, None).df) == 3
    assert len(apply_cleaning(multi, plan, {"drop_empty_rows": {"enabled": True}}).df) == 2


def test_fill_missing_requires_approval_and_logs():
    df = load("score\n1\n\n3\n5\n")
    plan = plan_for(df)
    assert apply_cleaning(df, plan, None).df["score"].isna().sum() == 1
    r = apply_cleaning(df, plan, {"fill_missing:score": {"enabled": True, "options": {"strategy": "median"}}})
    assert r.df["score"].tolist() == [1, 3, 3, 5]
    assert any(c["action"] == "fill_missing:score" and c["after"] == "3" for c in r.cell_changes)


def test_ambiguous_date_option_changes_interpretation():
    df = load("d\n01/02/2024\n03/04/2024\n")
    plan = plan_for(df)
    act = next(a for a in plan if a["id"] == "convert_type:d")
    assert act["category"] == "review" and act["options"][0]["name"] == "date_format"
    dm = apply_cleaning(df, plan, {"convert_type:d": {"enabled": True, "options": {"date_format": "%d/%m/%Y"}}})
    md = apply_cleaning(df, plan, {"convert_type:d": {"enabled": True, "options": {"date_format": "%m/%d/%Y"}}})
    assert dm.df["d"][0] == pd.Timestamp("2024-02-01") and md.df["d"][0] == pd.Timestamp("2024-01-02")


def test_decision_validation(sales):
    _, _, plan = sales
    with pytest.raises(DecisionError):
        resolve_decisions(plan, {"outliers:units": {"enabled": True, "options": {"handling": "delete_everything"}}})
    with pytest.raises(DecisionError):
        resolve_decisions(plan, {"outliers:units": {"enabled": True, "options": {"bogus": "x"}}})
    _, warnings = resolve_decisions(plan, {"not_an_action:x": {"enabled": True}})
    assert warnings and warnings[0]["params"]["ids"] == ["not_an_action:x"]
    assert render(warnings[0], "en") == "1 action in the recipe does not apply to this dataset and was ignored: not_an_action:x"


def test_text_missing_markers_need_review():
    df = load("answer,n\nNone,1\nyes,2\nmaybe,N/A\n")
    plan = plan_for(df)
    cats = {a["id"]: a["category"] for a in plan}
    assert cats["missing_markers:answer"] == "review"   # 'None' could be a real answer
    assert cats["missing_markers:n"] == "safe"          # 'N/A' can't be a number


def test_formula_like_text_survives_cleaning():
    df = load("v\n=1+1\n@SUM(A1)\n")
    r = apply_cleaning(df, plan_for(df), None)
    assert r.df["v"].tolist() == ["=1+1", "@SUM(A1)"]


# ─── EDA & insights ──────────────────────────────────────────

@pytest.fixture(scope="module")
def sales_clean(sales):
    raw, profile, plan = sales
    r = apply_cleaning(raw, plan, {a["id"]: {"enabled": True} for a in plan})
    infs = infer_frame(r.df)
    prof = profile_dataframe(r.df, infs)
    frame, provisional = analysis_frame(r.df, infs)
    return r, prof, run_eda(frame, prof, provisional)


def test_identifiers_excluded_from_correlations(sales_clean):
    _, _, eda = sales_clean
    assert "order_id" not in eda["correlation"]["pearson"]
    assert all("order_id" not in (r["a"], r["b"]) for r in eda["relationships"])
    assert any(e["column"] == "order_id" and "identifier" in e["reason"] for e in eda["excluded"])


def test_planted_signals_are_found(sales_clean):
    _, _, eda = sales_clean
    rel = {(r["a"], r["b"]): r for r in eda["relationships"]}
    ur = rel[("units", "revenue")]
    assert ur["value"] > 0.7 and ur["significant"]
    grp = next(g for g in eda["groups"] if g["group"] == "region" and g["measure"] == "unit_price")
    assert grp["significant"] and grp["groups"][0]["level"] == "North"
    trend = eda["trends"][0]
    count_test = next(t for t in trend["tests"] if t["series"] == "record count")
    assert trend["freq"] == "month" and count_test["spearman"] > 0.5 and count_test["q_value"] < 0.05


def test_correlation_values_match_scipy(sales_clean):
    from scipy import stats
    r, _, eda = sales_clean
    x, y = r.df["units"].astype(float), r.df["revenue"].astype(float)
    ok = x.notna() & y.notna()
    rel = next(p for p in eda["relationships"] if (p["a"], p["b"]) == ("units", "revenue"))
    assert rel["value"] == pytest.approx(stats.spearmanr(x[ok], y[ok])[0], abs=1e-4)
    assert rel["pearson"] == pytest.approx(stats.pearsonr(x[ok], y[ok])[0], abs=1e-4)
    assert rel["n"] == int(ok.sum())


def test_no_false_relationships_in_independent_noise():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({f"m{i}": rng.normal(size=300).round(3).astype(str) for i in range(6)}
                      | {"grp": rng.choice(list("ABC"), 300)}).astype(object)
    infs = infer_frame(df)
    prof = profile_dataframe(df, infs)
    eda = run_eda(*analysis_frame(df, infs)[:1], prof)
    assert not [r for r in eda["relationships"] if r["significant"]]


def test_insights_are_grounded(sales, sales_clean):
    raw, before, plan = sales
    r, prof, eda = sales_clean
    report = {"duplicates": {"remaining_exact": prof["exact_duplicate_rows"]}, "pending_review": []}
    insights = generate_insights(before, prof, report, eda, plan, r.decisions)
    cols = set(r.df.columns)
    assert insights
    for i in insights:
        assert set(i["evidence"]["columns"]) <= cols, i
        text = " ".join(render(i.get(k), "en") for k in ("title", "finding", "why", "action", "technical")).lower()
        if "caus" in text:  # causal language only appears as a disclaimer
            assert "not evidence" in text or "not show" in text or "may explain" in text or "before treating" in text, i
    cats = {i["category"] for i in insights}
    assert {"relationship", "difference", "trend"} <= cats
    rel = next(i for i in insights if i["category"] == "relationship" and set(i["evidence"]["columns"]) == {"units", "revenue"})
    tech = render(rel["technical"], "en")
    assert "ρ = " in tech and "n = " in tech
    assert "(q) = < 0.001" in tech  # a tiny q-value is shown as "< 0.001", never rounded to 0
    assert "أقل من" in render(rel["technical"], "ar")
