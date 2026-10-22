"""Analyze workspace: user-chosen columns, groupings, filters; results checked against pandas."""
import io
import json
import os
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))
import main  # noqa: E402
from core.i18n import collect_keys, load  # noqa: E402
from registration_data import build_rows, registration_csv_bytes  # noqa: E402

client = TestClient(main.app)
REG = pd.DataFrame(build_rows())


@pytest.fixture(scope="module")
def ds():
    r = client.post("/upload", files={"file": ("registrations.csv", registration_csv_bytes())})
    assert r.status_code == 200, r.text
    return r.json()


def explore(ds, **cfg):
    r = client.post(f"/datasets/{ds['dataset_id']}/explore", json=cfg)
    assert r.status_code == 200, r.text
    return r.json()


def section(res, sid):
    found = [s for s in res["sections"] if s["id"] == sid]
    assert found, [s["id"] for s in res["sections"]]
    return found[0]


def chart(sec, suffix):
    found = [c for c in sec["charts"] if c["id"].endswith(suffix)]
    assert found, [c["id"] for c in sec["charts"]]
    return found[0]


# ─── Column roles and catalog ────────────────────────────────

def test_catalog_roles_never_treat_codes_or_ids_as_measures(ds):
    by = {c["name"]: c for c in ds["explore_catalog"]["columns"]}
    for code in ("slot", "pon", "fat", "fat_port", "ring"):
        assert by[code]["role"] == "dimension" and by[code]["detected_role"] == "identifier", code
    for ident in ("id", "mobile", "sn", "clientname"):
        assert by[ident]["role"] in ("identifier", "text"), ident
    assert by["create_time_utc"]["role"] == "date"
    assert all(c["role"] != "measure" for c in by.values())
    assert by["profile"]["options"][0] == {"value": "FTTH-50M", "count": int((REG["profile"] == "FTTH-50M").sum())}


def test_recommended_setup(ds):
    rec = ds["explore_catalog"]["recommended"]
    assert rec["primary"] == "profile" and rec["secondary"] == ["city"]
    assert rec["breakdown"] == "status" and rec["date"] == "create_time_utc"


def test_eda_excludes_network_codes_from_correlations(ds):
    assert not set(ds["eda"]["columns_used"]["measures"]) & {"slot", "pon", "fat", "fat_port", "ring", "id"}


# ─── A. Per profile ──────────────────────────────────────────

def test_per_profile_counts_shares_and_ranking(ds):
    res = explore(ds, columns=["profile"])
    sec = section(res, "dist-profile")
    table = sec["tables"][0]
    expected = REG["profile"].value_counts()
    got = {r[1]: r[2] for r in table["rows"]}
    assert got == expected.to_dict()
    assert [r[0] for r in table["rows"]] == list(range(1, len(expected) + 1))
    assert sum(r[3] for r in table["rows"]) == pytest.approx(100, abs=0.05)
    bar = chart(sec, "dist-profile")
    assert sum(bar["series"][0]["values"]) == len(REG)
    top = sec["insights"][0]
    assert top["finding"][0]["params"]["value"] == expected.index[0]
    assert top["finding"][0]["params"]["count"] == int(expected.iloc[0])
    # Only the selected column is analyzed
    assert {s["id"] for s in res["sections"]} == {"dist-profile", "compare-profile"}


def test_status_distribution_per_profile_matches_crosstab(ds):
    res = explore(ds, columns=["status", "profile"])  # status is assigned automatically as the breakdown
    assert res["config"]["primary"] == "profile" and res["config"]["breakdown"] == "status"
    sec = section(res, "bd-profile-status")
    st = chart(sec, "bd-profile-status")
    ct = pd.crosstab(REG["profile"], REG["status"])
    for s in st["series"]:
        for lab, v, p in zip(st["labels"], s["values"], s["pct"]):
            assert v == ct.at[lab, s["name"]]
            assert p == pytest.approx(v / ct.loc[lab].sum() * 100, abs=0.01)


# ─── B. Per city ─────────────────────────────────────────────

def test_per_city_status_finds_planted_pending_gap(ds):
    res = explore(ds, columns=["city", "status"])
    sec = section(res, "bd-city-status")
    spread = [i for i in sec["insights"] if i["title"]["key"] == "ex.ins.status_spread_title"]
    pending = [i for i in spread if i["finding"]["params"]["status"] == "Pending"]
    assert pending and pending[0]["finding"]["params"]["hi"] == "Basra"
    share = (REG.loc[REG.city == "Basra", "status"] == "Pending").mean() * 100
    assert pending[0]["finding"]["params"]["hi_pct"] == pytest.approx(share, abs=0.1)
    assert pending[0]["limitations"][0]["key"] == "ex.note.association"


# ─── C. Profile × city ───────────────────────────────────────

def test_profile_by_city_heatmap_percentages_and_unusual_combination(ds):
    res = explore(ds, columns=["profile", "city"])
    sec = section(res, "cross-profile-city")
    heat = chart(sec, "-heat")
    ct = pd.crosstab(REG["profile"], REG["city"])
    m = heat["matrix"]
    for i, p in enumerate(heat["rows"]):
        for j, c in enumerate(heat["labels"]):
            assert m["counts"][i][j] == ct.at[p, c]
    for j in range(len(heat["labels"])):  # within each city the profile shares add up to 100%
        assert sum(m["pct_col"][i][j] for i in range(len(heat["rows"]))) == pytest.approx(100, abs=0.05)
    assert m["total"] == len(REG)
    unusual = [t for t in sec["tables"] if t["id"].endswith("-unusual")][0]["rows"]
    assert ["Business-1G", "Erbil"] == unusual[0][:2] and unusual[0][-1] > 3
    assert {n["key"] for n in sec["notes"]} == {"ex.note.pct_basis", "ex.note.association"}


# ─── Trends ──────────────────────────────────────────────────

def test_trend_per_profile_over_time(ds):
    res = explore(ds, columns=["create_time_utc", "profile"])
    sec = section(res, "trend-create_time_utc")
    total = chart(sec, "-total")
    months = pd.to_datetime(REG["create_time_utc"]).dt.to_period("M").value_counts().sort_index()
    assert total["freq"] == "month" and total["labels"] == [str(p) for p in months.index]
    assert total["series"][0]["values"] == months.tolist()
    by_profile = chart(sec, "-profile")
    assert sum(sum(s["values"]) for s in by_profile["series"]) <= len(REG)
    assert sec["insights"][0]["title"]["key"] == "ex.ins.trend_title_up"


def test_trend_by_city_and_status(ds):
    res = explore(ds, columns=["create_time_utc", "city", "status"])
    sec = section(res, "trend-create_time_utc")
    kinds = {c["id"].split("-")[-1]: c["type"] for c in sec["charts"]}
    assert kinds["city"] == "line" and kinds["status"] == "stacked"


# ─── Selection genuinely changes the computation ─────────────

def test_selection_controls_backend_computation(ds):
    city_only = explore(ds, columns=["city"], analyses=["distribution"])
    assert [s["id"] for s in city_only["sections"]] == ["dist-city"]
    assert "profile" not in json.dumps(city_only["sections"])


def test_unsupported_analysis_explains_and_suggests(ds):
    res = explore(ds, columns=["profile"], analyses=["trend", "cross", "numeric"])
    keys = {n["key"]: n for n in res["notes"]}
    assert "create_time_utc" in keys["ex.na.trend"]["params"]["suggest"]
    assert "city" in keys["ex.na.cross"]["params"]["suggest"]
    assert "ex.na.numeric_none" in keys
    assert res["sections"] == []


@pytest.mark.parametrize("cfg,key", [
    ({"columns": ["nope"]}, "ex.err.unknown_column"),
    ({"columns": ["profile"], "primary": "city"}, "ex.err.not_selected"),
    ({"columns": ["profile", "create_time_utc"], "primary": "create_time_utc"}, "ex.err.wrong_role"),
    ({"columns": ["profile"], "roles": {"profile": "date"}}, "ex.err.role_not_possible"),
    ({"columns": ["profile"], "analyses": ["magic"]}, "ex.err.bad_analysis"),
    ({"columns": ["profile"], "filters": {"nope": ["1"]}}, "ex.err.unknown_column"),
    ({"columns": ["profile"], "filters": {"city": "Basra"}}, "err.invalid_request"),  # schema error → 422
    ({"columns": ["profile"], "freq": "hourly"}, "ex.err.bad_value"),
    ({"columns": ["profile"], "date_from": "2024-05-01", "date_to": "2024-01-01"}, "ex.err.date_order"),
    ({"columns": ["profile"], "date_from": "not-a-date"}, "ex.err.not_a_date"),
])
def test_server_validates_configuration(ds, cfg, key):
    r = client.post(f"/datasets/{ds['dataset_id']}/explore", json=cfg)
    assert r.status_code == (422 if key == "err.invalid_request" else 400) and r.json()["error"]["key"] == key, r.text


# ─── Filters keep everything consistent ──────────────────────

def test_filters_apply_to_kpis_charts_tables_and_insights(ds):
    res = explore(ds, filters={"city": ["Basra", "Erbil"]}, date_from="2024-03-01", date_to="2024-08-31", freq="month")
    dt = pd.to_datetime(REG["create_time_utc"])
    sub = REG[REG.city.isin(["Basra", "Erbil"]) & (dt >= "2024-03-01") & (dt < "2024-09-01")]
    n = len(sub)
    assert res["records"] == {"total": len(REG), "filtered": n, "is_filtered": True}
    assert res["kpis"][0]["value"] == n
    dist = section(res, "dist-profile")
    assert dist["tables"][0]["total"][2] == n
    assert {r[1]: r[2] for r in dist["tables"][0]["rows"]} == sub["profile"].value_counts().to_dict()
    assert section(res, "dist-city")["tables"][0]["total"][2] == n
    trend = chart(section(res, "trend-create_time_utc"), "-total")
    assert sum(trend["series"][0]["values"]) == n
    assert trend["labels"][0] == "2024-03" and trend["labels"][-1] == "2024-08"
    assert section(res, "dist-profile")["insights"][0]["finding"][0]["params"]["total"] == n


def test_focus_compares_selected_profiles(ds):
    res = explore(ds, focus={"profile": ["FTTH-100M", "FTTH-25M"]})
    sec = section(res, "compare-profile")
    vol = chart(sec, "-volume")
    assert vol["labels"] == ["FTTH-100M", "FTTH-25M"]
    assert vol["series"][0]["values"] == [int((REG.profile == v).sum()) for v in vol["labels"]]


def test_filter_with_no_matches_is_reported(ds):
    res = explore(ds, filters={"city": ["Atlantis"]})
    assert res["records"]["filtered"] == 0 and res["sections"] == []
    assert any(n["key"] == "ex.note.no_rows" for n in res["notes"])


# ─── Numeric analysis only for real measures ─────────────────

def test_role_override_enables_numeric_analysis_with_warning(ds):
    res = explore(ds, columns=["profile", "fat_port"], roles={"fat_port": "measure"}, analyses=["numeric"])
    sec = section(res, "numeric")
    assert {c["type"] for c in sec["charts"]} >= {"histogram", "box"}
    assert any(n["key"] == "ex.note.measure_override" for n in res["notes"])
    stats_row = sec["tables"][0]["rows"][0]
    assert stats_row[0] == "fat_port" and stats_row[3] == pytest.approx(pd.to_numeric(REG["fat_port"]).median())


def test_numeric_measures_get_distributions_and_scatter():
    # (a sequential 0..59 column would be an identifier, so units are deliberately not sequential)
    units = [(i * 7) % 23 + 1 for i in range(60)]
    df = pd.DataFrame({"region": ["N", "S"] * 30, "units": units, "revenue": [u * 2.5 + (i % 7) for i, u in enumerate(units)]})
    up = client.post("/upload", files={"file": ("m.csv", df.to_csv(index=False).encode())}).json()
    res = explore(up, columns=["region", "units", "revenue"], analyses=["numeric"])
    types = [c["type"] for c in section(res, "numeric")["charts"]]
    assert types.count("histogram") == 2 and "scatter" in types


# ─── Exports and translations ────────────────────────────────

@pytest.mark.parametrize("fmt,lang", [("html", "en"), ("html", "ar"), ("xlsx", "en"), ("json", "ar")])
def test_export_uses_the_same_configuration(ds, fmt, lang):
    cfg = {"columns": ["profile", "city"], "filters": {"city": ["Basra"]}}
    r = client.post(f"/datasets/{ds['dataset_id']}/explore/export?format={fmt}&lang={lang}", json=cfg)
    assert r.status_code == 200, r.text
    n = int((REG.city == "Basra").sum())
    if fmt == "json":
        body = r.json()
        assert body["result"]["records"]["filtered"] == n and body["findings"]
    elif fmt == "xlsx":
        sheets = pd.read_excel(io.BytesIO(r.content), sheet_name=None)
        assert len(sheets) >= 4
    else:
        text = r.content.decode("utf-8")
        assert "Basra" in text and ('dir="rtl"' in text) == (lang == "ar")


def test_every_message_key_exists_in_both_languages(ds):
    res = explore(ds, columns=["create_time_utc", "profile", "city", "status", "mobile"], focus={"profile": ["FTTH-50M", "FTTH-25M"]})
    keys = collect_keys(res) | collect_keys(ds["explore_catalog"])
    for lang in ("en", "ar"):
        missing = sorted(k for k in keys if k not in load(lang))
        assert not missing, (lang, missing)


def test_frontend_ui_keys_exist_in_both_languages():
    root = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
    import re
    src = open(os.path.join(root, "explore.js"), encoding="utf-8").read() + open(os.path.join(root, "index.html"), encoding="utf-8").read()
    keys = set(re.findall(r"""t\('((?:ex|tab)\.[a-z_.]+)'""", src)) | set(re.findall(r'data-i18n(?:-[a-z]+)?="((?:ex|tab)\.[a-z_.]+)"', src))
    # keys built from a prefix + value (e.g. 'ex.role.' + r) are checked through their possible values
    prefixes = {k for k in keys if k.endswith(".")}
    keys -= prefixes
    keys |= {f"ex.role.{r}" for r in ("dimension", "date", "measure", "identifier", "text", "ignore")}
    keys |= {f"ex.{p}.{a}" for p in ("analysis", "analysis_help") for a in ("distribution", "breakdown", "trend", "cross", "compare", "numeric")}
    keys |= {f"ex.freq.{f}" for f in ("auto", "day", "week", "month", "quarter", "year")}
    keys |= {f"ex.ui.basis_note.{b}" for b in ("counts", "pct_col", "pct_row", "pct_total")}
    keys |= {f"ex.ui.type.{x}" for x in ("bar", "hbar", "doughnut", "stacked", "stacked100", "grouped", "line", "histogram",
                                          "heatmap", "box", "scatter")}
    keys |= {f"ex.{p}.{m}" for p in ("mode", "mode_help") for m in ("profile", "city", "cross", "custom")}
    keys |= {f"ex.metric_name.{m}" for m in ("records", "distinct", "sum", "mean", "median")}
    assert len(keys) > 30 and prefixes == {"ex.role.", "ex.analysis.", "ex.analysis_help.", "ex.freq.", "ex.ui.basis_note.",
                                           "ex.ui.type.", "ex.mode.", "ex.metric_name."}
    for lang in ("en", "ar"):
        assert not sorted(k for k in keys if k not in load(lang)), lang
