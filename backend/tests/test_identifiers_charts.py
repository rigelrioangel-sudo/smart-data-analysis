"""Numeric-looking identifiers, privacy and the Analyze main chart (modes, metrics, chart choice).

Uses the synthetic registration dataset (tests/registration_data.py) whose `user`
column holds 13-digit subscriber ids starting with 9642431."""
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
from core.explore import recommend_chart  # noqa: E402
from core.inference import infer_column  # noqa: E402
from registration_data import build_rows, registration_csv_bytes  # noqa: E402

client = TestClient(main.app)
REG = pd.DataFrame(build_rows())
USER_IDS = set(REG["user"])


@pytest.fixture(scope="module")
def ds():
    r = client.post("/upload", files={"file": ("registrations.csv", registration_csv_bytes())})
    assert r.status_code == 200, r.text
    return r.json()


def explore(ds, **cfg):
    r = client.post(f"/datasets/{ds['dataset_id']}/explore", json=cfg)
    assert r.status_code == 200, r.text
    return r.json()


def preset(ds, name, **extra):
    return explore(ds, **{**ds["explore_catalog"]["presets"][name], **extra})


def leaked(text):
    return sorted(u for u in USER_IDS if u in text)


# ─── Identifiers are identifiers ─────────────────────────────

def test_numeric_user_ids_are_sensitive_identifiers_kept_as_text(ds):
    user = next(c for c in ds["profile"]["columns"] if c["name"] == "user")
    assert user["role"] == "identifier" and user["sensitive"]
    assert "user" not in ds["eda"]["columns_used"]["measures"]
    assert not ds["eda"]["columns_used"]["measures"]          # no averages / correlations of ids or codes
    assert all(not (a.get("column") == "user" and a["type"] == "convert_type") for a in ds["plan"])
    assert all(k["id"] != "main_measure" for k in ds["summary"]["kpis"])   # no "typical user value" KPI
    assert all(ch["type"] != "breakdown" for ch in ds["summary"]["charts"])  # no average-by-group chart
    freq = [ch for ch in ds["summary"]["charts"] if ch["type"] == "frequency"]
    assert freq and freq[0]["column"] == "profile"               # replaced by record counts by category
    rows = client.get(f"/datasets/{ds['dataset_id']}/rows", params={"page_size": 5}).json()
    i = rows["columns"].index("user")
    assert all(r["values"][i] in USER_IDS for r in rows["rows"])  # exact text, no float formatting


def test_leading_zeros_and_long_digits_survive_cleaning():
    csv = "user,mobile,amount\n" + "".join(f"9642431{i:06d},0770{i:07d},{i * 12.5}\n" for i in range(40))
    up = client.post("/upload", files={"file": ("ids.csv", csv.encode())}).json()
    rows = client.get(f"/datasets/{up['dataset_id']}/rows", params={"page_size": 3}).json()
    assert rows["rows"][0]["values"][:2] == ["9642431000000", "07700000000"]
    roles = {c["name"]: (c["role"], c["sensitive"]) for c in up["profile"]["columns"]}
    assert roles["user"] == ("identifier", True) and roles["mobile"] == ("identifier", True)
    assert roles["amount"][0] == "measure"


def test_numeric_ids_from_excel_get_no_statistics():
    """Excel stores the ids as real numbers; they must still get no mean, sum, histogram or correlation."""
    df = pd.DataFrame({"user": [9642431000000 + i * 37 for i in range(40)], "units": [i % 9 + 1 for i in range(40)],
                       "city": ["Basra", "Erbil"] * 20})
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    up = client.post("/upload", files={"file": ("ids.xlsx", buf.getvalue())}).json()
    user = next(c for c in up["profile"]["columns"] if c["name"] == "user")
    assert user["role"] == "identifier" and user["sensitive"] and user["numeric"] is None
    legacy = up["analysis"]
    assert "user" not in legacy["numeric_summary"] and "user" not in legacy["correlation"] and "user" not in legacy["distributions"]
    payload = dict(up)
    payload.pop("preview")
    assert "9642431" not in json.dumps(payload)


@pytest.mark.parametrize("name,values,role", [
    ("user", [f"9642431{i:06d}" for i in range(30)], "identifier"),      # phone-like by value and name
    ("reference", [f"9647701{i:06d}" for i in range(30)], "identifier"),  # phone-like by value only
    ("amount", [str(100_000_000 + i * 7919) for i in range(30)], "measure"),  # large money amounts stay measures
    ("price", [str(i * 3.5) for i in range(30)], "measure"),
])
def test_phone_like_detection(name, values, role):
    assert infer_column(pd.Series(values, dtype=object), name).role == role


def test_identifiers_cannot_be_averaged_summed_or_grouped(ds):
    did = ds["dataset_id"]
    for metric in ("mean", "sum", "median"):
        r = client.post(f"/datasets/{did}/explore", json={"columns": ["profile", "user"], "metric": {"type": metric, "column": "user"}})
        assert r.status_code == 400 and r.json()["error"]["key"] == "ex.err.metric_measure"
    r = client.post(f"/datasets/{did}/explore", json={"columns": ["user"], "primary": "user"})
    assert r.status_code == 400
    r = client.post(f"/datasets/{did}/explore", json={"columns": ["profile"], "roles": {"user": "dimension"}})
    assert r.status_code == 400 and r.json()["error"]["key"] == "ex.err.role_not_possible"
    r = client.post(f"/datasets/{did}/explore", json={"columns": ["profile"], "filters": {"user": [REG["user"][0]]}})
    assert r.status_code == 400 and r.json()["error"]["key"] == "ex.err.not_filterable"
    cat = {c["name"]: c for c in ds["explore_catalog"]["columns"]}
    assert cat["user"]["allowed_roles"] == ["identifier", "ignore"] and cat["user"]["options"] is None
    metric_cols = {(m["type"], m.get("column")) for m in ds["explore_catalog"]["metrics"]}
    assert ("distinct", "user") in metric_cols and not {("mean", "user"), ("sum", "user")} & metric_cols


# ─── Privacy: no full user ids in charts, findings, reports ──

def test_no_user_ids_in_payload_charts_findings_or_reports(ds):
    payload = dict(ds)
    payload.pop("preview")   # the data preview shows the data itself, by design
    assert not leaked(json.dumps(payload))
    did = ds["dataset_id"]
    for name in ("profile", "city", "cross", "custom"):
        res = preset(ds, name, columns=ds["explore_catalog"]["presets"][name]["columns"] + ["user"],
                     metric={"type": "distinct", "column": "user"})
        assert not leaked(json.dumps(res)), name
    exports = [("quality-report", f) for f in ("html", "json", "xlsx")] + [("analysis-report", f) for f in ("html", "json", "xlsx")] \
        + [("executive-summary", f) for f in ("html", "json")] + [("cleaning-log", f) for f in ("json", "xlsx", "csv")]
    for art, fmt in exports:
        r = client.get(f"/datasets/{did}/export/{art}", params={"format": fmt})
        assert r.status_code == 200
        text = r.content.decode("utf-8", "ignore") if fmt != "xlsx" else \
            json.dumps({k: v.astype(str).to_dict() for k, v in pd.read_excel(io.BytesIO(r.content), sheet_name=None, dtype=str).items()})
        assert not leaked(text), (art, fmt)
    for fmt in ("html", "json", "xlsx"):
        r = client.post(f"/datasets/{did}/explore/export", params={"format": fmt},
                        json={"columns": ["profile", "user"], "metric": {"type": "distinct", "column": "user"}})
        assert r.status_code == 200 and not leaked(r.content.decode("utf-8", "ignore") if fmt != "xlsx" else "")


def test_cleaning_log_masks_personal_identifiers():
    csv = "user,city\n" + "".join(f" 9642431{i:06d} ,Basra\n" for i in range(25))  # stray spaces → a logged safe fix
    up = client.post("/upload", files={"file": ("ws.csv", csv.encode())}).json()
    changes = client.get(f"/datasets/{up['dataset_id']}/changes").json()["changes"]
    assert changes and all("9642431" not in str(c["before"]) + str(c["after"]) for c in changes)
    log = client.get(f"/datasets/{up['dataset_id']}/export/cleaning-log", params={"format": "json"}).json()
    assert log["cell_changes"] and "9642431" not in json.dumps(log["cell_changes"])
    plan = [a for a in up["plan"] if a.get("column") == "user"]
    assert plan and "9642431" not in json.dumps(plan)


# ─── Modes: counts, percentages, totals; chart == table ──────

def _single(res):
    m = res["main"]
    assert m["shape"] == "single"
    chart, table = m["chart"], m["table"]
    by_label = {r[1]: r for r in table["rows"]}
    for lab, v, p in zip(chart["labels"], chart["series"][0]["values"], chart["series"][0]["pct"]):
        if isinstance(lab, str):   # every bar equals its table row
            assert by_label[lab][2] == v and by_label[lab][3] == p
    return chart, table


@pytest.mark.parametrize("mode,col", [("profile", "profile"), ("city", "city")])
def test_per_profile_and_per_city_counts_and_percentages(ds, mode, col):
    res = preset(ds, mode)
    chart, table = _single(res)
    expected = REG[col].value_counts()
    assert {r[1]: r[2] for r in table["rows"]} == expected.to_dict()
    assert [r[2] for r in table["rows"]] == sorted(expected.values, reverse=True)   # ranking order
    assert sum(r[3] for r in table["rows"]) == pytest.approx(100, abs=0.05)
    assert table["total"][2] == len(REG) and res["main"]["total"] == len(REG)
    assert chart["explanation"]["key"] == "ex.denom.records"                       # denominator stated
    assert {s["id"] for s in res["sections"]} >= {f"bd-{col}-status", "trend-create_time_utc"}


def test_per_city_includes_profile_distribution(ds):
    res = preset(ds, "city")
    heat = next(c for s in res["sections"] if s["id"] == "cross-city-profile" for c in s["charts"] if c["type"] == "heatmap")
    ct = pd.crosstab(REG["city"], REG["profile"])
    for i, city in enumerate(heat["rows"]):
        for j, prof in enumerate(heat["labels"]):
            assert heat["matrix"]["counts"][i][j] == ct.at[city, prof]
            assert heat["matrix"]["pct_row"][i][j] == pytest.approx(ct.at[city, prof] / ct.loc[city].sum() * 100, abs=0.01)


def test_profile_by_city_totals(ds):
    res = preset(ds, "cross")
    m = res["main"]
    assert m["shape"] == "split" and m["chart"]["default"] == "stacked100"
    h = m["chart"]["heat"]
    ct = pd.crosstab(REG["profile"], REG["city"])
    for i, p in enumerate(h["rows"]):
        assert h["matrix"]["row_totals"][i] == ct.loc[p].sum()
        for j, c in enumerate(h["cols"]):
            assert h["matrix"]["counts"][i][j] == ct.at[p, c]
            assert sum(h["matrix"]["pct_row"][i]) == pytest.approx(100, abs=0.05)
    assert h["matrix"]["total"] == len(REG) and sum(h["matrix"]["col_totals"]) == len(REG)
    # stacked series = table columns
    for j, s in enumerate(m["chart"]["series"]):
        assert s["values"] == [row[1 + j] for row in m["table"]["rows"]]


def test_distinct_users_differ_from_records(ds):
    res = preset(ds, "profile", columns=ds["explore_catalog"]["presets"]["profile"]["columns"] + ["user"],
                 metric={"type": "distinct", "column": "user"})
    chart, table = _single(res)
    expected = REG.groupby("profile")["user"].nunique()
    assert {r[1]: r[2] for r in table["rows"]} == expected.to_dict()
    assert res["main"]["total"] == REG["user"].nunique() < len(REG)
    assert next(k for k in res["kpis"] if k["id"] == "metric")["value"] == REG["user"].nunique()
    assert next(k for k in res["kpis"] if k["id"] == "records")["value"] == len(REG)
    assert "doughnut" not in chart["alt_types"]   # overlapping shares: no part-of-whole chart
    assert chart["explanation"]["key"] == "ex.denom.distinct"


# ─── Custom selection and chart choice ───────────────────────

@pytest.mark.parametrize("args,expected", [
    (("single", 3, "records"), ("doughnut", ["bar", "hbar", "doughnut"])),
    (("single", 6, "records"), ("bar", ["bar", "hbar", "doughnut"])),
    (("single", 12, "records"), ("hbar", ["bar", "hbar"])),
    (("single", 3, "distinct"), ("bar", ["bar", "hbar"])),
    (("single", 3, "mean"), ("bar", ["bar", "hbar"])),
    (("split", 5, "records", False, 6), ("stacked100", ["stacked100", "stacked", "grouped", "heatmap"])),
    (("split", 20, "records", False, 6), ("heatmap", ["stacked100", "stacked", "grouped", "heatmap"])),
    (("split", 5, "distinct", False, 6), ("grouped", ["grouped", "heatmap"])),
    (("time",), ("line", ["line", "bar"])),
])
def test_chart_recommendation(args, expected):
    default, allowed = recommend_chart(*args)
    assert (default, allowed) == expected


def test_custom_selection_drives_the_main_chart(ds):
    status = explore(ds, columns=["status"])
    assert status["main"]["chart"]["type"] == "doughnut" and status["main"]["chart"]["labels"][0] == "Active"
    time = explore(ds, columns=["create_time_utc"], freq="month")
    ch = time["main"]["chart"]
    months = pd.to_datetime(REG["create_time_utc"]).dt.to_period("M").value_counts().sort_index()
    assert ch["type"] == "line" and ch["labels"] == [str(p) for p in months.index]   # chronological
    assert ch["series"][0]["values"] == months.tolist()
    nothing = explore(ds, columns=["user"])
    assert nothing["main"] is None and nothing["notes"][0]["key"] == "ex.main.none"
    split = explore(ds, columns=["status", "city"], primary="city", split="status")
    assert split["main"]["shape"] == "split" and split["config"]["split"] == "status"
    r = client.post(f"/datasets/{ds['dataset_id']}/explore", json={"columns": ["city"], "primary": "city", "split": "profile"})
    assert r.status_code == 400 and r.json()["error"]["key"] == "ex.err.bad_split"


def test_numeric_measures_get_histogram_or_scatter():
    units = [(i * 7) % 23 + 1 for i in range(60)]
    df = pd.DataFrame({"units": units, "revenue": [u * 2.5 + (i % 7) for i, u in enumerate(units)]})
    up = client.post("/upload", files={"file": ("m.csv", df.to_csv(index=False).encode())}).json()
    assert explore(up, columns=["units"])["main"]["chart"]["type"] == "histogram"
    assert explore(up, columns=["units", "revenue"])["main"]["chart"]["type"] == "scatter"
    df["region"] = ["N", "S"] * 30
    up = client.post("/upload", files={"file": ("m2.csv", df.to_csv(index=False).encode())}).json()
    res = explore(up, columns=["region", "revenue"], metric={"type": "mean", "column": "revenue"})
    vals = dict(zip(res["main"]["chart"]["labels"], res["main"]["chart"]["series"][0]["values"]))
    assert vals["N"] == pytest.approx(df[df.region == "N"].revenue.mean(), abs=1e-3)
    assert res["main"]["chart"]["series"][0]["pct"] is None and "doughnut" not in res["main"]["chart"]["alt_types"]


# ─── Filters and date aggregation on the main chart ──────────

def test_filters_and_dates_apply_to_main_chart(ds):
    res = preset(ds, "profile", filters={"city": ["Basra"]}, date_from="2024-04-01", date_to="2024-06-30")
    dt = pd.to_datetime(REG["create_time_utc"])
    sub = REG[(REG.city == "Basra") & (dt >= "2024-04-01") & (dt < "2024-07-01")]
    chart, table = _single(res)
    assert res["main"]["total"] == len(sub) == table["total"][2]
    assert {r[1]: r[2] for r in table["rows"]} == sub["profile"].value_counts().to_dict()
    time = explore(ds, columns=["create_time_utc"], filters={"city": ["Basra"]}, freq="week",
                   date_from="2024-04-01", date_to="2024-06-30")
    ch = time["main"]["chart"]
    assert sum(ch["series"][0]["values"]) == len(sub) and ch["freq"] == "week"
    assert ch["labels"] == sorted(ch["labels"])
