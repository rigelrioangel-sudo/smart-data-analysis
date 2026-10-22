import io
import json
import os
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import main  # noqa: E402

client = TestClient(main.app)

SAMPLE = pd.DataFrame({
    "name": ["Alice", "Bob", "Alice", "Cara"],
    "age": [30, None, 25, 41],
    "score": [1.5, 2.5, 3.0, 4.0],
})


def upload(filename, content):
    return client.post("/upload", files={"file": (filename, content)})


def assert_sample_analysis(body):
    a = body["analysis"]
    assert body["columns"] == ["name", "age", "score"]
    assert a["row_count"] == 4 and a["column_count"] == 3
    assert a["missing_counts"]["age"] == 1
    assert a["missing_pct"]["age"] == 25.0
    assert a["numeric_summary"]["score"]["mean"] == pytest.approx(2.75)
    assert a["categorical_summary"]["name"] == {"Alice": 2, "Bob": 1, "Cara": 1}
    assert set(a["correlation"]) == {"age", "score"}
    assert sum(a["distributions"]["score"]["values"]) == 4
    assert len(body["preview"]) == 4
    assert body["preview"][1]["age"] == ""  # missing value rendered as blank


# ─── Frontend / connection ───────────────────────────────────

def test_health():
    assert client.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize("path", ["/", "/style.css", "/script.js", "/static/script.js"])
def test_frontend_assets_served(path):
    assert client.get(path).status_code == 200


# ─── Supported formats ───────────────────────────────────────

def test_csv():
    r = upload("data.csv", SAMPLE.to_csv(index=False).encode())
    assert r.status_code == 200, r.text
    assert_sample_analysis(r.json())


def test_csv_semicolon_and_cp1252():
    content = "city;temp\nZürich;12\nMálaga;25\n".encode("cp1252")
    r = upload("data.csv", content)
    assert r.status_code == 200, r.text
    assert r.json()["columns"] == ["city", "temp"]
    assert "Zürich" in r.json()["analysis"]["categorical_summary"]["city"]


def test_xlsx():
    buf = io.BytesIO()
    SAMPLE.to_excel(buf, index=False, engine="openpyxl")
    r = upload("data.xlsx", buf.getvalue())
    assert r.status_code == 200, r.text
    assert_sample_analysis(r.json())


def test_xls():
    xlwt = pytest.importorskip("xlwt")
    wb = xlwt.Workbook()
    ws = wb.add_sheet("Sheet1")
    for c, col in enumerate(SAMPLE.columns):
        ws.write(0, c, col)
        for r_, v in enumerate(SAMPLE[col], start=1):
            if not pd.isna(v):
                ws.write(r_, c, v)
    buf = io.BytesIO()
    wb.save(buf)
    r = upload("data.xls", buf.getvalue())
    assert r.status_code == 200, r.text
    assert_sample_analysis(r.json())


def test_json_records():
    r = upload("data.json", SAMPLE.to_json(orient="records").encode())
    assert r.status_code == 200, r.text
    assert_sample_analysis(r.json())


def test_json_wrapped_nested():
    body = {"data": [{"id": 1, "user": {"name": "x"}}, {"id": 2, "user": {"name": "y"}}]}
    r = upload("data.json", json.dumps(body).encode())
    assert r.status_code == 200, r.text
    assert r.json()["columns"] == ["id", "user.name"]


def test_json_single_object_and_columnar():
    r = upload("one.json", json.dumps({"a": 1, "b": "x"}).encode())
    assert r.status_code == 200, r.text
    assert r.json()["analysis"]["row_count"] == 1
    r = upload("cols.json", json.dumps({"a": [1, 2], "b": ["x", "y"]}).encode())
    assert r.status_code == 200, r.text
    assert r.json()["analysis"]["row_count"] == 2


def test_json_lines_and_list_cells():
    r = upload("data.json", b'{"a": 1, "tags": [1, 2]}\n{"a": 2, "tags": [3]}\n')
    assert r.status_code == 200, r.text
    assert r.json()["preview"][0]["tags"] == "[1, 2]"


def test_uppercase_extension():
    r = upload("DATA.CSV", b"a,b\n1,2\n")
    assert r.status_code == 200, r.text


# ─── Edge cases ──────────────────────────────────────────────

def test_dates_detected():
    r = upload("d.csv", b"day,v,year\n2024-01-01,1,2020\n2024-03-01,2,2021\n")
    a = r.json()["analysis"]
    assert a["datetime_summary"]["day"]["range_days"] == 60
    assert "year" not in a["datetime_summary"]


def test_header_only_csv():
    r = upload("e.csv", b"a,b\n")
    assert r.status_code == 200, r.text
    assert r.json()["analysis"]["row_count"] == 0


def test_infinity_and_constant_columns():
    r = upload("i.csv", b"v,c\n1,5\ninf,5\n3,5\n")
    assert r.status_code == 200, r.text
    assert r.json()["analysis"]["numeric_summary"]["c"]["std"] == 0


@pytest.mark.parametrize("filename,content,status", [
    ("data.txt", b"a,b\n1,2\n", 400),
    ("data.csv", b"", 400),
    ("data.json", b"{not json", 400),
    ("data.xlsx", b"not an excel file", 400),
])
def test_bad_uploads_return_clean_errors(filename, content, status):
    r = upload(filename, content)
    assert r.status_code == status
    assert isinstance(r.json()["detail"], str)
