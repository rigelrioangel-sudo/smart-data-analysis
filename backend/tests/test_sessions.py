"""Public-site privacy: every visitor (browser session) sees only the datasets it uploaded."""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import main  # noqa: E402
from core.store import DatasetStore  # noqa: E402

CSV = b"profile,city,status\nA,Basra,Active\nB,Erbil,Pending\nA,Basra,Active\n"


def visitor():
    return TestClient(main.app)   # each client keeps its own cookies = its own browser


def test_session_cookie_is_private():
    r = visitor().get("/health")
    cookie = r.headers["set-cookie"].lower()
    assert "sdap_sid=" in cookie and "httponly" in cookie and "samesite=lax" in cookie
    assert r.headers["x-frame-options"] == "DENY" and r.headers["x-content-type-options"] == "nosniff"


def test_session_cookie_is_secure_behind_https_proxy():
    r = visitor().get("/health", headers={"x-forwarded-proto": "https"})
    assert "secure" in r.headers["set-cookie"].lower()


def test_other_visitors_cannot_see_or_touch_a_dataset():
    owner, stranger = visitor(), visitor()
    up = owner.post("/upload", files={"file": ("mine.csv", CSV)})
    assert up.status_code == 200 and up.headers["cache-control"] == "no-store"
    did = up.json()["dataset_id"]
    paths = [("get", f"/datasets/{did}"), ("get", f"/datasets/{did}/rows"), ("get", f"/datasets/{did}/changes"),
             ("get", f"/datasets/{did}/explore/columns"), ("get", f"/datasets/{did}/export/cleaned?format=csv"),
             ("get", f"/datasets/{did}/export/analysis-report?format=html"),
             ("post", f"/datasets/{did}/explore"), ("post", f"/datasets/{did}/explore/export?format=json"),
             ("post", f"/datasets/{did}/undo"), ("post", f"/datasets/{did}/reset"), ("delete", f"/datasets/{did}")]
    for method, path in paths:
        kwargs = {"json": {}} if method == "post" else {}
        r = getattr(stranger, method)(path, **kwargs)
        assert r.status_code == 404, (method, path, r.status_code)   # indistinguishable from "does not exist"
        assert b"Basra" not in r.content
    # The owner still has full access, and the stranger's delete did nothing
    assert owner.get(f"/datasets/{did}/rows").json()["total_rows"] == 3
    assert owner.delete(f"/datasets/{did}").status_code == 200
    assert owner.get(f"/datasets/{did}").status_code == 404


def test_forged_or_missing_cookie_gets_nothing():
    owner = visitor()
    did = owner.post("/upload", files={"file": ("mine.csv", CSV)}).json()["dataset_id"]
    anon = TestClient(main.app, cookies={"sdap_sid": "x" * 43})
    assert anon.get(f"/datasets/{did}").status_code == 404
    bad = TestClient(main.app, cookies={"sdap_sid": "../../etc"})
    r = bad.get(f"/datasets/{did}")
    assert r.status_code == 404 and "sdap_sid=" in r.headers["set-cookie"]   # invalid token replaced


def test_each_visitor_has_own_quota():
    store = DatasetStore(max_items=10, ttl=3600, max_per_owner=2)

    class Ds:
        def __init__(self, i):
            self.id, self.last_access = f"{i:032x}", 1e18

    for i in range(3):
        store.put(Ds(i), owner="alice")
    store.put(Ds(9), owner="bob")
    assert store.get(f"{0:032x}", "alice") is None          # alice's oldest dropped, not bob's
    assert store.get(f"{2:032x}", "alice") is not None and store.get(f"{9:032x}", "bob") is not None
    assert store.get(f"{9:032x}", "alice") is None


@pytest.mark.parametrize("origin,allowed", [("http://127.0.0.1:5500", True), ("https://evil.example", False)])
def test_cors_only_for_local_dev_servers(origin, allowed):
    r = visitor().options("/upload", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
    assert (r.headers.get("access-control-allow-origin") == origin) is allowed
