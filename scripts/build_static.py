"""Build the static site (GitHub Pages / any static host).

The page is the normal frontend; the Python backend (main.py + core/) is shipped as
plain files that the in-browser engine (frontend/py-worker.js) runs with Pyodide.
Uploaded files are analyzed in the visitor's browser and never sent anywhere.

Usage:  python scripts/build_static.py <output-dir>
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
BACKEND = ROOT / "backend"


def build(out: Path):
    if out.exists():
        for child in out.iterdir():          # keep a possible .git folder of a deployment checkout
            if child.name != ".git":
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    out.mkdir(parents=True, exist_ok=True)

    for src in FRONTEND.rglob("*"):
        if src.is_file():
            dest = out / src.relative_to(FRONTEND)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    (out / "config.js").write_text("// Static build: analysis runs in the browser (py-worker.js)\nwindow.SDAP_STATIC = true;\n",
                                   encoding="utf-8")

    files = ["backend/main.py"] + sorted(f"backend/core/{p.name}" for p in (BACKEND / "core").glob("*.py"))
    for rel in files:
        dest = out / "py" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    (out / "py" / "manifest.json").write_text(json.dumps({"files": files}, indent=1), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")   # serve files as-is (no Jekyll processing)
    (out / "404.html").write_text((out / "index.html").read_text(encoding="utf-8"), encoding="utf-8")
    return files


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "dist")
    built = build(target)
    print(f"Static site written to {target} ({len(built)} Python files)")
