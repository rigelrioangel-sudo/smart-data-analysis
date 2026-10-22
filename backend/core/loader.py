"""Safe file loading.

Files are parsed from memory (never written to disk). Values are kept exactly
as stored in the file: CSV cells stay strings, Excel/JSON cells keep their
native types. All type conversion happens later in the cleaning pipeline so
the original upload can always be shown and restored.
"""
import csv
import io
import json
import os

import pandas as pd

from .i18n import K, M, render

MAX_UPLOAD_BYTES = int(float(os.environ.get("SDAP_MAX_UPLOAD_MB", "50")) * 1024 * 1024)
MAX_COLUMNS = 500
MAX_ROWS = 2_000_000

SUPPORTED_EXTENSIONS = {".csv": "csv", ".xlsx": "xlsx", ".xls": "xls", ".json": "json"}

XLSX_MAGIC = b"PK\x03\x04"
XLS_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


class LoadError(Exception):
    """A user-facing problem with the uploaded file."""

    def __init__(self, msg, status_code: int = 400):
        self.msg = msg if isinstance(msg, dict) else M("err.generic", detail=str(msg))
        self.message = render(self.msg, "en")
        super().__init__(self.message)
        self.status_code = status_code


def detect_file_type(filename: str):
    ext = os.path.splitext(filename or "")[1].lower()
    return SUPPORTED_EXTENSIONS.get(ext)


def _decode_text(content: bytes):
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return content.decode("utf-16"), "utf-16"
    if b"\x00" in content[:8192]:
        raise LoadError(K("err.binary_file"))
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return content.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise LoadError(K("err.decode_failed"))


def _sniff_delimiter(text: str) -> str:
    sample = text[:65536]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _read_csv(content: bytes, meta: dict) -> pd.DataFrame:
    text, encoding = _decode_text(content)
    if not text.strip():
        raise LoadError(K("err.no_data"))
    delimiter = _sniff_delimiter(text)
    meta.update(encoding=encoding, delimiter=delimiter)
    try:
        # Blank lines are kept: in a one-column file they are missing values, not noise
        df = pd.read_csv(
            io.StringIO(text), sep=delimiter, dtype=str, keep_default_na=False,
            na_filter=False, skip_blank_lines=False,
        )
    except pd.errors.EmptyDataError:
        raise LoadError(K("err.no_data"))
    except pd.errors.ParserError as e:
        raise LoadError(M("err.csv_malformed", detail=str(e)))
    # Only blank lines at the very end of the file are dropped (and reported)
    trailing = 0
    if len(df):
        blank = (df == "").all(axis=1).to_numpy()
        while trailing < len(blank) and blank[len(blank) - 1 - trailing]:
            trailing += 1
    if trailing:
        df = df.iloc[: len(df) - trailing]
        meta.setdefault("notes", []).append(M("note.trailing_blank_lines", count=trailing))
    return df


def _read_excel(content: bytes, kind: str, meta: dict) -> pd.DataFrame:
    if content.startswith(XLSX_MAGIC):
        engine = "openpyxl"
    elif content.startswith(XLS_MAGIC):
        engine = "xlrd"
    else:
        raise LoadError(M("err.excel_invalid", ext=kind))
    try:
        book = pd.ExcelFile(io.BytesIO(content), engine=engine)
        sheets = book.sheet_names
        if not sheets:
            raise LoadError(K("err.excel_no_sheets"))
        df = book.parse(sheets[0], dtype=object, na_filter=False)
    except LoadError:
        raise
    except Exception as e:
        raise LoadError(M("err.excel_unreadable", detail=str(e)))
    meta.update(sheet=sheets[0], sheet_names=sheets)
    if len(sheets) > 1:
        meta.setdefault("notes", []).append(M("note.first_sheet_only", count=len(sheets), sheet=str(sheets[0])))
    return df


def _json_cell(v):
    # Nested structures are kept as JSON text so they stay hashable and visible
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


def _read_json(content: bytes, meta: dict) -> pd.DataFrame:
    text, encoding = _decode_text(content)
    meta["encoding"] = encoding
    if not text.strip():
        raise LoadError(K("err.no_data"))
    try:
        data = json.loads(text)
    except json.JSONDecodeError as first_error:
        # JSON Lines: one JSON value per line
        try:
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
            meta["layout"] = "json-lines"
        except json.JSONDecodeError:
            raise LoadError(M("err.json_malformed", detail=str(first_error)))

    if isinstance(data, dict):
        list_values = [v for v in data.values() if isinstance(v, list)]
        if data and all(isinstance(v, list) for v in data.values()) and len(data) > 1:
            lengths = {len(v) for v in data.values()}
            if len(lengths) != 1:
                raise LoadError(K("err.json_lengths"))
            meta["layout"] = "columns"
            df = pd.DataFrame(data)
            return df.map(_json_cell)
        if len(list_values) == 1:
            key = next(k for k, v in data.items() if isinstance(v, list))
            data = list_values[0]
            if not all(isinstance(r, dict) for r in data):
                meta["layout"] = "columns"
                return pd.DataFrame({key: [_json_cell(v) for v in data]})
            meta["layout"] = "wrapped-records"
        else:
            meta["layout"] = "single-record"
            data = [data]

    if not isinstance(data, list):
        raise LoadError(K("err.json_shape"))
    meta.setdefault("layout", "records")
    if all(isinstance(r, dict) for r in data):
        df = pd.json_normalize(data) if data else pd.DataFrame()
    else:
        df = pd.DataFrame({"value": data})
    return df.map(_json_cell) if not df.empty else df


def load_dataframe(filename: str, content: bytes):
    """Parse uploaded bytes into a raw (object-dtype) DataFrame plus file metadata."""
    kind = detect_file_type(filename)
    if not kind:
        raise LoadError(K("err.unsupported_type"))
    if not content:
        raise LoadError(K("err.empty_file"))
    if len(content) > MAX_UPLOAD_BYTES:
        raise LoadError(M("err.too_large", limit=MAX_UPLOAD_BYTES // (1024 * 1024)), 413)

    meta = {"file_type": kind, "size_bytes": len(content), "notes": []}
    if kind == "csv":
        df = _read_csv(content, meta)
    elif kind in ("xlsx", "xls"):
        df = _read_excel(content, kind, meta)
    else:
        df = _read_json(content, meta)

    if df.shape[1] == 0:
        raise LoadError(K("err.no_columns"))
    if df.shape[1] > MAX_COLUMNS:
        raise LoadError(M("err.too_many_columns", count=int(df.shape[1]), limit=MAX_COLUMNS), 413)
    if df.shape[0] > MAX_ROWS:
        raise LoadError(M("err.too_many_rows", count=int(df.shape[0]), limit=MAX_ROWS), 413)

    df.columns = [str(c) for c in df.columns]
    df = df.astype(object).reset_index(drop=True)
    # Treat float NaN produced by the readers (e.g. JSON null) as None
    df = df.where(pd.notna(df), None)
    return df, meta
