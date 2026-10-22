"""Centralized translation for backend-generated text.

The backend never stores translated prose. It emits language-neutral
messages: ``{"key": "...", "params": {...}}``. Messages are rendered with the
same locale files the browser uses (``frontend/locales/<lang>.json``), so the
interface and the exported reports always agree.

Parameter conventions (shared with frontend/i18n.js):
- str              → data (column names, cell values): inserted unchanged
- int / float      → localized number
- list             → each item rendered, joined with the locale's list separator
- {"key": ...}     → nested message (translated)
- {"date": iso}    → localized date
- {"period": label, "freq": f} → localized time period (month, quarter, ...)
Templates may be a plain string or a plural object ({"one": ..., "other": ...})
selected by the ``count`` parameter.
"""
import datetime as dt
import html as _html
import json
import math
import os
import re

LANGS = ("en", "ar")
DEFAULT_LANG = "en"
LOCALE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "locales"))
_cache = {}
# Keys requested but absent from a locale (checked by the test-suite; rendering falls back to English)
MISSING = set()
_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def M(key, **params):
    """A translatable message."""
    return {"key": key, "params": params} if params else {"key": key}


def K(key):
    """A translatable label without parameters (e.g. a type or severity name)."""
    return {"key": key}


def D(value):
    """A date parameter (ISO string or date)."""
    return {"date": value if isinstance(value, str) else value.isoformat()}


def P(label, freq):
    """A time-period parameter, e.g. P('2024-03', 'month')."""
    return {"period": str(label), "freq": freq}


def Q(q):
    """An adjusted p-value parameter: tiny values read "< 0.001" instead of rounding to 0."""
    if q is None:
        return None
    return K("fmt.q_lt_001") if q < 0.001 else round(float(q), 3)


def normalize_lang(lang):
    return lang if lang in LANGS else DEFAULT_LANG


def load(lang):
    lang = normalize_lang(lang)
    if lang not in _cache:
        with open(os.path.join(LOCALE_DIR, f"{lang}.json"), encoding="utf-8") as f:
            _cache[lang] = json.load(f)
    return _cache[lang]


def has_key(lang, key):
    return key in load(lang)


def plural_category(lang, n):
    try:
        n = abs(float(n))
    except (TypeError, ValueError):
        return "other"
    if lang == "ar":
        if not n.is_integer():
            return "other"
        n = int(n)
        if n == 0:
            return "zero"
        if n == 1:
            return "one"
        if n == 2:
            return "two"
        if 3 <= n % 100 <= 10:
            return "few"
        if 11 <= n % 100 <= 99:
            return "many"
        return "other"
    return "one" if n == 1 else "other"


def format_number(v, lang, digits=None):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and abs(v) >= 1000):
        s = f"{int(round(v)):,}"
    else:
        a = abs(v)
        if digits is None:
            digits = 0 if a >= 1000 else 1 if a >= 100 else 2 if a >= 1 else 3
        s = f"{v:,.{digits}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
    if lang == "ar":
        s = s.replace(",", "٬").replace(".", "٫").translate(_AR_DIGITS)
    return s


def format_date(value, lang):
    try:
        d = dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return str(value)
    months = load(lang).get("months")
    name = months[d.month - 1] if isinstance(months, list) and len(months) == 12 else d.strftime("%b")
    if lang == "ar":
        return f"{format_number(d.day, lang)} {name} {format_number(d.year, lang).replace('٬', '')}"
    return f"{name} {d.day}, {d.year}"


def format_period(label, freq, lang):
    label = str(label)
    if freq == "month" and re.match(r"^\d{4}-\d{2}$", label):
        y, m = label.split("-")
        months = load(lang).get("months")
        name = months[int(m) - 1] if isinstance(months, list) else m
        year = format_number(int(y), lang).replace(",", "").replace("٬", "")
        return f"{name} {year}"
    if freq == "quarter" and re.match(r"^\d{4}Q\d$", label):
        y, q = label.split("Q")
        return render(M("period.quarter", q=int(q), year=str(y) if lang == "en" else format_number(int(y), lang).replace("٬", "")), lang)
    if freq == "week" and "/" in label:
        return render(M("period.week_of", date=D(label.split("/")[0])), lang)
    if freq == "day" or re.match(r"^\d{4}-\d{2}-\d{2}$", label):
        return format_date(label, lang)
    if freq == "year":
        return format_number(int(label), lang).replace(",", "").replace("٬", "") if label.isdigit() else label
    return label


def _param(v, lang, html):
    if isinstance(v, dict):
        if "key" in v:
            return render(v, lang, html)
        if "date" in v:
            return format_date(v["date"], lang)
        if "period" in v:
            return format_period(v["period"], v.get("freq"), lang)
    if isinstance(v, (list, tuple)):
        sep = load(lang).get("list.separator", ", ")
        return sep.join(_param(x, lang, html) for x in v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return format_number(v, lang)
    if v is None:
        return "—"
    s = str(v)
    # Data values are inserted as-is; in HTML they are isolated for correct bidi rendering
    return f"<bdi>{_html.escape(s)}</bdi>" if html else s


def render(msg, lang=DEFAULT_LANG, html=False):
    """Render a message (or a list of messages) in `lang`. Missing keys fall back to English."""
    if msg is None:
        return ""
    if isinstance(msg, list):
        return " ".join(render(m, lang, html) for m in msg)
    if isinstance(msg, str):
        return _html.escape(msg) if html else msg
    lang = normalize_lang(lang)
    key, params = msg.get("key"), msg.get("params") or {}
    template = load(lang).get(key)
    if template is None:
        MISSING.add(f"{lang}:{key}")
        template = load(DEFAULT_LANG).get(key, key)
    if isinstance(template, dict):
        cat = plural_category(lang, params.get("count", 0))
        template = template.get(cat) or template.get("other") or next(iter(template.values()))
    if html:
        template = _html.escape(template)

    def sub(m):
        name = m.group(1)
        return _param(params[name], lang, html) if name in params else m.group(0)

    return _PLACEHOLDER.sub(sub, template)


def text(lang, key, **params):
    return render(M(key, **params), lang)


def collect_keys(obj, out=None):
    """All message keys referenced anywhere inside a (JSON-like) structure."""
    out = set() if out is None else out
    if isinstance(obj, dict):
        if isinstance(obj.get("key"), str) and set(obj) <= {"key", "params"}:
            out.add(obj["key"])
        for v in obj.values():
            collect_keys(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_keys(v, out)
    return out
