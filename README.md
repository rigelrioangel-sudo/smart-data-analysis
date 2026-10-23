# Smart Data Analysis Platform (DataSense Pro)

Upload a CSV, Excel (.xlsx/.xls) or JSON file and work through
**profile → review cleaning → analyze → export**. Everything is computed from
your data by the FastAPI backend; nothing is invented.

## Run

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py          # open http://127.0.0.1:8000
```

A messy demo file is in [samples/sales_messy.csv](samples/sales_messy.csv)
(regenerate it with `python backend/tests/sample_data.py`).

## Workflow

| Tab | What it shows |
|-----|---------------|
| Analyze | **Choose the columns to analyze** (search, select all / clear / recommended, override each column's role) and assign a primary grouping, additional groupings, a comparison category (e.g. status) and a date column. Runs per-group distributions, category mix per group, trends, grouping × grouping (heatmap, composition, unusual combinations), comparisons of chosen categories and — only for real numeric measures — distributions and medians. Filters (any category, date range) apply to every KPI, chart, table and finding. Charts can switch type, show their data and download as PNG; the analysis exports as HTML/XLSX/JSON. |
| Overview | Cleaned-data KPIs, column types, missingness, cleaning status |
| Data Quality | Profile of the **original** upload: issues by severity, per-column details, duplicates, name problems |
| Cleaning | Proposed actions split into **automatic safe fixes** (lossless, on by default) and **needs your review** (never applied without approval), with options, before/after report, operation log and changed cells. Undo / reset to original. |
| Data Preview | Server-side search, column filters, sorting and paging over the original or cleaned data; changed cells are highlighted with their original value |
| Statistics / Charts / Correlation | Descriptive stats, histograms, box plots, frequencies, trends, group comparisons, scatter plots, missing-data and outlier charts, type-aware relationships |
| Insights | Findings generated only from computed statistics, each citing its columns and numbers, plus limitations |
| Export | Cleaned data (CSV/XLSX/JSON), cleaning log, data-quality report, analysis report (HTML/XLSX/JSON) and the cleaning recipe |

### Cleaning guarantees
- The original upload is kept unchanged in memory; cleaning is always re-run from it, so results are reproducible (export the **recipe** and re-apply it).
- Rows are never deleted, missing values never filled and values never set to missing unless you approve that action. Original values of every changed cell are kept in the log.
- Outliers are flagged, not assumed wrong; the default handling adds a flag column.

### Analysis methods
- Numeric × numeric: Spearman ρ (and Pearson r); categorical × categorical: Cramér's V; categorical × numeric: η with Kruskal–Wallis.
- Identifiers, constant and free-text columns are excluded from relationships.
- p-values are Benjamini–Hochberg adjusted (q-values). Associations are never described as causes.

### Identifiers and privacy
- Columns named like a person/contact (`user`, `mobile`, `clientname` …) or holding long digit strings of one length (e.g. `9642431…`) are **personal identifiers**: kept as exact text (no number conversion, leading zeros kept), never averaged, summed, correlated, grouped or filtered by value, and their values are masked (`••••123`) in the profile, issues, cleaning plan and log, findings, charts and every report. They can only be counted (records or distinct values). The Data Preview and the cleaned-data export show the data itself, unmasked.
- Repeats of a personal identifier (one user, several registrations) are not reported as duplicate keys.

### Analyze workspace (registration-style data)
- Modes: **Per profile**, **Per city**, **Profile × city** and **Custom**; a metric selector (records, distinct values of an identifier, sum/average/median of a real numeric measure). The main chart, its exact table and KPIs appear at the top of the same page and refresh on every change; the chart type is recommended automatically (doughnut only for ≤ 6 shares that add up to 100%, horizontal bars for many/long labels, stacked 100% or heatmap for two groupings, line for dates) and can be switched.
- The default metric is the **number of records** (e.g. registrations). Averages and correlations are produced only for columns acting as numeric measures — never for IDs, phone numbers, serial numbers or network position codes (slot, PON, FDT/FAT, port, ring, OLT …), which are detected as codes and offered as groupings instead. Users can override any role; overriding a code to a measure shows a warning.
- Only the selected columns are computed, on the server; every column name, role and filter value is validated. Unsupported combinations are explained with a suggested column.
- Percentages always state their basis (within each group value, within each column value of a cross-table, or of all filtered records). Status differences are tested with chi-square / Cramér's V, unusual combinations with adjusted residuals (|r| ≥ 3), trends with Spearman ρ over complete periods (incomplete first/last periods are marked and excluded from change figures); all q-values are BH-adjusted within the analysis.
- Higher counts are never labelled good or bad; findings state the numbers, why they may matter and a suggested check.

## Live site (GitHub Pages)
**https://rigelrioangel-sudo.github.io/smart-data-analysis/** — free, public, HTTPS, no login.

The static build runs the same Python backend inside the visitor's browser with Pyodide (`frontend/py-worker.js`); uploaded files are analyzed on the visitor's device and are never sent to any server. The first visit downloads the engine (about 40 s), later visits use the browser cache.

Redeploy after changes:
```powershell
python scripts/build_static.py dist      # writes the static site (config.js sets SDAP_STATIC = true)
# then publish the contents of dist/ to the gh-pages branch
```

## Server deployment (each visitor sees only their own data)
- Every browser gets a random session token in an HttpOnly cookie (`sdap_sid`, `SameSite=Lax`, `Secure` behind HTTPS). Each upload is stored with that token; any other visitor asking for the dataset — even with the full link — gets `404`, exactly like a missing dataset. Uploads stay in memory only and expire after 3 hours.
- Limits: `SDAP_MAX_PER_SESSION` (default 3 open datasets per visitor), `SDAP_MAX_DATASETS` (default 40 for all visitors), `SDAP_MAX_UPLOAD_MB` (default 50). Cross-origin API calls are only allowed from local dev servers or `SDAP_CORS_ORIGINS`.
- Deploy with the `Dockerfile` (one worker: the data lives in that process). On Render: push this folder to a GitHub repository, then **New → Blueprint** and pick the repository (`render.yaml`). HTTPS is provided by the host.

## Limits & safety
- Upload limit 50 MB (`SDAP_MAX_UPLOAD_MB`), 500 columns, 2M rows; uploads are parsed in memory, never written to disk.
- Up to 8 datasets are kept in memory for 3 hours (`SDAP_MAX_DATASETS`, `SDAP_DATASET_TTL`).
- File signatures are checked; corrupted or malformed files return a clear error.
- XLSX exports store text as text (never formulas); CSV exports prefix formula-like text (`=`, `@`, `+x`, `-x`) with `'`.

## Tests

```powershell
cd backend
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests
```

- `tests/test_api.py`: file formats and the legacy `/upload` contract
- `tests/test_pipeline.py`: loader, inference, profiling, cleaning rules, statistics, insights
- `tests/test_workflow.py`: full HTTP workflow, exports, recipe reproducibility, validation and security
- `tests/test_identifiers_charts.py`: numeric-looking identifiers, privacy of exports, mode counts/percentages, metrics, chart choice
- `tests/test_explore.py`: Analyze workspace on a synthetic registration dataset (`tests/registration_data.py`, no real customer data): counts and percentages checked against pandas, filters, validation, exports, translations

## API (summary)

`POST /upload` · `GET /datasets/{id}` · `POST /datasets/{id}/clean` · `POST /datasets/{id}/undo` ·
`POST /datasets/{id}/reset` · `GET /datasets/{id}/rows` · `GET /datasets/{id}/changes` ·
`GET /datasets/{id}/export/{cleaned|cleaning-log|quality-report|analysis-report|recipe}?format=…` ·
`GET /datasets/{id}/explore/columns` · `POST /datasets/{id}/explore` · `POST /datasets/{id}/explore/export?format=html|xlsx|json&lang=en|ar` ·
`DELETE /datasets/{id}` · `GET /health`. Interactive docs at `/docs`.
