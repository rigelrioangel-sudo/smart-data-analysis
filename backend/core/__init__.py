"""Data processing for the Smart Data Analysis Platform.

Everything in this package is independent of FastAPI so it can be tested and
reused without the web layer:

- loader:     safe file reading that preserves original values
- inference:  per-column type / role inference and value parsing
- profiling:  data profile and data-quality issues
- cleaning:   cleaning plan (proposed actions) and deterministic application
- eda:        exploratory analysis on the cleaned data
- insights:   evidence-based findings derived from computed results
- explore:    user-configured analysis (chosen columns, groupings, filters) with charts and insights
- reporting:  CSV / Excel / JSON / HTML exports
- store:      in-memory dataset sessions
- service:    orchestration used by the API
"""
