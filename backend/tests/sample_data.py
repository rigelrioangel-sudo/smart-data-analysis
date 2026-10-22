"""Deterministic messy sales dataset with known, countable defects.

Real signal planted in the data:
- revenue = units × unit_price, so units and revenue are positively related
- the North region has higher unit prices (a real group difference)
- order volume grows over 2023-2024 (a real upward trend)

Defects planted (counts asserted in tests):
- 3 'region' values with stray whitespace, 'north'/'NORTH' label variants
- 4 'units' cells holding 'N/A', 2 holding 'abc', 3 negative units
- 'unit_price' with '$' prefixes, 'revenue' with thousands separators
- 2 invalid dates ('2024-02-30', '2023-13-01')
- 5 exact duplicate rows, 2 rows re-using an existing order_id
- a constant 'currency' column and an always-empty 'notes' column
"""
import csv
import io
import random

N_BASE = 240


def build_rows():
    rnd = random.Random(42)
    regions = ["North", "South", "East", "West"]
    products = ["Widget", "Gadget", "Gizmo"]
    rows = []
    for i in range(N_BASE):
        # Increasing density of orders over time → upward trend in monthly counts
        month_index = min(23, int((rnd.random() ** 0.6) * 24))
        year, month = 2023 + month_index // 12, month_index % 12 + 1
        day = rnd.randint(1, 28)
        region = regions[i % 4]
        price = round((14 if region == "North" else 9) + rnd.random() * 3, 2)
        units = rnd.randint(1, 20)
        rows.append({
            "order_id": f"{1000 + i}",
            "order_date": f"{year}-{month:02d}-{day:02d}",
            "Region": region,
            "product": products[i % 3],
            "units": str(units),
            "unit_price": f"{price:.2f}",
            "revenue": f"{units * price:.2f}",
            "returned": "yes" if i % 9 == 0 else "no",
            "currency": "USD",
            "notes": "",
        })

    # Formatting / label defects
    for i in (3, 7, 11):
        rows[i]["Region"] = f" {rows[i]['Region']} "
    rows[20]["Region"], rows[24]["Region"] = "north", "NORTH"
    for i in (30, 31, 32, 33):
        rows[i]["units"], rows[i]["revenue"] = "N/A", ""
    for i in (40, 41):
        rows[i]["units"] = "abc"
    for i in (50, 51, 52):
        rows[i]["units"] = "-" + rows[i]["units"]
    for i in range(60, 70):
        rows[i]["unit_price"] = "$" + rows[i]["unit_price"]
    rows[80]["revenue"] = "1,250.00"
    rows[81]["revenue"] = "2,400.50"
    rows[90]["order_date"] = "2024-02-30"
    rows[91]["order_date"] = "2023-13-01"

    # Duplicates
    dups = [dict(rows[i]) for i in (100, 101, 102, 103, 104)]
    key_dups = []
    for i in (110, 111):
        r = dict(rows[i])
        r["units"] = str(int(r["units"]) + 1)  # same id, conflicting content
        key_dups.append(r)
    return rows + dups + key_dups


COLUMNS = ["order_id", "order_date", "Region", "product", "units", "unit_price", "revenue", "returned", "currency", "notes"]


def sample_csv_bytes() -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(build_rows())
    return buf.getvalue().encode("utf-8")


if __name__ == "__main__":
    import os
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "..", "samples", "sales_messy.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as f:
        f.write(sample_csv_bytes())
    print("wrote", os.path.abspath(out))
