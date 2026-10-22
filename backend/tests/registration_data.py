"""Deterministic synthetic registration dataset with the same columns as the
operational registration export (no real customer data).

Columns: id, create_time_utc, user, profile, Created_by, sn, slot, pon, fdt,
fat, fat_port, ring, olt, city, mobile, clientname, status.

Real signal planted in the data (asserted in tests):
- monthly registrations grow through 2024 (a real upward trend)
- Basra has a much higher share of "Pending" registrations than other cities
- the "Business-1G" profile is concentrated in Erbil
- slot / pon / fat / fat_port / ring are small integers: positions, not quantities
- `user` holds 13-digit numeric subscriber ids starting with 9642431 (a sensitive identifier);
  900 distinct users across 1,200 records, so distinct users ≠ records
"""
import csv
import io
import random

N = 1200
N_USERS = 900
PROFILES = ["FTTH-50M", "FTTH-25M", "FTTH-100M", "FTTH-200M", "Business-1G"]
PROFILE_WEIGHTS = [40, 25, 20, 10, 5]
CITIES = ["Baghdad", "Basra", "Erbil", "Najaf", "Karbala", "Mosul"]
CITY_WEIGHTS = [35, 20, 15, 12, 10, 8]
COLUMNS = ["id", "create_time_utc", "user", "profile", "Created_by", "sn", "slot", "pon", "fdt", "fat",
           "fat_port", "ring", "olt", "city", "mobile", "clientname", "status"]


def build_rows(n=N, seed=7):
    rnd = random.Random(seed)
    staff = [f"agent{i:02d}" for i in range(1, 9)]
    users = [f"9642431{rnd.randrange(10 ** 6):06d}" for _ in range(N_USERS)]
    rows = []
    for i in range(n):
        # Later months are more likely → growing monthly volume
        month = min(12, 1 + int((rnd.random() ** 0.55) * 12))
        day = rnd.randint(1, 28)
        city = rnd.choices(CITIES, CITY_WEIGHTS)[0]
        if city == "Erbil" and rnd.random() < 0.25:
            profile = "Business-1G"
        else:
            profile = rnd.choices(PROFILES[:4], PROFILE_WEIGHTS[:4])[0]
        pending_p = 0.45 if city == "Basra" else 0.10
        r = rnd.random()
        status = "Pending" if r < pending_p else "Cancelled" if r < pending_p + 0.06 else "Active"
        agent = staff[i % len(staff)]
        rows.append({
            "id": str(100000 + i),
            "create_time_utc": f"2024-{month:02d}-{day:02d} {rnd.randint(0, 23):02d}:{rnd.randint(0, 59):02d}:{rnd.randint(0, 59):02d}",
            "user": users[i] if i < N_USERS else users[rnd.randrange(N_USERS)], "profile": profile, "Created_by": agent,
            "sn": f"ZTEG{rnd.randrange(16 ** 8):08X}",
            "slot": str(rnd.randint(1, 16)), "pon": str(rnd.randint(1, 16)),
            "fdt": f"FDT-{rnd.randint(1, 40)}", "fat": str(rnd.randint(1, 32)), "fat_port": str(rnd.randint(1, 8)),
            "ring": str(rnd.randint(1, 4)), "olt": f"OLT-{city[:3].upper()}-{rnd.randint(1, 3):02d}",
            "city": city, "mobile": f"07{rnd.choice('5789')}{rnd.randrange(10 ** 8):08d}",
            "clientname": f"Client {i:05d}", "status": status,
        })
    return rows


def registration_csv_bytes(n=N) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(build_rows(n))
    return buf.getvalue().encode("utf-8")
