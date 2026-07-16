#!/usr/bin/env python3
"""
Convert the Battery Network sweep CSV into the app's curated-provider snapshot.

Input:  socal_battery_locations_live.csv (produced by sd_county_live_locator.py;
        current snapshot scraped July 2026)
Output: scrapp/src/lib/curated/socalBatteries.data.json, consumed by
        scrapp/src/lib/curated/socalBatteries.ts

Regenerate from the current CSV:

    cd scrapp && npm run data:batteries

Or, after a fresh sweep of the upstream locator:

    python3 sd_county_live_locator.py
    python3 generate_socal_batteries_json.py

Rows sharing a google_place_id are the same physical site registered under
multiple Battery Network accounts; they are merged by unioning their accepted
materials. Only the fields the provider needs are kept (the raw_json and the
always-empty hours/notes columns are dropped).
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "socal_battery_locations_live.csv"
OUT_PATH = ROOT / "scrapp" / "src" / "lib" / "curated" / "socalBatteries.data.json"

KNOWN_MATERIALS = {
    "rechargeable",
    "singleuse",
    "cellphones",
    "highenergybattery",
    "ebike",
}

# In a North American number both the area code and the exchange start with 2-9.
VALID_PHONE = re.compile(r"^[2-9]\d{2}-[2-9]\d{2}-\d{4}$")


def build_address(row: dict) -> str:
    zip5 = row["zip"].split("-")[0].strip()
    return f"{row['street'].strip()}, {row['city'].strip()}, CA {zip5}"


def clean_phone(phone: str) -> str:
    """
    Drop the ~20% of upstream phone numbers that are corrupt.

    Battery Network stored "1" + the 10-digit number in a 12-character field, so
    the trailing digit was truncated off: "166-186-2890" is really 661-862-890_
    (Kern County, area code 661), and "194-920-6011" is 949-206-011_ (Orange).
    The lost digit is unrecoverable and a wrong number to dial is worse than no
    number, so these are dropped rather than displayed.
    """
    return phone if VALID_PHONE.match(phone) else ""


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records: dict[str, dict] = {}
    merged_dupes = 0
    for row in rows:
        place_id = row["google_place_id"].strip()
        materials = {m.strip() for m in row["materials"].split(";") if m.strip()}
        unknown = materials - KNOWN_MATERIALS
        if unknown:
            raise SystemExit(f"Unknown materials {unknown} on row: {row['name']}")

        existing = records.get(place_id)
        if existing:
            merged_dupes += 1
            existing["materials"] = sorted(set(existing["materials"]) | materials)
            continue

        record = {
            "id": place_id,
            "name": row["name"].strip(),
            "address": build_address(row),
            "materials": sorted(materials),
            "lat": round(float(row["lat"]), 5),
            "lng": round(float(row["lng"]), 5),
            "county": row["county"].strip(),
        }
        phone = clean_phone(row["phone"].strip())
        if phone:
            record["phone"] = phone
        records[place_id] = record

    out = sorted(records.values(), key=lambda r: (r["county"], r["name"], r["id"]))
    OUT_PATH.write_text(
        json.dumps(out, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    counts = Counter(r["county"] for r in out)
    with_phone = sum(1 for r in out if r.get("phone"))
    print(f"Read {len(rows)} rows -> {len(out)} sites ({merged_dupes} dupes merged)")
    print(f"  {with_phone} with a usable phone ({len(out) - with_phone} corrupt/absent)")
    for county, n in sorted(counts.items()):
        print(f"  {county}: {n}")
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
