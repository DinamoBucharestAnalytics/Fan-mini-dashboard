from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = Path(
    r"C:\Users\Razvan\Documents\Scouting\Data analysis department\Idei extrasportive"
    r"\DATA HARTA.xlsx"
)
OUTPUT_PATH = BASE_DIR / "data" / "ziua_drapelului_geocoded.csv"
USER_AGENT = "dinamo-fan-dashboard-ziua-drapelului/1.0"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
REQUEST_DELAY_SECONDS = 1.1
LAST_REQUEST_AT = 0.0

OUTPUT_COLUMNS = [
    "location",
    "count",
    "lat",
    "lon",
    "geocode_query",
    "display_name",
    "osm_type",
    "osm_id",
    "place_rank",
    "class",
    "type",
    "importance",
    "country",
    "country_code",
    "geocode_status",
]


def read_locations(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_excel(path, header=None, names=["location"])
    raw["location"] = raw["location"].astype(str).str.strip()
    raw = raw[raw["location"].ne("") & raw["location"].ne("nan")]

    counts = raw["location"].value_counts().rename_axis("location").reset_index(name="count")
    return counts.sort_values("location").reset_index(drop=True)


def read_cached_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["location"]: row for row in reader if row.get("location")}


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def ascii_text(value: str) -> str:
    text = "".join(
        char for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )
    return text.replace("ș", "s").replace("ț", "t").replace("Ș", "S").replace("Ț", "T")


def clean_location_part(value: str) -> str:
    text = value.strip()
    text = re.sub(r"\b(Municipiul|Dimos|Metropolitan City of|Province of)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(County|District|Region)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+-\s+", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,")


def add_candidate(candidates: list[str], value: str) -> None:
    value = re.sub(r"\s+", " ", value).strip(" ,")
    if value and value not in candidates:
        candidates.append(value)
    ascii_value = ascii_text(value)
    if ascii_value and ascii_value not in candidates:
        candidates.append(ascii_value)


def location_candidates(location: str) -> list[str]:
    candidates: list[str] = []
    special_cases = {
        "Dimos Lefkosias, Nicosia, Cyprus": ["Nicosia, Cyprus", "Lefkosia, Cyprus"],
        "Dimos Pafou, Paphos, Cyprus": ["Paphos, Cyprus"],
        "Metropolitan City of Florence, Tuscany, Italy": ["Florence, Italy", "Firenze, Italy"],
        "Metropolitan City of Rome Capital, Lazio, Italy": ["Rome, Italy", "Roma, Italy"],
        "Province of Ancona, Marche, Italy": ["Ancona, Italy"],
        "Municipiul Câmpulung Moldovenesc, Suceava County, Romania": ["Câmpulung Moldovenesc, Romania"],
        "Hochdorf District, Lucerne, Switzerland": ["Hochdorf, Lucerne, Switzerland"],
        "Toggenburg District, St. Gallen, Switzerland": ["Toggenburg, St. Gallen, Switzerland"],
    }
    for value in special_cases.get(location, []):
        add_candidate(candidates, value)

    parts = [part.strip() for part in location.split(",") if part and part.strip()]
    if len(parts) < 2:
        add_candidate(candidates, location)
        return candidates

    country = parts[-1]
    cleaned_parts = [clean_location_part(part) for part in parts]

    if len(parts) >= 3:
        if ascii_text(cleaned_parts[0]).lower() == ascii_text(cleaned_parts[1]).lower():
            add_candidate(candidates, f"{cleaned_parts[0]}, {country}")
        add_candidate(candidates, f"{cleaned_parts[0]}, {cleaned_parts[1]}, {country}")
        add_candidate(candidates, f"{cleaned_parts[0]}, {country}")
        add_candidate(candidates, f"{cleaned_parts[1]}, {country}")
    else:
        add_candidate(candidates, f"{cleaned_parts[0]}, {country}")

    add_candidate(candidates, ", ".join(cleaned_parts))
    add_candidate(candidates, location)

    return candidates


def wait_for_rate_limit() -> None:
    global LAST_REQUEST_AT
    elapsed = time.monotonic() - LAST_REQUEST_AT
    if LAST_REQUEST_AT and elapsed < REQUEST_DELAY_SECONDS:
        time.sleep(REQUEST_DELAY_SECONDS - elapsed)
    LAST_REQUEST_AT = time.monotonic()


def geocode_query(query: str, location: str) -> dict[str, object]:
    params = urllib.parse.urlencode(
        {
            "q": query,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": 1,
        }
    )
    request = urllib.request.Request(
        f"{NOMINATIM_URL}?{params}",
        headers={"User-Agent": USER_AGENT},
    )

    try:
        wait_for_rate_limit()
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "location": location,
            "geocode_query": query,
            "geocode_status": f"error: {exc}",
        }

    if not payload:
        return {
            "location": location,
            "geocode_query": query,
            "geocode_status": "not_found",
        }

    best = payload[0]
    address = best.get("address") or {}
    return {
        "location": location,
        "lat": best.get("lat"),
        "lon": best.get("lon"),
        "geocode_query": query,
        "display_name": best.get("display_name"),
        "osm_type": best.get("osm_type"),
        "osm_id": best.get("osm_id"),
        "place_rank": best.get("place_rank"),
        "class": best.get("category") or best.get("class"),
        "type": best.get("type"),
        "importance": best.get("importance"),
        "country": address.get("country"),
        "country_code": address.get("country_code"),
        "geocode_status": "ok",
    }


def geocode(location: str) -> dict[str, object]:
    last_result: dict[str, object] | None = None
    for query in location_candidates(location):
        result = geocode_query(query, location)
        last_result = result
        if result.get("geocode_status") == "ok":
            return result
    return last_result or {"location": location, "geocode_status": "not_found"}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    locations = read_locations(SOURCE_PATH)
    cached = read_cached_rows(OUTPUT_PATH)
    refresh_all = os.environ.get("REFRESH_GEOCODES") == "1"

    rows: list[dict[str, object]] = []
    requests_made = 0
    for idx, source_row in locations.iterrows():
        location = source_row["location"]
        count = int(source_row["count"])
        cached_row = cached.get(location)
        if (
            not refresh_all
            and cached_row
            and cached_row.get("geocode_status") == "ok"
            and cached_row.get("lat")
            and cached_row.get("lon")
            and cached_row.get("geocode_query")
        ):
            row = dict(cached_row)
            row["count"] = count
            rows.append(row)
            continue

        row = geocode(location)
        row["count"] = count
        rows.append(row)
        write_rows(OUTPUT_PATH, sorted(rows, key=lambda item: str(item.get("location", ""))))
        requests_made += 1
        print(f"{idx + 1}/{len(locations)} {row.get('geocode_status')}: {location}")

    rows = sorted(rows, key=lambda item: str(item.get("location", "")))
    write_rows(OUTPUT_PATH, rows)

    ok_count = sum(1 for row in rows if row.get("geocode_status") == "ok")
    not_ok = len(rows) - ok_count
    print(f"unique_locations={len(rows)}")
    print(f"requests_made={requests_made}")
    print(f"geocoded_ok={ok_count}")
    print(f"not_ok={not_ok}")
    print(f"output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
