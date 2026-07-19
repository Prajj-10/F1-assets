#!/usr/bin/env python3
"""Pipeline: toUpperCase78/formula1-datasets CSVs -> F1-Assets curated JSON.

Replaces the hand-maintained parts of `json_files/circuit_data.json` and
`json_files/car_specs.json` with data generated from the community-maintained
CSV datasets at https://github.com/toUpperCase78/formula1-datasets
(season calendar: circuit length / turns / lap records; season teams:
chassis / power unit). The app's contract is unchanged — it keeps reading the
same JSON files from F1-Assets; only *who writes them* changes (this script
instead of a human). See F1-GRID-HANDOFF.md §6.4.

Design rules (mirrors check_new_race.py conventions):
  * Standard library only — runs anywhere, including a GitHub Actions cron.
  * DRY-RUN BY DEFAULT. Prints a field-by-field review diff; nothing is
    written unless --write is passed. The upstream CSVs contain occasional
    typos ("Michael Schumacter"), so every change is meant to be reviewed.
  * Merge, never clobber:
      - circuit fields the CSV knows (lengthKm, corners, lapRecord) are
        updated; fields it doesn't (direction, layoutNote) are preserved.
      - a lap record's `team` is preserved only while the record itself
        (driver+year) is unchanged; a new record drops the stale team.
      - curated circuits absent from the CSV calendar are kept as-is.
      - car `notes`, `powerUnit` branding etc. are preserved; the CSV
        refreshes `chassis` and `powerUnitSupplier`.
  * Fixes known key drift: entries stored under non-Ergast ids
    (cota -> americas, marina -> marina_bay) are migrated so the app's
    Ergast-id lookups actually find them.

Usage:
  python3 generate_curated_data.py                    # newest season, dry run
  python3 generate_curated_data.py --season 2025      # explicit season
  python3 generate_curated_data.py --write            # apply changes
  python3 generate_curated_data.py --csv-dir ../formula1-datasets
                                                      # local clone instead of
                                                      # downloading from GitHub
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/toUpperCase78/formula1-datasets/master"

# ---------------------------------------------------------------------------
# Mappings and corrections
# ---------------------------------------------------------------------------

# CSV `Circuit Name` -> Ergast/Jolpica circuit id (the id the app queries by).
# Keys are normalised via _norm() before lookup, so accents / "the" / minor
# spelling drift in the CSV don't break matching.
CIRCUIT_IDS = {
    "albert park circuit": "albert_park",
    "shanghai international circuit": "shanghai",
    "suzuka circuit": "suzuka",
    "suzuka international racing course": "suzuka",
    "bahrain international circuit": "bahrain",
    "jeddah corniche circuit": "jeddah",
    "miami international autodrome": "miami",
    "autodromo internazionale enzo e dino ferrari": "imola",
    "autodromo enzo e dino ferrari": "imola",
    "circuit de monaco": "monaco",
    "circuit de barcelona catalunya": "catalunya",
    "circuit gilles villeneuve": "villeneuve",
    "red bull ring": "red_bull_ring",
    "silverstone circuit": "silverstone",
    "circuit de spa francorchamps": "spa",
    "hungaroring": "hungaroring",
    "circuit zandvoort": "zandvoort",
    "circuit park zandvoort": "zandvoort",
    "autodromo nazionale monza": "monza",
    "baku city circuit": "baku",
    "marina bay street circuit": "marina_bay",
    "circuit of americas": "americas",
    "autodromo hermanos rodriguez": "rodriguez",
    "autodromo jose carlos pace": "interlagos",
    "aurodromo jose carlos pace": "interlagos",  # upstream typo, 2025 CSV
    "las vegas strip circuit": "vegas",
    "las vegas street circuit": "vegas",
    "lusail international circuit": "losail",
    "yas marina circuit": "yas_marina",
    # 2026 newcomer — confirm the Ergast id once Jolpica serves the round.
    "madring": "madring",
}

# Legacy keys in circuit_data.json that never matched an Ergast id. Entries
# under these keys are migrated to the correct id (then updated normally).
KEY_MIGRATIONS = {
    "cota": "americas",
    "marina": "marina_bay",
}

# Known upstream data errors, applied after parsing. Extend as found.
NAME_CORRECTIONS = {
    "Michael Schumacter": "Michael Schumacher",
}

# CSV `Team` name -> candidate F1-Assets team ids, most likely first. The
# first candidate that already exists in car_specs.json is updated; if none
# exists the first is created (with a warning, so renames are noticed).
TEAM_IDS = {
    "mclaren": ["MCL"],
    "mercedes": ["MER"],
    "red bull racing": ["RBR"],
    "red bull": ["RBR"],
    "ferrari": ["FER"],
    "williams": ["WIL"],
    "racing bulls": ["RB"],
    "rb": ["RB"],
    "aston martin": ["AMR"],
    "haas": ["HAAS"],
    "haas f1 team": ["HAAS"],
    "kick sauber": ["SAU", "AUD"],
    "sauber": ["SAU", "AUD"],
    "audi": ["AUD", "SAU"],
    "alpine": ["ALP"],
    "cadillac": ["CAD"],
}


def _norm(name: str) -> str:
    """Normalise a CSV name for mapping lookups."""
    text = name.lower()
    for src, dst in (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"),
                     ("ú", "u"), ("ü", "u"), ("ã", "a"), ("ç", "c"), ("-", " ")):
        text = text.replace(src, dst)
    words = [w for w in re.split(r"\s+", text) if w and w != "the"]
    return " ".join(words)


# ---------------------------------------------------------------------------
# CSV acquisition
# ---------------------------------------------------------------------------

def _season_filenames(kind: str, season: int) -> list[str]:
    """Candidate filenames for one season's CSV.

    Upstream capitalisation drifts between years
    (Formula1_2024season_calendar.csv vs Formula1_2025Season_Calendar.csv),
    so several spellings are tried.
    """
    variants = {
        "calendar": ["Season_Calendar", "season_calendar"],
        "teams": ["Season_Teams", "season_teams"],
    }[kind]
    return [f"Formula1_{season}{suffix}.csv" for suffix in variants]


def _read_local(csv_dir: Path, names: list[str]) -> str | None:
    for name in names:
        path = csv_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8-sig")
    return None


def _read_remote(names: list[str]) -> str | None:
    for name in names:
        url = f"{RAW_BASE}/{name.replace(' ', '%20')}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read().decode("utf-8-sig")
        except urllib.error.HTTPError as err:
            if err.code != 404:
                raise
    return None


def load_csv(kind: str, season: int, csv_dir: Path | None) -> list[dict] | None:
    names = _season_filenames(kind, season)
    text = _read_local(csv_dir, names) if csv_dir else _read_remote(names)
    if text is None:
        return None
    return list(csv.DictReader(io.StringIO(text)))


# ---------------------------------------------------------------------------
# circuit_data.json
# ---------------------------------------------------------------------------

def build_circuit_updates(rows: list[dict], circuits: dict) -> list[str]:
    """Merge calendar rows into the `circuits` dict in place; return a log."""
    log: list[str] = []

    for old, new in KEY_MIGRATIONS.items():
        if old in circuits:
            circuits.setdefault(new, circuits.pop(old))
            log.append(f"[migrate] key '{old}' -> '{new}' (Ergast id fix)")

    for row in rows:
        raw_name = (row.get("Circuit Name") or "").strip()
        cid = CIRCUIT_IDS.get(_norm(raw_name))
        if cid is None:
            log.append(f"[WARN] unmapped circuit '{raw_name}' — row skipped; "
                       f"add it to CIRCUIT_IDS")
            continue

        entry = circuits.setdefault(cid, {})

        def set_field(key: str, value, entry=entry, cid=cid):
            old = entry.get(key)
            if value is None or old == value:
                return
            entry[key] = value
            log.append(f"[{cid}] {key}: {old!r} -> {value!r}")

        length = row.get("Circuit Length(km)")
        set_field("lengthKm", float(length) if length else None)
        turns = row.get("Turns")
        set_field("corners", int(turns) if turns else None)

        time_ = (row.get("Lap Record") or "").strip()
        driver = (row.get("Record Owner") or "").strip()
        driver = NAME_CORRECTIONS.get(driver, driver)
        year = (row.get("Record Year") or "").strip()
        if time_ and driver and year:
            old_rec = entry.get("lapRecord") or {}
            new_rec = {"time": time_, "driver": driver, "year": int(year)}
            same_holder = (old_rec.get("driver") == driver
                           and old_rec.get("year") == int(year))
            if same_holder and old_rec.get("team"):
                new_rec["team"] = old_rec["team"]  # keep curated team name
            if old_rec.get("layoutNote"):
                new_rec["layoutNote"] = old_rec["layoutNote"]
            if new_rec != old_rec:
                entry["lapRecord"] = new_rec
                log.append(f"[{cid}] lapRecord: {old_rec or None!r} -> {new_rec!r}")

    return log


# ---------------------------------------------------------------------------
# car_specs.json
# ---------------------------------------------------------------------------

def build_car_updates(rows: list[dict], specs: dict) -> list[str]:
    """Merge team rows into car_specs' `cars` dict in place; return a log."""
    log: list[str] = []
    cars = specs.setdefault("cars", {})

    for row in rows:
        raw_name = (row.get("Team") or "").strip()
        candidates = TEAM_IDS.get(_norm(raw_name))
        if not candidates:
            log.append(f"[WARN] unmapped team '{raw_name}' — row skipped; "
                       f"add it to TEAM_IDS")
            continue
        tid = next((c for c in candidates if c in cars), None)
        if tid is None:
            tid = candidates[0]
            cars[tid] = {}
            log.append(f"[WARN] team '{raw_name}' not in car_specs — "
                       f"created '{tid}'; review its notes/powerUnit by hand")
        entry = cars[tid]

        chassis = (row.get("Chassis") or "").strip()
        if chassis and entry.get("chassis") != chassis:
            log.append(f"[{tid}] chassis: {entry.get('chassis')!r} -> {chassis!r}")
            entry["chassis"] = chassis

        supplier = (row.get("Power Unit") or "").strip()
        if supplier and entry.get("powerUnitSupplier") != supplier:
            log.append(f"[{tid}] powerUnitSupplier: "
                       f"{entry.get('powerUnitSupplier')!r} -> {supplier!r}")
            entry["powerUnitSupplier"] = supplier
        if supplier and not entry.get("powerUnit"):
            entry["powerUnit"] = supplier  # branding stays hand-curated
            log.append(f"[{tid}] powerUnit: None -> {supplier!r} (from supplier)")

    return log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def newest_available_season(csv_dir: Path | None, this_year: int) -> int | None:
    for season in range(this_year, this_year - 4, -1):
        if load_csv("calendar", season, csv_dir):
            return season
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, default=None,
                        help="season year (default: newest with a calendar CSV)")
    parser.add_argument("--csv-dir", type=Path, default=None,
                        help="local clone of formula1-datasets (skips download)")
    parser.add_argument("--assets-dir", type=Path, default=Path(__file__).parent,
                        help="F1-Assets repo root (default: script's directory)")
    parser.add_argument("--only", choices=["circuits", "cars"], default=None,
                        help="update only one file (e.g. --only circuits while "
                             "the season's teams CSV lags behind)")
    parser.add_argument("--write", action="store_true",
                        help="apply changes (default is a dry-run diff)")
    args = parser.parse_args()

    json_dir = args.assets_dir / "json_files"
    circuit_path = json_dir / "circuit_data.json"
    specs_path = json_dir / "car_specs.json"
    if not circuit_path.exists() or not specs_path.exists():
        print(f"error: {json_dir} is missing the curated JSON files; "
              f"pass --assets-dir", file=sys.stderr)
        return 2

    from datetime import date
    season = args.season or newest_available_season(args.csv_dir, date.today().year)
    if season is None:
        print("error: no season calendar CSV found upstream", file=sys.stderr)
        return 2
    print(f"season: {season}  (mode: {'WRITE' if args.write else 'dry-run'})\n")

    log: list[str] = []
    circuit_log_start = 0

    calendar = None if args.only == "cars" else load_csv(
        "calendar", season, args.csv_dir)
    circuit_data = json.loads(circuit_path.read_text(encoding="utf-8"))
    if calendar:
        log += build_circuit_updates(calendar, circuit_data.setdefault("circuits", {}))
    elif args.only != "cars":
        log.append(f"[WARN] no {season} calendar CSV upstream — "
                   f"circuit_data.json untouched")

    circuit_changed = any(not l.startswith("[WARN]") for l in log)
    circuit_log_start = len(log)

    teams = None if args.only == "circuits" else load_csv(
        "teams", season, args.csv_dir)
    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    if teams:
        if specs.get("season") not in (None, season):
            log.append(f"[WARN] car_specs.json is for season "
                       f"{specs.get('season')}, CSV is {season} — chassis "
                       f"names may belong to the older cars; review carefully")
        log += build_car_updates(teams, specs)
    elif args.only != "circuits":
        log.append(f"[WARN] no {season} teams CSV upstream — "
                   f"car_specs.json untouched")

    print("\n".join(log) if log else "no changes")

    cars_changed = any(not l.startswith("[WARN]")
                       for l in log[circuit_log_start:])
    changed = [line for line in log if not line.startswith("[WARN]")]
    if args.write and changed:
        wrote = []
        if circuit_changed:
            circuit_path.write_text(
                json.dumps(circuit_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
            wrote.append(circuit_path.name)
        if cars_changed:
            specs_path.write_text(
                json.dumps(specs, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
            wrote.append(specs_path.name)
        print(f"\nwrote {' and '.join(wrote)} ({len(changed)} change(s))")
    elif changed:
        print(f"\n{len(changed)} change(s) pending — rerun with --write to apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
