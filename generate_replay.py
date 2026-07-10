#!/usr/bin/env python3
"""Generate a compact race replay file for the F1 Grid app.

Pulls timing + GPS data via FastF1, downsamples it to 1 Hz, and writes a
single gzipped JSON file that the app downloads from the F1-Assets repo.

Usage:
    pip install fastf1 numpy pandas
    python generate_replay.py 2024 5              # one race (2024, round 5)
    python generate_replay.py 2024 all            # every completed race in 2024
    python generate_replay.py 2018-2024           # backfill whole seasons
    python generate_replay.py all                 # backfill 2018 -> current year

Output:
    replays/index.json                 (catalog the app reads first)
    replays/<year>/<year>_<round>_R.json.gz

Commit the `replays/` folder to the F1-Assets repository.

File format (version 1) — everything the app needs for one race:
    meta      year, round, raceName, circuit, totalLaps, t0 = lights out
    track     one-lap polyline (x[], y[] in decimeters)
    drivers   number, code, name, team, color, final status, retirement time
    frames    per-driver x[]/y[] sampled at 1 Hz (null = no data / retired)
    posTimeline / lapTimeline   step functions of race position and lap
    stints    tyre compound + lap ranges + age
    pits      pit lane time and estimated stationary time per stop
    flags     track status segments (green/yellow/SC/VSC/red)
    weather   one sample per minute

Position data (posTimeline / posSource):
    2018-2022: FastF1 lap timing only. posTimeline updates at lap boundaries;
               posSource = "lap".
    2023+:     Also queries OpenF1's /position endpoint, which reports a car's
               position the moment it changes (not just at the line). Where
               available this replaces the per-driver lap-boundary series with
               a denser one, so overtakes show up mid-lap; posSource = "live".
               If OpenF1 has no matching session or the request fails, the
               race falls back to the lap-boundary series with no error.
"""

import gzip
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import fastf1

OUTPUT_ROOT = Path("replays")
SAMPLE_HZ = 1.0
TRACK_POINTS = 400

# FastF1 only carries full car position telemetry from 2018 onward.
FIRST_TELEMETRY_YEAR = 2018

# OpenF1's /position feed (sub-lap position changes) only covers 2023+.
OPENF1_BASE = "https://api.openf1.org/v1"
OPENF1_FIRST_YEAR = 2023

CACHE_DIR = Path("fastf1_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)  # FastF1 3.x requires it to exist
fastf1.Cache.enable_cache(str(CACHE_DIR))  # speeds up re-runs enormously


def _clean(value, default=None):
    """None-safe conversion of pandas/numpy scalars for JSON."""
    if value is None:
        return default
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return default
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 3)
    if isinstance(value, pd.Timedelta):
        return round(value.total_seconds(), 3)
    return value


def _seconds(td) -> float | None:
    if td is None or pd.isna(td):
        return None
    return td.total_seconds()


def position_source(year: int, openf1_used: bool) -> str:
    """How the race order/positions in this replay were derived.

    "live" only when OpenF1's sub-lap position feed was actually merged in for
    a meaningful share of the field; otherwise "lap" (FastF1 lap timing,
    updating at lap boundaries), which is always computed as the baseline.
    """
    return "live" if (year >= OPENF1_FIRST_YEAR and openf1_used) else "lap"


# Sessions list per year, fetched once and reused across every race in a
# batch run instead of re-querying OpenF1 for each round.
_openf1_sessions_cache: dict[int, list[dict]] = {}


def _openf1_get(path: str, **params) -> list[dict]:
    query = urllib.parse.urlencode(params)
    url = f"{OPENF1_BASE}/{path}?{query}" if query else f"{OPENF1_BASE}/{path}"
    last_exc = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"openf1 request failed after retries: {last_exc}")


def _openf1_session_key(year: int, race_t0_utc: pd.Timestamp) -> int | None:
    """Match this race to its OpenF1 session_key by nearest session date.

    Race weekends are on distinct calendar days, so the closest "Race" session
    start time for the season is a reliable, unambiguous match.
    """
    if year not in _openf1_sessions_cache:
        try:
            _openf1_sessions_cache[year] = _openf1_get(
                "sessions", year=year, session_name="Race",
            )
        except Exception as exc:
            print(f"  openf1: could not list {year} sessions ({exc})")
            _openf1_sessions_cache[year] = []

    sessions = _openf1_sessions_cache[year]
    best_key, best_diff = None, None
    for s in sessions:
        try:
            d = pd.Timestamp(s["date_start"])
        except Exception:
            continue
        if d.tzinfo is None:
            d = d.tz_localize("UTC")
        diff = abs((d - race_t0_utc).total_seconds())
        if best_diff is None or diff < best_diff:
            best_key, best_diff = s.get("session_key"), diff

    # A day+ off means we didn't actually find this race in OpenF1.
    if best_key is not None and best_diff is not None and best_diff < 86400:
        return best_key
    return None


def openf1_clock_offset(
    session_key: int,
    lights_out_utc: pd.Timestamp,
    fastf1_lap_starts: dict[tuple[int, int], float],
) -> float:
    """Median offset (s) between OpenF1's clock and FastF1's telemetry clock.

    The two feeds can run a few seconds apart, which makes OpenF1-sourced
    position changes lag the FastF1 car positions on screen. We line them up on
    a signal both report — per-driver lap start times — and return the median
    difference, to be subtracted from OpenF1 event times. Returns 0.0 if it
    can't be determined (then raw OpenF1 times are used, as before).
    """
    if not fastf1_lap_starts:
        return 0.0
    try:
        rows = _openf1_get("laps", session_key=session_key)
    except Exception:
        return 0.0

    deltas: list[float] = []
    for row in rows:
        start = row.get("date_start")
        drv = row.get("driver_number")
        lap = row.get("lap_number")
        if start is None or drv is None or lap is None:
            continue
        fastf1_t = fastf1_lap_starts.get((int(drv), int(lap)))
        if fastf1_t is None:
            continue
        try:
            openf1_t = (pd.Timestamp(start) - lights_out_utc).total_seconds()
        except Exception:
            continue
        deltas.append(openf1_t - fastf1_t)

    if not deltas:
        return 0.0
    deltas.sort()
    return deltas[len(deltas) // 2]  # median resists pit-lap/outlier noise


def fetch_openf1_positions(
    year: int,
    lights_out_utc: pd.Timestamp,
    duration: float,
    fastf1_lap_starts: dict[tuple[int, int], float],
) -> dict[str, list[list[float]]]:
    """Sub-lap position events per driver: {"num": [[t, position], ...]}.

    Never raises — any OpenF1 problem (no match, network error, empty
    response) yields {}, and the caller keeps the FastF1 lap-boundary
    timeline it already computed.
    """
    if year < OPENF1_FIRST_YEAR:
        return {}

    key = _openf1_session_key(year, lights_out_utc)
    if key is None:
        print("  openf1: no matching session found, using lap-boundary positions")
        return {}

    offset = openf1_clock_offset(key, lights_out_utc, fastf1_lap_starts)

    try:
        rows = _openf1_get("position", session_key=key)
    except Exception as exc:
        print(f"  openf1: position fetch failed ({exc}), using lap-boundary positions")
        return {}
    if not rows:
        return {}

    per_driver: dict[str, list[list[float]]] = {}
    for row in rows:
        try:
            # Subtract the OpenF1<->FastF1 clock offset so position changes
            # line up with the car positions on screen.
            t = (pd.Timestamp(row["date"]) - lights_out_utc).total_seconds() \
                - offset
        except Exception:
            continue
        if t < -30 or t > duration + 30:
            continue
        num = str(row.get("driver_number"))
        per_driver.setdefault(num, []).append(
            [round(max(t, 0.0), 1), int(row["position"])]
        )

    for num, events in per_driver.items():
        events.sort(key=lambda e: e[0])
        deduped: list[list[float]] = []
        for t, pos in events:
            if deduped and deduped[-1][1] == pos:
                continue
            deduped.append([t, pos])
        per_driver[num] = deduped

    if per_driver:
        print(f"  openf1: sub-lap positions for {len(per_driver)} drivers "
              f"(clock offset {offset:+.1f}s)")
    return per_driver


def build_replay(year: int, rnd: int) -> dict | None:
    session = fastf1.get_session(year, rnd, "R")
    session.load(laps=True, telemetry=True, weather=True, messages=True)

    if session.laps is None or len(session.laps) == 0:
        print(f"  no lap data for {year} round {rnd}, skipping")
        return None

    t_start = _seconds(session.session_start_time) or 0.0
    laps = session.laps
    t_end = float(np.nanmax(laps["Time"].dt.total_seconds())) + 60.0
    duration = t_end - t_start
    grid = np.arange(0.0, duration, 1.0 / SAMPLE_HZ)

    # ---------------- drivers + retirement detection ----------------
    drivers = []
    retirement_t: dict[str, float | None] = {}
    for _, row in session.results.iterrows():
        num = str(row["DriverNumber"])
        status = str(row["Status"])
        finished = status.startswith("Finished") or status.startswith("+")
        drv_laps = laps[laps["DriverNumber"] == num]
        ret_t = None
        if not finished and len(drv_laps) > 0:
            last = _seconds(drv_laps["Time"].max())
            if last is not None:
                ret_t = round(last - t_start, 1)
        retirement_t[num] = ret_t
        color = str(row.get("TeamColor") or "808080").lstrip("#") or "808080"
        drivers.append({
            "num": int(num),
            "code": str(row.get("Abbreviation") or f"#{num}"),
            "name": str(row.get("FullName") or f"Driver {num}"),
            "team": str(row.get("TeamName") or ""),
            "color": color,
            "grid": _clean(row.get("GridPosition"), 0),
            "finalPos": _clean(row.get("ClassifiedPosition"), None),
            "status": status,
            "retiredT": ret_t,
        })

    # ---------------- per-driver GPS frames (1 Hz) ----------------
    frames_cars = {}
    for num, pos in session.pos_data.items():
        if pos is None or len(pos) == 0:
            continue
        t = pos["SessionTime"].dt.total_seconds().to_numpy() - t_start
        x = pos["X"].to_numpy(dtype=float)
        y = pos["Y"].to_numpy(dtype=float)
        valid = ~(np.isnan(t) | np.isnan(x) | np.isnan(y))
        t, x, y = t[valid], x[valid], y[valid]
        if len(t) < 10:
            continue
        order = np.argsort(t)
        t, x, y = t[order], x[order], y[order]

        xi = np.interp(grid, t, x, left=np.nan, right=np.nan)
        yi = np.interp(grid, t, y, left=np.nan, right=np.nan)

        # Blank everything after retirement (+30 s so the car parks visibly).
        ret = retirement_t.get(str(num))
        if ret is not None:
            xi[grid > ret + 30] = np.nan
            yi[grid > ret + 30] = np.nan

        frames_cars[str(int(num))] = {
            "x": [None if np.isnan(v) else int(round(v)) for v in xi],
            "y": [None if np.isnan(v) else int(round(v)) for v in yi],
        }

    # ---------------- track polyline from the fastest lap ----------------
    if not frames_cars:
        print(f"  no position telemetry yet for {year} round {rnd}, skipping")
        return None

    fastest = laps.pick_fastest()
    tel = fastest.get_pos_data()
    step = max(1, len(tel) // TRACK_POINTS)
    track = {
        "x": [int(round(v)) for v in tel["X"].to_numpy()[::step]],
        "y": [int(round(v)) for v in tel["Y"].to_numpy()[::step]],
    }

    # ---------------- position + lap step-function timelines ----------------
    pos_timeline: dict[str, list] = {}
    lap_timeline: dict[str, list] = {}
    for num in frames_cars:
        drv_laps = laps[laps["DriverNumber"] == num].sort_values("LapNumber")
        pt, lt = [], []
        for _, lap in drv_laps.iterrows():
            lap_start = _seconds(lap["LapStartTime"])
            lap_end = _seconds(lap["Time"])
            if lap_start is not None and not pd.isna(lap["LapNumber"]):
                lt.append([round(lap_start - t_start, 1),
                           int(lap["LapNumber"])])
            if lap_end is not None and not pd.isna(lap["Position"]):
                pt.append([round(lap_end - t_start, 1),
                           int(lap["Position"])])
        # Seed the start with grid position so the pre-lap-1 order is sane.
        drv_meta = next(d for d in drivers if d["num"] == int(num))
        if drv_meta["grid"]:
            pt.insert(0, [0.0, int(drv_meta["grid"])])
        pos_timeline[num] = pt
        lap_timeline[num] = lt

    # ---------------- sub-lap positions from OpenF1 (2023+) ----------------
    t0 = session.t0_date
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    lights_out_utc = t0 + pd.Timedelta(seconds=t_start)

    # Per-(driver, lap) FastF1 lap-start times, used to calibrate OpenF1's clock.
    fastf1_lap_starts = {
        (int(num), int(lap)): start
        for num, lt in lap_timeline.items()
        for (start, lap) in lt
    }

    openf1_events = fetch_openf1_positions(
        year, lights_out_utc, duration, fastf1_lap_starts,
    )
    merged = 0
    for num, events in openf1_events.items():
        if num not in pos_timeline or not events:
            continue
        # Keep the grid-position seed at t=0 (pre-lap-1 order), then use
        # OpenF1's denser event stream for everything after the start.
        seed = [e for e in pos_timeline[num] if e[0] == 0.0][:1]
        pos_timeline[num] = seed + [e for e in events if e[0] > 0.0]
        merged += 1
    openf1_used = merged >= max(1, len(frames_cars) // 2)

    # ---------------- stints (tyres) ----------------
    stints: dict[str, list] = {}
    for num in frames_cars:
        drv_laps = laps[laps["DriverNumber"] == num]
        out = []
        for stint_no, chunk in drv_laps.groupby("Stint"):
            if pd.isna(stint_no):
                continue
            compound = str(chunk["Compound"].iloc[0] or "UNKNOWN")
            out.append([
                compound[:1].upper() if compound != "UNKNOWN" else "?",
                int(chunk["LapNumber"].min()),
                int(chunk["LapNumber"].max()),
                _clean(chunk["TyreLife"].iloc[0], 0),
            ])
        stints[num] = out

    # ---------------- pit stops ----------------
    pits: dict[str, list] = {}
    for num, car in frames_cars.items():
        drv_laps = laps[laps["DriverNumber"] == num]
        out = []
        for _, lap in drv_laps.iterrows():
            pit_in = _seconds(lap["PitInTime"])
            pit_out = _seconds(lap["PitOutTime"])
            if pit_in is None:
                continue
            lane_s = None
            stop_s = None
            if pit_out is not None and pit_out > pit_in:
                lane_s = round(pit_out - pit_in, 1)
                # Estimate stationary time: seconds within the pit window
                # where the car barely moves.
                i0 = max(0, int(pit_in - t_start))
                i1 = min(len(car["x"]) - 1, int(pit_out - t_start))
                still = 0
                for i in range(i0, i1):
                    x0, y0 = car["x"][i], car["y"][i]
                    x1, y1 = car["x"][i + 1], car["y"][i + 1]
                    if None in (x0, y0, x1, y1):
                        continue
                    if math.hypot(x1 - x0, y1 - y0) < 20:  # <2 m/s
                        still += 1
                stop_s = float(still) if still else None
            out.append([
                _clean(lap["LapNumber"], 0),
                round(pit_in - t_start, 1),
                lane_s,
                stop_s,
            ])
        if out:
            pits[num] = out

    # ---------------- flags (track status segments) ----------------
    flags = []
    ts = session.track_status
    if ts is not None and len(ts) > 0:
        times = ts["Time"].dt.total_seconds().to_numpy() - t_start
        codes = ts["Status"].astype(str).to_numpy()
        for i in range(len(times)):
            seg_start = max(0.0, times[i])
            seg_end = times[i + 1] if i + 1 < len(times) else duration
            if seg_end <= 0 or seg_end <= seg_start:
                continue
            flags.append([round(seg_start, 1), round(seg_end, 1),
                          int(codes[i]) if codes[i].isdigit() else 1])
    if not flags:
        flags = [[0.0, round(duration, 1), 1]]

    # ---------------- weather (per minute) ----------------
    weather = []
    wd = session.weather_data
    if wd is not None and len(wd) > 0:
        for _, row in wd.iterrows():
            t = _seconds(row["Time"])
            if t is None or t < t_start:
                continue
            weather.append([
                round(t - t_start, 0),
                _clean(row["AirTemp"]),
                _clean(row["TrackTemp"]),
                _clean(row["Humidity"]),
                _clean(row["WindSpeed"]),
                1 if bool(row["Rainfall"]) else 0,
            ])

    return {
        "version": 1,
        "year": year,
        "round": rnd,
        "raceName": str(session.event["EventName"]),
        "circuit": str(session.event["Location"]),
        "totalLaps": _clean(session.total_laps, 0),
        "posSource": position_source(year, openf1_used),
        "sampleHz": SAMPLE_HZ,
        "durationS": round(duration, 1),
        "track": track,
        "drivers": drivers,
        "frames": {"dt": 1.0 / SAMPLE_HZ, "cars": frames_cars},
        "posTimeline": pos_timeline,
        "lapTimeline": lap_timeline,
        "stints": stints,
        "pits": pits,
        "flags": flags,
        "weather": weather,
    }


def write_replay(replay: dict) -> Path:
    year, rnd = replay["year"], replay["round"]
    out_dir = OUTPUT_ROOT / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{year}_{rnd:02d}_R.json.gz"
    payload = json.dumps(replay, separators=(",", ":")).encode()
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(payload)
    print(f"  wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    return path


def update_index(entries: list[dict]) -> None:
    index_path = OUTPUT_ROOT / "index.json"
    existing = []
    if index_path.exists():
        existing = json.loads(index_path.read_text()).get("replays", [])
    by_key = {(e["year"], e["round"]): e for e in existing}
    for e in entries:
        by_key[(e["year"], e["round"])] = e
    merged = sorted(by_key.values(),
                    key=lambda e: (e["year"], e["round"]), reverse=True)
    index_path.write_text(
        json.dumps({"version": 1, "replays": merged}, indent=1))
    print(f"  index now lists {len(merged)} replays")


def completed_rounds(year: int) -> list[int]:
    """Rounds in [year] whose race has already finished.

    Robust to FastF1/pandas returning the race date as either tz-aware or
    tz-naive (older versions differed), which is what caused the historical
    'Already tz-aware' crash on the `all` path.
    """
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    now = pd.Timestamp.now(tz="UTC")
    done = []
    for _, event in schedule.iterrows():
        rnd = int(event["RoundNumber"])
        if rnd < 1:  # skip pre-season testing
            continue
        race_date = pd.Timestamp(event.get("Session5DateUtc"))
        if pd.isna(race_date):
            continue
        if race_date.tzinfo is None:
            race_date = race_date.tz_localize("UTC")
        if now > race_date:
            done.append(rnd)
    return done


def parse_jobs(argv: list[str]) -> list[tuple[int, str]]:
    """Turn CLI args into a list of (year, round_spec) jobs.

    Accepts:
        <year> <round>     e.g. 2024 5
        <year> all         e.g. 2024 all
        <year0>-<year1>    e.g. 2018-2024   (round defaults to 'all')
        all                backfill FIRST_TELEMETRY_YEAR -> current year
    """
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)

    year_arg = argv[1]
    round_arg = argv[2] if len(argv) > 2 else "all"
    current_year = pd.Timestamp.now(tz="UTC").year

    if year_arg == "all":
        years = list(range(FIRST_TELEMETRY_YEAR, current_year + 1))
    elif "-" in year_arg:
        low, high = year_arg.split("-", 1)
        years = list(range(int(low), int(high) + 1))
    else:
        years = [int(year_arg)]

    return [(year, round_arg) for year in years]


def main() -> None:
    jobs = parse_jobs(sys.argv)

    new_entries = []
    for year, round_arg in jobs:
        rounds = completed_rounds(year) if round_arg == "all" else [int(round_arg)]
        for rnd in rounds:
            print(f"Processing {year} round {rnd}...")
            try:
                replay = build_replay(year, rnd)
            except Exception as exc:  # noqa: BLE001 — keep batch runs going
                print(f"  FAILED: {exc}")
                continue
            if replay is None:
                continue
            path = write_replay(replay)
            new_entries.append({
                "year": year,
                "round": rnd,
                "raceName": replay["raceName"],
                "circuit": replay["circuit"],
                "file": str(path.relative_to(OUTPUT_ROOT)),
                "sizeKb": path.stat().st_size // 1024,
                "posSource": replay.get("posSource", "lap"),
            })

    if new_entries:
        update_index(new_entries)
    else:
        print("Nothing new to write.")


if __name__ == "__main__":
    main()
