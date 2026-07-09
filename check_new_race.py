#!/usr/bin/env python3
"""Cheap, dependency-free gate for the replay workflow.

Runs on every cron tick using only the Python standard library (no FastF1
install), so idle ticks cost a couple of seconds. It asks Jolpica for the most
recent classified race and decides whether that race still needs a replay:

  * nothing classified yet ................ run=false
  * race too fresh (telemetry unlikely) ... run=false
  * race too old (FastF1 never delivered) . run=false  (give up until next race)
  * already in replays/index.json ......... run=false
  * otherwise ............................. run=true  + year/round

It writes `run` / `year` / `round` to $GITHUB_OUTPUT for the workflow to read.
Diagnostics go to stderr so they show up in the Actions log.
"""

import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

JOLPICA = "https://api.jolpi.ca/ergast/f1"

# Tunables. A race is assumed to last ~2h; FastF1 usually has full telemetry
# ~20 min after it ends; if it hasn't shown up 12h later, stop trying.
RACE_DURATION_EST = timedelta(hours=2)
MIN_WAIT = timedelta(minutes=20)
GIVE_UP = timedelta(hours=12)


def emit(**outputs: str) -> None:
    """Write key=value lines to $GITHUB_OUTPUT (or stdout when run locally)."""
    text = "".join(f"{k}={v}\n" for k, v in outputs.items())
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)


def latest_classified_race():
    """(year, round, start_utc, name) for the most recent race with results."""
    url = f"{JOLPICA}/current/last/results.json"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.load(response)
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return None
    race = races[0]
    date = race.get("date")
    time = race.get("time", "13:00:00Z")
    start = datetime.fromisoformat(f"{date}T{time}".replace("Z", "+00:00"))
    return int(race["season"]), int(race["round"]), start, race.get("raceName", "")


def already_have(year: int, rnd: int) -> bool:
    index = Path("replays/index.json")
    if not index.exists():
        return False
    replays = json.loads(index.read_text()).get("replays", [])
    return any(e.get("year") == year and e.get("round") == rnd for e in replays)


def main() -> None:
    try:
        latest = latest_classified_race()
    except Exception as exc:  # noqa: BLE001 — a flaky gate must never crash the run
        print(f"gate: could not reach schedule ({exc}); skipping this tick",
              file=sys.stderr)
        emit(run="false")
        return

    if latest is None:
        print("gate: no classified race this season yet", file=sys.stderr)
        emit(run="false")
        return

    year, rnd, start, name = latest
    now = datetime.now(timezone.utc)
    ready_at = start + RACE_DURATION_EST + MIN_WAIT
    stale_at = start + RACE_DURATION_EST + GIVE_UP

    if now < ready_at:
        print(f"gate: {year} R{rnd} ({name}) too fresh; earliest attempt "
              f"{ready_at:%Y-%m-%d %H:%M}Z", file=sys.stderr)
        emit(run="false")
    elif now > stale_at:
        print(f"gate: {year} R{rnd} ({name}) still missing 12h on; giving up "
              f"until the next race", file=sys.stderr)
        emit(run="false")
    elif already_have(year, rnd):
        print(f"gate: {year} R{rnd} ({name}) already captured", file=sys.stderr)
        emit(run="false")
    else:
        print(f"gate: {year} R{rnd} ({name}) is ready — pulling", file=sys.stderr)
        emit(run="true", year=str(year), round=str(rnd))


if __name__ == "__main__":
    main()
