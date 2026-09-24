"""Add newly released Cricsheet matches so live predictions use recent form.

Usage:
    1. Download the latest IPL JSON zip from https://cricsheet.org/downloads/
       and extract it into data/raw/IPL Match Data/
    2. python scripts/update_history.py

Only matches not already known are parsed. They go into
data/processed/new_matches.csv, which the live pipeline reads alongside the
training data. The training CSV and the models are never modified.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from chimera.config import get_settings
from cricket_parser import parse_match


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=None, help="folder with Cricsheet match JSONs")
    args = ap.parse_args()

    s = get_settings()
    raw_dir = Path(args.raw_dir) if args.raw_dir else s.paths.raw_matches
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No match JSONs found in {raw_dir}")

    known = set(pd.read_csv(s.paths.history_csv, usecols=["match_id"])["match_id"].astype(str))
    existing_new = pd.read_csv(s.paths.new_matches_csv) if s.paths.new_matches_csv.exists() else None
    if existing_new is not None:
        known |= set(existing_new["match_id"].astype(str))

    rows, skipped = [], 0
    for f in files:
        if f.stem in known:
            continue
        try:
            with open(f) as fh:
                match = json.load(fh)
            if match.get("info", {}).get("event", {}).get("name", "Indian Premier League") != "Indian Premier League":
                skipped += 1
                continue
            rows.extend(parse_match(match, match_id=int(f.stem)))
        except Exception as e:  # one bad file should not stop the update
            print(f"  skipped {f.name}: {e}")
            skipped += 1

    if not rows:
        print(f"Nothing new. {len(known)} matches already known ({skipped} files skipped).")
        return
    new = pd.DataFrame(rows)
    out = pd.concat([existing_new, new], ignore_index=True) if existing_new is not None else new
    out.to_csv(s.paths.new_matches_csv, index=False)
    print(f"Added {new['match_id'].nunique()} matches ({len(new)} player rows), "
          f"latest {pd.to_datetime(new['date']).max().date()}. Saved to {s.paths.new_matches_csv}.")


if __name__ == "__main__":
    main()
