"""Historical data access and entity resolution.

Loads the per-player match history once and exposes fast lookups used by the
live feature builder: each player's prior matches, venue scoring history,
training-time fill constants, role encodings, credits, and resolvers that map
what a user types ("Virat Kohli", "RCB", "Chinnaswamy") onto the exact strings
the models were trained with ("V Kohli", "Royal Challengers Bengaluru", ...).
"""
from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import Settings
from .constants import (
    MODEL_ROLE_ENCODING,
    TEAM_ALIASES,
    USER_ROLE_TO_MODEL_ROLE,
    WEATHER_FEATURES,
    WICKETKEEPER_MIN_CAREER_STUMPINGS,
)

log = logging.getLogger(__name__)

# Raw columns the live feature builder needs. Everything else is derived.
RAW_COLUMNS = [
    "match_id", "player", "team", "opposition", "venue", "city", "date", "season",
    "toss_winner", "runs", "balls_faced", "fours", "sixes", "strike_rate",
    "batting_position", "wickets", "runs_conceded", "balls_bowled", "economy",
    "maidens", "total_wickets", "player_innings", "team_total",
    "total_fantasy_points", "stumpings",
]


@dataclass
class ResolvedName:
    input: str
    name: Optional[str]          # matched history name, None if unknown/new player
    method: str                  # exact | case_insensitive | surname_initial | subset | reversed_name | fuzzy | unresolved
    confidence: float


@dataclass
class ResolvedVenue:
    input: str
    venue: str
    city: Optional[str]
    method: str
    known: bool


def _season_label(value) -> str:
    return str(value).strip()


class DataStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        p = settings.paths

        train = pd.read_csv(p.history_csv, low_memory=False)
        self._train_full = train  # kept only to derive training-time constants

        frames = [train[[c for c in RAW_COLUMNS if c in train.columns]]]
        if p.new_matches_csv.exists():
            extra = pd.read_csv(p.new_matches_csv, low_memory=False)
            extra = extra[~extra["match_id"].isin(train["match_id"].unique())]
            if len(extra):
                log.info("Loaded %d rows from %s", len(extra), p.new_matches_csv.name)
                frames.append(extra[[c for c in RAW_COLUMNS if c in extra.columns]])

        hist = pd.concat(frames, ignore_index=True)
        hist["date"] = pd.to_datetime(hist["date"])
        hist["season"] = hist["season"].map(_season_label)
        if "stumpings" not in hist.columns:
            hist["stumpings"] = 0
        hist["stumpings"] = hist["stumpings"].fillna(0)
        hist = hist.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)

        # Derived per-row contributions, exactly as in 03_feature_engineering cell 13.
        hist["batting_contribution"] = np.where(
            hist["team_total"] > 0, hist["runs"] / hist["team_total"].where(hist["team_total"] > 0, 1), 0.0
        )
        hist["bowling_contribution"] = np.where(
            hist["total_wickets"] > 0, hist["wickets"] / hist["total_wickets"].where(hist["total_wickets"] > 0, 1), 0.0
        )
        self.history = hist

        # player -> their rows in date order (a view per player keeps lookups O(1))
        self._by_player = {name: g for name, g in hist.groupby("player", sort=False)}

        self._build_match_level()
        self._derive_training_constants()
        self._build_role_and_credit_lookups()
        self._build_resolvers()

        log.info(
            "DataStore ready: %d rows, %d players, %d matches, history through %s",
            len(hist), len(self._by_player), hist["match_id"].nunique(), hist["date"].max().date(),
        )

    # ------------------------------------------------------------------ history

    def player_history(self, player: str, before: pd.Timestamp) -> pd.DataFrame:
        """All of a player's matches strictly before `before`, oldest first."""
        g = self._by_player.get(player)
        if g is None:
            return self.history.iloc[0:0]
        return g[g["date"] < before]

    def has_player(self, name: str) -> bool:
        return name in self._by_player

    @property
    def last_history_date(self) -> pd.Timestamp:
        return self.history["date"].max()

    # --------------------------------------------------------- match-level data

    def _build_match_level(self) -> None:
        """Replicates 03_feature_engineering cells 34-37 (one row per match)."""
        h = self.history
        inns = (
            h[h["player_innings"] != 0]
            .drop_duplicates(subset=["match_id", "player_innings"])
            [["match_id", "player_innings", "team_total"]]
        )
        pivot = inns.pivot(index="match_id", columns="player_innings", values="team_total")
        pivot = pivot.rename(columns={1: "innings1_score", 2: "innings2_score"})
        for col in ("innings1_score", "innings2_score"):
            if col not in pivot.columns:
                pivot[col] = np.nan
        pivot = pivot[["innings1_score", "innings2_score"]].reset_index()

        meta = h.drop_duplicates(subset="match_id")[["match_id", "venue", "city", "date"]]
        ml = pivot.merge(meta, on="match_id", how="left")
        ml["total_runs"] = ml["innings1_score"] + ml["innings2_score"]  # NaN if an innings is missing
        self.match_level = ml.sort_values("date", kind="mergesort").reset_index(drop=True)

    def venue_innings_averages(self, venue: str, before: pd.Timestamp) -> dict:
        """Mean first/second innings and total runs at `venue` before `before` (NaN if none)."""
        ml = self.match_level
        prior = ml[(ml["venue"] == venue) & (ml["date"] < before)]
        return {
            "venue_avg_innings1": prior["innings1_score"].mean(),
            "venue_avg_innings2": prior["innings2_score"].mean(),
            "venue_avg_total_runs": prior["total_runs"].mean(),
        }

    def _derive_training_constants(self) -> None:
        """Fill values the training data used for missing features.

        Venue averages: read directly from training rows where the venue had no
        prior match (those rows hold the fill constant). Weather: mean over rows
        that had weather, as in 03_feature_engineering cell 51.
        """
        ml = self.match_level.copy()
        for src, dst in (
            ("innings1_score", "venue_avg_innings1"),
            ("innings2_score", "venue_avg_innings2"),
            ("total_runs", "venue_avg_total_runs"),
        ):
            ml[dst] = ml.groupby("venue")[src].transform(lambda x: x.shift(1).expanding().mean())

        cols = ("venue_avg_innings1", "venue_avg_innings2", "venue_avg_total_runs")
        train_ids = self._train_full[["match_id"]]
        per_row = train_ids.merge(ml[["match_id", *cols]], on="match_id", how="left")

        first_at_venue = ml.loc[ml["venue_avg_innings1"].isna(), "match_id"]
        t = self._train_full
        filled = t[t["match_id"].isin(first_at_venue)]
        self.venue_fill = {}
        for c in cols:
            if len(filled) and c in t.columns:
                self.venue_fill[c] = float(filled[c].median())
            else:
                self.venue_fill[c] = float(per_row[c].mean())

        mask = t["city"].fillna("Unknown") != "Unknown"
        self.weather_fill = {c: float(t.loc[mask, c].mean()) for c in WEATHER_FEATURES}

    # ------------------------------------------------------- roles and credits

    def _build_role_and_credit_lookups(self) -> None:
        t = self._train_full
        # Model role encoding: static per player in the training data.
        last = t.drop_duplicates(subset="player", keep="last")
        self._model_role_encoded = dict(zip(last["player"], last["role_encoded"].astype(int)))

        career_stumpings = self.history.groupby("player")["stumpings"].sum()
        self._keepers = set(career_stumpings[career_stumpings >= WICKETKEEPER_MIN_CAREER_STUMPINGS].index)

        credits = pd.read_csv(self.settings.paths.credits_csv)
        self._credit = dict(zip(credits["player"], credits["credit_value"].astype(float)))
        base_role = dict(zip(credits["player"], credits["role"]))
        # Apply the WK override here so both old and new credits files behave the same.
        self._optimizer_role = {
            p: ("wicketkeeper" if p in self._keepers else r) for p, r in base_role.items()
        }
        credits["opt_role"] = credits["player"].map(self._optimizer_role)
        self._median_credit_by_role = credits.groupby("opt_role")["credit_value"].median().to_dict()
        self._median_credit = float(credits["credit_value"].median())

    def model_role_encoded(self, player: Optional[str], user_role: Optional[str] = None) -> tuple[int, str]:
        """Role encoding the model was trained with, and where it came from."""
        if player and player in self._model_role_encoded:
            return self._model_role_encoded[player], "history"
        if player and player in self._by_player:
            return MODEL_ROLE_ENCODING[self._classify_from_history(player)], "derived"
        if user_role:
            return MODEL_ROLE_ENCODING[USER_ROLE_TO_MODEL_ROLE.get(user_role, "unknown")], "user"
        return MODEL_ROLE_ENCODING["unknown"], "default"

    def _classify_from_history(self, player: str) -> str:
        """Same rule as classify_role in 03_feature_engineering cell 26."""
        g = self._by_player[player]
        total_matches = len(g)
        career_runs = g["runs"].sum()
        career_wickets = g["wickets"].sum()
        career_balls_bowled = g["balls_bowled"].sum()
        avg_bat_position = g["batting_position"].mode().iloc[0]
        genuine_bowler = career_wickets >= 20 and career_balls_bowled >= 200
        genuine_batter = career_runs >= 500 or (career_runs >= 300 and avg_bat_position <= 7)
        if total_matches > 20:
            if genuine_batter and genuine_bowler:
                return "allrounder"
            if genuine_bowler:
                return "bowler"
            if genuine_batter:
                return "batter"
            return "unknown"
        if avg_bat_position == 0:
            return "bowler"
        if career_balls_bowled > 50 and avg_bat_position >= 7:
            return "bowler"
        if avg_bat_position <= 7 and career_balls_bowled > 30:
            return "allrounder"
        if avg_bat_position <= 7:
            return "batter"
        return "bowler"

    def optimizer_role(self, player: Optional[str], user_role: Optional[str] = None) -> tuple[str, str]:
        if user_role:
            return user_role, "user"
        if player and player in self._optimizer_role:
            return self._optimizer_role[player], "credits"
        if player and player in self._by_player:
            role = self._classify_from_history(player)
            if player in self._keepers:
                role = "wicketkeeper"
            return role, "derived"
        return "unknown", "default"

    def credit(self, player: Optional[str], role: str, user_credit: Optional[float] = None) -> tuple[float, str]:
        if user_credit is not None:
            return float(user_credit), "user"
        if player and player in self._credit:
            return self._credit[player], "credits"
        return float(self._median_credit_by_role.get(role, self._median_credit)), "role_median"

    # --------------------------------------------------------------- resolvers

    def _build_resolvers(self) -> None:
        h = self.history
        last_rows = h.drop_duplicates(subset="player", keep="last").set_index("player")
        self._player_last_team = last_rows["team"].to_dict()
        self._player_last_date = last_rows["date"].to_dict()

        self._known_players = sorted(set(self._by_player) | set(self._credit))
        self._lower_to_player = {p.lower(): p for p in self._known_players}
        self._by_surname: dict[str, list[str]] = {}
        for p in self._known_players:
            self._by_surname.setdefault(p.split()[-1].lower(), []).append(p)

        self._known_teams = sorted(h["team"].unique())
        self._alias_to_team = {}
        for canonical, aliases in TEAM_ALIASES.items():
            for a in aliases:
                self._alias_to_team[a] = canonical
        for t in self._known_teams:
            self._alias_to_team.setdefault(t.lower(), t)

        v = (h.sort_values("date", kind="mergesort")
             .groupby("venue")
             .agg(last_date=("date", "max"), city=("city", "last"), rows=("date", "size"))
             .reset_index())
        self._venues = v.sort_values("last_date", ascending=False).reset_index(drop=True)

    def canonical_team(self, name: str) -> tuple[str, bool]:
        key = name.strip().lower()
        if key in self._alias_to_team:
            return self._alias_to_team[key], True
        match = difflib.get_close_matches(key, list(self._alias_to_team), n=1, cutoff=0.8)
        if match:
            return self._alias_to_team[match[0]], True
        return name.strip(), False

    def _rank_by_recency(self, candidates: list[str], team: Optional[str]) -> list[str]:
        def key(p):
            same_team = 1 if (team and self._player_last_team.get(p) == team) else 0
            last = self._player_last_date.get(p, pd.Timestamp.min)
            return (same_team, last)
        return sorted(candidates, key=key, reverse=True)

    def resolve_player(self, raw: str, team: Optional[str] = None) -> ResolvedName:
        name = " ".join(raw.split())
        if name in self._by_player or name in self._credit:
            return ResolvedName(raw, name, "exact", 1.0)
        low = name.lower()
        if low in self._lower_to_player:
            return ResolvedName(raw, self._lower_to_player[low], "case_insensitive", 1.0)

        tokens = name.split()
        if len(tokens) >= 2:
            surname = tokens[-1].lower()
            first = tokens[0]
            candidates = []
            for known in self._by_surname.get(surname, []):
                ktoks = known.split()
                if len(ktoks) < 2:
                    continue
                kfirst = ktoks[0]
                is_initials = kfirst.isupper() and len(kfirst) <= 3
                if is_initials:
                    if kfirst[0].upper() == first[0].upper():
                        candidates.append(known)
                elif kfirst.lower() == first.lower():
                    candidates.append(known)
            if candidates:
                best = self._rank_by_recency(candidates, team)[0]
                conf = 0.95 if len(candidates) == 1 else 0.8
                return ResolvedName(raw, best, "surname_initial", conf)

            in_toks = [t.lower() for t in tokens]
            pool = set(self._by_surname.get(in_toks[-1], [])) | set(self._by_surname.get(in_toks[0], []))

            # every input word appears in the stored name: "Sai Sudharsan" -> "B Sai Sudharsan"
            subset = [k for k in pool if set(in_toks) <= {t.lower() for t in k.split()}]
            if subset:
                best = self._rank_by_recency(subset, team)[0]
                return ResolvedName(raw, best, "subset", 0.95 if len(subset) == 1 else 0.8)

            # given name stored last, surname as an initial: "Varun Chakravarthy" -> "CV Varun"
            reversed_hits = []
            for k in self._by_surname.get(in_toks[0], []):
                ktoks = k.split()
                if len(ktoks) >= 2 and ktoks[0].isupper() and len(ktoks[0]) <= 3 \
                        and in_toks[-1][0].upper() in ktoks[0]:
                    reversed_hits.append(k)
            if reversed_hits:
                best = self._rank_by_recency(reversed_hits, team)[0]
                return ResolvedName(raw, best, "reversed_name", 0.9 if len(reversed_hits) == 1 else 0.75)

        pool_all = self._known_players
        if team:
            team_players = [p for p in pool_all if self._player_last_team.get(p) == team]
            match = difflib.get_close_matches(name, team_players, n=1, cutoff=0.8)
            if match:
                return ResolvedName(raw, match[0], "fuzzy", 0.75)
        match = difflib.get_close_matches(name, pool_all, n=1, cutoff=0.88)
        if match:
            return ResolvedName(raw, match[0], "fuzzy", 0.7)
        return ResolvedName(raw, None, "unresolved", 0.0)

    def search_players(self, query: str, limit: int = 10) -> list[dict]:
        q = query.strip().lower()
        hits = [p for p in self._known_players if q in p.lower()]
        if not hits:
            hits = difflib.get_close_matches(query, self._known_players, n=limit, cutoff=0.6)
        hits = self._rank_by_recency(hits, None)[:limit]
        return [
            {"name": p, "last_team": self._player_last_team.get(p),
             "last_match": (self._player_last_date[p].date().isoformat() if p in self._player_last_date else None)}
            for p in hits
        ]

    def resolve_venue(self, raw: str, city: Optional[str] = None) -> ResolvedVenue:
        """Map free text onto the venue string most recently used in the data.

        Cricsheet renames grounds over time ("Eden Gardens" became
        "Eden Gardens, Kolkata"). A new match is recorded under the current
        name, so among all matching names we always take the most recent one,
        even when the input exactly equals an older name.
        """
        v = self._venues  # sorted most recent first
        name = raw.strip()
        low = name.lower()

        toks = [t for t in low.replace(",", " ").split() if len(t) > 2]

        def matches(s: str) -> bool:
            sl = s.lower()
            return sl == low or (bool(toks) and all(t in sl for t in toks))

        hits = v[v["venue"].apply(matches)]
        if len(hits):
            r = hits.iloc[0]
            method = "exact" if r["venue"] == name else "contains"
            return ResolvedVenue(raw, r["venue"], city or r["city"], method, True)

        match = difflib.get_close_matches(name, v["venue"].tolist(), n=1, cutoff=0.6)
        if match:
            r = v[v["venue"] == match[0]].iloc[0]
            return ResolvedVenue(raw, r["venue"], city or r["city"], "fuzzy", True)
        return ResolvedVenue(raw, name, city, "unresolved", False)

    def venues(self) -> list[dict]:
        return [
            {"venue": r.venue, "city": r.city, "last_match": r.last_date.date().isoformat(), "rows": int(r.rows)}
            for r in self._venues.itertuples()
        ]
