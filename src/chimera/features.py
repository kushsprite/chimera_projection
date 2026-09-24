"""Live feature construction for a match that has not been played yet.

Training built features with groupby + shift(1) + rolling/expanding over the
full dataset, so each row only ever saw matches strictly before it. For a live
match we do the equivalent directly: take each player's matches before the
match date and aggregate them the same way.

Every function below names the notebook cell it replicates. The validation
script (scripts/validate_live_features.py) recomputes features for real
2025-26 matches this way and checks them against the training CSV.

Known training quirk replicated on purpose:
    rolling_std_fantasy_10 is the std over the last 5 matches, not 10
    (03_feature_engineering cell 11 used rolling(5) for it).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from .constants import DEFAULT_BAT_POSITION_BY_ROLE, HOME_CITIES
from .data_store import DataStore


@dataclass
class PlayerSpec:
    input_name: str
    team: str
    role: Optional[str] = None               # user override: batter/bowler/allrounder/wicketkeeper
    batting_position: Optional[int] = None   # user override, e.g. from the announced lineup
    credit: Optional[float] = None           # user override, e.g. the real app's credit


@dataclass
class MatchSpec:
    team1: str
    team2: str
    venue: str
    match_date: date
    players: list[PlayerSpec]
    city: Optional[str] = None
    season: Optional[str] = None
    toss_winner: Optional[str] = None
    toss_decision: Optional[str] = None


@dataclass
class FeatureBuild:
    features: pd.DataFrame                   # one row per player, all model inputs + metadata
    sequences: np.ndarray                    # (n_players, seq_len, n_sequence_features) for the LSTM
    match: dict                              # resolved match info
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------- helpers

def _mean(x) -> float:
    return float(np.mean(x)) if len(x) else np.nan


def _std(x) -> float:
    # pandas default ddof=1, NaN with fewer than 2 values
    return float(np.std(x, ddof=1)) if len(x) >= 2 else np.nan


def _nz(v: float) -> float:
    return 0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def estimate_batting_position(prior: pd.DataFrame, window: int = 5) -> Optional[int]:
    """Mode of the last `window` batting positions; ties go to the most recent.

    0 means the player did not bat, and stays 0 if that is their usual outcome,
    because training used 0 the same way.
    """
    if prior.empty:
        return None
    recent = prior["batting_position"].astype(int).tolist()[-window:]
    counts = Counter(recent)
    top = max(counts.values())
    tied = {pos for pos, c in counts.items() if c == top}
    for pos in reversed(recent):
        if pos in tied:
            return int(pos)
    return int(recent[-1])


def player_form_features(prior: pd.DataFrame, venue: str, opposition: str, season: str) -> dict:
    """History-derived features for one player. `prior` = their matches before this one."""
    fp = prior["total_fantasy_points"].to_numpy(dtype=float)
    runs = prior["runs"].to_numpy(dtype=float)
    wkts = prior["wickets"].to_numpy(dtype=float)
    bat_c = prior["batting_contribution"].to_numpy(dtype=float)
    bowl_c = prior["bowling_contribution"].to_numpy(dtype=float)

    f = {}
    # cell 9 / 11: rolling form, min_periods=1, NaN -> 0 (cell 54)
    f["rolling_avg_fantasy_3"] = _nz(_mean(fp[-3:]))
    f["rolling_avg_fantasy_5"] = _nz(_mean(fp[-5:]))
    f["rolling_avg_fantasy_10"] = _nz(_mean(fp[-10:]))
    f["rolling_std_fantasy_5"] = _nz(_std(fp[-5:]))
    f["rolling_std_fantasy_10"] = _nz(_std(fp[-5:]))   # training quirk: window of 5
    f["rolling_avg_runs_5"] = _nz(_mean(runs[-5:]))
    f["rolling_avg_wickets_5"] = _nz(_mean(wkts[-5:]))
    f["matches_played"] = float(len(prior))

    # cell 6: career totals before this match
    f["Total_career_runs"] = float(runs.sum())
    f["Total_career_wickets"] = float(wkts.sum())

    # cell 12: same-season form, grouped by the season label
    s = prior.loc[prior["season"] == season, "total_fantasy_points"].to_numpy(dtype=float)
    f["expanding_season_fantasy_avg"] = _nz(_mean(s))
    f["expanding_season_fantasy_std"] = _nz(_std(s))

    # cell 15: contribution ratios
    f["rolling_batting_contribution_5"] = _nz(_mean(bat_c[-5:]))
    f["rolling_bowling_contribution_5"] = _nz(_mean(bowl_c[-5:]))

    # cell 21 / 33 / 54: venue and opposition history, first-appearance flag before filling
    v = prior.loc[prior["venue"] == venue, "total_fantasy_points"].to_numpy(dtype=float)
    o = prior.loc[prior["opposition"] == opposition, "total_fantasy_points"].to_numpy(dtype=float)
    v_avg, o_avg = _mean(v), _mean(o)
    f["venue_first_appearance"] = int(np.isnan(v_avg))
    f["opposition_first_appearance"] = int(np.isnan(o_avg))
    f["venue_avg_fantasy"] = _nz(v_avg)
    f["venue_std_fantasy"] = _nz(_std(v))
    f["opposition_avg_fantasy"] = _nz(o_avg)
    f["opposition_std_fantasy"] = _nz(_std(o))
    return f


def player_sequence(prior: pd.DataFrame, sequence_features: list[str], seq_len: int) -> np.ndarray:
    """Last `seq_len` matches, oldest first, left-padded with zeros (05_ensemble)."""
    data = prior[sequence_features].to_numpy(dtype=float)[-seq_len:]
    if len(data) < seq_len:
        pad = np.zeros((seq_len - len(data), len(sequence_features)))
        data = np.vstack([pad, data]) if len(data) else pad
    return data


# ------------------------------------------------------------------- builder

class FeatureBuilder:
    def __init__(self, store: DataStore, weather, sequence_features: list[str], seq_len: int):
        self.store = store
        self.weather = weather
        self.sequence_features = sequence_features
        self.seq_len = seq_len

    def build(
        self,
        match: MatchSpec,
        weather_override: Optional[dict] = None,
        use_actual: Optional[dict] = None,
        today: Optional[date] = None,
    ) -> FeatureBuild:
        """Build model-ready features for every player in `match`.

        weather_override: skip the weather service and use these 5 values.
        use_actual: validation only, {player_name: batting_position} to use
            real batting positions instead of estimates.
        """
        store = self.store
        warnings: list[str] = []
        when = pd.Timestamp(match.match_date)

        team1, ok1 = store.canonical_team(match.team1)
        team2, ok2 = store.canonical_team(match.team2)
        for raw, ok in ((match.team1, ok1), (match.team2, ok2)):
            if not ok:
                warnings.append(f"Unrecognised team '{raw}', used as given. Opposition history will be empty.")

        venue = store.resolve_venue(match.venue, match.city)
        if not venue.known:
            warnings.append(f"Unknown venue '{match.venue}'. Venue history features fall back to defaults.")
        city = venue.city or match.city
        season = match.season or str(match.match_date.year)

        toss_winner = None
        if match.toss_winner:
            toss_winner, _ = store.canonical_team(match.toss_winner)

        if when > store.last_history_date + pd.Timedelta(days=45):
            warnings.append(
                f"History ends {store.last_history_date.date()}. Recent form may be stale; "
                "run scripts/update_history.py with new Cricsheet files."
            )

        if weather_override is not None:
            wx, wx_source = dict(weather_override), "override"
        else:
            wx, wx_source = self.weather.get(city, match.match_date, today=today)

        venue_avgs = store.venue_innings_averages(venue.venue, when)
        venue_feats = {k: (store.venue_fill[k] if np.isnan(v) else float(v)) for k, v in venue_avgs.items()}

        rows, seqs, seen = [], [], set()
        for spec in match.players:
            team, _ = store.canonical_team(spec.team)
            opposition = team2 if team == team1 else team1
            res = store.resolve_player(spec.input_name, team)
            name = res.name or spec.input_name.strip()
            if name in seen:
                warnings.append(f"Duplicate player '{name}' ignored.")
                continue
            seen.add(name)

            if res.name is None:
                warnings.append(f"'{spec.input_name}' not found in history; treated as a debutant.")
            elif res.method == "fuzzy" or res.confidence < 0.9:
                warnings.append(f"'{spec.input_name}' matched to '{res.name}' ({res.method}); check this is right.")

            prior = store.player_history(res.name, when) if res.name else store.history.iloc[0:0]

            f = player_form_features(prior, venue.venue, opposition, season)
            f.update(venue_feats)
            f.update(wx)

            # cell 19: home flag, exact training rule
            f["is_home"] = int(HOME_CITIES.get(team) == city)
            # cell 55: toss, NaN when unknown (the predictor averages both outcomes)
            f["won_toss"] = np.nan if toss_winner is None else int(team == toss_winner)

            opt_role, opt_role_src = store.optimizer_role(res.name, spec.role)
            role_enc, role_src = store.model_role_encoded(res.name, spec.role)
            f["role_encoded"] = role_enc

            if use_actual is not None and name in use_actual:
                bat_pos, bat_src = int(use_actual[name]), "actual"
            elif spec.batting_position is not None:
                bat_pos, bat_src = int(spec.batting_position), "user"
            else:
                est = estimate_batting_position(prior)
                if est is None:
                    bat_pos, bat_src = DEFAULT_BAT_POSITION_BY_ROLE.get(opt_role, 0), "role_default"
                else:
                    bat_pos, bat_src = est, "last_5_mode"
            f["batting_position"] = bat_pos

            credit, credit_src = store.credit(res.name, opt_role, spec.credit)

            f.update({
                "player": name,
                "input_name": spec.input_name,
                "resolution": res.method,
                "team": team,
                "opposition": opposition,
                "has_history": bool(len(prior)),
                "role": opt_role,
                "role_source": opt_role_src,
                "model_role_source": role_src,
                "batting_position_source": bat_src,
                "credit_value": credit,
                "credit_source": credit_src,
                "last_match": prior["date"].max().date().isoformat() if len(prior) else None,
            })
            rows.append(f)
            seqs.append(player_sequence(prior, self.sequence_features, self.seq_len))

        features = pd.DataFrame(rows)
        sequences = np.stack(seqs) if seqs else np.zeros((0, self.seq_len, len(self.sequence_features)))

        match_info = {
            "team1": team1, "team2": team2, "venue": venue.venue, "venue_input": match.venue,
            "venue_resolution": venue.method, "city": city, "date": match.match_date.isoformat(),
            "season": season, "toss_winner": toss_winner, "toss_decision": match.toss_decision,
            "weather": {k: round(v, 2) for k, v in wx.items()}, "weather_source": wx_source,
            "venue_history": {k: round(v, 1) for k, v in venue_feats.items()},
            "history_through": store.last_history_date.date().isoformat(),
        }
        for team_name in (team1, team2):
            n = int((features["team"] == team_name).sum()) if len(features) else 0
            if n < 11:
                warnings.append(f"{team_name} has only {n} players in the squad.")
        return FeatureBuild(features, sequences, match_info, warnings)
