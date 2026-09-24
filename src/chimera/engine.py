"""One object that owns every loaded resource and runs the full pipeline.

Load it once (it reads the history CSV and both models) and reuse it:

    engine = ChimeraEngine()
    pred   = engine.predict(match_spec)
    team   = engine.optimize(pred.players)
    expl   = engine.explain(pred.match, team.team)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from .config import Settings, get_settings
from .data_store import DataStore
from .features import FeatureBuilder, MatchSpec
from .models import EnsemblePredictor
from .optimizer import TeamConstraints, select_team
from .rag.explainer import Explainer, Explanation, RagOptions
from .weather import WeatherService

log = logging.getLogger(__name__)

PLAYER_OUTPUT_COLUMNS = [
    "player", "input_name", "resolution", "team", "opposition", "role", "credit_value",
    "batting_position", "batting_position_source", "has_history", "matches_played", "last_match",
    "rolling_avg_fantasy_5", "rolling_avg_fantasy_10", "venue_avg_fantasy", "venue_first_appearance",
    "opposition_avg_fantasy", "is_home", "won_toss", "toss_known",
    "lgbm_pred", "lstm_pred", "ensemble_pred", "role_source", "credit_source",
]


@dataclass
class Prediction:
    match: dict
    players: pd.DataFrame           # full feature rows + predictions
    warnings: list[str] = field(default_factory=list)

    def public_players(self) -> pd.DataFrame:
        cols = [c for c in PLAYER_OUTPUT_COLUMNS if c in self.players.columns]
        return self.players[cols].sort_values("ensemble_pred", ascending=False).reset_index(drop=True)


@dataclass
class TeamSelection:
    team: pd.DataFrame
    credits_used: float
    projected_points: float
    warnings: list[str] = field(default_factory=list)


class ChimeraEngine:
    def __init__(self, settings: Optional[Settings] = None, explainer: Optional[Explainer] = None):
        self.settings = settings or get_settings()
        self.store = DataStore(self.settings)
        self.weather = WeatherService(self.settings, self.store.weather_fill)
        self.predictor = EnsemblePredictor(self.settings)
        self.builder = FeatureBuilder(self.store, self.weather,
                                      self.predictor.sequence_features, self.predictor.seq_len)
        self.explainer = explainer or Explainer(self.settings)

    def predict(self, match: MatchSpec, today: Optional[date] = None) -> Prediction:
        built = self.builder.build(match, today=today)
        players = self.predictor.predict(built.features, built.sequences)
        warnings = list(built.warnings)
        if players["won_toss"].isna().any():
            warnings.append("Toss not provided: predictions average both toss outcomes.")
        est = (players["batting_position_source"] != "user").sum()
        if est:
            warnings.append(
                f"Batting positions estimated for {est} players from their last 5 matches. "
                "Pass announced positions for better accuracy."
            )
        return Prediction(built.match, players, warnings)

    def optimize(self, players: pd.DataFrame, constraints: Optional[TeamConstraints] = None,
                 points_col: str = "ensemble_pred") -> TeamSelection:
        team, warnings = select_team(players, constraints, points_col)
        return TeamSelection(team, float(team["credit_value"].sum()), float(team["final_points"].sum()), warnings)

    def explain(self, match: dict, team: pd.DataFrame, options: Optional[RagOptions] = None) -> Explanation:
        return self.explainer.explain(match, team, options)

    def generate_team(self, match: MatchSpec, constraints: Optional[TeamConstraints] = None,
                      options: Optional[RagOptions] = None, explain: bool = True):
        pred = self.predict(match)
        sel = self.optimize(pred.players, constraints)
        expl = self.explain(pred.match, sel.team, options) if explain else None
        return pred, sel, expl
