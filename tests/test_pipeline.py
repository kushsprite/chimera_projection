"""Live pipeline fidelity, optimizer rules, and entity resolution."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from conftest import needs_data
from chimera.features import MatchSpec, PlayerSpec, estimate_batting_position
from chimera.optimizer import OptimizationError, TeamConstraints, select_team


def _spec_from_rows(g, toss=True):
    r0 = g.iloc[0]
    teams = g["team"].unique().tolist()
    return MatchSpec(teams[0], teams[-1], r0["venue"], pd.Timestamp(r0["date"]).date(),
                     [PlayerSpec(p, t) for p, t in zip(g["player"], g["team"])], city=r0["city"],
                     season=str(r0["season"]), toss_winner=r0["toss_winner"] if toss else None)


def _test_rows(engine):
    t = engine.store._train_full.copy()
    t["date"] = pd.to_datetime(t["date"])
    return t[t["date"].dt.year >= 2025]


@needs_data
def test_live_features_identical_to_training(engine):
    test = _test_rows(engine)
    cfg = engine.predictor.config
    feats = sorted(set(cfg["features_lgbm"]) | set(cfg["context_features"]))
    for mid in test["match_id"].unique()[:25]:
        g = test[test["match_id"] == mid]
        b = engine.builder.build(_spec_from_rows(g), use_actual=dict(zip(g["player"], g["batting_position"])))
        live = b.features.set_index("player").loc[g["player"]]
        np.testing.assert_allclose(live[feats].astype(float).values, g[feats].astype(float).values,
                                   atol=1e-9, err_msg=f"feature drift in match {mid}")


@needs_data
def test_live_path_reproduces_verified_mae(engine):
    test = _test_rows(engine)
    y, p = [], []
    for mid, g in test.groupby("match_id"):
        b = engine.builder.build(_spec_from_rows(g), use_actual=dict(zip(g["player"], g["batting_position"])))
        out = engine.predictor.predict(b.features, b.sequences).set_index("player")
        y += g["total_fantasy_points"].tolist()
        p += out.loc[g["player"], "ensemble_pred"].tolist()
    mae = float(np.mean(np.abs(np.array(p) - np.array(y))))
    assert abs(mae - 20.99) < 0.01, mae


@needs_data
def test_unknown_toss_averages_both_outcomes(engine):
    g = _test_rows(engine)
    g = g[g["match_id"] == g["match_id"].iloc[0]]
    b = engine.builder.build(_spec_from_rows(g, toss=False))
    unk = engine.predictor.predict(b.features, b.sequences)
    f1, f0 = b.features.copy(), b.features.copy()
    f1["won_toss"], f0["won_toss"] = 1, 0
    p1 = engine.predictor.predict(f1, b.sequences)["ensemble_pred"].values
    p0 = engine.predictor.predict(f0, b.sequences)["ensemble_pred"].values
    np.testing.assert_allclose(unk["ensemble_pred"].values, (p1 + p0) / 2, rtol=1e-6)


def test_batting_position_estimate_prefers_recent_on_tie():
    df = pd.DataFrame({"batting_position": [3, 3, 5, 5, 1, 7]})
    assert estimate_batting_position(df) == 5          # last 5 = 3,5,5,1,7 -> mode 5
    tie = pd.DataFrame({"batting_position": [4, 4, 6, 6]})
    assert estimate_batting_position(tie) == 6          # tie broken by most recent
    assert estimate_batting_position(pd.DataFrame({"batting_position": []})) is None


@needs_data
@pytest.mark.parametrize("raw,team,expected", [
    ("Virat Kohli", "RCB", "V Kohli"), ("Krunal Pandya", "RCB", "KH Pandya"),
    ("Hardik Pandya", "MI", "HH Pandya"), ("Quinton de Kock", "KKR", "Q de Kock"),
    ("Varun Chakravarthy", "KKR", "CV Varun"), ("Sai Sudharsan", "GT", "B Sai Sudharsan"),
    ("Rohit Sharma", "MI", "RG Sharma"), ("abhishek sharma", "SRH", "Abhishek Sharma"),
])
def test_player_resolution(engine, raw, team, expected):
    t, _ = engine.store.canonical_team(team)
    assert engine.store.resolve_player(raw, t).name == expected


@needs_data
def test_unknown_player_is_a_debutant(engine):
    spec = MatchSpec("RCB", "KKR", "Chinnaswamy", date(2026, 4, 10),
                     [PlayerSpec("Totally New Guy", "RCB", role="bowler"), PlayerSpec("Sunil Narine", "KKR")])
    b = engine.builder.build(spec)
    row = b.features.set_index("player").loc["Totally New Guy"]
    assert not row["has_history"] and row["matches_played"] == 0 and row["role"] == "bowler"
    assert row["batting_position"] == 9 and row["venue_first_appearance"] == 1
    assert np.allclose(b.sequences[0], 0)


@needs_data
def test_venue_resolves_to_current_name(engine):
    assert engine.store.resolve_venue("Eden Gardens").venue == "Eden Gardens, Kolkata"


def _pool(n_per_team=11):
    rng = np.random.default_rng(0)
    roles = ["wicketkeeper", "batter", "batter", "batter", "allrounder", "allrounder",
             "bowler", "bowler", "bowler", "bowler", "batter"]
    rows = []
    for t in ("A", "B"):
        for i in range(n_per_team):
            rows.append({"player": f"{t}{i}", "team": t, "role": roles[i % len(roles)],
                         "credit_value": float(rng.uniform(7.5, 10)), "ensemble_pred": float(rng.uniform(5, 60))})
    return pd.DataFrame(rows)


def test_optimizer_respects_every_rule():
    team, _ = select_team(_pool())
    c = TeamConstraints()
    assert len(team) == 11 and team["credit_value"].sum() <= c.budget + 1e-9
    counts = team["role"].value_counts()
    assert 1 <= counts.get("wicketkeeper", 0) <= 4 and 3 <= counts.get("batter", 0) <= 6
    assert 1 <= counts.get("allrounder", 0) <= 4 and 3 <= counts.get("bowler", 0) <= 6
    assert team["team"].value_counts().max() <= 7
    assert team["captain"].sum() == 1 and team["vice_captain"].sum() == 1
    assert team.loc[team["captain"], "ensemble_pred"].iloc[0] == team["ensemble_pred"].max()


def test_optimizer_include_exclude():
    pool = _pool()
    worst = pool.sort_values("ensemble_pred").iloc[0]["player"]
    best = pool.sort_values("ensemble_pred").iloc[-1]["player"]
    team, _ = select_team(pool, TeamConstraints(include=[worst], exclude=[best]))
    assert worst in set(team["player"]) and best not in set(team["player"])


def test_optimizer_relaxes_missing_keeper_and_rejects_impossible():
    pool = _pool()
    pool.loc[pool["role"] == "wicketkeeper", "role"] = "batter"
    team, warnings = select_team(pool)
    assert len(team) == 11 and any("wicketkeeper" in w for w in warnings)
    with pytest.raises(OptimizationError):
        select_team(pool, TeamConstraints(budget=50))
    with pytest.raises(OptimizationError):
        select_team(pool.head(8))
