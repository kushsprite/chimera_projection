"""Fantasy XI selection as an integer program (same model as 06_team_optimizer).

maximise   sum(points_i * x_i)
subject to sum(x_i) = 11
           sum(credit_i * x_i) <= budget
           role minimums / maximums
           at most `max_per_team` from either real team
           x_i = 1 for forced includes, 0 for excludes
"""
from __future__ import annotations

import re
import warnings as _warnings
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import pulp


class OptimizationError(ValueError):
    pass


@dataclass
class TeamConstraints:
    team_size: int = 11
    budget: float = 100.0
    min_wk: int = 1
    max_wk: int = 4
    min_batters: int = 3
    max_batters: int = 6
    min_allrounders: int = 1
    max_allrounders: int = 4
    min_bowlers: int = 3
    max_bowlers: int = 6
    max_per_team: int = 7
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


ROLE_LIMITS = {
    "wicketkeeper": ("min_wk", "max_wk"),
    "batter": ("min_batters", "max_batters"),
    "allrounder": ("min_allrounders", "max_allrounders"),
    "bowler": ("min_bowlers", "max_bowlers"),
}


def _var_name(i: int, player: str) -> str:
    return f"x{i}_" + re.sub(r"[^A-Za-z0-9]", "_", player)[:40]


def select_team(pool: pd.DataFrame, c: Optional[TeamConstraints] = None,
                points_col: str = "ensemble_pred") -> tuple[pd.DataFrame, list[str]]:
    """Pick the optimal XI from `pool`.

    pool needs columns: player, team, role, credit_value, and `points_col`.
    Returns (selected players with captain/vice_captain/final_points, warnings).
    """
    c = c or TeamConstraints()
    warnings: list[str] = []
    required = {"player", "team", "role", "credit_value", points_col}
    missing = required - set(pool.columns)
    if missing:
        raise OptimizationError(f"Pool is missing columns: {sorted(missing)}")

    pool = pool.drop_duplicates(subset="player").reset_index(drop=True)
    if len(pool) < c.team_size:
        raise OptimizationError(f"Need at least {c.team_size} players, got {len(pool)}.")

    unknown_names = set(c.include + c.exclude) - set(pool["player"])
    if unknown_names:
        warnings.append(f"Ignored include/exclude names not in the pool: {sorted(unknown_names)}")
    clash = set(c.include) & set(c.exclude)
    if clash:
        raise OptimizationError(f"Players both included and excluded: {sorted(clash)}")

    prob = pulp.LpProblem("fantasy_xi", pulp.LpMaximize)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", DeprecationWarning)
        x = {i: pulp.LpVariable(_var_name(i, p), cat="Binary") for i, p in enumerate(pool["player"])}
    pts = pool[points_col].astype(float).tolist()
    cred = pool["credit_value"].astype(float).tolist()

    prob += pulp.lpSum(pts[i] * x[i] for i in x)
    prob += pulp.lpSum(x.values()) == c.team_size
    prob += pulp.lpSum(cred[i] * x[i] for i in x) <= c.budget

    available_after_exclude = pool[~pool["player"].isin(c.exclude)]
    for role, (lo_attr, hi_attr) in ROLE_LIMITS.items():
        idx = [i for i, r in enumerate(pool["role"]) if r == role]
        lo, hi = getattr(c, lo_attr), getattr(c, hi_attr)
        n_avail = int((available_after_exclude["role"] == role).sum())
        if n_avail < lo:
            warnings.append(f"Only {n_avail} {role}(s) available, relaxed minimum from {lo} to {n_avail}.")
            lo = n_avail
        prob += pulp.lpSum(x[i] for i in idx) >= lo
        prob += pulp.lpSum(x[i] for i in idx) <= hi

    for team in pool["team"].unique():
        idx = [i for i, t in enumerate(pool["team"]) if t == team]
        prob += pulp.lpSum(x[i] for i in idx) <= c.max_per_team

    for i, p in enumerate(pool["player"]):
        if p in c.include:
            prob += x[i] == 1
        elif p in c.exclude:
            prob += x[i] == 0

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", DeprecationWarning)  # PuLP 3.x announces 4.0 API changes
        solver = pulp.PULP_CBC_CMD(msg=0) if hasattr(pulp, "PULP_CBC_CMD") else pulp.COIN_CMD(msg=0)
        prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise OptimizationError(
            f"No valid team exists under these constraints (solver status: {status}). "
            "Try raising the budget, loosening role limits, or removing forced includes."
        )

    chosen = [i for i in x if x[i].value() is not None and x[i].value() > 0.5]
    team = (pool.iloc[chosen]
            .sort_values(points_col, ascending=False, kind="mergesort")
            .reset_index(drop=True))
    team["captain"] = False
    team["vice_captain"] = False
    team.loc[0, "captain"] = True
    team.loc[1, "vice_captain"] = True
    team["multiplier"] = 1.0
    team.loc[0, "multiplier"] = 2.0
    team.loc[1, "multiplier"] = 1.5
    team["final_points"] = team[points_col] * team["multiplier"]
    return team, warnings
