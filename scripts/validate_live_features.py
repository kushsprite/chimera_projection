"""Prove the live pipeline matches training, and measure realistic live accuracy.

For every 2025-26 test match, rebuild each player's features using only data
from before that match, then:
  1. check every model feature equals the training CSV value
  2. MAE with actual batting positions and toss  (should equal the verified 20.99)
  3. MAE as used live: estimated batting positions, toss unknown

Run after changing anything in src/chimera/features.py or data_store.py.
"""
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from chimera.engine import ChimeraEngine
from chimera.features import MatchSpec, PlayerSpec


def main():
    eng = ChimeraEngine()
    eng.weather.coords = lambda c: None  # historical dates use the exact training lookup; stay offline
    cfg = eng.predictor.config
    feats = sorted(set(cfg["features_lgbm"]) | set(cfg["context_features"]))
    csv = eng.store._train_full.copy()
    csv["date"] = pd.to_datetime(csv["date"])
    test = csv[csv["date"].dt.year >= 2025]

    def spec(g, toss):
        r0 = g.iloc[0]
        teams = g["team"].unique().tolist()
        return MatchSpec(teams[0], teams[-1], r0["venue"], r0["date"].date(),
                         [PlayerSpec(p, t) for p, t in zip(g["player"], g["team"])], city=r0["city"],
                         season=str(r0["season"]), toss_winner=r0["toss_winner"] if toss else None)

    t0 = time.time()
    worst = dict.fromkeys(feats, 0.0)
    res = {"actual": ([], []), "live": ([], [])}
    bat_exact = 0
    for _, g in test.groupby("match_id"):
        b = eng.builder.build(spec(g, True), use_actual=dict(zip(g["player"], g["batting_position"])))
        live = b.features.set_index("player").loc[g["player"]]
        for f in feats:
            worst[f] = max(worst[f], float(np.max(np.abs(live[f].astype(float).values - g[f].astype(float).values))))
        out = eng.predictor.predict(b.features, b.sequences).set_index("player")
        res["actual"][0].extend(g["total_fantasy_points"])
        res["actual"][1].extend(out.loc[g["player"], "ensemble_pred"])

        b2 = eng.builder.build(spec(g, False))
        out2 = eng.predictor.predict(b2.features, b2.sequences).set_index("player")
        res["live"][0].extend(g["total_fantasy_points"])
        res["live"][1].extend(out2.loc[g["player"], "ensemble_pred"])
        bat_exact += int((out2.loc[g["player"], "batting_position"].values == g["batting_position"].values).sum())

    mae = {k: float(np.mean(np.abs(np.array(p) - np.array(y)))) for k, (y, p) in res.items()}
    drift = {f: v for f, v in worst.items() if v > 1e-9}
    print(f"{test['match_id'].nunique()} matches, {len(test)} player rows, {time.time() - t0:.0f}s\n")
    print("Feature fidelity:", "PASS, all features identical to training" if not drift else f"FAIL {drift}")
    print(f"MAE with actual batting order + toss: {mae['actual']:.2f}  (verified offline: {cfg.get('verified_test_mae')})")
    print(f"MAE live (estimated batting order, toss unknown): {mae['live']:.2f}")
    print(f"Batting position estimate exactly right: {bat_exact / len(test):.1%}")
    naive = np.mean(np.abs(test["rolling_avg_fantasy_5"] - test["total_fantasy_points"]))
    print(f"Naive baseline (last-5 average): {naive:.2f}")
    if drift or abs(mae["actual"] - cfg.get("verified_test_mae", mae["actual"])) > 0.01:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
