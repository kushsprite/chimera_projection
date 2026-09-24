"""Pick and explain a fantasy XI from the command line.

    python scripts/predict_match.py data/fixtures/upcoming.example.json
    python scripts/predict_match.py my_fixtures.json --match 2 --no-explain
    python scripts/predict_match.py my_fixtures.json --offline      # skip live news

Fixture format (see data/fixtures/upcoming.example.json):
    team1, team2, venue, date (YYYY-MM-DD), team1_squad, team2_squad,
    optional: city, toss_winner, constraints {...}, rag {urls: [...], manual_texts: [...]}
Squad entries are names, or objects {"name", "role", "batting_position", "credit"}.
"""
import argparse
import json
import textwrap
from datetime import date

import _bootstrap  # noqa: F401

from chimera.engine import ChimeraEngine
from chimera.features import MatchSpec, PlayerSpec
from chimera.optimizer import TeamConstraints
from chimera.rag import RagOptions


def _player(entry, team) -> PlayerSpec:
    if isinstance(entry, str):
        return PlayerSpec(entry, team)
    return PlayerSpec(entry["name"], team, entry.get("role"), entry.get("batting_position"), entry.get("credit"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture_file")
    ap.add_argument("--match", type=int, default=0, help="index into the file's matches list")
    ap.add_argument("--no-explain", action="store_true")
    ap.add_argument("--offline", action="store_true", help="skip live news fetching")
    args = ap.parse_args()

    data = json.load(open(args.fixture_file))
    matches = data.get("matches", data if isinstance(data, list) else [data])
    m = matches[args.match]

    spec = MatchSpec(
        team1=m["team1"], team2=m["team2"], venue=m["venue"], match_date=date.fromisoformat(m["date"]),
        players=[_player(p, m["team1"]) for p in m["team1_squad"]] + [_player(p, m["team2"]) for p in m["team2_squad"]],
        city=m.get("city"), season=m.get("season"), toss_winner=m.get("toss_winner"),
        toss_decision=m.get("toss_decision"),
    )
    constraints = TeamConstraints(**m.get("constraints", {}))
    rag = RagOptions(**m.get("rag", {}))
    if args.offline:
        rag.use_google_news = rag.use_gnews = False

    eng = ChimeraEngine()
    pred, sel, expl = eng.generate_team(spec, constraints, rag, explain=not args.no_explain)
    info = pred.match
    print(f"\n{info['team1']} vs {info['team2']}  |  {info['venue']}  |  {info['date']}")
    print(f"weather ({info['weather_source']}): {info['weather']}")
    for w in pred.warnings + sel.warnings:
        print(f"  ! {w}")
    print(f"\nXI: {sel.credits_used:.1f}/100 credits, {sel.projected_points:.1f} projected points\n")
    for r in sel.team.itertuples():
        tag = "(C) " if r.captain else "(VC)" if r.vice_captain else "    "
        print(f"  {tag} {r.player:<20} {r.team:<28} {r.role:<12} {r.credit_value:>4.1f}  {r.ensemble_pred:>5.1f}")
    if expl:
        print(f"\nExplanation ({expl.provider}{', ' + expl.model if expl.model else ''}; "
              f"{expl.documents_found} docs, retrieval: {expl.retrieval_backend}):\n")
        print(textwrap.fill(expl.text, 100, replace_whitespace=False))
        for c in expl.citations:
            print(f"  [{c['id']}] {c['title']} - {c['publisher'] or c['source']} {c['url'] or ''}")
        for n in expl.source_notes:
            print(f"  - {n}")
        for w in expl.warnings:
            print(f"  ! {w}")


if __name__ == "__main__":
    main()
