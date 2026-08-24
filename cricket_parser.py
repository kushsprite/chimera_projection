import json


TEAM_NAME_MAP = {
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Deccan Chargers": "Deccan Chargers",  # no change, just documenting
}

def normalize_team_name(name: str)-> str:
    return TEAM_NAME_MAP.get(name,name)



with open("data/raw/IPL Match Data/335982.json") as f:
    match = json.load(f)

def compute_batting_points(runs: int, balls: int, fours: int, sixes: int) -> int:
    current_points = 4 + runs * 1 + fours * 1 + sixes * 2
    if runs >= 100:
        current_points += 30
    if runs >= 50 and runs < 100:
        current_points += 20
    if balls >= 10:
        strike_rate = (runs / balls) * 100
        if strike_rate > 170:
            current_points += 6
        elif strike_rate >= 150:
            current_points += 4
        elif strike_rate >= 130:
            current_points += 2
        elif strike_rate < 60:
            current_points -= 6
        elif strike_rate <= 70:
            current_points -= 4
        elif strike_rate <= 80:
            current_points -= 2
    return current_points

def compute_bowling_points(wickets: int, runs_conceded: int, balls_bowled: int, maidens: int) -> int:
    points = 0
    points += wickets * 12
    points += maidens * 8
    if wickets >= 5:
        points += 25
    elif wickets >= 4:
        points += 12

    if balls_bowled >= 6:
        economy = (runs_conceded / balls_bowled) * 6
        if economy < 5:
            points += 6
        elif economy <= 6:
            points += 4
        elif economy <= 7:
            points += 2
        elif economy >= 12:
            points -= 12
        elif economy >= 9:
            points -= 6

    return points



def parse_match(match, match_id = "Unknown"):
    player_cards = []
    wicket_cards = []
    scorecards = []
    info = match["info"]
    innings = match["innings"]

    venue = info.get("venue", "Unknown")
    city = info.get("city", "Unknown")
    date = info.get("dates", "Unknown")[0]
    season = info.get("season", "Unknown")
    toss_winner = normalize_team_name(info.get("toss", {}).get("winner", "Unknown"))
    toss_decision = info.get("toss", {}).get("decision", "Unknown")
    all_teams = [normalize_team_name(t) for t in info["players"].keys()]

    # ---- MATCH LEVEL STATS (computed once) ----
    batting_position_inn1 = {}
    batting_position_inn2 = {}
    innings1_total = 0
    innings2_total = 0
    innings1_wickets = 0
    innings2_wickets = 0

    for i, innings_data in enumerate(innings, start=1):
        positions = batting_position_inn1 if i == 1 else batting_position_inn2
        for over in innings_data["overs"]:
            for delivery in over["deliveries"]:
                batter = delivery["batter"]
                if batter not in positions:
                    positions[batter] = len(positions) + 1
                if i == 1:
                    innings1_total += delivery["runs"]["total"]
                else:
                    innings2_total += delivery["runs"]["total"]
                for wicket in delivery.get("wickets", []):
                    if i == 1:
                        innings1_wickets += 1
                    else:
                        innings2_wickets += 1

    # ---- PLAYER LEVEL STATS ----
    for team_name, players in info["players"].items():
        team_name = normalize_team_name(team_name)
        for player in players:
            runs_scored = 0
            balls_faced = 0
            sixes = 0
            fours = 0
            wickets = 0
            runs_conceded = 0
            balls_bowled = 0
            maidens = 0

            for innings_data in innings:
                for over in innings_data["overs"]:
                    over_balls = 0
                    over_runs = 0
                    for delivery in over["deliveries"]:
                        if delivery["batter"] == player:
                            is_wide = delivery.get("extras", {}).get("wides", 0) > 0
                            runs_scored += delivery["runs"]["batter"]
                            if not is_wide:
                                balls_faced += 1
                            if delivery["runs"]["batter"] == 6:
                                sixes += 1
                            if delivery["runs"]["batter"] == 4:
                                fours += 1
                        if delivery["bowler"] == player:
                            is_wide = delivery.get("extras", {}).get("wides", 0) > 0
                            is_noball = delivery.get("extras", {}).get("noballs", 0) > 0
                            is_legal = not is_wide and not is_noball
                            byes = delivery.get("extras", {}).get("byes", 0)
                            legbyes = delivery.get("extras", {}).get("legbyes", 0)
                            over_runs += delivery["runs"]["total"] - byes - legbyes
                            if is_legal:
                                balls_bowled += 1
                            runs_conceded += delivery["runs"]["total"] - byes - legbyes
                            if is_legal:
                                over_balls += 1
                            for wicket in delivery.get("wickets", []):
                                if wicket["kind"] != "run out":
                                    wickets += 1
                    if over_runs == 0 and over_balls == 6:
                        maidens += 1

            # determine innings and team total
            if player in batting_position_inn1:
                player_innings = 1
                team_total = innings1_total
            elif player in batting_position_inn2:
                player_innings = 2
                team_total = innings2_total
            else:
                player_innings = 0
                team_total = innings1_total if team_name == all_teams[0] else innings2_total

            # wickets fallen against the bowler's opponent
            total_wickets = innings2_wickets if team_name == all_teams[0] else innings1_wickets

            bat_position = batting_position_inn1.get(player, batting_position_inn2.get(player, 0))
            opposition = all_teams[1] if team_name == all_teams[0] else all_teams[0]

            scorecards.append({
                "player": player, "team": team_name, "opposition": opposition,
                "venue": venue, "city": city, "date": date, "season": season,
                "toss_winner": toss_winner, "toss_decision": toss_decision,
                "runs": runs_scored, "balls": balls_faced, "fours": fours, "sixes": sixes,
                "strike_rate": round((runs_scored / balls_faced) * 100, 2) if balls_faced > 0 else 0,
                "batting_position": bat_position, "player_innings": player_innings,
                "team_total": team_total,
                "fantasy_points": compute_batting_points(runs_scored, balls_faced, fours, sixes),
            })

            wicket_cards.append({
                "player": player, "team": team_name, "wicket": wickets,
                "runs_given": runs_conceded, "balls_delivered": balls_bowled, "maiden": maidens,
                "Bowling_economy": round((runs_conceded / balls_bowled) * 6, 2) if balls_bowled > 0 else 0,
                "total_wickets": total_wickets, "player_innings": player_innings,
                "fantasy_points": compute_bowling_points(wickets, runs_conceded, balls_bowled, maidens),
            })

    for scorecard, wicket_card in zip(scorecards, wicket_cards):
        player_cards.append({
            "match_id": "match_id",
            "player": scorecard["player"], "team": scorecard["team"],
            "opposition": scorecard["opposition"], "venue": scorecard["venue"],
            "city": scorecard["city"], "date": scorecard["date"], "season": scorecard["season"],
            "toss_winner": scorecard["toss_winner"], "toss_decision": scorecard["toss_decision"],
            "runs": scorecard["runs"], "balls_faced": scorecard["balls"],
            "fours": scorecard["fours"], "sixes": scorecard["sixes"],
            "strike_rate": scorecard["strike_rate"], "batting_position": scorecard["batting_position"],
            "wickets": wicket_card["wicket"], "runs_conceded": wicket_card["runs_given"],
            "balls_bowled": wicket_card["balls_delivered"], "maidens": wicket_card["maiden"],
            "economy": wicket_card["Bowling_economy"], "total_wickets": scorecard.get("total_wickets", wicket_card["total_wickets"]),
            "player_innings": scorecard["player_innings"], "team_total": scorecard["team_total"],
            "total_fantasy_points": scorecard["fantasy_points"] + wicket_card["fantasy_points"],
        })

    return player_cards





player_cards = parse_match(match)
print(json.dumps(player_cards, indent=2))
    




        










    




