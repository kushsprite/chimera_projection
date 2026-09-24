"""Constants copied verbatim from the training notebooks.

Do not "fix" these. The models were trained on features computed with these
exact values, including known quirks (see notes below). Changing them here
without retraining creates training/serving skew.
"""

# From 03_feature_engineering.ipynb, cell 18.
# Quirk: RCB maps to "Bengaluru", but older rows use city "Bangalore", so RCB
# home games before the rename were is_home=0 in training. Replicated on purpose.
HOME_CITIES = {
    "Mumbai Indians": "Mumbai",
    "Chennai Super Kings": "Chennai",
    "Royal Challengers Bangalore": "Bengaluru",
    "Royal Challengers Bengaluru": "Bengaluru",
    "Kolkata Knight Riders": "Kolkata",
    "Sunrisers Hyderabad": "Hyderabad",
    "Rajasthan Royals": "Jaipur",
    "Punjab Kings": "Mohali",
    "Kings XI Punjab": "Mohali",
    "Delhi Capitals": "Delhi",
    "Delhi Daredevils": "Delhi",
    "Gujarat Titans": "Ahmedabad",
    "Lucknow Super Giants": "Lucknow",
    "Deccan Chargers": "Hyderabad",
    "Kochi Tuskers Kerala": "Kochi",
    "Pune Warriors": "Pune",
    "Rising Pune Supergiants": "Pune",
}

# The prediction models were trained on this 4-class encoding (no wicketkeeper).
MODEL_ROLE_ENCODING = {"batter": 0, "bowler": 1, "allrounder": 2, "unknown": 3}

# Wicketkeepers were a batter-like category before the WK role existed.
USER_ROLE_TO_MODEL_ROLE = {
    "batter": "batter",
    "wicketkeeper": "batter",
    "bowler": "bowler",
    "allrounder": "allrounder",
    "unknown": "unknown",
}

# Optimizer roles (5-class). WK rule from 06_team_optimizer.ipynb.
OPTIMIZER_ROLES = ("wicketkeeper", "batter", "allrounder", "bowler", "unknown")
WICKETKEEPER_MIN_CAREER_STUMPINGS = 3

# Default batting position for a player with no history, if the caller gives a role.
DEFAULT_BAT_POSITION_BY_ROLE = {
    "batter": 4,
    "wicketkeeper": 5,
    "allrounder": 6,
    "bowler": 9,
    "unknown": 0,
}

# Current franchise names and the aliases people actually type.
TEAM_ALIASES = {
    "Chennai Super Kings": ["csk", "chennai", "chennai super kings"],
    "Mumbai Indians": ["mi", "mumbai", "mumbai indians"],
    "Royal Challengers Bengaluru": [
        "rcb", "royal challengers", "royal challengers bengaluru",
        "royal challengers bangalore", "bangalore", "bengaluru",
    ],
    "Kolkata Knight Riders": ["kkr", "kolkata", "kolkata knight riders"],
    "Sunrisers Hyderabad": ["srh", "hyderabad", "sunrisers", "sunrisers hyderabad"],
    "Rajasthan Royals": ["rr", "rajasthan", "rajasthan royals"],
    "Punjab Kings": ["pbks", "kxip", "punjab", "punjab kings", "kings xi punjab"],
    "Delhi Capitals": ["dc", "delhi", "delhi capitals", "delhi daredevils"],
    "Gujarat Titans": ["gt", "gujarat", "gujarat titans"],
    "Lucknow Super Giants": ["lsg", "lucknow", "lucknow super giants"],
}

TEAM_SHORT = {
    "Chennai Super Kings": "CSK",
    "Mumbai Indians": "MI",
    "Royal Challengers Bengaluru": "RCB",
    "Kolkata Knight Riders": "KKR",
    "Sunrisers Hyderabad": "SRH",
    "Rajasthan Royals": "RR",
    "Punjab Kings": "PBKS",
    "Delhi Capitals": "DC",
    "Gujarat Titans": "GT",
    "Lucknow Super Giants": "LSG",
}

WEATHER_FEATURES = ("weather_temp", "weather_humidity", "weather_dew", "weather_windspeed", "weather_precip")
WEATHER_HOUR_WINDOW = (15, 21)  # local hours averaged in training, inclusive
