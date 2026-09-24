# Chimera: AI-Powered Cricket Fantasy Team Optimization Platform

Chimera turns raw IPL ball-by-ball data into a fantasy XI for an upcoming match. Give it
the two squads, the venue and the date; it builds each player's features from their
match history, predicts fantasy points with a LightGBM + LSTM ensemble, picks the optimal
11 under fantasy-league rules with integer programming, and explains the pick using live
pre-match news retrieved at request time (online RAG).

```
squads + venue + date
   -> live features (form, venue, opposition, batting slot, weather forecast)
   -> LightGBM + LSTM ensemble -> projected points per player
   -> PuLP optimizer -> XI with captain / vice-captain
   -> news search -> chunk -> embed -> retrieve -> LLM -> cited explanation
```

## Results

Evaluated on every 2025-26 match (148 matches, 3,542 player rows), trained on 2008-2024:

| Setting | Ensemble MAE |
|---|---|
| Naive baseline (last-5 average) | 23.74 |
| Offline, actual batting order and toss known | **20.99** |
| **Live**, batting order estimated, toss unknown | **21.78** |

The live number is the honest one for real use. The gap comes almost entirely from
batting position (the model's most important feature): before the toss it is estimated
as the mode of the player's last 5 matches, which is exactly right 55% of the time (the
best of several estimators tested). Passing the announced batting order recovers the
offline accuracy. The unknown toss costs about 0.01; predictions average both outcomes.

`scripts/validate_live_features.py` reproduces all three numbers.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                    # add GROQ_API_KEY and GEMINI_API_KEY (both free)
python scripts/smoke_test_live.py       # checks weather, news, embeddings, LLM connections
python scripts/predict_match.py data/fixtures/upcoming.example.json
```

Or open `notebooks/08_live_pipeline.ipynb`, which walks through each stage.

Requires `data/processed/` and `models/` (see "Getting the data"). No keys are strictly
required: without them, retrieval uses TF-IDF and the explanation is a plain summary.

Names can be written normally ("Virat Kohli", "Varun Chakravarthy"); they are matched to
Cricsheet's forms ("V Kohli", "CV Varun"), and uncertain matches come back as warnings.
Teams accept abbreviations (RCB, KKR, MI...), venues accept partial names ("Chinnaswamy").
Per-player overrides in a squad: `role` (useful for debutants), `batting_position` (from
the announced lineup), `credit` (the real app's price).

## How the live pipeline stays faithful to training

A model is only as good as the features it's served. Training computed features with
`groupby` + `shift(1)` + rolling windows over the whole dataset; the live builder
(`src/chimera/features.py`) computes each one directly from a player's matches before
the match date. To prove they're the same, `validate_live_features.py` rebuilds features
for every test match and compares them to the training CSV: **all features identical**,
and the live path reproduces the verified 20.99 MAE exactly. Both checks also run in the
test suite.

Training quirks are replicated deliberately, not fixed, because the models learned them:
`rolling_std_fantasy_10` uses a 5-match window, the RCB home-city mapping treats
"Bangalore" rows as away, and missing values use the exact fill constants from training.

## Online RAG

Document sources share one interface, so adding a source is one class
(`src/chimera/rag/sources.py`):

| Source | Key | What you get |
|---|---|---|
| Google News RSS | none | headlines and summaries for the match window |
| GNews API | `GNEWS_API_KEY` (optional) | real article URLs, full text fetched |
| URLs you pass | none | full article text (e.g. a Cricbuzz preview) |
| Manual text | none | anything you paste |

Documents are filtered to the week before the match and to the two teams, deduplicated,
chunked (300 words, 50 overlap), embedded with Gemini `gemini-embedding-2`, and retrieved
with several focused queries (pitch, weather/dew, team news, key players). Embeddings are
cached on disk by content hash, so the same text never costs quota twice. If Gemini is
unavailable, retrieval falls back to a local TF-IDF index automatically.

Generation uses Groq (`openai/gpt-oss-120b`) with Gemini as fallback. Because providers
retire model names often, the code discovers an available model if the configured one
disappears. The prompt includes each player's role and projected batting slot so the
model doesn't guess them, sources are numbered so claims are cited as `[n]`, and the
"projections, not guarantees" caveat is added by code rather than generated.

## Keeping data current

Live predictions use form up to the latest match in the history. After new matches:

```bash
# download the latest IPL JSONs from cricsheet.org into data/raw/IPL Match Data/
python scripts/update_history.py
```

New matches go to `data/processed/new_matches.csv`, which the live pipeline reads
alongside the training data. The training CSV and models are not modified.

## Repo structure

```
chimera_projection/
├── src/
│   ├── cricket_parser.py        # match parsing + fantasy scoring
│   └── chimera/
│       ├── config.py            # paths (resolved from repo root) and settings
│       ├── constants.py         # training constants, team aliases
│       ├── data_store.py        # history, credits, player/team/venue resolution
│       ├── features.py          # live feature builder
│       ├── weather.py           # forecast/archive/cache/climatology
│       ├── models.py            # LightGBM + LSTM ensemble
│       ├── optimizer.py         # PuLP team selection
│       ├── engine.py            # ties it all together
│       └── rag/                 # sources, chunking, embeddings, retriever, llm, explainer
├── scripts/                     # validation, smoke test, CLI, history updates
├── tests/                       # 32 tests (pytest)
├── notebooks/                   # 01-07 development pipeline, 08 live pipeline walkthrough
├── data/                        # untracked except the fixtures example
└── models/                      # untracked
```

## Tests

```bash
pytest -q
```

Covers feature fidelity against training, MAE reproduction, toss averaging, name
resolution, every optimizer rule, and the RAG components (against responses shaped like
the real services). Tests never call paid or rate-limited APIs.

## Getting the data

The notebooks expect Cricsheet IPL JSON files at `data/raw/IPL Match Data/*.json`:

```bash
mkdir -p data/raw
curl -o ipl.zip https://cricsheet.org/downloads/ipl_json.zip
unzip ipl.zip -d "data/raw/IPL Match Data"
```

Then run notebooks 02-06 to produce `data/processed/` and `models/`. Cricsheet's data is
CC BY 4.0; credit Cricsheet.org if you publish anything built from it.

`data/` and `models/` are gitignored: the raw archive is thousands of files, model
binaries don't diff well, and both are regenerable from the notebooks.

## Known limitations

- **Breakout performances are unpredictable by construction.** Every feature is
  historical. In the 2025 opener Krunal Pandya took 3/29 and was Player of the Match;
  the model ranked him last of 24.
- **Batting order before the toss is a guess.** See Results; pass announced positions
  when you have them.
- **Weather features slightly hurt offline MAE** (21.21 vs 20.94 for LightGBM). City-level
  averages over 15:00-21:00 are probably too coarse to capture dew. Kept because live
  conditions matter to users, but the trade-off is measured.
- **Google News RSS gives headlines, not full articles**, unless the optional
  `googlenewsdecoder` package is installed. GNews or pasted URLs give full text.
- **The LLM can still introduce outside knowledge.** Grounding and prompt rules reduce it;
  they don't eliminate it.
- **Credits are synthetic**, derived from career stats and form, not a real app's prices.

## Tech stack

Python, pandas, NumPy, LightGBM, PyTorch, PuLP, Gemini API (embeddings), Groq API
(generation), Open-Meteo (weather), Google News RSS / GNews (news).

Not yet built: backend API, frontend, Docker, cloud deployment.
