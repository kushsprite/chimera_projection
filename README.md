# Chimera — AI-Powered Cricket Fantasy Team Optimization Platform

Chimera turns raw IPL ball-by-ball data into a fantasy XI. It parses match data into
per-player fantasy scores, engineers features (form, venue, opposition, weather),
trains a LightGBM + LSTM ensemble to predict next-match fantasy points, picks an
optimal 11-player team under standard fantasy-league constraints (credits, role mix,
max-per-team) using integer linear programming, and generates a natural-language
explanation of the pick with the Gemini API.

## Project status

This section reflects what's actually been run and verified, not just written —
checked directly against each notebook's saved execution history.

**Working end-to-end, with confirmed output:**
- Match parsing (`src/cricket_parser.py` + `02_build_dataset.ipynb`) → `data/processed/player_match_stats.csv`
- Feature engineering + weather enrichment (`03_feature_engineering.ipynb`) → `data/processed/player_match_features.csv`, `models/lgbm_baseline.pkl`, `models/lgbm_weather.pkl`
- LSTM sequence model (`04_sequence_model.ipynb`) → `models/best_lstm.pt`, checkpointed automatically during training
- Team optimizer (`06_team_optimizer.ipynb`) → demonstrated successfully on two real matches, with real ILP output (11-player team, captain/vice-captain, credits used, total points)

**Not completed, or not yet verified:**

1. **Ensemble finalization (`05_ensemble.ipynb`) is unconfirmed.** The cells that align
   the LSTM and LightGBM predictions, blend them, sweep blend weights, and actually
   save `models/lgbm_final.pkl`, `models/lstm_final.pt`, and `models/ensemble_config.json`
   show no execution history in the notebook as currently saved. `06_team_optimizer.ipynb`
   does load `lgbm_final.pkl` successfully, so the file likely exists locally from an
   earlier run — but there's no reproducible record that the *current* version of this
   notebook regenerates it. **Needs a clean, top-to-bottom re-run to confirm.**
2. **LLM explainer (`07_llm_explainer.ipynb`) has never been run.** Zero of its 9 cells
   have any execution history — the Gemini integration, prompt construction, and
   retry/backoff logic are fully written but completely untested end-to-end. Needs a
   `GEMINI_API_KEY` and a first real run.
3. **No evaluation at scale.** The optimizer has been demonstrated on two individual
   sample matches, not backtested across a held-out season. There's no accuracy or
   ROI metric anywhere in the repo yet (e.g. predicted vs. actual fantasy points across
   many matches).
4. **`01_explore.ipynb` has never been run.** Low-stakes — it's a scratch notebook with
   no required downstream output — but worth knowing it's not a working example as-is.
5. **No application layer.** Everything above lives in notebooks. There's no backend
   API, frontend, or deployment configuration anywhere in the repo — no server, no UI,
   no Dockerfile, no CI/CD, no cloud config. The repo right now is 100% Python/ML
   pipeline; nothing here demonstrates Java, TypeScript, React/Angular/Node/Spring
   Boot, or a cloud deployment yet.
6. **Minor cleanup:**
   - `02_build_dataset.ipynb` (cell 11) writes to `player_match_features.csv` in
     addition to `player_match_stats.csv` — looks like a leftover copy/paste from
     notebook 3's save cell. Harmless (notebook 3 overwrites it) but worth removing.
   - Several inspection-only cells (column dumps, shape/date checks) across notebooks
     2, 3, 4, and 6 have no execution history. No functional impact, but it means the
     notebooks as committed don't reflect a single clean run top-to-bottom — worth a
     fresh run before treating any of them as a reference example.

## Pipeline

The project is a sequence of notebooks, run in order. Each stage reads the previous
stage's output from `data/` or `models/`.

| Stage | Notebook | Input | Output | Status |
|---|---|---|---|---|
| 1. Explore | `01_explore.ipynb` | one raw match JSON | sanity checks on the Cricsheet schema | ❌ never run |
| 2. Build dataset | `02_build_dataset.ipynb` | `data/raw/IPL Match Data/*.json` (via `src/cricket_parser.py`) | `data/processed/player_match_stats.csv` | ✅ confirmed |
| 3. Feature engineering | `03_feature_engineering.ipynb` | `player_match_stats.csv` + weather API | `data/processed/player_match_features.csv`, `models/lgbm_baseline.pkl`, `models/lgbm_weather.pkl` | ✅ confirmed |
| 4. Sequence model | `04_sequence_model.ipynb` | `player_match_features.csv` | `models/best_lstm.pt` | ✅ confirmed |
| 5. Ensemble | `05_ensemble.ipynb` | features + baseline models | `models/lgbm_final.pkl`, `models/lstm_final.pt`, `models/ensemble_config.json` | ⚠️ needs re-run to confirm |
| 6. Team optimizer | `06_team_optimizer.ipynb` | ensemble models + `data/processed/player_credits.csv` | optimal fantasy XI (ILP via PuLP) | ✅ confirmed on 2 sample matches |
| 7. LLM explainer | `07_llm_explainer.ipynb` | optimizer output | natural-language writeup (Gemini API) | ❌ never run |

`src/cricket_parser.py` holds the shared logic: parsing a raw Cricsheet match into
per-player "cards" and computing Dream11-style batting/bowling fantasy points. This
file is complete — no TODOs or stubs.

## Repo structure

```
chimera_projection/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/              # 01–07, run in order
├── src/
│   └── cricket_parser.py   # match parsing + fantasy point scoring
├── data/
│   ├── raw/                # untracked — raw Cricsheet match JSON
│   └── processed/          # untracked — CSVs/JSON produced by notebooks 02–03
└── models/                 # untracked — trained model artifacts (.pkl / .pt)
```

## Setup

```bash
git clone <repo-url>
cd chimera_projection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root for the LLM explainer (notebook 07):

```
GEMINI_API_KEY=your_key_here
```

## Getting the data

`data/raw/` is not tracked in git (see below), so it needs to be populated before
running the pipeline. The notebooks expect IPL match files in Cricsheet's JSON
format, one file per match, at `data/raw/IPL Match Data/*.json`:

```bash
mkdir -p data/raw
curl -o ipl.zip https://cricsheet.org/downloads/ipl_json.zip
unzip ipl.zip -d "data/raw/IPL Match Data"
```

Cricsheet's data is CC BY 4.0 — if you publish anything built from it, credit
Cricsheet.org.

Once `data/raw/` is populated, run `notebooks/02_build_dataset.ipynb` through
`07_llm_explainer.ipynb` in order to regenerate everything in `data/processed/`
and `models/`. Given the status notes above, notebook 5 in particular should be
run fresh from a clean kernel rather than assumed to still be valid.

## Why `data/` and `models/` aren't in the repo

Both are gitignored on purpose:

- **Size and file type.** The raw match archive is thousands of small JSON files,
  and the trained artifacts (`best_lstm.pt`, the LightGBM `.pkl` files) are binary
  blobs that don't diff or compress well in git — they'd bloat every clone.
- **They're regenerable.** Everything in `data/processed/` and `models/` is a
  deterministic output of `notebooks/02`–`06` given the raw data, so committing
  them is duplicating what the code already reproduces.
- **Licensing.** The raw data comes from a third party (Cricsheet, CC BY 4.0), so
  redistributing a copy of it inside a separate repo is best avoided even though
  the license permits it, if attribution isn't wired into this repo yet.

If you'd rather have the data/model files versioned instead of regenerated, the
usual options are Git LFS for the model binaries, or a tool like DVC pointing at
cloud storage for both `data/` and `models/` — happy to wire either of those in if
you want it.

## Tech stack

- **Data:** pandas, NumPy, Cricsheet ball-by-ball JSON
- **Weather enrichment:** Open-Meteo geocoding + historical weather APIs
- **Modeling:** LightGBM (XGBoost also available), PyTorch (LSTM), simple ensemble
- **Team selection:** PuLP (integer linear programming) under credit/role/team-count constraints
- **Explanation layer:** Gemini API (`google-genai`)

Currently Python-only end to end. No frontend, backend service, or cloud deployment
exists in this repo yet — see "Project status" above.
