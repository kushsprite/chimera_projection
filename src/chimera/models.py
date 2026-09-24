"""Load the trained LightGBM + LSTM and produce ensemble predictions.

Import order matters on macOS: lightgbm before torch (plus the OpenMP flag
set in chimera/__init__.py), otherwise the process can crash on load.
"""
from __future__ import annotations

import json
import logging
import pickle

import lightgbm  # noqa: F401  (must be imported before torch, see module docstring)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import Settings

log = logging.getLogger(__name__)


class CricketLSTM(nn.Module):
    """Architecture must match 04_sequence_model.ipynb exactly to load the weights."""

    def __init__(self, seq_features: int, context_features: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=seq_features, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size + context_features, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, seq, context):
        lstm_out, _ = self.lstm(seq)
        last_output = lstm_out[:, -1, :]
        combined = torch.cat([last_output, context], dim=1)
        x = self.dropout(self.relu(self.fc1(combined)))
        return self.fc2(x).squeeze(1)


def _pick_device(pref: str) -> torch.device:
    pref = (pref or "cpu").lower()
    if pref == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if pref == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS requested but unavailable, using CPU")
        return torch.device("cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable, using CPU")
        return torch.device("cpu")
    return torch.device(pref)


class EnsemblePredictor:
    def __init__(self, settings: Settings):
        p = settings.paths
        with open(p.ensemble_config) as f:
            cfg = json.load(f)
        self.config = cfg
        self.sequence_features: list[str] = cfg["sequence_features"]
        self.context_features: list[str] = cfg["context_features"]
        self.seq_len: int = int(cfg["seq_len"])
        self.w_lgbm = float(cfg.get("lgbm_weight", 0.5))
        self.w_lstm = float(cfg.get("lstm_weight", 0.5))

        with open(p.lgbm_model, "rb") as f:
            self.lgbm = pickle.load(f)
        # The model remembers its own training columns. Trust that over any list.
        self.lgbm_features: list[str] = list(self.lgbm.feature_name_)

        self.device = _pick_device(settings.device)
        self.lstm = CricketLSTM(len(self.sequence_features), len(self.context_features))
        state = torch.load(p.lstm_model, map_location="cpu", weights_only=True)
        self.lstm.load_state_dict(state)
        self.lstm.to(self.device).eval()
        log.info("Models loaded (LightGBM %d features, LSTM on %s)", len(self.lgbm_features), self.device)

    def _predict_once(self, features: pd.DataFrame, sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lgbm_pred = self.lgbm.predict(features[self.lgbm_features].astype(float))
        ctx = torch.tensor(features[self.context_features].astype(float).to_numpy(), dtype=torch.float32)
        seq = torch.tensor(sequences, dtype=torch.float32)
        with torch.no_grad():
            lstm_pred = self.lstm(seq.to(self.device), ctx.to(self.device)).cpu().numpy()
        return np.asarray(lgbm_pred, dtype=float), np.asarray(lstm_pred, dtype=float)

    def predict(self, features: pd.DataFrame, sequences: np.ndarray) -> pd.DataFrame:
        """Return features with lgbm_pred, lstm_pred, ensemble_pred columns added.

        If won_toss is unknown (NaN) for a row, predict with won_toss=1 and =0
        and average, rather than guessing a toss result.
        """
        if features.empty:
            out = features.copy()
            for c in ("lgbm_pred", "lstm_pred", "ensemble_pred"):
                out[c] = []
            return out

        unknown = features["won_toss"].isna()
        if unknown.any():
            f1 = features.copy(); f1.loc[unknown, "won_toss"] = 1
            f0 = features.copy(); f0.loc[unknown, "won_toss"] = 0
            lg1, ls1 = self._predict_once(f1, sequences)
            lg0, ls0 = self._predict_once(f0, sequences)
            lg, ls = (lg1 + lg0) / 2, (ls1 + ls0) / 2
        else:
            lg, ls = self._predict_once(features, sequences)

        out = features.copy()
        out["lgbm_pred"] = lg
        out["lstm_pred"] = ls
        out["ensemble_pred"] = self.w_lgbm * lg + self.w_lstm * ls
        out["toss_known"] = ~unknown
        return out
