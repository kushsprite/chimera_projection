"""Match-day weather features.

Training averaged hourly Open-Meteo values over 15:00-21:00 local time
(precipitation summed). This module reproduces that exact aggregation for any
date, choosing the right source:

    1. training lookup   exact values the model saw, for historical matches
    2. live cache        anything fetched before
    3. forecast API      today and up to ~16 days ahead (also recent past)
    4. archive API       older past dates
    5. climatology       this city's historical average (same month if available)
    6. training mean     the value training used to fill missing weather
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import date
from typing import Optional

import requests

from .config import Settings
from .constants import WEATHER_HOUR_WINDOW

log = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
HOURLY = "temperature_2m,relative_humidity_2m,dew_point_2m,wind_speed_10m,precipitation"

FORECAST_MAX_DAYS_AHEAD = 15
FORECAST_MAX_DAYS_BACK = 80  # forecast API serves ~3 months of past data
CITY_NAME_OVERRIDES = {"Bangalore": "Bengaluru", "New Chandigarh": "Mullanpur"}
_KEY_MAP = {"temp": "weather_temp", "humidity": "weather_humidity", "dew": "weather_dew",
            "windspeed": "weather_windspeed", "precip": "weather_precip"}


def extract_match_window(weather_json: dict, match_date: str) -> Optional[dict]:
    """Same aggregation as extract_match_day_weather in 03_feature_engineering."""
    hourly = weather_json.get("hourly") or {}
    times = hourly.get("time") or []
    lo, hi = WEATHER_HOUR_WINDOW
    buckets = {k: [] for k in ("temp", "humidity", "dew", "windspeed", "precip")}
    src = {"temp": "temperature_2m", "humidity": "relative_humidity_2m", "dew": "dew_point_2m",
           "windspeed": "wind_speed_10m", "precip": "precipitation"}
    for i, t in enumerate(times):
        if not t.startswith(match_date):
            continue
        hour = int(t.split("T")[1].split(":")[0])
        if lo <= hour <= hi:
            vals = {k: hourly[src[k]][i] for k in buckets}
            if any(v is None for v in vals.values()):
                continue
            for k, v in vals.items():
                buckets[k].append(v)
    if not buckets["temp"]:
        return None
    out = {k: sum(v) / len(v) for k, v in buckets.items() if k != "precip"}
    out["precip"] = sum(buckets["precip"])
    return out


class WeatherService:
    def __init__(self, settings: Settings, training_fill: dict):
        self.settings = settings
        self.training_fill = training_fill
        p = settings.paths
        self._lock = threading.Lock()

        self._coords = self._read_json(p.city_coords_json) or {}
        self._training_lookup = self._read_json(p.weather_lookup_json) or {}
        self._live_cache = self._read_json(p.weather_live_cache_json) or {}

    @staticmethod
    def _read_json(path) -> Optional[dict]:
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _persist(self) -> None:
        p = self.settings.paths
        with self._lock:
            p.weather_live_cache_json.parent.mkdir(parents=True, exist_ok=True)
            with open(p.weather_live_cache_json, "w") as f:
                json.dump(self._live_cache, f)
            with open(p.city_coords_json, "w") as f:
                json.dump(self._coords, f)

    # ------------------------------------------------------------------ coords

    def coords(self, city: str) -> Optional[tuple[float, float]]:
        c = self._coords.get(city)
        if c and c.get("lat") is not None:
            return c["lat"], c["lon"]
        latlon = self._geocode(city)
        if latlon:
            self._coords[city] = {"lat": latlon[0], "lon": latlon[1]}
            self._persist()
        return latlon

    def _geocode(self, city: str) -> Optional[tuple[float, float]]:
        name = CITY_NAME_OVERRIDES.get(city, city)
        try:
            r = requests.get(GEOCODE_URL, params={"name": name, "count": 10}, timeout=10)
            r.raise_for_status()
            results = r.json().get("results") or []
        except Exception as e:  # network or parse failure
            log.warning("Geocoding failed for %s: %s", city, e)
            return None
        for res in results:  # prefer India, IPL's home
            if res.get("country_code") == "IN":
                return res["latitude"], res["longitude"]
        return (results[0]["latitude"], results[0]["longitude"]) if results else None

    # --------------------------------------------------------------- fetching

    def _fetch(self, url: str, lat: float, lon: float, day: str) -> Optional[dict]:
        params = {"latitude": lat, "longitude": lon, "start_date": day, "end_date": day,
                  "hourly": HOURLY, "timezone": "auto"}
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code != 200:
                log.warning("Weather %s returned %s: %s", url, r.status_code, r.text[:160])
                return None
            return extract_match_window(r.json(), day)
        except Exception as e:
            log.warning("Weather request failed (%s): %s", url, e)
            return None

    def _climatology(self, city: str, day: date) -> tuple[Optional[dict], str]:
        """Average of this city's historical match-day weather: same month if we
        have it (IPL is played Mar-May), otherwise all months for the city."""
        city_rows = {k: v for k, v in self._training_lookup.items() if v and k.split(",")[0] == city}
        month = f"-{day.month:02d}-"
        rows = [v for k, v in city_rows.items() if month in k]
        label = "climatology (same city, same month)"
        if not rows:
            rows, label = list(city_rows.values()), "climatology (same city, all months)"
        if not rows:
            return None, ""
        return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}, label

    # ----------------------------------------------------------------- public

    def get(self, city: Optional[str], match_day: date, today: Optional[date] = None) -> tuple[dict, str]:
        """Return (weather feature dict, source description)."""
        if not city or city == "Unknown":
            return dict(self.training_fill), "training_mean (no city)"
        today = today or date.today()
        day = match_day.isoformat()
        key = f"{city},{day}"

        hit = self._training_lookup.get(key)
        if hit:
            return self._to_features(hit), "training_lookup"
        hit = self._live_cache.get(key)
        if hit:
            return self._to_features(hit["values"]), f"cache ({hit['source']})"

        latlon = self.coords(city)
        values, source = None, None
        if latlon:
            delta = (match_day - today).days
            if -FORECAST_MAX_DAYS_BACK <= delta <= FORECAST_MAX_DAYS_AHEAD:
                values, source = self._fetch(FORECAST_URL, *latlon, day), "open-meteo forecast"
            if values is None and delta < -5:
                values, source = self._fetch(ARCHIVE_URL, *latlon, day), "open-meteo archive"

        if values is not None:
            with self._lock:
                self._live_cache[key] = {"values": values, "source": source}
            self._persist()
            return self._to_features(values), source

        clim, label = self._climatology(city, match_day)
        if clim:
            return self._to_features(clim), label
        return dict(self.training_fill), "training_mean"

    @staticmethod
    def _to_features(values: dict) -> dict:
        return {_KEY_MAP[k]: float(v) for k, v in values.items() if k in _KEY_MAP}
