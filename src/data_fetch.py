"""
Data fetching module for sports APIs.
Handles fixture data and odds retrieval with caching, retries, and error handling.
"""

import time
from typing import Dict, List, Optional, Any
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import settings
from src.logger import setup_logger

logger = setup_logger(__name__)

API_SPORTS_BASE_URL = "https://v3.football.api-sports.io"

LEAGUE_MAPPING = {
    "premier_league": 39,
    "la_liga": 140,
    "serie_a": 135,
    "bundesliga": 78,
    "ligue_1": 61,
}


class APIClient:
    """Robust API client with retry logic and caching."""

    def __init__(self, timeout: int = None, max_retries: int = None):
        self.timeout = timeout or settings.REQUEST_TIMEOUT
        self.max_retries = max_retries or settings.MAX_RETRIES
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=settings.RETRY_BACKOFF,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def get(self, url: str, headers: Dict[str, str] = None, **kwargs) -> Optional[Dict]:
        try:
            logger.debug(f"Fetching: {url}")
            response = self.session.get(
                url,
                headers=headers,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None


class SportsDataFetcher:
    """Fetch sports data from APIs with caching."""

    def __init__(self):
        self.api_client = APIClient()
        self._cache = {}

    @staticmethod
    def _get_cache_key(func_name: str, *args, **kwargs) -> str:
        return f"{func_name}_{args}_{sorted(kwargs.items())}"

    def _get_cached(self, key: str, ttl: int = None) -> Optional[Any]:
        ttl = ttl or settings.CACHE_TTL
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < ttl:
                return value
            else:
                del self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        if settings.CACHE_ENABLED:
            self._cache[key] = (value, time.time())

    def fetch_fixtures(
        self,
        sport: str = "soccer",
        league: str = "premier_league",
        season: int = 2026
    ) -> pd.DataFrame:
        """Fetch upcoming fixtures from API."""
        league_id = LEAGUE_MAPPING.get(league, league)
        cache_key = self._get_cache_key("fixtures", sport, league_id, season)
        
        if settings.CACHE_ENABLED:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        url = f"{API_SPORTS_BASE_URL}/fixtures"
        params = {
            "league": league_id,
            "next": 10
        }
        headers = {"x-apisports-key": settings.API_SPORTS_KEY}

        data = self.api_client.get(url, headers=headers, params=params)
        
        if not data:
            logger.warning(f"No data returned for league {league}")
            return pd.DataFrame()

        if data.get("errors"):
            logger.error(f"API Sports Error: {data.get('errors')}")

        try:
            fixtures = []
            for item in data.get("response", []):
                fixture_info = item.get("fixture", {})
                teams_info = item.get("teams", {})
                fixtures.append({
                    "fixture_id": fixture_info.get("id"),
                    "date": fixture_info.get("date"),
                    "status": fixture_info.get("status", {}).get("short"),
                    "home_team": teams_info.get("home", {}).get("name"),
                    "away_team": teams_info.get("away", {}).get("name"),
                    "home_team_id": teams_info.get("home", {}).get("id"),
                    "away_team_id": teams_info.get("away", {}).get("id"),
                })
            
            df = pd.DataFrame(fixtures)
            logger.info(f"Fetched {len(df)} fixtures for league ID {league_id}")
            self._set_cache(cache_key, df)
            return df
            
        except Exception as e:
            logger.error(f"Error parsing fixtures: {e}")
            return pd.DataFrame()

    def fetch_odds(self, event_id: str, region: str = "us") -> Dict[str, float]:
        cache_key = self._get_cache_key("odds", event_id, region)
        if settings.CACHE_ENABLED:
            cached = self._get_cached(cache_key, ttl=1800)
            if cached is not None:
                return cached

        url = f"https://api.the-odds-api.com/v4/sports/soccer/events/{event_id}/odds"
        params = {
            "apiKey": settings.ODDS_API_KEY,
            "regions": region,
            "markets": "h2h"
        }

        data = self.api_client.get(url, params=params)
        if not data or "bookmakers" not in data:
            return {}

        try:
            odds_dict = {}
            for bookmaker in data.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        odds_dict[outcome.get("name", "unknown")] = outcome.get("price", 0.0)
            self._set_cache(cache_key, odds_dict)
            return odds_dict
        except Exception as e:
            logger.error(f"Error parsing odds: {e}")
            return {}

    def fetch_team_stats(self, team_id: int, season: int = 2026) -> Dict[str, Any]:
        cache_key = self._get_cache_key("team_stats", team_id, season)
        if settings.CACHE_ENABLED:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        url = f"{API_SPORTS_BASE_URL}/teams/statistics"
        params = {"team": team_id, "season": season}
        headers = {"x-apisports-key": settings.API_SPORTS_KEY}

        data = self.api_client.get(url, headers=headers, params=params)
        if not data or "response" not in data:
            return {}

        try:
            stats = data.get("response", {})
            self._set_cache(cache_key, stats)
            return stats
        except Exception as e:
            logger.error(f"Error parsing team stats: {e}")
            return {}

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("Cache cleared")


_fetcher = None


def get_fetcher() -> SportsDataFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = SportsDataFetcher()
    return _fetcher
