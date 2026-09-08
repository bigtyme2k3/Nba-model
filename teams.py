"""
teams.py
--------
Single source of truth for NBA team names, conferences, and timezones.

Every other script imports from here instead of keeping its own copy of
the team list. Team display names match ESPN's `displayName` field so
scraped data joins cleanly across sources.
"""

from __future__ import annotations

# team display name -> ESPN numeric team id (site.api.espn.com .../teams)
TEAM_ESPN_ID: dict[str, int] = {
    "Atlanta Hawks": 1,
    "Boston Celtics": 2,
    "Brooklyn Nets": 17,
    "Charlotte Hornets": 30,
    "Chicago Bulls": 4,
    "Cleveland Cavaliers": 5,
    "Dallas Mavericks": 6,
    "Denver Nuggets": 7,
    "Detroit Pistons": 8,
    "Golden State Warriors": 9,
    "Houston Rockets": 10,
    "Indiana Pacers": 11,
    "LA Clippers": 12,
    "Los Angeles Lakers": 13,
    "Memphis Grizzlies": 29,
    "Miami Heat": 14,
    "Milwaukee Bucks": 15,
    "Minnesota Timberwolves": 16,
    "New Orleans Pelicans": 3,
    "New York Knicks": 18,
    "Oklahoma City Thunder": 25,
    "Orlando Magic": 19,
    "Philadelphia 76ers": 20,
    "Phoenix Suns": 21,
    "Portland Trail Blazers": 22,
    "Sacramento Kings": 23,
    "San Antonio Spurs": 24,
    "Toronto Raptors": 28,
    "Utah Jazz": 26,
    "Washington Wizards": 27,
}

TEAM_ABBR: dict[str, str] = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

EASTERN_CONFERENCE = {
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Detroit Pistons", "Indiana Pacers",
    "Miami Heat", "Milwaukee Bucks", "New York Knicks", "Orlando Magic",
    "Philadelphia 76ers", "Toronto Raptors", "Washington Wizards",
}

WESTERN_CONFERENCE = {
    "Dallas Mavericks", "Denver Nuggets", "Golden State Warriors", "Houston Rockets",
    "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Minnesota Timberwolves",
    "New Orleans Pelicans", "Oklahoma City Thunder", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Utah Jazz",
}

# Used for the "long travel" / east-west feature in the models.
TEAM_TIMEZONE: dict[str, str] = {
    "Atlanta Hawks": "US/Eastern", "Boston Celtics": "US/Eastern",
    "Brooklyn Nets": "US/Eastern", "Charlotte Hornets": "US/Eastern",
    "Chicago Bulls": "US/Central", "Cleveland Cavaliers": "US/Eastern",
    "Dallas Mavericks": "US/Central", "Denver Nuggets": "US/Mountain",
    "Detroit Pistons": "US/Eastern", "Golden State Warriors": "US/Pacific",
    "Houston Rockets": "US/Central", "Indiana Pacers": "US/Eastern",
    "LA Clippers": "US/Pacific", "Los Angeles Lakers": "US/Pacific",
    "Memphis Grizzlies": "US/Central", "Miami Heat": "US/Eastern",
    "Milwaukee Bucks": "US/Central", "Minnesota Timberwolves": "US/Central",
    "New Orleans Pelicans": "US/Central", "New York Knicks": "US/Eastern",
    "Oklahoma City Thunder": "US/Central", "Orlando Magic": "US/Eastern",
    "Philadelphia 76ers": "US/Eastern", "Phoenix Suns": "US/Mountain",
    "Portland Trail Blazers": "US/Pacific", "Sacramento Kings": "US/Pacific",
    "San Antonio Spurs": "US/Central", "Toronto Raptors": "US/Eastern",
    "Utah Jazz": "US/Mountain", "Washington Wizards": "US/Eastern",
}

TEAM_NAMES = sorted(TEAM_ESPN_ID)

# Alias map for normalizing scraped/CSV team fields (abbr or short name -> full name).
TEAM_ALIASES: dict[str, str] = {abbr: full for full, abbr in TEAM_ABBR.items()}
TEAM_ALIASES.update({full: full for full in TEAM_NAMES})


def normalize_team(value: str) -> str:
    """Best-effort mapping of any team string we might scrape to the canonical full name."""
    v = str(value or "").strip()
    if v in TEAM_ALIASES:
        return TEAM_ALIASES[v]
    for full in TEAM_NAMES:
        if v and (v == full or v in full or full.endswith(v)):
            return full
    return v
