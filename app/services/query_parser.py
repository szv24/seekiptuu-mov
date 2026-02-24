"""
Intelligent query parser: classifies user intent and extracts structured
parameters from natural-language movie questions.

Two-stage approach:
  1. Fast path: regex/keyword matching handles most of the queries instantly.
  2. LLM fallback: for ambiguous queries, ask the model to classify.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

KNOWN_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "Foreign",
    "History", "Horror", "Music", "Mystery", "Romance",
    "Science Fiction", "TV Movie", "Thriller", "War", "Western",
]

_GENRE_PATTERN = "|".join(re.escape(g) for g in sorted(KNOWN_GENRES, key=len, reverse=True))


class Intent(str, Enum):
    LOOKUP = "lookup"
    RECOMMEND = "recommend"
    COMPARE = "compare"
    TOP_RATED = "top_rated"
    CAST_CREW = "cast_crew"
    GENERAL = "general"


@dataclass
class ParsedQuery:
    intent: Intent = Intent.GENERAL
    titles: list[str] = field(default_factory=list)
    genre: str | None = None
    year: int | None = None
    year_from: int | None = None
    year_to: int | None = None
    person: str | None = None
    sort_by: str | None = None
    limit: int = 10
    raw_message: str = ""

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "titles": self.titles,
            "genre": self.genre,
            "year": self.year,
            "year_from": self.year_from,
            "year_to": self.year_to,
            "person": self.person,
            "sort_by": self.sort_by,
            "limit": self.limit,
        }


# ── Regex helpers ──────────────────────────────────────────────────────

_QUOTED_TITLE = re.compile(r'["\u201c\u201d]([^"\u201c\u201d]+)["\u201c\u201d]')
_YEAR_EXACT = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
_YEAR_RANGE = re.compile(
    r"\b(?:from|between)\s+(19[5-9]\d|20[0-3]\d)\s*(?:to|and|-)\s*(19[5-9]\d|20[0-3]\d)\b", re.I
)
_TOP_N = re.compile(r"\btop\s+(\d{1,3})\b", re.I)

_LOOKUP_PATTERNS = [
    re.compile(r"\b(?:tell\s+me\s+about|what\s+is|info(?:rmation)?\s+(?:about|on)|plot\s+of|overview\s+of|describe)\b", re.I),
    re.compile(r"\bwhat(?:'s|\s+is)\s+the\s+(?:movie|film|plot|story)\b", re.I),
]

_RECOMMEND_PATTERNS = [
    re.compile(r"\b(?:recommend|suggest|give\s+me|show\s+me|find\s+me|any\s+good|looking\s+for)\b", re.I),
    re.compile(r"\bmovies?\s+like\b", re.I),
    re.compile(r"\bsimilar\s+to\b", re.I),
]

_COMPARE_PATTERNS = [
    re.compile(r"\b(?:compare|versus|vs\.?|difference\s+between|better)\b", re.I),
]

_TOP_RATED_PATTERNS = [
    re.compile(r"\b(?:best|top|highest[\s-]rated|most\s+popular|greatest|all[\s-]time)\b", re.I),
]

_CAST_CREW_PATTERNS = [
    re.compile(r"\bwho\s+(?:directed|starred|acted|is\s+the\s+director|is\s+in|are\s+the\s+(?:actors?|cast))\b", re.I),
    re.compile(r"\b(?:directed\s+by|starring|movies?\s+(?:with|by|starring|featuring))\b", re.I),
    re.compile(r"\b(?:cast|director|actors?|actress(?:es)?)\s+(?:of|in|for)\b", re.I),
]

_TITLE_ABOUT_PATTERN = re.compile(
    r"(?:(?:tell\s+me\s+)?about|what(?:'s|\s+is)\s+(?:the\s+)?(?:movie\s+)?|"
    r"info\s+(?:on|about)|plot\s+of|overview\s+of|describe)\s+(.+?)(?:\?|$)",
    re.I,
)

_DIRECTED_PERSON = re.compile(
    r"(?:directed|starring|(?:movies?\s+)?(?:with|by|featuring))\s+(.+?)(?:\?|$)", re.I
)

_WHO_DIRECTED = re.compile(r"who\s+directed\s+(.+?)(?:\?|$)", re.I)


def _extract_genre(text: str) -> str | None:
    text_lower = text.lower()
    if "sci-fi" in text_lower or "scifi" in text_lower:
        return "Science Fiction"
    m = re.search(_GENRE_PATTERN, text, re.I)
    if m:
        matched = m.group(0)
        for g in KNOWN_GENRES:
            if g.lower() == matched.lower():
                return g
    return None


def _extract_titles(text: str) -> list[str]:
    """Extract movie titles from quoted strings."""
    return _QUOTED_TITLE.findall(text)


def _extract_person(text: str) -> str | None:
    for pat in (_DIRECTED_PERSON, _WHO_DIRECTED):
        m = pat.search(text)
        if m:
            name = m.group(1).strip().strip("?.!")
            name = re.sub(r"(?i)\b(?:the\s+)?(?:movie|film)\b", "", name).strip()
            if len(name) > 2:
                return name

    return None


def _extract_title_from_about(text: str) -> str | None:
    m = _TITLE_ABOUT_PATTERN.search(text)
    if m:
        title = m.group(1).strip().strip("?.!")
        title = re.sub(r"(?i)\b(?:the\s+)?(?:movie|film)\s*", "", title).strip()
        if len(title) > 1:
            return title
    return None


def _match_any(patterns: list[re.Pattern], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def parse_query(message: str) -> ParsedQuery:
    """
    Parse a user message into structured intent + parameters.
    Pure function: no LLM call, no I/O.
    """
    result = ParsedQuery(raw_message=message)
    text = message.strip()

    result.titles = _extract_titles(text)
    result.genre = _extract_genre(text)

    year_range = _YEAR_RANGE.search(text)
    if year_range:
        result.year_from = int(year_range.group(1))
        result.year_to = int(year_range.group(2))
    else:
        year_match = _YEAR_EXACT.search(text)
        if year_match:
            result.year = int(year_match.group(1))

    top_n = _TOP_N.search(text)
    if top_n:
        result.limit = min(int(top_n.group(1)), 50)

    # Intent classification (order matters: most specific first)

    if _match_any(_COMPARE_PATTERNS, text) and len(result.titles) >= 2:
        result.intent = Intent.COMPARE
        return result

    if _match_any(_COMPARE_PATTERNS, text) and " and " in text.lower():
        parts = re.split(r"\band\b", text, flags=re.I)
        if len(parts) >= 2:
            result.intent = Intent.COMPARE
            if not result.titles:
                result.titles = [p.strip().strip("?.!") for p in parts[-2:]]
            return result

    if _match_any(_CAST_CREW_PATTERNS, text):
        result.intent = Intent.CAST_CREW
        result.person = _extract_person(text)
        if not result.person and result.titles:
            pass  # "who directed Inception" -> title already captured
        if not result.person:
            title_from_about = _extract_title_from_about(text)
            if title_from_about and not result.titles:
                result.titles = [title_from_about]
        return result

    if _match_any(_LOOKUP_PATTERNS, text):
        result.intent = Intent.LOOKUP
        if not result.titles:
            title_from_about = _extract_title_from_about(text)
            if title_from_about:
                result.titles = [title_from_about]
        return result

    if _match_any(_TOP_RATED_PATTERNS, text):
        result.intent = Intent.TOP_RATED
        result.sort_by = "rating"
        return result

    if _match_any(_RECOMMEND_PATTERNS, text):
        result.intent = Intent.RECOMMEND
        return result

    # Fallback: if there's a quoted title, treat it as a lookup
    if result.titles:
        result.intent = Intent.LOOKUP
        return result

    result.intent = Intent.GENERAL
    return result


LLM_CLASSIFY_PROMPT = """\
You are a query classifier for a movie database assistant.
Classify the user message into ONE intent and extract parameters.

Intents: lookup, recommend, compare, top_rated, cast_crew, general

Return ONLY valid JSON (no markdown):
{{"intent": "...", "titles": [...], "genre": "...", "year": ..., "person": "..."}}

Omit fields that are null.

User message: {message}"""
