"""POST /chat — the core AI endpoint that combines query parsing, DB retrieval, and LLM response."""

import logging

from fastapi import APIRouter

from app.models import ChatRequest, ChatResponse
from app.services.database import DatabaseService, MovieFilters
from app.services.llm import OllamaService
from app.services.query_parser import Intent, ParsedQuery, parse_query

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

_db: DatabaseService | None = None
_llm: OllamaService | None = None


def init_router(db: DatabaseService, llm: OllamaService) -> None:
    global _db, _llm
    _db = db
    _llm = llm


def _get_services() -> tuple[DatabaseService, OllamaService]:
    assert _db is not None and _llm is not None, "chat router not initialized"
    return _db, _llm


def _retrieve_movies(parsed: ParsedQuery, db: DatabaseService) -> list[dict]:
    """Execute the right DB query based on parsed intent."""

    if parsed.intent == Intent.LOOKUP:
        results = []
        for title in parsed.titles:
            found = db.search_movies(MovieFilters(title=title, limit=3))
            results.extend(found)
        if not results and parsed.titles:
            found = db.search_movies(MovieFilters(title=parsed.titles[0], limit=5))
            results.extend(found)
        if results:
            detailed = [db.get_movie_detail(m["id"]) for m in results[:5]]
            return [d for d in detailed if d is not None]
        return results

    if parsed.intent == Intent.RECOMMEND:
        return db.search_movies(MovieFilters(
            genre=parsed.genre,
            year=parsed.year,
            year_from=parsed.year_from,
            year_to=parsed.year_to,
            sort_by="rating",
            limit=parsed.limit,
        ))

    if parsed.intent == Intent.COMPARE:
        results = []
        for title in parsed.titles[:5]:
            found = db.search_movies(MovieFilters(title=title, limit=1))
            if found:
                detail = db.get_movie_detail(found[0]["id"])
                if detail:
                    results.append(detail)
        return results

    if parsed.intent == Intent.TOP_RATED:
        return db.get_top_rated(
            genre=parsed.genre,
            year=parsed.year,
            limit=parsed.limit,
        )

    if parsed.intent == Intent.CAST_CREW:
        if parsed.person:
            return db.get_movies_by_person(parsed.person, limit=parsed.limit)
        if parsed.titles:
            results = []
            for title in parsed.titles:
                found = db.search_movies(MovieFilters(title=title, limit=1))
                if found:
                    detail = db.get_movie_detail(found[0]["id"])
                    if detail:
                        results.append(detail)
            return results
        return []

    # GENERAL — broad search with whatever params we extracted
    return db.search_movies(MovieFilters(
        title=parsed.titles[0] if parsed.titles else None,
        genre=parsed.genre,
        year=parsed.year,
        sort_by="popularity",
        limit=parsed.limit,
    ))


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a natural-language movie question.

    The system parses the intent, queries the database, and uses an LLM
    to generate a conversational response grounded in the retrieved data.
    """
    db, llm = _get_services()
    parsed = parse_query(request.message)

    logger.info("Parsed intent=%s params=%s", parsed.intent.value, parsed.to_dict())

    movies = _retrieve_movies(parsed, db)
    llm_response = await llm.generate(request.message, movies)

    return ChatResponse(
        message=llm_response,
        intent=parsed.intent.value,
        params=parsed.to_dict(),
        movies=movies,
    )
