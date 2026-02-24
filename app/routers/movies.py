"""Structured movie CRUD endpoints."""

import logging

from fastapi import APIRouter, HTTPException, Query

from app.models import MovieDetail, MovieListResponse, MovieSummary
from app.services.database import DatabaseService, MovieFilters

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/movies", tags=["movies"])

_db: DatabaseService | None = None


def init_router(db: DatabaseService) -> None:
    global _db
    _db = db


def _get_db() -> DatabaseService:
    assert _db is not None, "movies router not initialized"
    return _db


@router.get("", response_model=MovieListResponse)
def list_movies(
    title: str | None = Query(None, description="Search by title (partial match)"),
    genre: str | None = Query(None, description="Filter by genre name"),
    year: int | None = Query(None, description="Filter by exact release year"),
    director: str | None = Query(None, description="Filter by director name"),
    actor: str | None = Query(None, description="Filter by actor name"),
    sort_by: str = Query("popularity", description="Sort field: popularity, rating, year, title, revenue"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """Search and filter movies with optional parameters."""
    db = _get_db()
    filters = MovieFilters(
        title=title,
        genre=genre,
        year=year,
        director=director,
        actor=actor,
        sort_by=sort_by,
        limit=limit,
        offset=offset,
    )
    movies = db.search_movies(filters)
    return MovieListResponse(count=len(movies), movies=movies)


@router.get("/{movie_id}", response_model=MovieDetail)
def get_movie(movie_id: int):
    """Get full details for a single movie by its TMDB id."""
    db = _get_db()
    movie = db.get_movie_detail(movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
    return movie
