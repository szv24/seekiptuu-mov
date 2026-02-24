"""SQLite database service which manages connection management and query helpers."""

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MovieFilters:
    title: str | None = None
    genre: str | None = None
    year: int | None = None
    year_from: int | None = None
    year_to: int | None = None
    director: str | None = None
    actor: str | None = None
    sort_by: str = "popularity"
    limit: int = 20
    offset: int = 0


class DatabaseService:
    VALID_SORT_COLUMNS = {
        "popularity": "m.popularity DESC",
        "rating": "m.vote_average DESC",
        "year": "m.year DESC",
        "title": "m.title ASC",
        "revenue": "m.revenue DESC",
    }

    def __init__(self, db_path: Path):
        self._db_path = db_path
        logger.info("DatabaseService initialized with %s", db_path)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
        finally:
            conn.close()

    def health_check(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1 FROM movies LIMIT 1")
            return True
        except Exception:
            logger.exception("Database health check failed")
            return False

    def search_movies(self, filters: MovieFilters) -> list[dict]:
        clauses: list[str] = []
        params: list = []

        if filters.title:
            clauses.append("m.title LIKE ?")
            params.append(f"%{filters.title}%")
        if filters.genre:
            clauses.append(
                "m.id IN (SELECT mg.movie_id FROM movie_genres mg "
                "JOIN genres g ON g.id = mg.genre_id WHERE g.name LIKE ?)"
            )
            params.append(f"%{filters.genre}%")
        if filters.year:
            clauses.append("m.year = ?")
            params.append(filters.year)
        if filters.year_from:
            clauses.append("m.year >= ?")
            params.append(filters.year_from)
        if filters.year_to:
            clauses.append("m.year <= ?")
            params.append(filters.year_to)
        if filters.director:
            clauses.append(
                "m.id IN (SELECT d.movie_id FROM directors d WHERE d.name LIKE ?)"
            )
            params.append(f"%{filters.director}%")
        if filters.actor:
            clauses.append(
                "m.id IN (SELECT c.movie_id FROM cast_members c WHERE c.name LIKE ?)"
            )
            params.append(f"%{filters.actor}%")

        where = " AND ".join(clauses) if clauses else "1=1"
        order = self.VALID_SORT_COLUMNS.get(filters.sort_by, "m.popularity DESC")

        sql = f"""
            SELECT m.id, m.title, m.year, m.overview, m.runtime,
                   m.vote_average, m.vote_count, m.popularity, m.release_date,
                   m.tagline
            FROM movies m
            WHERE {where}
            ORDER BY {order}
            LIMIT ? OFFSET ?
        """
        params.extend([filters.limit, filters.offset])

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [self._enrich_movie(dict(r)) for r in rows]

    def get_movie_detail(self, movie_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT m.* FROM movies m WHERE m.id = ?""", (movie_id,)
            ).fetchone()
            if not row:
                return None

            movie = dict(row)

            movie["genres"] = [
                r["name"]
                for r in conn.execute(
                    "SELECT g.name FROM genres g "
                    "JOIN movie_genres mg ON g.id = mg.genre_id "
                    "WHERE mg.movie_id = ?",
                    (movie_id,),
                ).fetchall()
            ]
            movie["directors"] = [
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM directors WHERE movie_id = ?", (movie_id,)
                ).fetchall()
            ]
            movie["cast"] = [
                {"name": r["name"], "character": r["character"]}
                for r in conn.execute(
                    "SELECT name, character FROM cast_members "
                    "WHERE movie_id = ? ORDER BY cast_order LIMIT 10",
                    (movie_id,),
                ).fetchall()
            ]

            rating_row = conn.execute(
                "SELECT AVG(rating) as avg_rating, COUNT(*) as count "
                "FROM ratings WHERE movie_id = ?",
                (movie_id,),
            ).fetchone()
            movie["user_rating_avg"] = (
                round(rating_row["avg_rating"], 2) if rating_row["avg_rating"] else None
            )
            movie["user_rating_count"] = rating_row["count"]

        return movie

    def get_top_rated(
        self,
        genre: str | None = None,
        year: int | None = None,
        limit: int = 10,
    ) -> list[dict]:
        filters = MovieFilters(
            genre=genre, year=year, sort_by="rating", limit=limit
        )
        return self.search_movies(filters)

    def get_movies_by_person(self, name: str, limit: int = 10) -> list[dict]:
        """Find movies where a person appears as cast or director."""
        with self._connect() as conn:
            ids = conn.execute(
                "SELECT DISTINCT movie_id FROM ("
                "  SELECT movie_id FROM cast_members WHERE name LIKE ? "
                "  UNION "
                "  SELECT movie_id FROM directors WHERE name LIKE ?"
                ") sub LIMIT ?",
                (f"%{name}%", f"%{name}%", limit),
            ).fetchall()

        movie_ids = [r["movie_id"] for r in ids]
        if not movie_ids:
            return []

        return [self.get_movie_detail(mid) for mid in movie_ids if self.get_movie_detail(mid)]

    def get_genres(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT name FROM genres ORDER BY name").fetchall()
        return [r["name"] for r in rows]

    def _enrich_movie(self, movie: dict) -> dict:
        """Attach genres, directors, and top cast to a movie dict."""
        mid = movie["id"]
        with self._connect() as conn:
            movie["genres"] = [
                r["name"]
                for r in conn.execute(
                    "SELECT g.name FROM genres g "
                    "JOIN movie_genres mg ON g.id = mg.genre_id "
                    "WHERE mg.movie_id = ?",
                    (mid,),
                ).fetchall()
            ]
            movie["directors"] = [
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM directors WHERE movie_id = ?", (mid,)
                ).fetchall()
            ]
            movie["cast"] = [
                {"name": r["name"], "character": r["character"]}
                for r in conn.execute(
                    "SELECT name, character FROM cast_members "
                    "WHERE movie_id = ? ORDER BY cast_order LIMIT 5",
                    (mid,),
                ).fetchall()
            ]
        return movie
