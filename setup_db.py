"""
setup_db.py — Build a normalized SQLite database from TMDB 5000 + MovieLens data.

Data sources (expected in the workspace):
  tmdb5000/tmdb_5000_movies.csv
  tmdb5000/tmdb_5000_credits.csv
  ml-latest-small/ratings.csv
  ml-latest-small/links.csv

Output: movies.db
"""

import csv
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "movies.db"

TMDB_MOVIES_CSV = BASE_DIR / "tmdb5000" / "tmdb_5000_movies.csv"
TMDB_CREDITS_CSV = BASE_DIR / "tmdb5000" / "tmdb_5000_credits.csv"
ML_RATINGS_CSV = BASE_DIR / "ml-latest-small" / "ratings.csv"
ML_LINKS_CSV = BASE_DIR / "ml-latest-small" / "links.csv"

MAX_CAST_PER_MOVIE = 10

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS movies (
    id              INTEGER PRIMARY KEY,
    title           TEXT NOT NULL,
    year            INTEGER,
    overview        TEXT,
    runtime         REAL,
    budget          INTEGER,
    revenue         INTEGER,
    popularity      REAL,
    vote_average    REAL,
    vote_count      INTEGER,
    original_language TEXT,
    tagline         TEXT,
    status          TEXT,
    release_date    TEXT
);

CREATE TABLE IF NOT EXISTS genres (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS movie_genres (
    movie_id INTEGER NOT NULL REFERENCES movies(id),
    genre_id INTEGER NOT NULL REFERENCES genres(id),
    PRIMARY KEY (movie_id, genre_id)
);

CREATE TABLE IF NOT EXISTS cast_members (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_id   INTEGER NOT NULL REFERENCES movies(id),
    name       TEXT NOT NULL,
    character  TEXT,
    cast_order INTEGER
);

CREATE TABLE IF NOT EXISTS directors (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_id INTEGER NOT NULL REFERENCES movies(id),
    name     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ratings (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_id  INTEGER NOT NULL REFERENCES movies(id),
    user_id   INTEGER NOT NULL,
    rating    REAL NOT NULL,
    timestamp INTEGER
);

CREATE INDEX IF NOT EXISTS idx_movies_title        ON movies(title);
CREATE INDEX IF NOT EXISTS idx_movie_genres_movie   ON movie_genres(movie_id);
CREATE INDEX IF NOT EXISTS idx_movie_genres_genre   ON movie_genres(genre_id);
CREATE INDEX IF NOT EXISTS idx_cast_members_movie   ON cast_members(movie_id);
CREATE INDEX IF NOT EXISTS idx_directors_movie      ON directors(movie_id);
CREATE INDEX IF NOT EXISTS idx_ratings_movie        ON ratings(movie_id);
"""


def parse_year(release_date: str) -> int | None:
    if release_date and len(release_date) >= 4:
        try:
            return int(release_date[:4])
        except ValueError:
            return None
    return None


def safe_json_loads(text: str) -> list:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []


def load_movies_and_genres(cur: sqlite3.Cursor) -> set[int]:
    """Parse tmdb_5000_movies.csv -> movies + genres + movie_genres. Returns set of loaded movie ids."""
    movie_ids = set()
    seen_genres: dict[int, str] = {}

    with open(TMDB_MOVIES_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie_id = int(row["id"])
            movie_ids.add(movie_id)

            year = parse_year(row.get("release_date", ""))
            runtime = float(row["runtime"]) if row.get("runtime") else None
            budget = int(row["budget"]) if row.get("budget") else None
            revenue = int(row["revenue"]) if row.get("revenue") else None
            popularity = float(row["popularity"]) if row.get("popularity") else None
            vote_avg = float(row["vote_average"]) if row.get("vote_average") else None
            vote_cnt = int(row["vote_count"]) if row.get("vote_count") else None

            cur.execute(
                """INSERT OR IGNORE INTO movies
                   (id, title, year, overview, runtime, budget, revenue,
                    popularity, vote_average, vote_count, original_language,
                    tagline, status, release_date)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    movie_id,
                    row["title"],
                    year,
                    row.get("overview"),
                    runtime,
                    budget,
                    revenue,
                    popularity,
                    vote_avg,
                    vote_cnt,
                    row.get("original_language"),
                    row.get("tagline"),
                    row.get("status"),
                    row.get("release_date"),
                ),
            )

            genres_list = safe_json_loads(row.get("genres", "[]"))
            for g in genres_list:
                gid, gname = g["id"], g["name"]
                if gid not in seen_genres:
                    seen_genres[gid] = gname
                    cur.execute("INSERT OR IGNORE INTO genres (id, name) VALUES (?, ?)", (gid, gname))
                cur.execute(
                    "INSERT OR IGNORE INTO movie_genres (movie_id, genre_id) VALUES (?, ?)",
                    (movie_id, gid),
                )

    return movie_ids


def load_credits(cur: sqlite3.Cursor, movie_ids: set[int]) -> None:
    """Parse tmdb_5000_credits.csv -> cast_members + directors."""
    with open(TMDB_CREDITS_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie_id = int(row["movie_id"])
            if movie_id not in movie_ids:
                continue

            cast_list = safe_json_loads(row.get("cast", "[]"))
            cast_list.sort(key=lambda c: c.get("order", 9999))
            for member in cast_list[:MAX_CAST_PER_MOVIE]:
                cur.execute(
                    "INSERT INTO cast_members (movie_id, name, character, cast_order) VALUES (?,?,?,?)",
                    (movie_id, member["name"], member.get("character"), member.get("order")),
                )

            crew_list = safe_json_loads(row.get("crew", "[]"))
            for person in crew_list:
                if person.get("job") == "Director":
                    cur.execute(
                        "INSERT INTO directors (movie_id, name) VALUES (?, ?)",
                        (movie_id, person["name"]),
                    )


def build_ml_to_tmdb_map() -> dict[int, int]:
    """Build a mapping from MovieLens movieId -> TMDB id using links.csv."""
    mapping: dict[int, int] = {}
    with open(ML_LINKS_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ml_id = int(row["movieId"])
            tmdb_raw = row.get("tmdbId", "").strip()
            if tmdb_raw:
                try:
                    mapping[ml_id] = int(float(tmdb_raw))
                except ValueError:
                    continue
    return mapping


def load_ratings(cur: sqlite3.Cursor, movie_ids: set[int]) -> int:
    """Parse ratings.csv -> ratings table. Returns count of inserted rows."""
    ml_to_tmdb = build_ml_to_tmdb_map()
    valid_tmdb_ids = movie_ids & set(ml_to_tmdb.values())

    inserted = 0
    batch: list[tuple] = []

    with open(ML_RATINGS_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ml_movie_id = int(row["movieId"])
            tmdb_id = ml_to_tmdb.get(ml_movie_id)
            if tmdb_id is None or tmdb_id not in valid_tmdb_ids:
                continue

            batch.append((
                tmdb_id,
                int(row["userId"]),
                float(row["rating"]),
                int(row["timestamp"]),
            ))

            if len(batch) >= 5000:
                cur.executemany(
                    "INSERT INTO ratings (movie_id, user_id, rating, timestamp) VALUES (?,?,?,?)",
                    batch,
                )
                inserted += len(batch)
                batch.clear()

    if batch:
        cur.executemany(
            "INSERT INTO ratings (movie_id, user_id, rating, timestamp) VALUES (?,?,?,?)",
            batch,
        )
        inserted += len(batch)

    return inserted


def print_summary(cur: sqlite3.Cursor) -> None:
    tables = ["movies", "genres", "movie_genres", "cast_members", "directors", "ratings"]
    print("\n=== Database Summary ===")
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"  {table:20s}: {count:>8,} rows")

    print("\n=== Sample: Top 5 highest-rated movies (min 50 user ratings) ===")
    cur.execute("""
        SELECT m.title, m.year, m.vote_average,
               GROUP_CONCAT(DISTINCT g.name) AS genres,
               GROUP_CONCAT(DISTINCT d.name) AS directors
        FROM movies m
        LEFT JOIN movie_genres mg ON mg.movie_id = m.id
        LEFT JOIN genres g ON g.id = mg.genre_id
        LEFT JOIN directors d ON d.movie_id = m.id
        WHERE m.vote_count >= 50
        GROUP BY m.id
        ORDER BY m.vote_average DESC
        LIMIT 5
    """)
    for row in cur.fetchall():
        title, year, avg, genres, directors = row
        print(f"  {title} ({year}) — {avg}/10 | Genres: {genres} | Director(s): {directors}")


def main() -> None:
    for path in (TMDB_MOVIES_CSV, TMDB_CREDITS_CSV, ML_RATINGS_CSV, ML_LINKS_CSV):
        if not path.exists():
            print(f"ERROR: Missing data file: {path}", file=sys.stderr)
            sys.exit(1)

    if DB_PATH.exists():
        os.remove(DB_PATH)
        print(f"Removed existing {DB_PATH.name}")

    t0 = time.perf_counter()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    cur = conn.cursor()

    print("Creating schema...")
    cur.executescript(SCHEMA_SQL)

    print("Loading movies & genres...")
    movie_ids = load_movies_and_genres(cur)
    conn.commit()
    print(f"  Loaded {len(movie_ids)} movies")

    print("Loading cast & directors...")
    load_credits(cur, movie_ids)
    conn.commit()

    print("Loading ratings (MovieLens -> TMDB mapping)...")
    n_ratings = load_ratings(cur, movie_ids)
    conn.commit()
    print(f"  Linked {n_ratings:,} ratings")

    print_summary(cur)

    conn.close()
    elapsed = time.perf_counter() - t0
    print(f"\nDone. Database written to {DB_PATH}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
