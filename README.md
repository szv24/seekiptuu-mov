# Movie AI Agent

A conversational REST API that answers natural-language movie questions by combining intelligent SQL query parsing with LLM-powered response generation, backed by the TMDB 5000 + MovieLens dataset in SQLite.

## Approach: Combining Structured Data with LLM

The core design principle is **retrieval-augmented generation (RAG) over a local SQL database**.
Instead of letting the LLM hallucinate movie facts, we retrieve verified data first and
only use the LLM to synthesise a natural-language answer on top of it.

The `POST /chat` pipeline has three stages:

```
User message
    |
    v
1. QUERY PARSER  (regex + keyword matching, zero latency)
    |- classifies intent: lookup / recommend / compare / top_rated / cast_crew / general
    |- extracts structured params: title, genre, year, actor, director
    |
    v
2. DATABASE SERVICE  (parameterized SQL against SQLite)
    |- translates intent + params into the right query
    |- returns precise, trusted movie data (title, cast, ratings, plot, etc.)
    |
    v
3. LLM SERVICE  (Ollama phi3:mini)
    |- receives the original question + retrieved movie data as context
    |- generates a conversational response grounded in the data
    |- system prompt forbids inventing facts not present in the context
    |
    v
Response  { "message": "<LLM text>", "movies": [<structured data>], "intent": "..." }
```

**Why this split matters:**

- **Accuracy** -- Factual data (ratings, cast, years) always comes from the database, not the
  LLM. The model cannot hallucinate a wrong director or fabricate a rating.
- **Speed** -- The regex-based query parser resolves 80%+ of queries instantly without an LLM
  round-trip. Only the final response-generation step hits the model.
- **Transparency** -- The API returns both the LLM's prose *and* the raw structured data, so
  the client can verify or display the data independently.
- **Graceful degradation** -- If Ollama is down, the API still returns the correct movie data
  with a note that the LLM is unavailable. The structured endpoints (`GET /movies`) work
  entirely without the LLM.

## Prerequisites

- **Python 3.11+** (tested on 3.14)
- **Ollama** installed and running — https://ollama.com
- ~2.3 GB free disk for the `phi3:mini` model
- CUDA GPU with >= 4 GB VRAM (recommended) or CPU-only (slower)

## Quick Start

```bash
# 1. Create virtual environment & install dependencies
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt

# 2. Build the SQLite database
python setup_db.py

# 3. Pull the LLM model
ollama pull phi3:mini

# 4. Start the API server
uvicorn app.main:app --reload
```

The server starts at **http://localhost:8000**.
Interactive API docs at **http://localhost:8000/docs**.

## API Endpoints

### `GET /health`
Service health — DB connectivity and Ollama/model availability.

### `GET /movies`
Structured movie search with optional query parameters:

| Parameter  | Type   | Description                         |
|------------|--------|-------------------------------------|
| `title`    | string | Partial title match                 |
| `genre`    | string | Genre filter (e.g. `Action`)        |
| `year`     | int    | Exact release year                  |
| `director` | string | Director name (partial match)       |
| `actor`    | string | Actor name (partial match)          |
| `sort_by`  | string | `popularity`, `rating`, `year`, `title`, `revenue` |
| `limit`    | int    | Results per page (1-100, default 20)|
| `offset`   | int    | Pagination offset                   |

### `GET /movies/{movie_id}`
Full detail for a single movie including cast, director, genres, and user ratings.

### `POST /chat`
The core AI endpoint. Send a natural-language question and receive a conversational answer grounded in database results.

**Request:**
```json
{ "message": "Recommend action movies from 2020" }
```

**Response:**
```json
{
  "message": "Here are some great action movies from 2020...",
  "intent": "recommend",
  "params": { "intent": "recommend", "genre": "Action", "year": 2020, ... },
  "movies": [ ... ]
}
```

Supported intent types: `lookup`, `recommend`, `compare`, `top_rated`, `cast_crew`, `general`.

## Usage Examples

Once the server is running (`uvicorn app.main:app --reload`), you can try these with
`curl`, PowerShell, or the interactive docs at http://localhost:8000/docs.

### Check service health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "database": true,
  "ollama": { "ollama_reachable": true, "model_loaded": true, "model": "phi3:mini" }
}
```

### Search movies by genre and year

```bash
curl "http://localhost:8000/movies?genre=Action&year=2015&sort_by=rating&limit=3"
```

```json
{
  "count": 3,
  "movies": [
    {
      "id": 76341,
      "title": "Mad Max: Fury Road",
      "year": 2015,
      "vote_average": 7.2,
      "genres": ["Adventure", "Action", "Thriller", "Science Fiction"],
      "directors": ["George Miller"],
      "cast": [{"name": "Tom Hardy", "character": "Max Rockatansky"}, ...]
    },
    ...
  ]
}
```

### Find movies by director

```bash
curl "http://localhost:8000/movies?director=Christopher+Nolan&sort_by=rating"
```

### Get full details for a single movie

```bash
curl http://localhost:8000/movies/155
```

```json
{
  "id": 155,
  "title": "The Dark Knight",
  "year": 2008,
  "overview": "Batman raises the stakes in his war on crime...",
  "vote_average": 8.2,
  "genres": ["Drama", "Action", "Thriller", "Crime"],
  "directors": ["Christopher Nolan"],
  "cast": [
    {"name": "Christian Bale", "character": "Bruce Wayne"},
    {"name": "Heath Ledger", "character": "Joker"},
    ...
  ],
  "user_rating_avg": 4.24,
  "user_rating_count": 259
}
```

### Chat: movie lookup

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Tell me about Inception\"}"
```

```json
{
  "message": "Inception (2010), directed by Christopher Nolan, is a mind-bending sci-fi thriller...",
  "intent": "lookup",
  "params": {"intent": "lookup", "titles": ["Inception"], ...},
  "movies": [...]
}
```

### Chat: recommendations

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Recommend sci-fi movies from 2014\"}"
```

### Chat — who directed / movies with actor

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Who directed The Godfather?\"}"
```

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Movies with Tom Hanks\"}"
```

### Chat: top rated

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Top 5 highest rated horror movies\"}"
```

### Chat: compare two movies

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Compare \\\"The Godfather\\\" and \\\"Goodfellas\\\"\"}"
```

> **Tip (Windows PowerShell):** Use `Invoke-RestMethod` instead of curl:
> ```powershell
> # GET
> Invoke-RestMethod http://localhost:8000/movies?genre=Action
>
> # POST /chat
> $body = '{"message": "Best comedy movies of 2010"}'
> Invoke-RestMethod -Uri http://localhost:8000/chat -Method Post -Body $body -ContentType "application/json"
> ```

## Project Structure

```
.
├── app/
│   ├── main.py              # FastAPI app, lifespan, middleware
│   ├── config.py            # Settings (DB path, Ollama URL, model)
│   ├── models.py            # Pydantic request/response schemas
│   ├── routers/
│   │   ├── movies.py        # GET /movies, GET /movies/{id}
│   │   └── chat.py          # POST /chat
│   └── services/
│       ├── database.py      # SQLite queries
│       ├── query_parser.py  # Intent classification & parameter extraction
│       └── llm.py           # Ollama HTTP client
├── tests/
│   ├── test_query_parser.py # 25 unit tests
│   └── test_api.py          # 11 integration tests
├── setup_db.py              # Database builder
├── movies.db                # SQLite database (generated)
├── requirements.txt
└── README.md
```

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests mock the Ollama service, so they run without a GPU or model pulled.

## Configuration

Environment variables (prefix `MOVIE_AGENT_`):

| Variable                  | Default                  | Description           |
|---------------------------|--------------------------|-----------------------|
| `MOVIE_AGENT_DB_PATH`    | `./movies.db`            | Path to SQLite DB     |
| `MOVIE_AGENT_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL   |
| `MOVIE_AGENT_OLLAMA_MODEL` | `phi3:mini`            | Ollama model name     |
| `MOVIE_AGENT_OLLAMA_TIMEOUT` | `180`                | LLM request timeout (s)|
| `MOVIE_AGENT_LOG_LEVEL`  | `INFO`                   | Logging level         |
