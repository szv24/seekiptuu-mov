"""Ollama LLM service which calls the local HTTP API for text generation."""

import json
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a knowledgeable and friendly movie expert assistant. "
    "Answer the user's question using ONLY the movie data provided below. "
    "Be conversational but concise, aim for 2-4 sentences unless the user "
    "asks for detail. If the data doesn't contain enough information, say so "
    "honestly. Never invent facts not present in the data."
)


def _format_movie_context(movies: list[dict]) -> str:
    if not movies:
        return "(No movies found in the database matching the query.)"

    lines: list[str] = []
    for m in movies[:15]:
        parts = [f"- {m.get('title', '?')}"]
        if m.get("year"):
            parts[0] += f" ({m['year']})"
        if m.get("vote_average"):
            parts.append(f"  Rating: {m['vote_average']}/10")
        if m.get("genres"):
            g = m["genres"] if isinstance(m["genres"], list) else [m["genres"]]
            parts.append(f"  Genres: {', '.join(g)}")
        if m.get("directors"):
            d = m["directors"] if isinstance(m["directors"], list) else [m["directors"]]
            parts.append(f"  Director(s): {', '.join(d)}")
        if m.get("cast"):
            cast_names = [
                c["name"] if isinstance(c, dict) else c for c in m["cast"][:5]
            ]
            parts.append(f"  Cast: {', '.join(cast_names)}")
        if m.get("overview"):
            parts.append(f"  Plot: {m['overview'][:300]}")
        if m.get("user_rating_avg") is not None:
            parts.append(
                f"  User rating: {m['user_rating_avg']}/5 "
                f"({m.get('user_rating_count', 0)} ratings)"
            )
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


class OllamaService:
    def __init__(
        self,
        base_url: str = settings.ollama_base_url,
        model: str = settings.ollama_model,
        timeout: float = settings.ollama_timeout,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def health_check(self) -> dict:
        """Check if Ollama is reachable and the model is available."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                model_loaded = any(self._model in m for m in models)
                return {
                    "ollama_reachable": True,
                    "model_loaded": model_loaded,
                    "model": self._model,
                    "available_models": models,
                }
        except Exception as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return {
                "ollama_reachable": False,
                "model_loaded": False,
                "model": self._model,
                "error": str(exc),
            }

    async def generate(
        self,
        question: str,
        movie_data: list[dict],
    ) -> str:
        context = _format_movie_context(movie_data)
        prompt = f"MOVIE DATA:\n{context}\n\nUSER QUESTION: {question}"

        payload = {
            "model": self._model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 512,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                body = resp.json()
                return body.get("response", "").strip()
        except httpx.TimeoutException:
            logger.error("Ollama request timed out after %.0fs", self._timeout)
            return (
                "I'm sorry, the language model took too long to respond. "
                "Here's the raw data I found, you can see the 'movies' field in the response."
            )
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama HTTP error: %s", exc)
            return f"LLM service error ({exc.response.status_code}). Returning raw data instead."
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s", self._base_url)
            return (
                "The LLM service (Ollama) is not reachable. "
                "Please ensure Ollama is running. Returning raw movie data."
            )

    async def classify_intent(self, message: str) -> dict | None:
        """Use the LLM to classify a query when regex parsing is uncertain."""
        from app.services.query_parser import LLM_CLASSIFY_PROMPT

        prompt = LLM_CLASSIFY_PROMPT.format(message=message)
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 200},
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self._base_url}/api/generate", json=payload
                )
                resp.raise_for_status()
                text = resp.json().get("response", "")
                return json.loads(text)
        except Exception:
            logger.debug("LLM intent classification failed, falling back", exc_info=True)
            return None
