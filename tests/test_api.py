"""Integration tests for API endpoints using FastAPI TestClient.

The Ollama LLM is mocked so tests run without a GPU / running model.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_returns_200(self, client):
        with patch.object(
            app.state.llm, "health_check",
            new_callable=AsyncMock,
            return_value={"ollama_reachable": True, "model_loaded": True, "model": "phi3:mini", "available_models": ["phi3:mini"]},
        ):
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["database"] is True
        assert "status" in body


class TestMoviesEndpoint:
    def test_list_movies_default(self, client):
        resp = client.get("/movies")
        assert resp.status_code == 200
        body = resp.json()
        assert "movies" in body
        assert body["count"] > 0

    def test_list_movies_filter_genre(self, client):
        resp = client.get("/movies?genre=Action&limit=5")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] <= 5
        for m in body["movies"]:
            assert "Action" in m["genres"]

    def test_list_movies_filter_year(self, client):
        resp = client.get("/movies?year=2010&limit=5")
        assert resp.status_code == 200
        for m in resp.json()["movies"]:
            assert m["year"] == 2010

    def test_get_movie_by_id(self, client):
        resp = client.get("/movies/19995")  # Avatar
        assert resp.status_code == 200
        body = resp.json()
        assert body["title"] == "Avatar"
        assert "genres" in body
        assert "cast" in body
        assert "directors" in body

    def test_get_movie_not_found(self, client):
        resp = client.get("/movies/999999999")
        assert resp.status_code == 404


class TestChatEndpoint:
    MOCK_LLM_RESPONSE = "Here are some great action movies for you!"

    def _mock_generate(self):
        return patch.object(
            app.state.llm,
            "generate",
            new_callable=AsyncMock,
            return_value=self.MOCK_LLM_RESPONSE,
        )

    def test_chat_recommend(self, client):
        with self._mock_generate():
            resp = client.post("/chat", json={"message": "Recommend action movies from 2015"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["intent"] == "recommend"
        assert body["message"] == self.MOCK_LLM_RESPONSE
        assert isinstance(body["movies"], list)

    def test_chat_lookup(self, client):
        with self._mock_generate():
            resp = client.post("/chat", json={"message": "Tell me about Avatar"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["intent"] == "lookup"

    def test_chat_top_rated(self, client):
        with self._mock_generate():
            resp = client.post("/chat", json={"message": "Best comedy movies"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["intent"] == "top_rated"
        assert body["params"]["genre"] == "Comedy"

    def test_chat_empty_message_rejected(self, client):
        resp = client.post("/chat", json={"message": ""})
        assert resp.status_code == 422

    def test_chat_cast_crew(self, client):
        with self._mock_generate():
            resp = client.post("/chat", json={"message": "Movies with Tom Hanks"})
        assert resp.status_code == 200
        assert resp.json()["intent"] == "cast_crew"
