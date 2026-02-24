"""Pydantic request/response schemas for the Movie AI Agent API."""

from pydantic import BaseModel, Field


# Shared sub-models

class CastMember(BaseModel):
    name: str
    character: str | None = None


class MovieSummary(BaseModel):
    id: int
    title: str
    year: int | None = None
    overview: str | None = None
    runtime: float | None = None
    vote_average: float | None = None
    vote_count: int | None = None
    popularity: float | None = None
    release_date: str | None = None
    tagline: str | None = None
    genres: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    cast: list[CastMember] = Field(default_factory=list)


class MovieDetail(MovieSummary):
    budget: int | None = None
    revenue: int | None = None
    original_language: str | None = None
    status: str | None = None
    user_rating_avg: float | None = None
    user_rating_count: int | None = None


# Request / Response

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, examples=[
        "Recommend action movies from 2020",
        "Tell me about Inception",
        "Who directed The Godfather?",
    ])


class ChatResponse(BaseModel):
    message: str
    intent: str
    params: dict
    movies: list[MovieSummary] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    database: bool
    ollama: dict


class MovieListResponse(BaseModel):
    count: int
    movies: list[MovieSummary]
