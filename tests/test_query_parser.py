"""Unit tests for the query parser — no DB, no LLM, no I/O."""

import pytest

from app.services.query_parser import Intent, parse_query


class TestLookupIntent:
    def test_tell_me_about(self):
        r = parse_query("Tell me about Inception")
        assert r.intent == Intent.LOOKUP
        assert "Inception" in (r.titles[0] if r.titles else r.raw_message)

    def test_quoted_title(self):
        r = parse_query('What is the movie "The Matrix"?')
        assert r.intent == Intent.LOOKUP
        assert "The Matrix" in r.titles

    def test_plot_of(self):
        r = parse_query("What is the plot of Interstellar?")
        assert r.intent == Intent.LOOKUP

    def test_describe(self):
        r = parse_query("Describe the movie Gladiator")
        assert r.intent == Intent.LOOKUP


class TestRecommendIntent:
    def test_recommend_genre(self):
        r = parse_query("Recommend action movies")
        assert r.intent == Intent.RECOMMEND
        assert r.genre == "Action"

    def test_suggest_year(self):
        r = parse_query("Suggest comedy movies from 2020")
        assert r.intent == Intent.RECOMMEND
        assert r.genre == "Comedy"
        assert r.year == 2020

    def test_looking_for(self):
        r = parse_query("I'm looking for sci-fi movies")
        assert r.intent == Intent.RECOMMEND
        assert r.genre == "Science Fiction"

    def test_movies_like(self):
        r = parse_query('Movies like "Inception"')
        assert r.intent == Intent.RECOMMEND
        assert "Inception" in r.titles

    def test_show_me(self):
        r = parse_query("Show me some thriller movies")
        assert r.intent == Intent.RECOMMEND
        assert r.genre == "Thriller"


class TestCompareIntent:
    def test_compare_two_quoted(self):
        r = parse_query('Compare "The Godfather" and "Goodfellas"')
        assert r.intent == Intent.COMPARE
        assert len(r.titles) == 2

    def test_vs(self):
        r = parse_query('"Alien" vs "Aliens"')
        assert r.intent == Intent.COMPARE
        assert len(r.titles) == 2


class TestTopRatedIntent:
    def test_best_of_year(self):
        r = parse_query("Best movies of 2015")
        assert r.intent == Intent.TOP_RATED
        assert r.year == 2015

    def test_highest_rated_genre(self):
        r = parse_query("Highest rated horror movies")
        assert r.intent == Intent.TOP_RATED
        assert r.genre == "Horror"

    def test_top_n(self):
        r = parse_query("Top 5 comedy movies")
        assert r.intent == Intent.TOP_RATED
        assert r.genre == "Comedy"
        assert r.limit == 5

    def test_greatest_all_time(self):
        r = parse_query("Greatest movies of all time")
        assert r.intent == Intent.TOP_RATED


class TestCastCrewIntent:
    def test_who_directed(self):
        r = parse_query("Who directed Inception?")
        assert r.intent == Intent.CAST_CREW

    def test_movies_with_person(self):
        r = parse_query("Movies with Tom Hanks")
        assert r.intent == Intent.CAST_CREW
        assert r.person is not None
        assert "Tom Hanks" in r.person

    def test_cast_of(self):
        r = parse_query('Cast of "Pulp Fiction"')
        assert r.intent == Intent.CAST_CREW
        assert "Pulp Fiction" in r.titles

    def test_directed_by(self):
        r = parse_query("Movies directed by Christopher Nolan")
        assert r.intent == Intent.CAST_CREW
        assert "Christopher Nolan" in r.person


class TestParameterExtraction:
    def test_year_extraction(self):
        r = parse_query("Best action movies from 2019")
        assert r.year == 2019

    def test_genre_sci_fi_alias(self):
        r = parse_query("Recommend scifi movies")
        assert r.genre == "Science Fiction"

    def test_genre_case_insensitive(self):
        r = parse_query("Best COMEDY movies")
        assert r.genre == "Comedy"

    def test_multiple_quoted_titles(self):
        r = parse_query('Compare "Inception" and "Interstellar"')
        assert len(r.titles) == 2
        assert "Inception" in r.titles
        assert "Interstellar" in r.titles


class TestGeneralFallback:
    def test_generic_question(self):
        r = parse_query("What are some fun things to watch on a rainy day?")
        assert r.intent == Intent.GENERAL

    def test_empty_falls_to_general(self):
        r = parse_query("hello")
        assert r.intent == Intent.GENERAL
