"""Tests for agent.retriever.KeywordSearchRetriever."""

import pytest

from agent.retriever import KeywordSearchRetriever

KB_DIR = "kb"


@pytest.fixture(scope="module")
def retriever():
    return KeywordSearchRetriever(KB_DIR)


# ---------------------------------------------------------------------------
# Phrase search
# ---------------------------------------------------------------------------

class TestPhraseSearch:
    def test_phrase_match_score_is_2(self, retriever):
        results = retriever.search("encrypted in transit and at rest")
        phrase_hits = [r for r in results if r.match_type == "phrase"]
        assert phrase_hits, "expected at least one phrase match"
        assert all(r.score == 2.0 for r in phrase_hits)

    def test_phrase_match_outscores_keyword(self, retriever):
        results = retriever.search("14 days")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "results must be sorted by score desc"
        assert results[0].match_type == "phrase"
        assert results[0].score == 2.0

    def test_phrase_match_cites_correct_file(self, retriever):
        results = retriever.search("encrypted in transit and at rest")
        phrase_files = [r.file for r in results if r.match_type == "phrase"]
        assert any("security_compliance" in f for f in phrase_files)

    def test_phrase_match_lines_format(self, retriever):
        results = retriever.search("14 days")
        for r in results:
            parts = r.lines.split("-")
            assert len(parts) == 2
            start, end = int(parts[0]), int(parts[1])
            assert start >= 1
            assert end >= start


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------

class TestKeywordSearch:
    def test_keyword_score_between_0_and_1(self, retriever):
        results = retriever.search("annual refund policy")
        kw_results = [r for r in results if r.match_type == "keyword"]
        for r in kw_results:
            assert 0 < r.score <= 1.0

    def test_keyword_matched_terms_nonempty(self, retriever):
        results = retriever.search("annual refund policy")
        for r in results:
            assert r.matched_terms, "matched_terms must not be empty"

    def test_stopwords_excluded_from_keywords(self, retriever):
        # "the", "is", "a" are stopwords — searching only stopwords should return no keyword hits
        results = retriever.search("the is a")
        kw_results = [r for r in results if r.match_type == "keyword"]
        assert kw_results == []

    def test_relevant_file_surfaces_for_refund_query(self, retriever):
        results = retriever.search("refund annual subscription")
        files = [r.file for r in results]
        assert any("refunds_cancellation" in f for f in files)

    def test_relevant_file_surfaces_for_sso_query(self, retriever):
        results = retriever.search("SSO plan EU")
        files = [r.file for r in results]
        assert any("security_compliance" in f for f in files)

    def test_results_sorted_by_score_descending(self, retriever):
        results = retriever.search("how many seats does business plan include")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_results_for_unknown_term(self, retriever):
        results = retriever.search("xyzzy foobar nonexistent")
        assert results == []


# ---------------------------------------------------------------------------
# Citation helper
# ---------------------------------------------------------------------------

class TestCitation:
    def test_citation_dict_keys(self, retriever):
        results = retriever.search("Business plan")
        for r in results:
            c = r.citation()
            assert set(c.keys()) == {"file", "lines"}
            assert isinstance(c["file"], str)
            assert isinstance(c["lines"], str)
