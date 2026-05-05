"""AnswerGenerator context relevance (keyword + LLM-as-judge paths)."""

from __future__ import annotations

from src.generation.generator import AnswerGenerator


class _StubLLMYes:
    model = "stub"

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, model: str = None, stream: bool = False, timeout=None):
        return "YES\n"


class _StubLLMNo:
    model = "stub"

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, model: str = None, stream: bool = False, timeout=None):
        return "NO\n"


class _StubLLMUnavailable:
    model = "stub"

    def is_available(self) -> bool:
        return False

    def generate(self, prompt: str, model: str = None, stream: bool = False, timeout=None):
        raise AssertionError("should not be called")


def test_parse_yes_no():
    g = AnswerGenerator(_StubLLMYes())
    assert g._parse_yes_no_answer("YES") is True
    assert g._parse_yes_no_answer("no\nextra") is False
    assert g._parse_yes_no_answer("") is None


def test_keyword_relevance_fallback_when_llm_unavailable():
    g = AnswerGenerator(_StubLLMUnavailable())
    q = "EVALCORPUS_TOKEN_ALPHA capital France Paris"
    ctx = [{"content": "EVALCORPUS_TOKEN_ALPHA The capital of France is Paris."}]
    assert g.check_context_relevance(q, ctx) is True


def test_llm_judge_majority_yes():
    g = AnswerGenerator(_StubLLMYes())
    q = "What is documented about databases?"
    contexts = [
        {"content": "PostgreSQL supports SQL queries."},
        {"content": "Another line also about databases."},
    ]
    assert g.check_context_relevance(q, contexts) is True


def test_llm_judge_majority_no():
    g = AnswerGenerator(_StubLLMNo())
    q = "What is documented about databases?"
    contexts = [
        {"content": "PostgreSQL supports SQL queries."},
        {"content": "Another line also about databases."},
    ]
    assert g.check_context_relevance(q, contexts) is False
