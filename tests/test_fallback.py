"""Real tests for model->provider routing and fallback mapping."""
import pytest

from app.fallback import ANTHROPIC_FALLBACK, OPENAI_FALLBACK, get_fallback_model, get_provider


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-4", "gpt-3.5-turbo"),
        ("gpt-4-turbo", "gpt-3.5-turbo"),
        ("gpt-4o", "gpt-3.5-turbo"),
    ],
)
def test_openai_fallbacks(model, expected):
    assert get_fallback_model("openai", model) == expected


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-3-opus-20240229", "claude-3-sonnet-20240229"),
        ("claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"),
    ],
)
def test_anthropic_fallbacks(model, expected):
    assert get_fallback_model("anthropic", model) == expected


def test_model_without_fallback_returns_none():
    assert get_fallback_model("openai", "gpt-3.5-turbo") is None
    assert get_fallback_model("anthropic", "claude-3-haiku-20240307") is None


def test_unknown_model_and_provider_return_none():
    assert get_fallback_model("openai", "some-future-model") is None
    assert get_fallback_model("azure", "gpt-4") is None


def test_fallback_maps_are_consistent_with_lookup():
    for model, fallback in OPENAI_FALLBACK.items():
        assert get_fallback_model("openai", model) == fallback
        assert get_provider(fallback) == "openai"
    for model, fallback in ANTHROPIC_FALLBACK.items():
        assert get_fallback_model("anthropic", model) == fallback
        assert get_provider(fallback) == "anthropic"


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-3-opus-20240229", "anthropic"),
        ("claude-3-haiku-20240307", "anthropic"),
        ("gpt-4", "openai"),
        ("gpt-3.5-turbo", "openai"),
        ("anything-else", "openai"),  # documented default in get_provider
    ],
)
def test_get_provider(model, expected):
    assert get_provider(model) == expected
