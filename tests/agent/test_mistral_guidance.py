"""Regression tests for Mistral-family operational guidance."""

import pytest

from agent.mistral_guidance import (
    MISTRAL_MODEL_OPERATIONAL_GUIDANCE,
    get_mistral_operational_guidance,
    is_mistral_model,
)


@pytest.mark.parametrize(
    "model_name",
    [
        "mistral-medium-latest",
        "mistral-small-latest",
        "ministral-8b-latest",
        "codestral-latest",
        "pixtral-large-latest",
        "openrouter/mistralai/mistral-medium-3.5",
    ],
)
def test_mistral_family_detection_covers_product_names(model_name):
    assert is_mistral_model(model_name)


@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-5.6-terra",
        "gemini-3.6-flash",
        "gemma-4-26b-a4b-it",
        "claude-opus-4-6",
        "deepseek-v4-flash",
    ],
)
def test_mistral_detection_does_not_capture_other_families(model_name):
    assert not is_mistral_model(model_name)


def test_mistral_guidance_is_present_when_enforcement_is_active():
    assert (
        get_mistral_operational_guidance("mistral-medium-latest", True)
        == MISTRAL_MODEL_OPERATIONAL_GUIDANCE
    )


def test_mistral_guidance_is_absent_when_enforcement_is_disabled():
    assert get_mistral_operational_guidance("mistral-medium-latest", False) is None


@pytest.mark.parametrize(
    "model_name",
    ["gpt-5.6-terra", "gemini-3.6-flash", "claude-opus-4-6", None],
)
def test_non_mistral_models_never_receive_mistral_guidance(model_name):
    assert get_mistral_operational_guidance(model_name, True) is None


def test_guidance_explicitly_preserves_non_action_conversations():
    # This is the key guard against turning Medium into a generic coding agent.
    assert "requests to do something from requests to explain" in MISTRAL_MODEL_OPERATIONAL_GUIDANCE
    assert "Do not default to a coding-agent posture" in MISTRAL_MODEL_OPERATIONAL_GUIDANCE
    assert "Verify completion" in MISTRAL_MODEL_OPERATIONAL_GUIDANCE
