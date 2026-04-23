"""Public interface for the Antese writing agent.

This module stays intentionally lightweight to avoid circular imports during
FastAPI bootstrapping.
"""

from writing_agent.models import GENRE_CARD_IDS, STYLE_REASON_TAGS, WRITING_GOALS, WRITING_TASK_TYPES


def get_antese_capabilities():
    from writing_agent.service import get_antese_capabilities as _get_antese_capabilities

    return _get_antese_capabilities()


def run_antese(*args, **kwargs):
    from writing_agent.service import run_antese as _run_antese

    return _run_antese(*args, **kwargs)


def get_antese_execution_history(*args, **kwargs):
    from writing_agent.service import get_antese_execution_history as _get_antese_execution_history

    return _get_antese_execution_history(*args, **kwargs)


def get_antese_versions(*args, **kwargs):
    from writing_agent.service import get_antese_versions as _get_antese_versions

    return _get_antese_versions(*args, **kwargs)


def get_antese_feedback(*args, **kwargs):
    from writing_agent.service import get_antese_feedback as _get_antese_feedback

    return _get_antese_feedback(*args, **kwargs)


def get_antese_style_profiles_catalog(*args, **kwargs):
    from writing_agent.service import get_antese_style_profiles_catalog as _get_antese_style_profiles_catalog

    return _get_antese_style_profiles_catalog(*args, **kwargs)


def get_antese_genre_card_catalog(*args, **kwargs):
    from writing_agent.service import get_antese_genre_card_catalog as _get_antese_genre_card_catalog

    return _get_antese_genre_card_catalog(*args, **kwargs)


def get_antese_inspiration_profile_catalog(*args, **kwargs):
    from writing_agent.service import get_antese_inspiration_profile_catalog as _get_antese_inspiration_profile_catalog

    return _get_antese_inspiration_profile_catalog(*args, **kwargs)


def create_or_update_antese_style_profile(*args, **kwargs):
    from writing_agent.service import create_or_update_antese_style_profile as _create_or_update_antese_style_profile

    return _create_or_update_antese_style_profile(*args, **kwargs)


def create_or_update_antese_genre_card(*args, **kwargs):
    from writing_agent.service import create_or_update_antese_genre_card as _create_or_update_antese_genre_card

    return _create_or_update_antese_genre_card(*args, **kwargs)


def submit_antese_feedback(*args, **kwargs):
    from writing_agent.service import submit_antese_feedback as _submit_antese_feedback

    return _submit_antese_feedback(*args, **kwargs)


__all__ = [
    "GENRE_CARD_IDS",
    "STYLE_REASON_TAGS",
    "WRITING_GOALS",
    "WRITING_TASK_TYPES",
    "create_or_update_antese_genre_card",
    "create_or_update_antese_style_profile",
    "get_antese_capabilities",
    "get_antese_execution_history",
    "get_antese_feedback",
    "get_antese_genre_card_catalog",
    "get_antese_inspiration_profile_catalog",
    "get_antese_style_profiles_catalog",
    "get_antese_versions",
    "run_antese",
    "submit_antese_feedback",
]
