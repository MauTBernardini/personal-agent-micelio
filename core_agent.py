"""Compatibility facade for the refactored email triage backend.

The implementation now lives inside the `email_agent` package. This module keeps a
stable import surface for older callers while the codebase is organized into
smaller, more maintainable modules.
"""

from email_agent import *  # noqa: F401,F403

