"""
AstraGuard 2.3 — Test Suite Reset Fixture
Forces ProfileRegistry singleton to reload ASQD 2.3 config before tests.
"""

import sys


def reset_profile_registry():
    """Clear the ProfileRegistry singleton so it reloads from current config files."""
    from astraguard_core.context_resolver import profiles as _prof_mod
    _prof_mod.ProfileRegistry._instance = None
