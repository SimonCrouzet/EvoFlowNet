"""Test suite for evoflownet.

This file makes ``tests`` a package. Without it, mypy resolves test modules by
bare filename (``test_package`` rather than ``tests.test_package``), so the
``tests.*`` configuration override silently fails to match and the relaxed
annotation rules never apply.
"""
