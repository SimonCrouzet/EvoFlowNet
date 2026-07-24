"""Smoke tests for the installed package.

These exist to catch packaging mistakes -- a renamed module, a missing py.typed
marker, a version that does not match the installed distribution -- which are
easy to introduce and awkward to debug once other tests start failing for
apparently unrelated reasons.
"""

from importlib.metadata import version
from importlib.resources import files

import evoflownet


def test_package_is_importable():
    assert evoflownet.__name__ == "evoflownet"


def test_version_matches_installed_distribution():
    assert evoflownet.__version__ == version("evoflownet")


def test_version_is_not_the_source_tree_fallback():
    # __init__ falls back to "0.0.0.dev0" when the distribution metadata is
    # missing, which happens if the package is imported from a source tree that
    # was never installed. In a correctly configured environment it must not.
    assert evoflownet.__version__ != "0.0.0.dev0"


def test_ships_type_information():
    # A missing py.typed makes the package opaque to downstream type checkers
    # while everything still imports fine, so nothing else would catch this.
    assert files("evoflownet").joinpath("py.typed").is_file()
