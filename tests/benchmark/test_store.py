"""Tests for the result store, and above all for when it refuses a cached run.

The store's job is to be right about staleness in both directions. Handing back
a result produced by code that has since changed silently mixes versions inside
one table; re-running a result that could not have changed costs hours. Both
failures are cheap to write and expensive to notice, so the import-graph walk
that decides between them is tested against a synthetic package where the
answer is known by construction.
"""

import json
from typing import Any

import pytest

from evoflownet.benchmark.store import (
    ResultStore,
    RunRecord,
    dependency_closure,
    fingerprint,
    package_fingerprint,
)

# A package whose import graph is small enough to reason about by eye. Each
# module exists to pin one behaviour of the walk.
SOURCES = {
    "__init__.py": "",
    "alpha.py": "from pkg import beta\nfrom pkg.deep import gamma\n",
    "beta.py": "import pkg.leaf\n",
    "leaf.py": "import json\n",
    "lonely.py": "VALUE = 1\n",
    "deep/__init__.py": "",
    "deep/gamma.py": "from . import sibling\nfrom ..beta import THING\n",
    "deep/sibling.py": "",
    "cyclic_a.py": "import pkg.cyclic_b\n",
    "cyclic_b.py": "import pkg.cyclic_a\n",
    "typed.py": (
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from pkg import lonely\n"
    ),
    "ambiguous.py": "from pkg.deep import gamma, Thing\n",
    "algorithms/__init__.py": "",
    "algorithms/genetic.py": "from pkg import leaf\n",
}

# Splatted into RunRecord and ResultStore.stamp, which is why the values are
# Any: inferred as `object` the splat fails every field's declared type.
RECORD_FIELDS: dict[str, Any] = {
    "task": "t",
    "method": "m",
    "seed": 0,
    "protocol": "P",
    "best": 1.0,
    "regret": 0.0,
    "diversity": 0.5,
    "feasible_fraction": 1.0,
    "oracle_calls": 10,
    "proposals": 20,
}


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    """A synthetic package, installed as the thing the store fingerprints."""
    root = tmp_path / "pkg"
    for name, body in SOURCES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    monkeypatch.setattr("evoflownet.benchmark.store._package_root", lambda: root)
    return root


@pytest.fixture
def store(pkg, tmp_path):
    """A store over the synthetic package."""
    del pkg
    return ResultStore(tmp_path / "results")


def test_fingerprint_is_one_entry_per_module(pkg):
    del pkg
    digests = fingerprint()
    assert set(digests) == {
        "pkg",
        "pkg.algorithms",
        "pkg.algorithms.genetic",
        "pkg.alpha",
        "pkg.ambiguous",
        "pkg.beta",
        "pkg.cyclic_a",
        "pkg.cyclic_b",
        "pkg.deep",
        "pkg.deep.gamma",
        "pkg.deep.sibling",
        "pkg.leaf",
        "pkg.lonely",
        "pkg.typed",
    }
    assert len(digests) == len(SOURCES)


def test_fingerprint_is_stable_and_content_addressed(pkg):
    before = fingerprint()
    assert fingerprint() == before
    (pkg / "lonely.py").write_text("VALUE = 2\n")
    after = fingerprint()
    assert after["pkg.lonely"] != before["pkg.lonely"]
    assert {k: v for k, v in after.items() if k != "pkg.lonely"} == {
        k: v for k, v in before.items() if k != "pkg.lonely"
    }


def test_closure_is_transitive(pkg):
    del pkg
    assert dependency_closure(["pkg.alpha"]) == (
        "pkg.alpha",
        "pkg.beta",
        "pkg.deep.gamma",
        "pkg.deep.sibling",
        "pkg.leaf",
    )


def test_closure_excludes_unreachable_modules(pkg):
    del pkg
    assert "pkg.lonely" not in dependency_closure(["pkg.alpha"])


def test_closure_ignores_external_imports(pkg):
    del pkg
    assert dependency_closure(["pkg.leaf"]) == ("pkg.leaf",)


def test_relative_imports_resolve(pkg):
    del pkg
    # `from . import sibling` and `from ..beta import THING`, resolved against
    # the package containing pkg/deep/gamma.py.
    closure = dependency_closure(["pkg.deep.gamma"])
    assert "pkg.deep.sibling" in closure
    assert "pkg.beta" in closure


def test_type_checking_imports_are_included(pkg):
    del pkg
    assert "pkg.lonely" in dependency_closure(["pkg.typed"])


def test_cycles_terminate(pkg):
    del pkg
    assert dependency_closure(["pkg.cyclic_a"]) == ("pkg.cyclic_a", "pkg.cyclic_b")


def test_from_import_prefers_the_submodule_and_falls_back_to_the_package(pkg):
    del pkg
    # `from pkg.deep import gamma, Thing`: gamma is a module, Thing is not, so
    # the first resolves to the submodule and the second to pkg.deep itself.
    closure = dependency_closure(["pkg.ambiguous"])
    assert "pkg.deep.gamma" in closure
    assert "pkg.deep" in closure


def test_unknown_entry_point_is_an_error(pkg):
    del pkg
    with pytest.raises(ValueError, match="no such module"):
        dependency_closure(["pkg.nonexistent"])


def test_stamp_stores_only_the_declared_closure(store):
    record = store.stamp(depends_on=["pkg.alpha"], **RECORD_FIELDS)
    assert set(record.source) == set(dependency_closure(["pkg.alpha"]))


def test_unrelated_change_leaves_a_record_usable(pkg, store):
    store.append(store.stamp(depends_on=["pkg.alpha"], **RECORD_FIELDS))
    (pkg / "lonely.py").write_text("VALUE = 99\n")

    reopened = ResultStore(store.root)
    assert set(reopened.usable("t", "m")) == {0}
    assert reopened.missing("t", "m", [0]) == []
    assert reopened.stale("t", "m") == {}


def test_change_to_a_depended_module_marks_a_record_stale(pkg, store):
    store.append(store.stamp(depends_on=["pkg.alpha"], **RECORD_FIELDS))
    # Two hops from the entry point: alpha imports beta imports leaf.
    (pkg / "leaf.py").write_text("import json\n\nVALUE = 2\n")

    reopened = ResultStore(store.root)
    assert reopened.usable("t", "m") == {}
    assert reopened.missing("t", "m", [0]) == [0]
    assert reopened.stale("t", "m") == {0: ("pkg.leaf",)}


def test_a_deleted_dependency_marks_a_record_stale(pkg, store):
    store.append(store.stamp(depends_on=["pkg.alpha"], **RECORD_FIELDS))
    (pkg / "leaf.py").unlink()

    assert ResultStore(store.root).stale("t", "m") == {0: ("pkg.leaf",)}


def test_stamp_without_depends_on_falls_back_to_everything(pkg, store):
    record = store.stamp(**RECORD_FIELDS)
    assert set(record.source) == set(fingerprint())
    # The root package hashes under a bare name, which must not be mistaken
    # for the bare package names the old scheme used.
    assert "pkg" in record.source
    assert not record.per_package

    store.append(record)
    (pkg / "lonely.py").write_text("VALUE = 99\n")
    assert ResultStore(store.root).stale("t", "m") == {0: ("pkg.lonely",)}


def test_a_record_without_a_fingerprint_is_treated_as_current(store):
    store.append(RunRecord(**RECORD_FIELDS))
    assert set(store.usable("t", "m")) == {0}


def write_legacy(store, source):
    """Append a record in the superseded per-package format."""
    path = store.root / "t"
    path.mkdir(parents=True, exist_ok=True)
    (path / "m.jsonl").write_text(json.dumps({**RECORD_FIELDS, "source": source}) + "\n")


def test_old_format_records_still_load_and_are_judged_current(pkg, store):
    del pkg
    write_legacy(store, package_fingerprint())

    reopened = ResultStore(store.root)
    assert reopened.load("t", "m")[0].per_package
    assert set(reopened.usable("t", "m")) == {0}


def test_old_format_records_go_stale_on_a_package_change(pkg, store):
    write_legacy(store, package_fingerprint())
    (pkg / "algorithms" / "genetic.py").write_text("from pkg import leaf\n\nVALUE = 2\n")

    assert ResultStore(store.root).stale("t", "m") == {0: ("algorithms",)}


def test_old_format_records_do_not_match_module_hashes(pkg, store):
    del pkg, store
    # The two schemes share no keys, so a legacy record compared against the
    # per-module fingerprint would trivially look current. It must not.
    record = RunRecord(**RECORD_FIELDS, source={"algorithms": "0000000000000000"})
    assert record.per_package
    assert record.stale_against(fingerprint()) == ("algorithms",)


def test_bless_restores_a_stale_record_without_widening_it(pkg, store):
    store.append(store.stamp(depends_on=["pkg.alpha"], **RECORD_FIELDS))
    declared = set(store.load("t", "m")[0].source)
    (pkg / "leaf.py").write_text("import json\n\nVALUE = 2\n")

    reopened = ResultStore(store.root)
    assert reopened.bless("t", "m") == 1
    assert set(reopened.usable("t", "m")) == {0}
    assert set(reopened.load("t", "m")[0].source) == declared


def test_bless_keeps_an_old_format_record_in_its_own_format(pkg, store):
    write_legacy(store, package_fingerprint())
    (pkg / "algorithms" / "genetic.py").write_text("VALUE = 2\n")

    reopened = ResultStore(store.root)
    assert reopened.bless("t", "m") == 1
    assert reopened.stale("t", "m") == {}
    assert set(reopened.load("t", "m")[0].source) == {"algorithms"}
