"""Pins the ``Py``-prefixed deprecation shim added in 0.7.0.

Plain words: 21 classes used to be spelled ``raptorbt.PyBacktestConfig`` and
are now spelled ``raptorbt.BacktestConfig``. The old spelling still works for
one release so nobody's code breaks on upgrade, but it prints a warning telling
them what to change. These tests make sure both halves of that promise hold:
the new name works, and the old name works *and still warns*.

Why the warning half matters: a shim that stops warning is indistinguishable
from a permanent alias, and the point of this one is that it goes away. If
someone silences the warning, the rename never finishes and 0.8.0 breaks users
who were never told.

The removal itself is pinned by ``test_removal_is_scheduled`` below, which
fails the moment the version reaches 0.8.0 -- that failure is the reminder to
delete the shim, the ``.pyi`` aliases, and this file.
"""

import warnings

import pytest

import raptorbt


def test_renamed_map_is_not_empty():
    """The shim exists at all. Guards against the map being emptied by accident."""
    assert raptorbt._RENAMED, "deprecation map is empty; was the shim deleted early?"


@pytest.mark.parametrize("old_name", sorted(raptorbt._RENAMED))
def test_old_name_resolves_and_warns(old_name):
    """Old spelling still works, and says so loudly."""
    new_name = raptorbt._RENAMED[old_name]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old_obj = getattr(raptorbt, old_name)

    assert len(caught) == 1, f"{old_name} should warn exactly once, got {len(caught)}"
    assert issubclass(caught[0].category, DeprecationWarning)
    message = str(caught[0].message)
    assert new_name in message, "the warning must name the replacement"
    assert "0.8.0" in message, "the warning must say when the old name disappears"

    # Same object, not a copy -- isinstance checks and identity comparisons in
    # user code must keep working across the rename.
    assert old_obj is getattr(raptorbt, new_name)


@pytest.mark.parametrize("new_name", sorted(raptorbt._RENAMED.values()))
def test_new_name_is_public_and_silent(new_name):
    """New spelling is exported and does not warn."""
    assert new_name in raptorbt.__all__, f"{new_name} missing from __all__"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(raptorbt, new_name)

    assert not caught, f"{new_name} is the canonical name and must not warn"


def test_old_names_are_not_advertised():
    """Deprecated names resolve but stay out of ``__all__``.

    ``from raptorbt import *`` must not hand anyone a name that is about to be
    deleted.
    """
    leaked = sorted(set(raptorbt._RENAMED) & set(raptorbt.__all__))
    assert not leaked, f"deprecated names still advertised in __all__: {leaked}"


def test_unknown_attribute_still_raises_attribute_error():
    """The shim must not swallow genuine typos into a confusing warning."""
    with pytest.raises(AttributeError, match="NoSuchThing"):
        raptorbt.NoSuchThing


def test_removal_is_scheduled():
    """Fails when 0.8.0 is cut. That failure is the instruction to finish the job.

    When this goes red: delete ``_RENAMED``, ``__getattr__`` and ``__dir__``
    from ``python/raptorbt/__init__.py``, delete the deprecated-alias block at
    the bottom of ``python/raptorbt/_raptorbt.pyi``, and delete this file.
    """
    version = raptorbt.__version__
    if version == "unknown":  # source checkout without install metadata
        pytest.skip("no install metadata; version gate cannot be evaluated")

    major, minor = (int(part) for part in version.split(".")[:2])
    assert (major, minor) < (0, 8), (
        f"raptorbt is {version}: the Py* deprecation window has closed. "
        "Remove the shim, the .pyi aliases, and this test file."
    )
