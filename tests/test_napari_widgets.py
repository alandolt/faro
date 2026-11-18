"""Toolbar buttons of the pertzlab pymmcore-widgets build per scope (``NAPARI_WIDGETS``)."""

from __future__ import annotations

import logging

import pytest

from faro.microscope.pymmcore import PyMMCoreMicroscope
from faro.napari_launcher import (
    PERTZLAB_WIDGETS,
    _add_pertzlab_widgets,
    _scope_widgets,
)


class Scope:
    def __init__(self, widgets):
        self.NAPARI_WIDGETS = widgets


def test_default_is_all_widgets():
    assert PyMMCoreMicroscope.NAPARI_WIDGETS is None
    assert _scope_widgets(Scope(None)) == PERTZLAB_WIDGETS
    assert _scope_widgets(object()) == PERTZLAB_WIDGETS  # no attribute


def test_scope_picks_widgets():
    assert _scope_widgets(Scope(["Stage Map"])) == {
        "Stage Map": PERTZLAB_WIDGETS["Stage Map"]
    }
    assert _scope_widgets(Scope([])) == {}


def test_unknown_widget_is_skipped_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="faro.napari_launcher"):
        widgets = _scope_widgets(Scope(["Stage Map", "Nope"]))
    assert list(widgets) == ["Stage Map"]
    assert "Nope" in caplog.text


@pytest.fixture
def toolbar(monkeypatch):
    """The plugin's DOCK_WIDGETS, with every pertzlab widget class present."""
    pytest.importorskip("napari_micromanager")
    import pymmcore_widgets
    from napari_micromanager._gui_objects import _toolbar

    for class_name, _ in PERTZLAB_WIDGETS.values():
        if not hasattr(pymmcore_widgets, class_name):
            monkeypatch.setattr(pymmcore_widgets, class_name, object, raising=False)
    docks = dict(_toolbar.DOCK_WIDGETS)
    monkeypatch.setattr(_toolbar, "DOCK_WIDGETS", docks)
    return docks


def test_only_scope_widgets_get_a_button(toolbar):
    _add_pertzlab_widgets(Scope(["Stage Map"]))
    assert "Stage Map" in toolbar
    assert "Continuous Focus" not in toolbar


def test_no_widgets_no_buttons(toolbar):
    before = dict(toolbar)
    _add_pertzlab_widgets(Scope([]))
    assert toolbar == before


def test_all_widgets_by_default(toolbar):
    _add_pertzlab_widgets(Scope(None))
    assert set(PERTZLAB_WIDGETS) <= set(toolbar)
