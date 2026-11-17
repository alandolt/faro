"""Controls of the "Illumination Control" dock of ``faro-napari``.

The resolution of ``LIGHT_PROPERTIES`` and ``EXTRA_CONTROLS`` entries runs on
a stub core. The widget test builds the dock offscreen on the Micro-Manager
demo config in a subprocess, so its ``QApplication`` does not change the
other tests.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from faro.microscope.pymmcore import ConfigGroupControl, ContinuousFocusControl
from faro.napari_launcher import (
    _illumination_dock_factory,
    _resolve_extra_controls,
    _resolve_light_properties,
)

REPO = Path(__file__).resolve().parents[1]


class StubCore:
    def __init__(self, autofocus="PFS"):
        self.props = {
            "LED": ["L.15 Intensity", "State"],
            "Camera": ["Binning", "Gain"],
            "PFS": ["Offset"],
        }
        self.groups = {"Binning": ["1x1", "2x2"], "Empty": []}
        self.autofocus = autofocus

    def getLoadedDevices(self):
        return tuple(self.props)

    def getDevicePropertyNames(self, device):
        return tuple(self.props[device])

    def getAvailableConfigGroups(self):
        return tuple(self.groups)

    def getAvailableConfigs(self, group):
        return tuple(self.groups[group])

    def getAutoFocusDevice(self):
        return self.autofocus


def test_light_properties_unchanged():
    rows = _resolve_light_properties(
        StubCore(),
        [("LED", "L.15", "LedEpi"), ("Camera", "Gain"), ("Nope", "X")],
    )
    assert rows == [
        ("LED", "L.15 Intensity", "LedEpi"),
        ("Camera", "Gain", "Camera::Gain"),
    ]


def test_extra_controls_resolve_in_order():
    rows = _resolve_extra_controls(
        StubCore(),
        [
            ConfigGroupControl("Binning"),
            ContinuousFocusControl(),
            ("Camera", "Gain", "Gain"),
            ConfigGroupControl("Binning", label="Bin"),
            ContinuousFocusControl("PFS"),
        ],
    )
    assert rows == [
        ("group", "Binning", "Binning"),
        ("focus", "PFS", "Continuous focus"),
        ("property", ("Camera", "Gain"), "Gain"),
        ("group", "Binning", "Bin"),
        ("focus", "PFS", "PFS"),
    ]


def test_missing_extra_controls_are_skipped_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="faro.napari_launcher"):
        rows = _resolve_extra_controls(
            StubCore(autofocus=""),
            [
                ConfigGroupControl("NoSuchGroup"),
                ConfigGroupControl("Empty"),
                ContinuousFocusControl(),
                ("Nope", "X"),
                ("Camera", "NoSuchProperty"),
                "not an entry",
            ],
        )
    assert rows == []
    assert len(caplog.records) == 6


def test_no_attributes_keep_upstream_widget():
    class Plain:
        pass

    assert _illumination_dock_factory(Plain()) is None


_WIDGET_SCRIPT = textwrap.dedent(
    """
    import threading

    from pymmcore_plus import CMMCorePlus
    from pymmcore_widgets import PresetsWidget, PropertyWidget
    from qtpy.QtWidgets import QApplication, QCheckBox, QLabel

    from faro.microscope.pymmcore import ConfigGroupControl, ContinuousFocusControl
    from faro.napari_launcher import _illumination_dock_factory

    app = QApplication([])
    core = CMMCorePlus()
    core.loadSystemConfiguration()

    class CountingLock:
        def __init__(self):
            self._lock = threading.Lock()
            self.n = 0

        def acquire(self, blocking=True):
            got = self._lock.acquire(blocking)
            self.n += got
            return got

        def release(self):
            self._lock.release()

        __enter__ = acquire

        def __exit__(self, *exc):
            self.release()

    class Scope:
        LIGHT_PROPERTIES = [("Camera", "Gain", "Gain")]
        EXTRA_CONTROLS = [
            ConfigGroupControl("Camera", label="Camera mode"),
            ContinuousFocusControl("Autofocus"),
            ("Camera", "Binning", "Binning"),
            ConfigGroupControl("Missing"),
        ]
        _pfs_lock = CountingLock()

    scope = Scope()
    widget = _illumination_dock_factory(scope)(parent=None, mmcore=core)
    labels = [w.text() for w in widget.findChildren(QLabel)]
    for text in ("Gain", "Camera mode", "Autofocus", "Binning"):
        assert text in labels, labels
    presets = widget.findChildren(PresetsWidget)
    assert len(presets) == 1
    assert len(widget.findChildren(PropertyWidget)) == 2  # Gain, Binning

    box = widget.findChildren(QCheckBox)[0]
    core.enableContinuousFocus(False)
    box.setChecked(True)
    assert core.isContinuousFocusEnabled()
    assert "locked" in [w.text() for w in widget.findChildren(QLabel)]
    box.setChecked(False)
    assert not core.isContinuousFocusEnabled()
    assert scope._pfs_lock.n > 0

    # A poll while the MDA thread holds the lock is skipped, not blocked.
    from faro.napari_launcher import _continuous_focus_toggle

    scope._pfs_lock.acquire()
    _continuous_focus_toggle(core, scope._pfs_lock)  # polls once
    scope._pfs_lock.release()
    print("OK")
    """
)


def test_dock_widget_offscreen():
    pytest.importorskip("pymmcore_widgets")
    from pymmcore_plus import find_micromanager

    if not find_micromanager():
        pytest.skip("Micro-Manager (demo config) is not installed")
    result = subprocess.run(
        [sys.executable, "-c", _WIDGET_SCRIPT],
        cwd=REPO,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr
