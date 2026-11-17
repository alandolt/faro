"""Controls of the "Illumination Control" dock of ``faro-napari``.

The resolution of ``NAPARI_CONTROLS`` entries runs on a stub core. The widget
test builds the dock offscreen on the Micro-Manager demo config in a
subprocess, so its ``QApplication`` does not change the other tests.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from faro.microscope.pymmcore import (
    ConfigGroupControl,
    ContinuousFocusControl,
    DeviceControl,
)
from faro.napari_launcher import _illumination_dock_factory, _resolve_controls

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


def test_properties_match_by_name_or_prefix():
    rows = _resolve_controls(
        StubCore(),
        [("LED", "L.15", "LedEpi"), ("Camera", "Gain"), ("Nope", "X")],
    )
    assert rows == [
        ("property", ("LED", "L.15 Intensity"), "LedEpi"),
        ("property", ("Camera", "Gain"), "Camera::Gain"),
    ]


def test_controls_resolve_in_order():
    camera = DeviceControl("Camera")
    rows = _resolve_controls(
        StubCore(),
        [
            "Light",
            ("LED", "L.15", "LedEpi"),
            "Camera",
            ConfigGroupControl("Binning"),
            camera,
            ContinuousFocusControl(),
            ConfigGroupControl("Binning", label="Bin"),
            ContinuousFocusControl("PFS"),
        ],
    )
    assert rows == [
        ("heading", None, "Light"),
        ("property", ("LED", "L.15 Intensity"), "LedEpi"),
        ("heading", None, "Camera"),
        ("group", "Binning", "Binning"),
        ("device", camera, "Camera"),
        ("focus", "PFS", "Continuous focus"),
        ("group", "Binning", "Bin"),
        ("focus", "PFS", "PFS"),
    ]


def test_missing_controls_are_skipped_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="faro.napari_launcher"):
        rows = _resolve_controls(
            StubCore(autofocus=""),
            [
                ConfigGroupControl("NoSuchGroup"),
                ConfigGroupControl("Empty"),
                ContinuousFocusControl(),
                DeviceControl("NoSuchDevice"),
                ("Nope", "X"),
                ("Camera", "NoSuchProperty"),
                42,
            ],
        )
    assert rows == []
    assert len(caplog.records) == 7


def test_no_controls_keep_upstream_widget():
    class Plain:
        pass

    class Empty:
        NAPARI_CONTROLS = []

    assert _illumination_dock_factory(Plain()) is None
    assert _illumination_dock_factory(Empty()) is None


_WIDGET_SCRIPT = textwrap.dedent(
    """
    import threading

    from pymmcore_plus import CMMCorePlus
    from pymmcore_widgets import PresetsWidget, PropertyWidget
    from qtpy.QtWidgets import QApplication, QCheckBox, QLabel

    from faro.microscope.pymmcore import (
        ConfigGroupControl,
        ContinuousFocusControl,
        DeviceControl,
    )
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
        NAPARI_CONTROLS = [
            "Light",
            ("Camera", "Gain", "Gain"),
            "Camera",
            ConfigGroupControl("Camera", label="Camera mode"),
            ContinuousFocusControl("Autofocus"),
            ("Camera", "Binning", "Binning"),
            ConfigGroupControl("Missing"),
            DeviceControl("Dichroic"),
        ]
        _pfs_lock = CountingLock()

    scope = Scope()
    widget = _illumination_dock_factory(scope)(parent=None, mmcore=core)
    labels = [w.text() for w in widget.findChildren(QLabel)]
    for text in ("Light", "Gain", "Camera", "Camera mode", "Autofocus", "Binning",
                 "Dichroic"):
        assert text in labels, labels
    presets = widget.findChildren(PresetsWidget)
    assert len(presets) == 1
    writable = [
        p for p in core.getDevicePropertyNames("Dichroic")
        if not core.isPropertyReadOnly("Dichroic", p)
    ]
    assert writable
    # Gain, Binning, and one per writable Dichroic property.
    assert len(widget.findChildren(PropertyWidget)) == 2 + len(writable)

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
