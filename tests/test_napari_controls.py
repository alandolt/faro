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
    PropertyToggle,
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
            PropertyToggle("LED", "L.15 Intensity", 50, "LED on"),
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
        ("toggle", ("LED", "L.15 Intensity", 50, 0), "LED on"),
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
                PropertyToggle("Nope", "X", 1, "x"),
                PropertyToggle("Camera", "NoSuchProperty", 1, "x"),
                PropertyToggle("LED", "L.15", 1, "prefix only"),
                42,
            ],
        )
    assert rows == []
    assert len(caplog.records) == 10


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
        PropertyToggle,
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
            PropertyToggle("Camera", "Gain", 3, "Gain on"),
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

    boxes = {b.text(): b for b in widget.findChildren(QCheckBox)}
    gain = boxes["Gain on"]
    # The demo camera starts at Gain 0 (off), so the first tick sets on_value.
    core.setProperty("Camera", "Gain", 0)
    assert not gain.isChecked()
    gain.setChecked(True)
    assert core.getProperty("Camera", "Gain") == "3"
    # Changed while on (e.g. with the slider): the box stays ticked and
    # remembers the value for the next tick.
    core.setProperty("Camera", "Gain", 5)
    assert gain.isChecked()
    gain.setChecked(False)
    assert core.getProperty("Camera", "Gain") == "0"
    gain.setChecked(True)
    assert core.getProperty("Camera", "Gain") == "5"
    # A slider dragged down to off passes small values; they are not kept.
    for value in (4, 3, 2, 1, 0):
        core.setProperty("Camera", "Gain", value)
    assert not gain.isChecked()
    gain.setChecked(True)
    assert core.getProperty("Camera", "Gain") == "5"

    box = boxes["on"]
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


_CLOSE_SCRIPT = textwrap.dedent(
    """
    from napari._qt.widgets.qt_viewer_dock_widget import QtViewerDockWidget
    from qtpy import sip
    from qtpy.QtWidgets import QApplication, QWidget

    from faro.napari_launcher import _hide_docks_on_close

    app = QApplication([])

    class QtViewer:
        # destroyOnClose only removes the dock from a napari Viewer.
        viewer = None

    class Window:
        def _add_dock_widget(self, widget, name, floating=False, tabify=False):
            return QtViewerDockWidget(self.qt_viewer, widget, name=name)

    win = Window()
    win.qt_viewer = QtViewer()
    _hide_docks_on_close(win)
    dock = win._add_dock_widget(QWidget(), "Illumination Control")
    dock.show()
    dock.title.close_button.click()
    app.processEvents()
    assert not sip.isdeleted(dock)
    assert not dock.isVisible(), "the close button did not hide the dock"
    dock.show()
    assert dock.isVisible()
    print("OK")
    """
)


def test_dock_close_button_hides_the_dock():
    # napari's close button deletes the dock, but napari-micromanager keeps it
    # and shows it again on the next toolbar click.
    pytest.importorskip("napari")
    result = subprocess.run(
        [sys.executable, "-c", _CLOSE_SCRIPT],
        cwd=REPO,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr


def test_mda_handler_slots_run_on_main_thread_in_order():
    """Slots of the plugin's MDA handler run on the main thread, in order.

    With psygnal, MDA events are sent on the MDA thread. The plugin's handler
    expects them on the GUI thread in order, as the Qt backend delivers them;
    otherwise the first frame is lost (see
    ``_deliver_mda_handler_on_main_thread``).
    """
    import threading
    from types import SimpleNamespace

    from psygnal import Signal, SignalGroup, emit_queued

    from faro.napari_launcher import _deliver_mda_handler_on_main_thread

    class Events(SignalGroup):
        sequenceStarted = Signal(object)
        frameReady = Signal(object, object)
        sequenceFinished = Signal(object)

    events = Events()
    calls = []

    def record(name):
        def slot(*args):
            calls.append((name, threading.current_thread() is threading.main_thread()))

        return slot

    connections = [
        (events.frameReady, record("frame")),
        (events.sequenceStarted, record("started")),
        (events.sequenceFinished, record("finished")),
    ]
    for signal, slot in connections:
        signal.connect(slot)
    handler = SimpleNamespace(_connections=connections)
    win = SimpleNamespace(_core_link=SimpleNamespace(_mda_handler=handler))

    _deliver_mda_handler_on_main_thread(win)

    def run():
        events.sequenceStarted.emit("seq")
        events.frameReady.emit("img", "event")
        events.sequenceFinished.emit("seq")

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    assert calls == []
    emit_queued()
    assert calls == [("started", True), ("frame", True), ("finished", True)]
