"""Launch napari-micromanager on a faro microscope class.

``python -m napari_micromanager -c some.cfg`` only loads a Micro-Manager
``.cfg``. Many scopes need Python-side setup that a cfg cannot express --
the DMD wake/blank loop and hold exposure, the ROI-follows-binning hook, the
laser keepalive, the per-scope MDA engine. This launcher builds the
microscope class first and hands its configured core to
``MainWindow(viewer, mmcore=...)``. The rest mirrors upstream's
``napari_micromanager.__main__``.

Scope names come from the ``faro.microscopes`` entry points of the installed
lab packages (see ``faro.microscope.registry``).

Examples
--------
    uv run faro-napari demo
    uv run faro-napari moench                      # needs faro-pertzlab
    uv run faro-napari moench --kwarg uncropped=True
    uv run faro-napari mypkg.scopes:MyScope --kwarg port=3

Inside the napari console, ``mic`` and ``mmcore`` are available, so e.g.
``mic.calibrate_dmd(CyanStim)`` can be run interactively.

The widgets in ``PERTZLAB_WIDGETS`` get a button in the plugin's toolbar. They
come from the pertzlab branch of pymmcore-widgets, installed with
``uv sync --extra pertzlab-widgets``. A scope picks a subset with
``NAPARI_WIDGETS`` (see ``PyMMCoreMicroscope``).
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import logging
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

from faro.microscope.registry import available_scopes, load_scope_class

logger = logging.getLogger(__name__)

# Widgets from the pertzlab pymmcore-widgets branch that get a button in the
# napari-micromanager toolbar: toolbar label -> (pymmcore_widgets class name,
# MDI6 icon name). Every scope gets all of them unless it lists the labels it
# wants in ``NAPARI_WIDGETS``.
PERTZLAB_WIDGETS = {
    "Stage Map": ("StageMapWidget", "map_marker_multiple"),
    "Continuous Focus": ("ContinuousFocusWidget", "focus_auto"),
}


def parse_kwarg(item: str) -> tuple[str, Any]:
    """Parse a ``key=value`` CLI pair; the value is Python-literal-parsed.

    ``uncropped=False`` (or ``false``) yields ``False``, not a truthy string.
    Values that are not Python literals stay strings, so unquoted Windows
    paths work.
    """
    key, sep, raw = item.partition("=")
    if not sep:
        raise SystemExit(f"--kwarg expects key=value, got {item!r}.")
    keywords = {"true": True, "false": False, "none": None}
    if raw.strip().lower() in keywords:
        return key.strip(), keywords[raw.strip().lower()]
    try:
        return key.strip(), ast.literal_eval(raw)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return key.strip(), raw


def _cache_snapped_image(core) -> None:
    """Let ``getImage()`` be called more than once per ``snapImage()``.

    napari-micromanager reads the snap buffer in its ``imageSnapped`` slot
    (``CoreViewerLink._image_snapped``), and the slot runs synchronously
    inside ``snapImage()`` on the psygnal backend. ``CMMCorePlus.snap()`` --
    what the Snap button calls -- then reads the buffer a second time. The
    Andor SDK3 and PVCAM adapters hand out each snapped frame only once, so
    the second read raises "Camera image buffer read failed" (the demo
    camera does not, which is why ``faro-napari demo`` never showed it).

    The first read of each camera channel after a snap is kept, and later
    callers get a copy of it until the next snap. While an MDA runs the cache
    is bypassed: the plugin does not read snaps then, and faro's engines keep
    the plain single-read path.
    """
    snap_image = core.snapImage
    get_image = core.getImage
    cache: dict[int, Any] = {}

    def snapImage(*args, **kwargs):
        cache.clear()
        return snap_image(*args, **kwargs)

    def getImage(numChannel=None, *, fix=True):
        if core.mda.is_running():
            return get_image(numChannel, fix=fix)
        # None means the first channel, so getImage() and getImage(0) (e.g.
        # from getTaggedImage) share one read. The raw frame is cached and
        # fixImage, which leaves a fixed image unchanged, is applied per call.
        channel = numChannel or 0
        if channel not in cache:
            cache[channel] = get_image(numChannel, fix=False)
        img = cache[channel].copy()
        return core.fixImage(img) if fix else img

    core.snapImage = snapImage
    core.getImage = getImage


def _stage_map_widget(*, parent, mmcore):
    """Build a ``StageMapWidget`` that shows the MDA widget's positions.

    The map has no position list of its own; it binds to an existing one. The
    MDA dock is opened first so there is a list to bind to. Without it the map
    would start empty until a list is picked from the map's toolbar.

    The plugin docks the returned widget as a tab next to the MDA dock, after
    this factory returns. So the MDA dock is moved below the map on the next
    pass of the event loop, and both are visible at the same time.
    """
    from pymmcore_widgets import StageMapWidget
    from qtpy.QtCore import QTimer

    parent._show_dock_widget("MDA")
    widget = StageMapWidget(parent=parent, mmcore=mmcore)
    widget.bindToFirstPositionTable()
    QTimer.singleShot(0, partial(_dock_mda_below, parent, widget))
    return widget


def _dock_mda_below(parent, widget) -> None:
    """Dock the MDA widget below the dock of ``widget``, not as a tab.

    ``splitDockWidget`` adds the second dock as a tab if the first one is in a
    tab group, and the plugin tabifies every dock it adds. So both docks are
    taken out of their tab groups first.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QDockWidget

    top = widget.parent()
    while top is not None and not isinstance(top, QDockWidget):
        top = top.parent()
    bottom = parent._dock_widgets.get("MDA")
    if top is None or bottom is None:
        logger.warning("Stage map or MDA dock not found; not docking them together.")
        return

    win = parent.viewer.window._qt_window
    for dock in (top, bottom):
        win.removeDockWidget(dock)
    win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, top)
    win.splitDockWidget(top, bottom, Qt.Orientation.Vertical)
    # removeDockWidget hides a dock.
    top.show()
    bottom.show()


def _continuous_focus_widget(mic, *, parent, mmcore):
    """Build a ``ContinuousFocusWidget`` that holds the scope's PFS lock.

    The widget polls the autofocus device every second from the GUI thread. On
    Moench, a PFS access while the MDA engine reloads the autofocus device
    crashes the process. So when the scope has a ``_pfs_lock``, every read and
    write of the widget takes it.
    """
    from pymmcore_widgets import ContinuousFocusWidget

    lock = getattr(mic, "_pfs_lock", None)
    if lock is None:
        return ContinuousFocusWidget(parent=parent, mmcore=mmcore)

    def locked(func):
        def call(*args):
            with lock:
                return func(*args)

        return call

    return ContinuousFocusWidget(
        parent=parent,
        mmcore=mmcore,
        enabled_source=locked(mmcore.isContinuousFocusEnabled),
        locked_source=locked(mmcore.isContinuousFocusLocked),
        set_enabled=locked(mmcore.enableContinuousFocus),
    )


def _scope_widgets(mic) -> dict[str, tuple[str, str]]:
    """The ``PERTZLAB_WIDGETS`` entries the scope asks for in ``NAPARI_WIDGETS``.

    None (or no attribute) means all of them. Labels that are not in
    ``PERTZLAB_WIDGETS`` are skipped with a warning.
    """
    wanted = getattr(mic, "NAPARI_WIDGETS", None)
    if wanted is None:
        return dict(PERTZLAB_WIDGETS)
    for label in wanted:
        if label not in PERTZLAB_WIDGETS:
            logger.warning(
                "NAPARI_WIDGETS: unknown widget %r. Known: %s",
                label,
                ", ".join(PERTZLAB_WIDGETS),
            )
    return {
        label: PERTZLAB_WIDGETS[label] for label in wanted if label in PERTZLAB_WIDGETS
    }


def _add_pertzlab_widgets(mic) -> None:
    """Add the scope's ``PERTZLAB_WIDGETS`` buttons to the plugin's toolbar.

    The plugin builds its toolbar buttons from its module-level
    ``DOCK_WIDGETS`` when ``MainWindow`` is created, so this must run before
    that. A widget is built the first time its button is clicked, as
    ``factory(parent=window, mmcore=core)``.

    Only the widgets in the scope's ``NAPARI_WIDGETS`` are added (all of them
    when it is None). Without the ``pertzlab-widgets`` extra,
    pymmcore_widgets has none of these widgets and the plugin runs as is. A
    widget missing from a pertzlab build is skipped with a warning.
    """
    widgets = _scope_widgets(mic)
    if not widgets:
        return

    import pymmcore_widgets
    from fonticon_mdi6 import MDI6
    from napari_micromanager._gui_objects._toolbar import DOCK_WIDGETS

    if not any(
        hasattr(pymmcore_widgets, class_name)
        for class_name, _ in PERTZLAB_WIDGETS.values()
    ):
        logger.info(
            "pymmcore-widgets is not the pertzlab build; no extra toolbar "
            "buttons (`uv sync --extra pertzlab-widgets` adds them)."
        )
        return

    factories = {
        "StageMapWidget": _stage_map_widget,
        "ContinuousFocusWidget": partial(_continuous_focus_widget, mic),
    }
    for label, (class_name, icon_name) in widgets.items():
        if not hasattr(pymmcore_widgets, class_name):
            logger.warning(
                "pymmcore_widgets has no %s. Skipping the %r button.",
                class_name,
                label,
            )
            continue
        factory = factories.get(class_name, getattr(pymmcore_widgets, class_name))
        DOCK_WIDGETS[label] = (factory, getattr(MDI6, icon_name))


def _hide_docks_on_close(win) -> None:
    """Make the close button of the plugin's dock widgets hide the dock.

    napari's close button removes the dock and deletes it. The plugin keeps
    each dock in ``win._dock_widgets`` and calls ``show()`` on it when its
    toolbar button is clicked again, which then raises "wrapped C/C++ object
    ... has been deleted". The plugin sets ``_close_btn = False`` to drop the
    button, but napari has built the title bar by then, so it stays.

    Hiding keeps the widget and its state, so a reopened MDA dock still has
    its sequence.
    """
    add_dock_widget = win._add_dock_widget

    def add_hiding_dock_widget(*args, **kwargs):
        dock = add_dock_widget(*args, **kwargs)
        # The title bar looks up destroyOnClose on each click.
        if hasattr(dock, "destroyOnClose"):
            dock.destroyOnClose = dock.hide
        return dock

    win._add_dock_widget = add_hiding_dock_widget


def _deliver_mda_handler_on_main_thread(win) -> None:
    """Run the slots of the plugin's MDA handler on the GUI thread, in order.

    The plugin's ``_NapariMDAHandler`` is written for the Qt signal backend,
    which delivers ``sequenceStarted``, ``frameReady`` and
    ``sequenceFinished`` on the GUI thread in the order they were sent. It is
    not a ``QObject``, so ``deliver_gui_slots_on_main_thread`` leaves its
    slots on the MDA thread. There, two things go wrong:

    - ``_on_mda_started`` is only posted to the GUI thread, but frames are
      queued on the MDA thread at once. When it runs, it replaces the frame
      queue, so the first frame is lost and stays blank in the layer.
    - ``_on_mda_finished`` resets ``viewer.dims`` from the MDA thread, which
      updates napari's widgets off the GUI thread.

    The handler's connections are made again with ``thread="main"``. faro's
    own frame callbacks are separate connections and stay on the MDA thread.
    Call this only when the core's signals are psygnal signals. A core
    installed later by the plugin (``set_core``) gets a new handler that
    keeps the plain connections.
    """
    handler = getattr(getattr(win, "_core_link", None), "_mda_handler", None)
    connections = getattr(handler, "_connections", None)
    if not connections:
        logger.warning("napari-micromanager MDA handler not found; not rerouted.")
        return
    for signal, slot in connections:
        signal.disconnect(slot)
        signal.connect(slot, thread="main")


def _resolve_property(core, loaded, device, name, label, exact=False):
    """Return ``(device, property, label)`` for one property entry, or None.

    A name matches a property exactly, else (unless ``exact``) as the prefix
    of one (for names an adapter builds at runtime). Without a label, ``device::property`` is
    shown. An unloaded device or an unmatched name is skipped with a warning.
    """
    if device not in loaded:
        logger.warning("NAPARI_CONTROLS: device %r is not loaded.", device)
        return None
    props = core.getDevicePropertyNames(device)
    if name in props:
        match = name
    elif exact:
        match = None
    else:
        match = next((p for p in props if p.startswith(name)), None)
    if match is None:
        logger.warning(
            "NAPARI_CONTROLS: %r has no property %r. Properties: %s",
            device,
            name,
            ", ".join(props),
        )
        return None
    return device, match, label or f"{device}::{match}"


def _resolve_controls(core, entries) -> list[tuple[str, Any, str]]:
    """Map ``NAPARI_CONTROLS`` entries to ``(kind, target, label)`` rows.

    ``kind`` is one of:

    - ``"heading"``: target None, label the heading text;
    - ``"property"``: target ``(device, property)``;
    - ``"group"``: target the config group;
    - ``"focus"``: target the autofocus device;
    - ``"device"``: target the ``DeviceControl``, label the device;
    - ``"toggle"``: target ``(device, property, on_value, off_value)``.

    A missing device, property, group (or a group without presets), a missing
    autofocus device or an unknown entry is skipped with a warning.
    """
    from faro.microscope.pymmcore import (
        ConfigGroupControl,
        ContinuousFocusControl,
        DeviceControl,
        PropertyToggle,
    )

    loaded = set(core.getLoadedDevices())
    rows = []
    for entry in entries:
        if isinstance(entry, str):
            rows.append(("heading", None, entry))
        elif isinstance(entry, ConfigGroupControl):
            groups = core.getAvailableConfigGroups()
            if entry.group not in groups:
                logger.warning(
                    "NAPARI_CONTROLS: no config group %r. Groups: %s",
                    entry.group,
                    ", ".join(groups),
                )
            elif not core.getAvailableConfigs(entry.group):
                logger.warning("NAPARI_CONTROLS: group %r has no presets.", entry.group)
            else:
                rows.append(("group", entry.group, entry.label or entry.group))
        elif isinstance(entry, ContinuousFocusControl):
            device = core.getAutoFocusDevice()
            if not device:
                logger.warning(
                    "NAPARI_CONTROLS: no autofocus device is set. Skipping %r.",
                    entry.label,
                )
            else:
                rows.append(("focus", device, entry.label))
        elif isinstance(entry, DeviceControl):
            if entry.device not in loaded:
                logger.warning(
                    "NAPARI_CONTROLS: device %r is not loaded.", entry.device
                )
            else:
                rows.append(("device", entry, entry.device))
        elif isinstance(entry, PropertyToggle):
            # Exact only: the check box writes, so a prefix must not pick
            # another property.
            prop = _resolve_property(
                core, loaded, entry.device, entry.property, entry.label, exact=True
            )
            if prop is not None:
                rows.append(
                    ("toggle", (*prop[:2], entry.on_value, entry.off_value), prop[2])
                )
        elif isinstance(entry, (tuple, list)) and len(entry) in (2, 3):
            device, name, *rest = entry
            prop = _resolve_property(
                core, loaded, device, name, rest[0] if rest else None
            )
            if prop is not None:
                rows.append(("property", prop[:2], prop[2]))
        else:
            logger.warning("NAPARI_CONTROLS: unknown entry %r.", entry)
    return rows


def _continuous_focus_toggle(mmcore, lock=None, parent=None):
    """An on/off check box for continuous focus with its locked state.

    The state is polled every second. ``lock`` (the scope's ``_pfs_lock``),
    if given, is held for every autofocus access. A poll that finds it taken
    is skipped, so the GUI thread does not wait for the MDA thread.
    """
    from contextlib import nullcontext

    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QWidget

    widget = QWidget(parent)
    box = QCheckBox("on")
    state = QLabel()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(box)
    layout.addWidget(state)
    layout.addStretch()

    def poll() -> None:
        if lock is not None and not lock.acquire(blocking=False):
            return
        try:
            enabled = mmcore.isContinuousFocusEnabled()
            in_focus = enabled and mmcore.isContinuousFocusLocked()
        except Exception as e:
            state.setText("error")
            state.setToolTip(str(e))
            return
        finally:
            if lock is not None:
                lock.release()
        box.blockSignals(True)
        box.setChecked(enabled)
        box.blockSignals(False)
        state.setText("locked" if in_focus else "not locked" if enabled else "")
        state.setToolTip("")

    def set_enabled(on: bool) -> None:
        try:
            with lock if lock is not None else nullcontext():
                mmcore.enableContinuousFocus(on)
        except Exception:
            logger.exception(
                "Could not turn continuous focus %s.", "on" if on else "off"
            )
        poll()

    box.toggled.connect(set_enabled)
    timer = QTimer(widget)
    timer.timeout.connect(poll)
    timer.start(1000)
    poll()
    return widget


def _same_value(a, b) -> bool:
    """Whether two property values are equal, as numbers when both are."""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def _property_toggle(mmcore, device, prop, on_value, off_value, label):
    """A check box that switches ``device::prop`` between on and ``off_value``.

    Unticking remembers the property's value and sets ``off_value``; ticking
    sets the remembered value, or ``on_value`` if there is none (or the
    property's value when the box is built, if on). Only an untick remembers:
    a slider dragged down to off passes small values that must not be kept.
    The box is ticked while the property is not at ``off_value`` and follows
    changes made elsewhere. A failed set is logged and the box goes back to
    the property's value.
    """
    from qtpy.QtWidgets import QCheckBox
    from superqt.utils import ensure_main_thread

    box = QCheckBox(label)
    box.setToolTip(
        f"{device}::{prop}: off {off_value}; on the last value, first {on_value}"
    )
    last_on = [on_value]

    def read():
        try:
            return mmcore.getProperty(device, prop)
        except Exception:
            logger.exception("Could not read %s::%s.", device, prop)
            return None

    def remember(value) -> None:
        if value is not None and not _same_value(value, off_value):
            last_on[0] = value

    def sync(value=None) -> None:
        if value is None:
            value = read()
            if value is None:
                return
        box.blockSignals(True)
        box.setChecked(not _same_value(value, off_value))
        box.blockSignals(False)

    def set_on(on: bool) -> None:
        if not on:
            remember(read())
        value = last_on[0] if on else off_value
        try:
            mmcore.setProperty(device, prop, value)
        except Exception:
            logger.exception("Could not set %s::%s to %r.", device, prop, value)
            sync()

    @ensure_main_thread
    def on_core_changed(dev, name, value) -> None:
        if dev == device and name == prop:
            with contextlib.suppress(RuntimeError):  # box already deleted
                sync(value)

    box.toggled.connect(set_on)
    mmcore.events.propertyChanged.connect(on_core_changed)
    box.destroyed.connect(
        lambda *_: mmcore.events.propertyChanged.disconnect(on_core_changed)
    )
    value = read()
    remember(value)
    sync(value)
    return box


def _illumination_dock_factory(mic):
    """Return the "Illumination Control" widget class for ``mic``, or None.

    The widget shows ``mic.NAPARI_CONTROLS`` top to bottom. None when the
    scope does not set it. The class is called as
    ``factory(parent=..., mmcore=...)``.
    """
    entries = list(getattr(mic, "NAPARI_CONTROLS", None) or [])
    if not entries:
        return None

    import re

    from pymmcore_widgets import PresetsWidget, PropertiesWidget, PropertyWidget
    from qtpy.QtGui import QFont
    from qtpy.QtWidgets import QLabel, QSizePolicy

    lock = getattr(mic, "_pfs_lock", None)

    class ControlsWidget(PropertiesWidget):
        """The rows of ``NAPARI_CONTROLS``.

        Rebuilt, like ``PropertiesWidget``, when a config is loaded.
        """

        def __init__(self, *, parent=None, mmcore=None):
            # PropertiesWidget.__init__ calls rebuild(), which reads this.
            self._wanted = entries
            super().__init__(parent=parent, mmcore=mmcore)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        def rebuild(self) -> None:
            layout = self.layout()
            while layout.count():
                layout.takeAt(0).widget().deleteLater()
            for row, (kind, target, label) in enumerate(
                _resolve_controls(self._mmc, self._wanted)
            ):
                if kind == "toggle":
                    # In the value column, under the property it switches.
                    box = _property_toggle(self._mmc, *target, label)
                    layout.addWidget(box, row, 1)
                    continue
                name = QLabel(label)
                if kind == "heading":
                    font = QFont(name.font())
                    font.setBold(True)
                    name.setFont(font)
                    layout.addWidget(name, row, 0, 1, 2)
                    continue
                if kind == "device":
                    # Every property of the device, each with its own label.
                    value = PropertiesWidget(
                        device_label=re.compile(f"^{re.escape(target.device)}$"),
                        is_read_only=None if target.include_read_only else False,
                        mmcore=self._mmc,
                    )
                    name.setToolTip(f"all properties of {target.device}")
                    layout.addWidget(name, row, 0)
                    layout.addWidget(value, row, 1)
                    continue
                if kind == "property":
                    name.setToolTip("::".join(target))
                    value = PropertyWidget(*target, mmcore=self._mmc)
                elif kind == "group":
                    name.setToolTip(f"config group {target}")
                    value = PresetsWidget(target, mmcore=self._mmc)
                else:
                    name.setToolTip(f"continuous focus of {target}")
                    value = _continuous_focus_toggle(self._mmc, lock)
                layout.addWidget(name, row, 0)
                layout.addWidget(value, row, 1)

    return ControlsWidget


def _replace_illumination_widget(mic) -> None:
    """Fill the "Illumination Control" dock from ``mic.NAPARI_CONTROLS``.

    napari-micromanager's widget shows every Intensity/Power/Level property
    of the config. A scope with ``NAPARI_CONTROLS`` gets exactly the rows it
    lists (see ``PyMMCoreMicroscope``). Like ``_add_pertzlab_widgets``, this
    must run before ``MainWindow`` is created.
    """
    from napari_micromanager._gui_objects._toolbar import DOCK_WIDGETS

    key = "Illumination Control"
    factory = _illumination_dock_factory(mic)
    if factory is not None:
        DOCK_WIDGETS[key] = (factory, DOCK_WIDGETS[key][1])


FARO_ICON = Path(__file__).parent / "resources" / "faro_icon.png"


def _set_faro_app_id() -> None:
    """Give the process its own Windows taskbar group.

    napari's ``get_qapp`` sets napari's app ID, so the taskbar groups the
    window with napari and shows napari's icon. Call after ``get_qapp`` and
    before the first window is created; a window keeps the ID it opened with.
    """
    import sys

    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("faro.napari")


def _set_faro_icon(viewer) -> None:
    """Show the faro icon on the napari window and in the taskbar.

    napari sets its own application icon when it creates the QApplication;
    this replaces it once the viewer exists.
    """
    from qtpy.QtGui import QIcon
    from qtpy.QtWidgets import QApplication

    icon = QIcon(str(FARO_ICON))
    QApplication.instance().setWindowIcon(icon)
    viewer.window._qt_window.setWindowIcon(icon)


def open_napari(mic, viewer=None):
    """Open napari-micromanager on the core of ``mic``, set up as in ``faro-napari``.

    For notebooks and scripts that build the microscope themselves::

        from faro.napari_launcher import open_napari

        mic = Ti2E()
        viewer, mm_widget = open_napari(mic)

    Compared with ``MainWindow(viewer, mmcore=mic.mmc)``, this also:

    - fills the "Illumination Control" dock from ``mic.NAPARI_CONTROLS``;
    - adds the toolbar buttons of ``mic.NAPARI_WIDGETS`` (pertzlab build of
      pymmcore-widgets);
    - runs the widgets' core-event slots on the GUI thread (psygnal backend);
    - lets ``getImage()`` be read more than once per ``snapImage()``;
    - makes the close button of the plugin's docks hide them.

    ``viewer`` is the napari viewer to dock into; a new one when None. Call
    it on the main thread. In Jupyter, napari enables the Qt event loop
    (``%gui qt``) when the viewer is created. Returns ``(viewer, window)``.
    The event loop is not started (``faro-napari`` calls ``napari.run()``)
    and nothing is torn down when the window closes.
    """
    import napari
    from napari.qt import get_qapp
    from napari_micromanager.main_window import MainWindow

    from faro.core.qt import deliver_gui_slots_on_main_thread

    # Once per core: a second call (a re-run notebook cell) must not wrap the
    # cache in another cache.
    if not getattr(mic.mmc, "_faro_snap_cache", False):
        _cache_snapped_image(mic.mmc)
        mic.mmc._faro_snap_cache = True

    # psygnal delivers the core's events on the thread that sends them (the
    # MMCore notification thread, the MDA thread). The widgets connect after
    # this call, so their slots run on the GUI thread; faro's callbacks do
    # not change.
    get_qapp()
    routed = deliver_gui_slots_on_main_thread(mic.mmc)
    if not routed:
        logger.info(
            "Core events (%s) are not routed to the GUI thread.",
            type(mic.mmc.events).__name__,
        )
    _add_pertzlab_widgets(mic)
    _replace_illumination_widget(mic)
    if viewer is None:
        viewer = napari.Viewer()
    win = MainWindow(viewer, mmcore=mic.mmc)
    _hide_docks_on_close(win)
    if routed:
        _deliver_mda_handler_on_main_thread(win)
    # No name=: a named dock of the napari-micromanager window crashes.
    dw = viewer.window.add_dock_widget(win, area="top")
    if hasattr(dw, "_close_btn"):
        dw._close_btn = False
    _set_faro_icon(viewer)
    return viewer, win


def main(args: Sequence[str] | None = None) -> None:
    """Build the microscope, then run napari-micromanager on its core."""
    parser = argparse.ArgumentParser(
        prog="faro-napari",
        description="Run napari-micromanager on a faro microscope class.",
        epilog=f"Registered scopes: {', '.join(sorted(available_scopes()))}",
    )
    parser.add_argument(
        "scope", help="registered scope name, or 'module.path:ClassName'"
    )
    parser.add_argument(
        "--kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="keyword argument for the microscope class (repeatable)",
    )
    parsed_args = parser.parse_args(args)

    try:
        cls = load_scope_class(parsed_args.scope)
    except LookupError as e:
        parser.error(str(e))
    mic = cls(**dict(parse_kwarg(item) for item in parsed_args.kwarg))

    # Imported only after the microscope is built: pymmcore_widgets sets
    # PYMM_SIGNALS_BACKEND=qt when unset, and faro needs the psygnal backend.
    import napari
    from napari.qt import get_qapp

    get_qapp()
    _set_faro_app_id()
    viewer, _ = open_napari(mic)
    viewer.update_console({"mic": mic, "mmcore": mic.mmc})
    napari.run()

    # The window is closed, but its widgets are still connected to the core.
    # Block the signals so the teardown does not call into deleted Qt objects.
    mic.mmc.events.block()
    mic.mmc.mda.events.block()
    # The plugin does not cancel an MDA on a core it does not own. Cancel it
    # and wait, so the devices are not unloaded while the engine still uses
    # them.
    mic.mmc.mda.cancel()
    while mic.mmc.mda.is_running():
        time.sleep(0.1)
    # Explicit teardown: the atexit hook of PyMMCoreMicroscope only holds a
    # weakref, and IPython's exit hook frees ``mic`` (the console namespace
    # held the last reference) before that hook runs.
    mic._teardown_hardware()


if __name__ == "__main__":
    main()
