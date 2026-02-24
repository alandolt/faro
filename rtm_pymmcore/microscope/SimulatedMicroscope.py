import os
import threading
import time as _time

import numpy as np
import pymmcore_plus
from pymmcore_plus.mda._engine import MDAEngine
from useq import MDAEvent

from rtm_pymmcore.controller import Analyzer, Controller
from rtm_pymmcore.data_structures import ImgType
from rtm_pymmcore.microscope.abstract_microscope import AbstractMicroscope

# Channel name → simulation rendering mode
_CHANNEL_MODE_MAP = {
    "Brightfield": 0,
    "Nucleus": 1,
    "Membrane": 2,
    "Stim": 0,  # stimulation events render as brightfield
}


# ---------------------------------------------------------------------------
# CMMCorePlus subclass – renders from simulation, syncs stage & channels
# ---------------------------------------------------------------------------


class _SimulatedCMMCorePlus(pymmcore_plus.CMMCorePlus):
    """CMMCorePlus whose image-retrieval methods return simulation frames.

    * Live view (``getLastImage``) and snap (``getImage``) return renders
      from the cell simulator instead of the demo camera noise.
    * The current XY-stage position is forwarded to the simulator's
      ``camera_offset`` so that panning in napari moves the simulated FOV.
    * The current channel preset is mapped to a rendering mode
      (brightfield / nucleus / membrane).
    * A background thread continuously advances the simulation physics
      so that cells move even when no image is acquired.
    """

    def __init__(self):
        super().__init__()
        self._simulator = None
        self._sim_lock = threading.Lock()
        self._sim_thread = None
        self._sim_running = False

    def _set_simulator(self, simulator):
        self._simulator = simulator
        self._start_simulation_thread()

    # -- background physics -------------------------------------------------

    def _start_simulation_thread(self):
        self._sim_running = True
        self._sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._sim_thread.start()

    def _stop_simulation_thread(self):
        self._sim_running = False
        if self._sim_thread is not None:
            self._sim_thread.join(timeout=2)

    def _simulation_loop(self):
        last = _time.perf_counter()
        while self._sim_running:
            now = _time.perf_counter()
            dt = (now - last) * 0.3  # scale for smoother physics
            last = now
            with self._sim_lock:
                self._simulator.update(dt=dt)
            _time.sleep(0.033)  # ~30 fps

    # -- tolerant property access -------------------------------------------

    def setProperty(self, label, propName, propValue):
        """Silently ignore properties for devices not loaded in the simulation."""
        try:
            if label not in self.getLoadedDevices():
                return
        except Exception:
            return
        super().setProperty(label, propName, propValue)

    # -- helpers ------------------------------------------------------------

    def _sync_stage_to_sim(self):
        """Read the demo XY-stage position and set it as camera_offset."""
        try:
            x = self.getXPosition()
            y = self.getYPosition()
            self._simulator.camera_offset = np.array([x, y])
        except Exception:
            pass

    def _current_render_mode(self) -> int:
        """Map the active channel preset to a rendering mode integer."""
        try:
            ch = self.getCurrentConfig("Channel")
            return _CHANNEL_MODE_MAP.get(ch, 0)
        except Exception:
            return 0

    def _render_sim_frame(self) -> np.ndarray:
        """Render one grayscale frame from the simulation."""
        import cv2

        self._sync_stage_to_sim()
        mode = self._current_render_mode()
        with self._sim_lock:
            # Set fluorescence values so the renderer draws the correct channel
            for cell in self._simulator._cells:
                self._simulator._update_cell_fluorescence(cell, mode)
            img = self._simulator.renderer.render_cells(
                self._simulator._cells,
                mode=mode,
                camera_offset=tuple(self._simulator.camera_offset),
                focal_plane=self._simulator.focal_plane,
            )
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -- snap (single-frame capture) ----------------------------------------

    def snap(self, numChannel=None, *, fix=True):
        if self._simulator is None:
            return super().snap(numChannel=numChannel, fix=fix)
        self.snapImage()  # trigger demo camera for timing / signals
        img = self._render_sim_frame()
        self.events.imageSnapped.emit(img)
        return img

    def getImage(self, *args, **kwargs):
        if self._simulator is None:
            return super().getImage(*args, **kwargs)
        return self._render_sim_frame()

    # -- continuous acquisition (live view) ---------------------------------

    def getLastImage(self, *args, **kwargs):
        if self._simulator is None:
            return super().getLastImage(*args, **kwargs)
        return self._render_sim_frame()

    def popNextImage(self, *args, **kwargs):
        if self._simulator is None:
            return super().popNextImage(*args, **kwargs)
        return self._render_sim_frame()


# ---------------------------------------------------------------------------
# Pass-through DMD (identity affine transform)
# ---------------------------------------------------------------------------


class _SimulatedDMD:
    """Fake DMD whose affine_transform is an identity / resize.

    The simulation camera space *is* the DMD space, so no geometric
    correction is needed.  If the mask dimensions differ from the
    viewport the mask is nearest-neighbor resized.
    """

    def __init__(self, width: int, height: int):
        self.name = "SimSLM"
        self.width = width
        self.height = height
        self.affine = np.eye(3)  # "already calibrated"

    def affine_transform(self, img: np.ndarray) -> np.ndarray:
        import cv2

        img = np.asarray(img, dtype=np.uint8)
        if img.shape[0] != self.height or img.shape[1] != self.width:
            img = cv2.resize(
                img,
                (self.width, self.height),
                interpolation=cv2.INTER_NEAREST,
            )
        if img.max() == 1:
            img = img * 255
        return img

    def calibrate(self, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Tolerant MDA engine – accepts any channel / property, intercepts SLM masks
# ---------------------------------------------------------------------------


class _SimulatedMDAEngine(MDAEngine):
    """MDA engine that silently ignores missing channels / device properties
    and forwards SLM images to the cell simulator.

    The demo configuration does not contain the channel presets or devices
    defined in a real experiment (e.g. ``CyanStim``, ``TTL_ERK``, ``Spectra``).
    This engine wraps every hardware-facing call with error handling so that
    the acquisition loop runs without crashing, while ``frameReady`` still
    fires for every event.

    When an ``SLMImage`` is attached to an event the mask is applied to the
    simulator's optogenetic stimulation API instead of being sent to a
    (non-existent) hardware SLM.
    """

    def _set_event_slm_image(self, event: MDAEvent) -> None:
        """Override: apply SLM mask to simulator instead of hardware."""
        sim = getattr(self.mmcore, "_simulator", None)
        if sim is not None and event.slm_image is not None:
            mask = np.asarray(event.slm_image.data)
            mask = mask > 0  # must be boolean for OptogeneticCell.stimulate
            lock = getattr(self.mmcore, "_sim_lock", None)
            if lock is not None:
                with lock:
                    sim.apply_optogenetic_stimulator(mask)
            else:
                sim.apply_optogenetic_stimulator(mask)

    def _exec_event_slm_image(self, img) -> None:
        """Override: no-op – SLM display handled by _set_event_slm_image."""
        pass

    def setup_single_event(self, event: MDAEvent) -> None:
        # XY position – works with the demo XYStage
        self._set_event_xy_position(event)

        # Z position
        if event.z_pos is not None:
            try:
                self._set_event_z(event)
            except Exception:
                pass

        # SLM image – handled by _set_event_slm_image override
        if event.slm_image is not None:
            self._set_event_slm_image(event)

        # Channel – config may not exist in demo
        try:
            self._set_event_channel(event)
        except Exception:
            pass

        # Exposure
        if event.exposure is not None:
            try:
                self.mmcore.setExposure(event.exposure)
            except Exception:
                pass

        # Device properties (e.g. laser power) – devices may not exist
        if event.properties is not None:
            for dev, prop, value in event.properties:
                try:
                    self.mmcore.setProperty(dev, prop, value)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Controller that replaces camera frames with simulation renders
# ---------------------------------------------------------------------------


class _SimulationController(Controller):
    """Controller whose ``_on_frame_ready`` renders from the cell simulator.

    Each imaging channel is rendered with a configurable rendering mode so
    that multi-channel experiments produce distinct synthetic images.
    Stimulation events are handled by the MDA engine (SLM mask applied
    directly to the simulator).  The simulation physics are advanced once
    per timestep via the background thread.
    """

    def __init__(
        self,
        analyzer,
        mmc,
        queue,
        use_autofocus_event,
        simulator,
        channel_rendering_modes,
        dmd=None,
    ):
        super().__init__(
            analyzer,
            mmc,
            queue,
            use_autofocus_event,
            dmd=dmd,
        )
        self._sim = simulator
        self._channel_rendering_modes = channel_rendering_modes
        self._last_timestep = -1

    # -- rendering helpers --------------------------------------------------

    def _render_channel(self, rendering_mode: int) -> np.ndarray:
        """Render one channel image via the simulator's renderer."""
        import cv2

        lock = getattr(self._mmc, "_sim_lock", None)
        if lock is not None:
            with lock:
                for cell in self._sim._cells:
                    self._sim._update_cell_fluorescence(cell, rendering_mode)
                img = self._sim.renderer.render_cells(
                    self._sim._cells,
                    mode=rendering_mode,
                    camera_offset=tuple(self._sim.camera_offset),
                    focal_plane=self._sim.focal_plane,
                )
        else:
            for cell in self._sim._cells:
                self._sim._update_cell_fluorescence(cell, rendering_mode)
            img = self._sim.renderer.render_cells(
                self._sim._cells,
                mode=rendering_mode,
                camera_offset=tuple(self._sim.camera_offset),
                focal_plane=self._sim.focal_plane,
            )
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -- core override ------------------------------------------------------

    def _on_frame_ready(self, img: np.ndarray, event: MDAEvent) -> None:
        metadata = event.metadata

        # Stimulation events – mask already applied by MDA engine
        if metadata["img_type"] == ImgType.IMG_STIM:
            return

        # Imaging events (IMG_RAW / IMG_OPTOCHECK)
        # Move the virtual stage so the renderer shows the correct FOV
        fov_x = metadata.get("fov_x", 0)
        fov_y = metadata.get("fov_y", 0)
        lock = getattr(self._mmc, "_sim_lock", None)
        if lock is not None:
            with lock:
                self._sim.camera_offset = np.array([fov_x, fov_y])
        else:
            self._sim.camera_offset = np.array([fov_x, fov_y])

        # Render the channel corresponding to the current buffer position
        channel_idx = len(self._frame_buffer)
        rendering_mode = self._channel_rendering_modes.get(channel_idx, 0)
        sim_img = self._render_channel(rendering_mode)
        self._frame_buffer.append(sim_img)

        if metadata["last_channel"]:
            frame_complete = np.stack(self._frame_buffer, axis=-1)
            frame_complete = np.moveaxis(frame_complete, -1, 0)
            self._frame_buffer = []

            # Mark timestep (physics run continuously in background thread)
            timestep = metadata.get("timestep", -1)
            if timestep != self._last_timestep:
                self._last_timestep = timestep

            self._results = self._analyzer.run(frame_complete, event)


# ---------------------------------------------------------------------------
# Public microscope class
# ---------------------------------------------------------------------------


class SimulatedMicroscope(AbstractMicroscope):
    """Microscope backed by *microscope-cell-simulation* instead of real hardware.

    Drop-in replacement for any concrete microscope class (``Jungfrau``,
    ``MMDemo``, …).  A Micro-Manager demo configuration drives the MDA
    engine while every camera frame is replaced by a simulation-rendered
    image.

    Key simulation features
    -----------------------
    * **Live view**: napari-micromanager shows the cell simulation (not
      demo camera noise).  Cells move continuously via a background thread.
    * **Stage movement**: panning the XY stage in the GUI moves the
      simulated field of view across the cell culture.
    * **Channels**: three channel presets (Brightfield / Nucleus / Membrane)
      map to rendering modes 0 / 1 / 2.
    * **Optogenetic stimulation**: a pass-through DMD forwards stimulation
      masks to the simulator's ``apply_optogenetic_stimulator()`` API.

    Install the simulation extra before use::

        pip install rtm-pymmcore[sim]

    Parameters
    ----------
    micromanager_path : str
        Path to the Micro-Manager installation (must contain
        ``MMConfig_demo.cfg``).
    sim_width, sim_height : int
        Size of the simulated cell culture area (pixels).
    nb_cells : int
        Number of cells to seed in the simulation.
    cell_type : str
        One of ``"optogenetic"``, ``"drug"``, ``"normal"``, ``"mixed"``.
    viewport_width, viewport_height : int
        Size of the rendered image (camera field of view).
    base_radius : float
        Base cell radius in simulation units.
    rng_seed : int
        Random seed for reproducibility.
    concentration : float
        Initial drug concentration (only relevant for ``cell_type="drug"``).
    drug_type : str
        Drug effect type: ``"growth"``, ``"mobility"``, or ``"apoptosis"``.
    channel_rendering_modes : dict[int, int] | None
        Mapping from channel index to the renderer's ``mode``.
        Defaults to ``{0: 1, 1: 2}`` (channel 0 → nucleus fluorescence,
        channel 1 → membrane fluorescence).  Available modes:

        ===== ========================
        Mode  Description
        ===== ========================
        0     Brightfield
        1     Nucleus fluorescence
        2     Membrane fluorescence
        ===== ========================
    brownian_d : float
        Brownian motion diffusion coefficient controlling cell migration
        speed.  Lower values yield slower migration.  Default is ``5.0``.
    """

    CHANNEL_GROUP = "Channel"
    USE_AUTOFOCUS_EVENT = False
    USE_ONLY_PFS = False
    SET_ROI_REQUIRED = False

    def __init__(
        self,
        micromanager_path: str = "C:\\Program Files\\Micro-Manager-2.0",
        sim_width: int = 1500,
        sim_height: int = 1500,
        nb_cells: int = 120,
        cell_type: str = "optogenetic",
        viewport_width: int = 512,
        viewport_height: int = 512,
        base_radius: float = 20.0,
        rng_seed: int = 0,
        concentration: float = 0.01,
        drug_type: str = "growth",
        channel_rendering_modes: dict = None,
        brownian_d: float = 5.0,
    ):
        super().__init__()

        # -- Micro-Manager demo backend (needed for the MDA engine) ----------
        self.micromanager_path = micromanager_path
        pymmcore_plus.use_micromanager(self.micromanager_path)
        self.micromanager_config = os.path.join(
            self.micromanager_path, "MMConfig_demo.cfg"
        )
        self.mmc = _SimulatedCMMCorePlus()

        # -- Cell simulation -------------------------------------------------
        from rtm_pymmcore.microscope_simulation.microscope_sim_optimized import (
            MicroscopeSimOptmized,
        )

        self.simulator = MicroscopeSimOptmized(
            width=sim_width,
            height=sim_height,
            nb_cells=nb_cells,
            cell_type=cell_type,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            base_radius=base_radius,
            rng_seed=rng_seed,
            concentration=concentration,
            drug_type=drug_type,
            brownian_d=brownian_d,
        )
        self.mmc._set_simulator(self.simulator)

        self.channel_rendering_modes = channel_rendering_modes or {0: 1, 1: 2}

        # -- Pass-through DMD for stimulation --------------------------------
        self.dmd = _SimulatedDMD(viewport_width, viewport_height)

        self.init_scope()

    # ------------------------------------------------------------------

    def init_scope(self):
        """Load demo config, define simulation channels, centre the stage."""
        self.mmc.loadSystemConfiguration(self.micromanager_config)
        self._setup_simulation_channels()
        self.mmc.setChannelGroup(self.CHANNEL_GROUP)
        self.register_engine()

        # Centre the viewport in the simulation world
        cx = (self.simulator.width - self.simulator.viewport_width) / 2
        cy = (self.simulator.height - self.simulator.viewport_height) / 2
        try:
            self.mmc.setXYPosition(cx, cy)
            self.mmc.waitForDevice(self.mmc.getXYStageDevice())
        except Exception:
            pass

    def _setup_simulation_channels(self):
        """Replace demo channel presets with simulation rendering modes."""
        group = self.CHANNEL_GROUP
        # Clear existing presets
        if group in self.mmc.getAvailableConfigGroups():
            for preset in list(self.mmc.getAvailableConfigs(group)):
                self.mmc.deleteConfig(group, preset)
        else:
            self.mmc.defineConfigGroup(group)

        # Define new presets using the demo Emission filter wheel
        self.mmc.defineConfig(group, "Brightfield", "Emission", "State", "1")
        self.mmc.defineConfig(group, "Nucleus", "Emission", "State", "2")
        self.mmc.defineConfig(group, "Membrane", "Emission", "State", "3")
        self.mmc.defineConfig(group, "Stim", "Emission", "State", "4")

        # Start in brightfield
        self.mmc.setConfig(group, "Brightfield")

    def register_engine(self, force: bool = False) -> None:
        """Register the tolerant MDA engine that accepts any channel/property."""
        if hasattr(self, "engine") and self.engine is not None and not force:
            return
        self.engine = _SimulatedMDAEngine(self.mmc)
        self.mmc.mda.set_engine(self.engine)

    def calibrate_dmd(self, **kwargs):
        """No-op – the simulated DMD needs no calibration."""
        pass

    def run_experiment(self, df_acquire):
        """Run an acquisition using the cell simulation as image source."""
        self.analyzer = Analyzer(self.pipeline)
        self.register_engine()
        self.controller = _SimulationController(
            self.analyzer,
            self.mmc,
            self.queue,
            self.USE_AUTOFOCUS_EVENT,
            simulator=self.simulator,
            channel_rendering_modes=self.channel_rendering_modes,
            dmd=self.dmd,
        )
        pymmcore_plus.configure_logging(stderr_level="WARNING")
        self.controller.run(df_acquire=df_acquire)

    def post_experiment(self):
        """No-op – nothing to clean up for the simulation."""
        pass
