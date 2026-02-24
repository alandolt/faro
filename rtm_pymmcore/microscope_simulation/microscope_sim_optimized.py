"""Optimized microscope simulation."""

import numpy as np
import time
from typing import Optional, List, Union, Literal
from .cell_optogenetic import OptogeneticCell
from .cell_drug import DrugResponseCell
from .renderer import Renderer
from .spatial_grid import SpatialGrid
from .cell_base import CellBase, update_all_cells_parallel, check_collision
from .cell_normal import NormalCell
import cv2


class MicroscopeSimOptmized:
    """Optmized microscope simulation with modular cell types."""

    def __init__(
        self,
        width: int = 1500,
        height: int = 1500,
        nb_cells: int = 120,
        cell_type: str = "optogenetic",
        viewport_width: int = 512,
        viewport_height: int = 512,
        base_radius: float = 20.0,
        rng_seed: int = 0,
        cell_mix: dict = None,
        concentration: float = 0.01,
        drug_type: Literal["growth", "mobility", "apoptosis"] = "growth",
        brownian_d: float = 5.0,
    ):
        self.width = width
        self.height = height
        self.nb_cells = nb_cells
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.base_radius = base_radius
        self.cell_type = cell_type
        self.cell_mix = cell_mix
        self.concentration = concentration
        self.drug_type = drug_type
        self.brownian_d = brownian_d
        # Initialize components
        self.renderer = Renderer(viewport_width, viewport_height)
        self.spatial_grid = SpatialGrid(width, height, base_radius * 3)

        # Camera settings
        self.camera_offset = np.array([0.0, 0.0])
        self.focal_plane = 0.0

        # State tracking
        self.state_devices = {}
        self.mode = 0
        self._last_time = time.perf_counter()
        self._objectif_dict = {"10x": 10, "20x": 20, "40x": 40}

        # Initialize cells based on type
        self._rng = np.random.RandomState(rng_seed)

        # Create cell objects
        self._cells = self._create_cells()
        self._init_numpy_arrays()

        # Add objective property
        self.current_objectiv: int = 10

    def _create_cells(
        self,
    ) -> List[Union[OptogeneticCell, DrugResponseCell, NormalCell]]:
        """Create cells of specific type."""
        cells = []
        if self.cell_mix is None:
            for i in range(self.nb_cells):
                seed = self._rng.randint(0, 10000)

                if self.cell_type == "optogenetic":
                    cell = OptogeneticCell(
                        self.width,
                        self.height,
                        self.base_radius,
                        vertices=24,
                        seed=seed,
                    )
                elif self.cell_type == "drug":
                    cell = DrugResponseCell(
                        self.width,
                        self.height,
                        self.base_radius,
                        vertices=24,
                        seed=seed,
                    )
                elif self.cell_type == "normal":
                    cell = NormalCell(
                        self.width,
                        self.height,
                        self.base_radius,
                        vertices=24,
                        seed=seed,
                    )
                elif self.cell_type == "mixed":
                    cell_choice = self._rng.choice(
                        ["normal", "optogenetic"], p=[0.7, 0.3]
                    )
                    if cell_choice == "normal":
                        cell = NormalCell(
                            self.width,
                            self.height,
                            self.base_radius,
                            vertices=24,
                            seed=seed,
                        )
                    else:
                        cell = OptogeneticCell(
                            self.width,
                            self.height,
                            self.base_radius,
                            vertices=24,
                            seed=seed,
                        )
                else:
                    raise ValueError(f"Unknown cell type: {self.cell_type}")

                cells.append(cell)
        else:
            for cell_type, count in self.cell_mix.items():
                for _ in range(count):
                    seed = self._rng.randint(0, 10000)

                    if cell_type == "normal":
                        cell = NormalCell(
                            self.width,
                            self.height,
                            self.base_radius,
                            vertices=24,
                            seed=seed,
                        )
                    elif cell_type == "optogenetic":
                        cell = OptogeneticCell(
                            self.width,
                            self.height,
                            self.base_radius,
                            vertices=24,
                            seed=seed,
                        )
                    elif cell_type == "drug":
                        cell = DrugResponseCell(
                            self.width,
                            self.height,
                            self.base_radius,
                            vertices=24,
                            seed=seed,
                        )
                    else:
                        continue

                    cells.append(cell)

            self._rng.shuffle(cells)
            self.nb_cells = len(cells)

        return cells

    def _init_numpy_arrays(self):
        """Initialize numpy arrays from cell objects for fast physics."""
        self.centers = np.zeros((self.nb_cells, 2), dtype=np.float64)
        self.velocities = np.zeros((self.nb_cells, 2), dtype=np.float64)
        self.radii = np.zeros((self.nb_cells, 24), dtype=np.float64)
        self.base_radii = np.zeros(self.nb_cells, dtype=np.float64)
        self.areas = np.zeros(self.nb_cells, dtype=np.float64)
        self.angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)

        for i, cell in enumerate(self._cells):
            self.centers[i] = cell.center
            self.velocities[i] = cell.vel
            self.radii[i] = cell.r
            self.base_radii[i] = cell.base_r
            self.areas[i] = cell.area0

    def _sync_arrays_to_cells(self):
        """Sync numpy arrays back to cell objects."""
        for i, cell in enumerate(self._cells):
            cell.center = self.centers[i].copy()
            cell.vel = self.velocities[i].copy()
            cell.r = self.radii[i].copy()

    def update(self, dt: float = 0.016) -> None:
        """Update simulation using fast parallel physics and cell objects."""
        update_all_cells_parallel(
            self.centers,
            self.velocities,
            self.radii,
            self.angles,
            self.base_radii,
            self.areas,
            self.width,
            self.height,
            dt,
            brownian_d=self.brownian_d,
        )

        self._sync_arrays_to_cells()

        for cell in self._cells:
            if hasattr(cell, "update_behavior"):
                cell.update_behavior(dt)

        self._handle_collisions_with_spatial_grid()

    def _handle_collisions_with_spatial_grid(self) -> None:
        """Use SpatialGrid for collision detection."""
        self.spatial_grid.clear()

        for i, cell in enumerate(self._cells):
            self.spatial_grid.add_cell(cell, i)

        checked_pairs = set()
        for i, cell in enumerate(self._cells):
            potential = self.spatial_grid.get_potential_collisions(cell, i)

            for j in potential:
                if (i, j) not in checked_pairs and (j, i) not in checked_pairs:
                    if cell.check_collision(self._cells[j]):
                        if isinstance(cell, NormalCell):
                            cell.respond_to_collision(self._cells[j])
                        if isinstance(self._cells[j], NormalCell):
                            self._cells[j].respond_to_collision(cell)

                    checked_pairs.add((i, j))

    def apply_optogenetic_stimulator(self, mask: np.ndarray) -> None:
        """Apply optogenetic stimulation to cells."""
        if self.cell_type != "optogenetic":
            return

        for cell in self._cells:
            if isinstance(cell, OptogeneticCell):
                cell.stimulate(mask, camera_offset=self.camera_offset)

        for i, cell in enumerate(self._cells):
            self.radii[i] = cell.r
            self.velocities[i] = cell.vel

    def apply_drug(self, concentration: float, drug_type: str = "growth") -> None:
        """Apply drug to all cells."""
        if self.cell_type != "drug":
            return

        for cell in self._cells:
            if isinstance(cell, DrugResponseCell):
                cell.apply_drug(concentration, drug_type)

        for i, cell in enumerate(self._cells):
            self.base_radii[i] = cell.base_r
            self.radii[i] = cell.r
            self.areas[i] = cell.area0

    def snap_frame(
        self, mask: Optional[np.ndarray] = None, intensity: float = 1.0, exposure=1.0
    ) -> np.ndarray:
        """Capture a frame with optional stimulation."""
        now = time.perf_counter()
        dt = (now - self._last_time) * 0.3
        self._last_time = now
        self.update(dt)

        if mask is not None and self.cell_type == "optogenetic":
            self.apply_optogenetic_stimulator(mask)
        elif mask is not None and self.cell_type == "drug":
            if np.any(mask):
                self.apply_drug(self.concentration, self.drug_type)

        self._update_mode()
        self._update_objectif()

        for cell in self._cells:
            self._update_cell_fluorescence(cell, self.mode)

        img = self.renderer.render_cells(
            self._cells, self.mode, tuple(self.camera_offset), self.focal_plane
        )

        img = (
            (img.astype(np.float32) * intensity * exposure)
            .clip(0, 255)
            .astype(np.uint8)
        )

        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _update_mode(self) -> None:
        """Update rendering mode based on state devices."""
        if "Filter Wheel" not in self.state_devices or "LED" not in self.state_devices:
            self.mode = 0
            return
        filter_label = self.state_devices["Filter Wheel"]["label"]
        led_label = self.state_devices["LED"]["label"]

        if filter_label == "mScarlet3(569/582)" and led_label == "ORANGE":
            self.mode = 1
        elif filter_label == "miRFP670(642/670)" and led_label == "RED":
            self.mode = 2
        else:
            self.mode = 0

    def _update_objectif(self) -> None:
        """Update the objective used based on the state device."""
        if "Objective" not in self.state_devices:
            return

        objectif_label = self.state_devices["Objective"]["label"]

        if objectif_label not in ("10x", "20x", "40x"):
            return

        self.current_objectiv = self._objectif_dict[objectif_label]

        if self.current_objectiv == 10:
            dof = 6.0
        elif self.current_objectiv == 20:
            dof = 4.0
        else:
            dof = 1.5

        self.renderer.set_objective(self.current_objectiv, dof)

    def _update_cell_fluorescence(self, cell: CellBase, mode: int) -> None:
        """Update cell fluorescence based on mode."""
        if isinstance(cell, NormalCell):
            if mode == 1:
                if not cell.has_nucleus_marker:
                    cell.nucleus_fluorescence = 0.0
            elif mode == 2:
                if not cell.has_membrane_marker:
                    cell.membrane_fluorescence[:] = 0.0

        if mode == 0:
            cell.nucleus_fluorescence = 0.0
            cell.membrane_fluorescence[:] = 0.0
        elif mode == 1:
            cell.nucleus_fluorescence = 1.0
        elif mode == 2:
            cell.membrane_fluorescence[:] = 1.0

    def get_visible_cells(self) -> List[CellBase]:
        """Get cells visible in current viewport."""
        return self.renderer._get_visible_cells(self._cells, tuple(self.camera_offset))

    def set_focal_plane(self, z: float) -> None:
        """Set focal plane position."""
        self.focal_plane = z

    def reset(self) -> None:
        """Reset simulation."""
        self._cells = self._create_cells()
        self._init_numpy_arrays()
        self._last_time = time.perf_counter()
