"""Protrusion cell behaviour module.

A cell that responds to optogenetic stimulation with persistent area
increase (protrusions) instead of migration.  The magnitude of the
response depends on:

* **stimulation position** – vertices further from the cell centre
  receive a larger protrusion gain (edge stimulation > centre
  stimulation).
* **opto-RTK expression** – a random per-cell scalar drawn from a
  Beta(2, 5) distribution; higher expression → stronger response.
"""

import numpy as np
from .cell_optogenetic import OptogeneticCell
from .cell_base import polygon_area


class ProtrusionCell(OptogeneticCell):

    def __init__(self, *args, protrusion_gain: float = 0.03, **kwargs):
        super().__init__(*args, protrusion_gain=protrusion_gain, impulse=0.0, **kwargs)

        # --- No migration ---
        self.friction = 100.0
        self.brownian_d = 0.0

        # --- Slow shape relaxation so protrusions persist ---
        self.curvature_relax = 0.05
        self.radial_relax = 0.03

        # --- Per-cell opto-RTK expression (non-steerable covariate) ---
        self.opto_rtk_expression: float = float(self._rng.beta(2, 5))

        # --- Baseline area for measuring change ---
        self.initial_area: float = self.area0

    # ------------------------------------------------------------------

    def stimulate(self, mask: np.ndarray, camera_offset=(0, 0)) -> None:
        """Apply optogenetic stimulation that increases cell area.

        Unlike the parent ``OptogeneticCell`` this variant:

        * does **not** apply a velocity impulse (no migration);
        * modulates the protrusion gain by the distance of the
          stimulated vertices from the cell centre (edge > centre);
        * scales the gain by ``opto_rtk_expression``;
        * grows ``base_r`` and ``area0`` so the area increase persists
          across subsequent physics updates.
        """
        if mask is None or not mask.any():
            self.is_stimulated = False
            return

        # --- vertex-to-viewport conversion (same as OptogeneticCell) ---
        vertices = self.vertices_positions - np.array(camera_offset)
        inside = (
            (vertices[:, 0] >= 0)
            & (vertices[:, 0] < mask.shape[1])
            & (vertices[:, 1] >= 0)
            & (vertices[:, 1] < mask.shape[0])
        )

        if not inside.any():
            self.is_stimulated = False
            return

        ix = vertices[inside, 0].astype(int)
        iy = vertices[inside, 1].astype(int)
        hit = mask[iy, ix]

        if not hit.any():
            self.is_stimulated = False
            return

        self.is_stimulated = True
        idx = np.where(inside)[0][hit]

        # --- Distance-dependent gain ---
        hit_positions = self.vertices_positions[idx]
        distances = np.linalg.norm(hit_positions - self.center, axis=1)
        # Normalise: 0 at centre, ~1 at the cell edge
        position_factor = np.mean(distances) / self.base_r
        position_factor = np.clip(position_factor, 0.3, 1.5)

        effective_gain = (
            self.protrusion_gain * position_factor * self.opto_rtk_expression
        )

        # --- Apply protrusion ---
        self.r[idx] += effective_gain * self.base_r
        self.r = np.clip(self.r, 0.4 * self.base_r, 2.5 * self.base_r)

        # --- Grow base_r & area0 so the physics update preserves the
        #     larger size instead of shrinking back. ---
        new_area = polygon_area(self.vertices_positions)
        if new_area > 0 and self.area0 > 0:
            scale = np.sqrt(new_area / self.area0)
            self.base_r *= scale
            self.area0 = new_area
