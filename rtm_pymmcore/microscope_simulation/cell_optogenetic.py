"""Optogenetic cell behaviour module."""

import numpy as np
from .cell_base import CellBase


class OptogeneticCell(CellBase):

    def __init__(
        self, *args, protrusion_gain: float = 0.05, impulse: float = 10.0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.protrusion_gain = protrusion_gain
        self.impulse = impulse
        self.is_stimulated = False

    def stimulate(self, mask: np.ndarray, camera_offset=(0, 0)) -> None:
        """Apply optogenetic stimulation based on mask.

        Parameters
        ----------
        mask : np.ndarray
            Boolean stimulation mask in viewport (image) coordinates.
        camera_offset : tuple
            The (x, y) camera offset used to convert world coordinates
            to viewport coordinates: ``viewport_pos = world_pos - offset``.
        """
        if mask is None or not mask.any():
            self.is_stimulated = False
            return

        # Convert world coordinates to viewport (image) coordinates
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

        # Apply protrusion
        self.r[idx] += self.protrusion_gain * self.base_r
        self.r = np.clip(self.r, 0.4 * self.base_r, 2.2 * self.base_r)
        self._conserve_area()

        # Apply impulse toward stimulated region (in world coordinates)
        hit_vertices_world = self.vertices_positions[idx]
        target = np.mean(hit_vertices_world, axis=0)
        direction = target - self.center
        norm = np.linalg.norm(direction)
        if norm > 0:
            self.vel += (direction / norm) * self.impulse
