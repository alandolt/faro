"""Normal cell without special behaviors."""

import numpy as np
from .cell_base import CellBase


class NormalCell(CellBase):
    """Normal cell that only responds to physics and collisions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.friction = 4.0
        self.brownian_d = 60.0
        self.curvature_relax = 0.08
        self.radial_relax = 0.03
        self.ruffle_std = 0.04

        self.has_nucleus_marker = np.random.random() > 0.5
        self.has_membrane_marker = np.random.random() > 0.5

        if self.has_nucleus_marker:
            self.nucleus_fluorescence = 0.7 + np.random.random() * 0.3
        else:
            self.nucleus_fluorescence = 0.0

        if self.has_membrane_marker:
            self.membrane_fluorescence = np.full(
                self.vertices, 0.7 + np.random.random() * 0.3
            )
        else:
            self.membrane_fluorescence = np.zeros(self.vertices)

    def respond_to_collision(self, other: "CellBase") -> None:
        """Normal cells can deform slightly when collided."""
        dvec = other.center - self.center
        dvec[0] -= self.width * np.round(dvec[0] / self.width)
        dvec[1] -= self.height * np.round(dvec[1] / self.height)

        if np.linalg.norm(dvec) > 0:
            collision_angle = np.arctan2(dvec[1], dvec[0])
            for i, angle in enumerate(self.angles):
                angle_diff = abs(angle - collision_angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                if angle_diff < np.pi / 3:
                    self.r[i] *= 0.95
            self.r = np.clip(self.r, 0.4 * self.base_r, 2.2 * self.base_r)
            self._conserve_area()

    def update_behavior(self, dt: float) -> None:
        """Normal cells slowly return to round shape."""
        self.r += 0.01 * (self.base_r - self.r) * dt
