"""Optimized rendering using OpenCV."""

import numpy as np
import cv2
from typing import List, Tuple
from .cell_base import CellBase


class Renderer:
    """Fast renderer using OpenCV."""

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        self.contrast = 0.55
        self.brightness = 0.78
        self.blur_radius = 3
        self.noise_std = 10
        self.margins = 100

        self.dof_map = {10: 6.0, 20: 3.0, 40: 1.5}
        self.objective = 10
        self.dof = self.dof_map[self.objective]

        self.max_blur_radius = 25
        self.min_opacity = 0.2
        self.crop_dim = 512

    def render_cells(
        self,
        cells: List[CellBase],
        mode: int = 0,
        camera_offset: Tuple[float, float] = (0, 0),
        focal_plane: float = 0.0,
    ) -> np.ndarray:
        """Render cells to image array using OpenCV."""
        img = np.full((self.height, self.width, 3), 0, dtype=np.uint8)
        visible_cells = self._get_visible_cells(cells, camera_offset)
        for cell in visible_cells:
            self._draw_cell(img, cell, mode, camera_offset, focal_plane)
        img = self._apply_filters(img, mode)
        img = self._crop_and_rescale(img)
        return img

    def _get_visible_cells(
        self, cells: List[CellBase], camera_offset: Tuple[float, float]
    ) -> List[CellBase]:
        margin = self.margins
        vx, vy = camera_offset
        visible = []
        for cell in cells:
            if (
                vx - margin <= cell.center[0] <= vx + self.width + margin
                and vy - margin <= cell.center[1] <= vy + self.height + margin
            ):
                visible.append(cell)
        return visible

    def _draw_smooth_cell(self, img, center, vertices, color, thickness=-1):
        n_interp = 100
        angles_interp = np.linspace(0, 2 * np.pi, n_interp, endpoint=False)
        angles_orig = np.linspace(0, 2 * np.pi, len(vertices), endpoint=False)
        radii_orig = np.linalg.norm(vertices - center, axis=1)
        radii_interp = np.interp(
            angles_interp, angles_orig, radii_orig, period=2 * np.pi
        )
        smooth_verts = np.zeros((n_interp, 2))
        smooth_verts[:, 0] = center[0] + radii_interp * np.cos(angles_interp)
        smooth_verts[:, 1] = center[1] + radii_interp * np.sin(angles_interp)
        smooth_pts = smooth_verts.astype(np.int32)
        if thickness == -1:
            cv2.fillPoly(img, [smooth_pts], color, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(
                img, [smooth_pts], True, color, thickness, lineType=cv2.LINE_AA
            )

    def _draw_cell(self, img, cell, mode, camera_offset, focal_plane):
        kernel_size, opacity = self._compute_blur_and_opacity(
            cell.z_position, focal_plane
        )
        vertices = cell.vertices_positions - camera_offset
        center_screen = cell.center - camera_offset
        cell_radius = cell.base_r

        if (
            center_screen[0] < -self.margins
            or center_screen[0] > self.width + self.margins
            or center_screen[1] < -self.margins
            or center_screen[1] > self.height + self.margins
        ):
            return

        if mode == 0:  # brightfield
            cell_img = np.full((self.height, self.width, 3), 0, dtype=np.uint8)
            layers = 10
            for i in range(layers, 0, -1):
                s = i / layers
                shade = 80 + int(100 * s)
                color = (shade, shade, 255)
                scaled_verts = center_screen + (vertices - center_screen) * s
                self._draw_smooth_cell(
                    cell_img, center_screen, scaled_verts, color, thickness=-1
                )
            self._draw_smooth_cell(
                cell_img, center_screen, vertices, (0, 0, 0), thickness=2
            )
            nucleus_pos = tuple(center_screen.astype(int))
            nucleus_radius = int(0.4 * cell_radius)
            cv2.circle(
                cell_img,
                nucleus_pos,
                nucleus_radius,
                (150, 60, 60),
                -1,
                lineType=cv2.LINE_AA,
            )
            if kernel_size > 0:
                cell_img = cv2.GaussianBlur(
                    cell_img, (kernel_size, kernel_size), kernel_size / 3.0
                )
            cell_opacity = opacity * 1.0
            img[:] = cv2.addWeighted(img, 1.0, cell_img, cell_opacity, 0)

        elif mode == 1:  # nucleus fluorescence
            if cell.nucleus_fluorescence > 0:
                fluor_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                nucleus_pos = tuple(center_screen.astype(int))
                nucleus_radius = int(0.55 * cell_radius)
                fluorescence_intensity = cell.nucleus_fluorescence * opacity
                for i in range(5, 0, -1):
                    radius = int(nucleus_radius * (1.0 + 0.2 * (5 - i)))
                    intensity = int(255 * fluorescence_intensity * (i / 5.0))
                    color = (8, 8, intensity)
                    cv2.circle(
                        fluor_img, nucleus_pos, radius, color, -1, lineType=cv2.LINE_AA
                    )
                cv2.circle(
                    fluor_img,
                    nucleus_pos,
                    int(nucleus_radius * 0.6),
                    (8, 8, 255),
                    -1,
                    lineType=cv2.LINE_AA,
                )
                base_blur = 3
                if kernel_size > 0:
                    total_blur = base_blur + (kernel_size * 2)
                    total_blur = total_blur if total_blur % 2 == 1 else total_blur + 1
                    fluor_img = cv2.GaussianBlur(
                        fluor_img, (total_blur, total_blur), kernel_size * 0.8
                    )
                else:
                    fluor_img = cv2.GaussianBlur(fluor_img, (base_blur, base_blur), 1)
                img[:] = cv2.add(img, fluor_img)

        elif mode == 2:  # membrane fluorescence
            if np.any(cell.membrane_fluorescence > 0):
                fluor_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                avg_fluorescence = np.mean(cell.membrane_fluorescence) * opacity
                membrane_color = (
                    int(8 * avg_fluorescence),
                    int(8 * avg_fluorescence),
                    int(255 * avg_fluorescence),
                )
                self._draw_smooth_cell(
                    fluor_img, center_screen, vertices, membrane_color, thickness=-1
                )
                base_blur = 7
                if kernel_size > 0:
                    total_blur = base_blur + (base_blur * 2)
                    total_blur = total_blur if total_blur % 2 == 1 else total_blur + 1
                    fluor_img = cv2.GaussianBlur(
                        fluor_img, (total_blur, total_blur), kernel_size * 0.6
                    )
                else:
                    fluor_img = cv2.GaussianBlur(fluor_img, (7, 7), 2)
                img[:] = cv2.add(img, fluor_img)

    def _apply_filters(self, img, mode):
        if mode == 0:  # brightfield
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float32)
            img = (img - 128.0) * self.contrast + 128.0
            img = np.clip(img, 0, 255).astype(np.uint8)
            kernel_size = 2 * self.blur_radius + 1
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            noise = np.random.normal(0, self.noise_std, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = img.astype(np.float32)
            img = img * self.brightness
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            kernel_size = 2 * self.blur_radius + 1
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 1.0)
        return img

    def set_objective(self, mag: int, dof: float) -> None:
        self.objective = mag
        self.dof = dof

    def _compute_blur_and_opacity(self, cell_z, focal_plane):
        z_dist_um = abs(cell_z - focal_plane)
        if z_dist_um <= self.dof:
            return 0, 1.0
        out_um = z_dist_um - self.dof
        norm = out_um / max(1.0, self.dof * 4.0)
        radius = int(min(self.max_blur_radius, (norm**0.75) * self.max_blur_radius))
        if radius > 0:
            kernel = radius if radius % 2 == 1 else radius + 1
        else:
            kernel = 0
        opacity = max(
            self.min_opacity, 1.0 / (1.0 + 0.12 * (out_um / max(1.0, self.dof)))
        )
        return kernel, float(opacity)

    def _crop_and_rescale(self, img):
        if self.objective == 10:
            self.crop_dim = 512
            return img
        elif self.objective == 20:
            self.crop_dim = 512 * (10 / self.objective)
        else:
            self.crop_dim = 512 * (10 / self.objective)
        x_start = int((self.width - self.crop_dim) / 2)
        y_start = int((self.height - self.crop_dim) / 2)
        crop_img = img[x_start:-x_start, y_start:-y_start]
        rescaled_img = cv2.resize(
            crop_img, (self.width, self.height), interpolation=cv2.INTER_CUBIC
        )
        return rescaled_img
