"""Two-axis stimulation of a percentage of each cell.

Extends the single-axis logic of :class:`StimPercentageOfCell` with a
second (minor-axis) parameter that controls the width of the
stimulation band perpendicular to the major axis.

Metadata keys read by :meth:`get_stim_mask`:

* ``stim_position_major`` (float, 0-1):
      0 → cutoff at cell centre (whole half stimulated, low protrusion);
      1 → cutoff at cell edge (only the tip stimulated, high protrusion).
* ``stim_position_minor`` (float, 0-1):
      0 → very narrow band along the major axis;
      1 → full cell width (no perpendicular constraint).
"""

from .base_stimulation import Stim
import math
import numpy as np
import skimage
from skimage.morphology import disk

from skimage.morphology import dilation as skimage_dilation


class StimTwoAxisPercentage(Stim):
    """Stimulate a region of each cell controlled by two axis parameters."""

    def __init__(self):
        self.use_labels = True
        self.use_imgs = False

    def get_stim_mask(
        self, label_images, metadata: dict = None, img: np.ndarray = None
    ) -> tuple:
        label_image = label_images["labels"]
        h, w = label_image.shape
        light_map = np.zeros_like(label_image, dtype=bool)
        props = skimage.measure.regionprops(label_image)

        if metadata is None:
            metadata = {}

        stim_major = float(metadata.get("stim_position_major", 0.5))
        stim_minor = float(metadata.get("stim_position_minor", 1.0))

        selem = disk(5)

        try:
            # stim_major: 0 = centre (extent = 0 → cutoff at centroid,
            #                         stimulates half the cell)
            #             1 = edge   (extent = 0.5 → cutoff at cell tip,
            #                         stimulates only the very edge)
            extent = stim_major * 0.5

            for prop in props:
                label = prop.label

                minr, minc, maxr, maxc = prop.bbox
                pad = 6
                r0 = max(0, minr - pad)
                r1 = min(h, maxr + pad)
                c0 = max(0, minc - pad)
                c1 = min(w, maxc + pad)

                sub_labels = label_image[r0:r1, c0:c1]
                single_label_sub = sub_labels == label

                if not single_label_sub.any():
                    continue

                y0, x0 = prop.centroid
                orientation = prop.orientation

                # --- Major axis cutoff (same geometry as StimPercentageOfCell) ---
                x2 = x0 - math.sin(orientation) * extent * prop.axis_major_length
                y2 = y0 - math.cos(orientation) * extent * prop.axis_major_length

                length = 0.5 * prop.axis_minor_length
                x3 = x2 + length * math.cos(-orientation)
                y3 = y2 + length * math.sin(-orientation)

                ys = np.arange(r0, r1)
                xs = np.arange(c0, c1)
                x_coords_sub, y_coords_sub = np.meshgrid(xs, ys)

                # Cross-product mask (one side of the cutoff line)
                v1_x = x3 - x2
                v1_y = y3 - y2
                v2_x = x3 - x_coords_sub
                v2_y = y3 - y_coords_sub
                cross_product = v1_x * v2_y - v1_y * v2_x
                major_mask = cross_product > 0

                # --- Minor axis band ---
                # Perpendicular distance from the major axis (through centroid)
                # Major axis direction: (-sin(orientation), -cos(orientation))
                # Minor axis direction: (cos(-orientation), sin(-orientation))
                minor_dx = math.cos(-orientation)
                minor_dy = math.sin(-orientation)
                dx = x_coords_sub - x0
                dy = y_coords_sub - y0
                perp_dist = np.abs(dx * minor_dx + dy * minor_dy)
                max_perp = stim_minor * 0.5 * max(prop.axis_minor_length, 1.0)
                minor_mask = perp_dist <= max_perp

                # Expand label region slightly
                expanded_sub = skimage_dilation(single_label_sub, footprint=selem)

                stim_mask_sub = major_mask & minor_mask & expanded_sub

                light_map[r0:r1, c0:c1] = np.logical_or(
                    light_map[r0:r1, c0:c1], stim_mask_sub
                )

            return light_map.astype("uint8"), None

        except Exception as e:
            print(e)
            return np.zeros_like(label_image), None
