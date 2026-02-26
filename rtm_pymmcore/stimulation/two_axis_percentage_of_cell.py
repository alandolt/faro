"""Fixed-size stimulation patch placed in polar coordinates on each cell.

Places a circular disk of fixed pixel radius at a position defined by
two parameters relative to the cell's geometry:

Metadata keys read by :meth:`get_stim_mask`:

* ``stim_angle`` (float, 0-1):
      Angular position around the cell circumference, relative to the
      cell's major axis.  0 and 1 both map to the major-axis tip;
      0.5 maps to the opposite tip; 0.25 / 0.75 to the minor-axis
      sides.
* ``stim_radial`` (float, 0-1):
      Radial position from cell centre (0) to cell edge (1).
      The edge distance is computed from the cell's elliptical shape
      at the chosen angle.
"""

from .base_stimulation import Stim
import math
import numpy as np
import skimage
from skimage.morphology import disk as morph_disk
from skimage.draw import disk as draw_disk
from skimage.morphology import dilation as skimage_dilation


# Fixed stimulation patch radius in pixels.
STIM_PATCH_RADIUS = 10


class StimTwoAxisPercentage(Stim):
    """Place a fixed-size circular stimulation patch on each cell."""

    def __init__(self, patch_radius: int = STIM_PATCH_RADIUS):
        self.use_labels = True
        self.use_imgs = False
        self.patch_radius = patch_radius

    def get_stim_mask(
        self, label_images, metadata: dict = None, img: np.ndarray = None
    ) -> tuple:
        label_image = label_images["labels"]
        h, w = label_image.shape
        light_map = np.zeros((h, w), dtype=bool)
        props = skimage.measure.regionprops(label_image)

        if metadata is None:
            metadata = {}

        stim_angle = float(metadata.get("stim_angle", 0.0))
        stim_radial = float(metadata.get("stim_radial", 0.7))

        selem = morph_disk(5)

        try:
            # Local angle relative to cell's major axis (full circle).
            local_theta = stim_angle * 2.0 * math.pi

            for prop in props:
                label = prop.label

                y0, x0 = prop.centroid
                orientation = prop.orientation

                # Semi-axes (ensure > 0 to avoid division by zero).
                a = max(prop.axis_major_length / 2.0, 1.0)  # semi-major
                b = max(prop.axis_minor_length / 2.0, 1.0)  # semi-minor

                # Distance from centroid to ellipse boundary at local_theta.
                cos_t = math.cos(local_theta)
                sin_t = math.sin(local_theta)
                r_edge = (a * b) / math.sqrt((b * cos_t) ** 2 + (a * sin_t) ** 2)

                # Convert local angle to image-space angle.
                # skimage orientation: angle of major axis w.r.t. row (y) axis.
                # Major-axis direction in (x, y): (-sin(ori), -cos(ori)).
                major_angle = math.atan2(-math.cos(orientation), -math.sin(orientation))
                global_theta = major_angle + local_theta

                # Patch centre in image coordinates.
                patch_cx = x0 + stim_radial * r_edge * math.cos(global_theta)
                patch_cy = y0 + stim_radial * r_edge * math.sin(global_theta)

                # Draw fixed-size disk.
                rr, cc = draw_disk(
                    (patch_cy, patch_cx), self.patch_radius, shape=(h, w)
                )

                # Restrict to dilated cell region.
                minr, minc, maxr, maxc = prop.bbox
                pad = self.patch_radius + 6
                r0 = max(0, minr - pad)
                r1 = min(h, maxr + pad)
                c0 = max(0, minc - pad)
                c1 = min(w, maxc + pad)

                sub_labels = label_image[r0:r1, c0:c1]
                single_label_sub = sub_labels == label

                if not single_label_sub.any():
                    continue

                expanded_sub = skimage_dilation(single_label_sub, footprint=selem)

                # Build full-size mask for this cell's dilated region.
                cell_mask = np.zeros((h, w), dtype=bool)
                cell_mask[r0:r1, c0:c1] = expanded_sub

                # Combine: disk AND dilated cell.
                patch_mask = np.zeros((h, w), dtype=bool)
                patch_mask[rr, cc] = True
                patch_mask &= cell_mask

                light_map |= patch_mask

            return light_map.astype("uint8"), None

        except Exception as e:
            print(e)
            return np.zeros_like(label_image), None
