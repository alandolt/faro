import numpy as np
import numpy.typing as npt
from skimage.draw import disk
from skimage.measure import regionprops
import pandas as pd


# TODO return also labels_stim, list in which the stimulated cells are marked
class Stim:
    """
    Base class for all stimulators. Specific implementations should inherit
    from this class and override the get_stim_mask method.
    """

    required_metadata: set[str] = set()

    def __init__(self):
        self.use_labels = True
        self.use_imgs = True

    def get_stim_mask(
        self, label_images: dict, metadata: dict, img: np.ndarray,
        tracks: "pd.DataFrame | None" = None,
    ) -> npt.NDArray[np.uint8]:
        """
        Parameters:
        label_images (dict): Segmentation results (e.g. {"labels": np.ndarray}).
        metadata (dict): Event metadata.
        img (np.ndarray): Raw image.
        tracks (pd.DataFrame, optional): Tracked cells for the current FOV.
            Contains columns like 'label', 'particle', 'x', 'y', 'timestep'.
            Use to make per-cell stimulation decisions based on tracking.

        Returns:
        np.ndarray: The stimulation mask.
        list: A list of labels that were stimulated.
        """
        raise NotImplementedError("Subclasses should implement this!")


class StimWholeFOV(Stim):
    """
    Stimulate the whole FOV.
    """

    def __init__(self):
        self.use_labels = False
        self.use_imgs = False

    def get_stim_mask(
        self, label_images: dict, metadata: dict = None, img: np.array = None
    ) -> npt.NDArray[np.uint8]:
        return np.ones((img.shape[-2], img.shape[-1]), dtype=np.uint8), None


class StimTopEdgeMeta(Stim):
    """Illuminate the top *fraction* of each cell's y-extent.

    Unlike ``StimTopEdge`` (in the notebook), the fraction is read from
    ``metadata["stim_fraction"]`` at runtime, so it can vary per-event.

    Declares ``required_metadata = {"stim_fraction"}`` so that
    ``pipeline.validate_pipeline()`` flags events that forget to set it.
    """

    required_metadata: set[str] = {"stim_fraction"}

    def __init__(self):
        super().__init__()
        self.use_labels = True
        self.use_imgs = False

    def get_stim_mask(self, label_images, metadata=None, img=None, tracks=None):
        from skimage.morphology import disk as _disk, dilation

        labels = label_images["labels"]
        stim_mask = np.zeros(labels.shape, dtype=np.uint8)
        fraction = metadata["stim_fraction"]
        selem = _disk(3)

        for prop in regionprops(labels):
            minr, minc, maxr, maxc = prop.bbox
            y_cutoff = minr + fraction * (maxr - minr)

            cell_mask = labels == prop.label
            rows, cols = np.where(cell_mask)
            top_pixels = rows < y_cutoff
            if not top_pixels.any():
                continue

            local = np.zeros_like(labels, dtype=np.uint8)
            local[rows[top_pixels], cols[top_pixels]] = 1
            local = dilation(local, footprint=selem)
            stim_mask = np.maximum(stim_mask, local)

        return stim_mask, None


class StimNothing(Stim):
    """Use when you don't want to stimulate. Returns empty stimulation mask."""

    def get_stim_mask(
        self, label_image: np.ndarray, metadata: dict = None, img: np.array = None
    ) -> npt.NDArray[np.uint8]:
        return np.zeros_like(label_image), [1, 2, 3, 4]  # some dummy values
