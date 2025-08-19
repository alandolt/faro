"""OME-Zarr image processing pipeline implementation."""

import os
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import zarr
from useq import MDAEvent


import rtm_pymmcore.segmentation.base_segmentation as base_segmentation
import rtm_pymmcore.stimulation.base_stimulation as base_stimulation
import rtm_pymmcore.tracking.abstract_tracker as abstract_tracker
import rtm_pymmcore.feature_extraction.abstract_fe as abstract_fe
from rtm_pymmcore.data_structures import Fov, ImgType
from rtm_pymmcore.utils import labels_to_particles, create_folders
import dataclasses
import tifffile

from ome_zarr.writer import (
    initialize_array,
    write_single_timepoint,
    initialize_label_arrays,
    write_single_label_timepoint,
)
from ome_zarr.scale import Scaler

from rtm_pymmcore.img_proc_pipeline.ImageProcessingPipeline import (
    ImageProcessingPipeline,
)


class ImageProcessingPipelineOmeZarr(ImageProcessingPipeline):
    """
    OME-Zarr based image processing pipeline.

    This pipeline processes images and stores them in OME-Zarr format,
    enabling efficient storage and access of multidimensional imaging data
    with proper metadata and multiscale pyramids.
    """

    def __init__(
        self,
        storage_path: str,
        segmentators: List[base_segmentation.Segmentator] = None,
        feature_extractor: abstract_fe.FeatureExtractor = None,
        stimulator: base_stimulation.Stim = None,
        tracker: abstract_tracker.Tracker = None,
        feature_extractor_optocheck: abstract_fe.FeatureExtractor = None,
        zarr_scaler: Scaler = None,
        zarr_storage_options: Dict[str, Any] = None,
        zarr_storage_options_labels: Dict[str, Any] = None,
    ):
        """
        Initialize the OME-Zarr pipeline.

        Args:
            storage_path: Path where OME-Zarr stores will be created
            segmentators: List of segmentation algorithms
            feature_extractor: Feature extraction algorithm
            stimulator: Stimulation algorithm
            tracker: Cell tracking algorithm
            feature_extractor_optocheck: Feature extractor for optogenetic checks
            zarr_scaler: Scaler for creating multiscale pyramids
            zarr_storage_options: Storage options for Zarr arrays (chunks, shards, etc.)
        """
        self.zarr_scaler = zarr_scaler
        self.zarr_storage_options = zarr_storage_options
        self.zarr_storage_options_labels = zarr_storage_options_labels

        self.segmentators = segmentators
        self.feature_extractor = feature_extractor
        self.stimulator = stimulator
        self.tracker = tracker
        self.feature_extractor_optocheck = feature_extractor_optocheck
        self.storage_path = storage_path
        folders = ["tracks"]

        if self.stimulator is not None:
            folders.extend(["stim"])
        if self.feature_extractor_optocheck is not None:
            folders.append("optocheck")
            if hasattr(feature_extractor_optocheck, "extra_folders"):
                folders.extend(feature_extractor_optocheck.extra_folders)

        create_folders(self.storage_path, folders)

        # Storage for initialized Zarr arrays
        self._zarr_arrays = {}
        self._zarr_label_arrays = {}

    def _get_or_create_zarr_arrays(
        self,
        fov_id: int,
        shape: tuple,
        dtype: np.dtype,
        array_type: str = "image",
        name: str = "image_data",
    ) -> tuple:
        """
        Get or create Zarr arrays for a specific FOV.

        Args:
            fov_id: Field of view identifier
            shape: Shape of the array (T, C, Z, Y, X)
            dtype: Data type of the array
            array_type: Type of array ("image" or "labels")

        Returns:
            Tuple of (arrays, datasets)
        """
        key = f"{fov_id}_{array_type}_{name}"  # Include name to distinguish different label types

        # Check the appropriate dictionary based on array type
        if array_type == "image":
            cache_dict = self._zarr_arrays
        else:
            cache_dict = self._zarr_label_arrays

        if key not in cache_dict:
            # Create Zarr store for this FOV
            store_path = os.path.join(self.storage_path, f"{fov_id:03d}.ome.zarr")
            zarr_group = zarr.group(
                store=store_path, overwrite=False
            )  # Allow overwrite for testing

            if array_type == "image":
                # Create image arrays
                arrays, datasets = initialize_array(
                    shape=shape,
                    dtype=dtype,
                    group=zarr_group,
                    scaler=self.zarr_scaler,
                    axes="tcyx",  # Assuming 4D: Time, Channel, Y, X
                    storage_options=self.zarr_storage_options,
                    name=name,
                )
                cache_dict[key] = (arrays, datasets)

            elif array_type == "labels":

                arrays, datasets = initialize_label_arrays(
                    shape=tuple(shape),
                    dtype=dtype,
                    group=zarr_group,
                    label_name=name,
                    scaler=self.zarr_scaler,
                    axes="tyx",
                    storage_options=self.zarr_storage_options_labels,
                )
                cache_dict[key] = (arrays, datasets)

        return cache_dict[key]

    def store_img(
        self, img: np.ndarray, metadata: Dict[str, Any], path: str, folder: str
    ) -> None:
        """
        Store an image in OME-Zarr format.

        Args:
            img: Image array to store
            metadata: Metadata containing FOV and timepoint information
            path: Base storage path (unused for Zarr, kept for compatibility)
            folder: Folder type (used to determine storage strategy)
        """
        fov_id = metadata.get("fov", 0)
        timestep = metadata.get("timestep", 0)

        if folder in ["raw", "image_data"]:
            # Store as main image data
            # Determine shape for full experiment
            # This is a simplified approach - in practice you might want to
            # pre-determine the full experiment shape
            if len(img.shape) == 3:  # (C, Y, X)
                # Assume this is part of a time series - we'll need to know max timepoints
                max_timepoints = (
                    metadata.get("max_timestep", 100) + 1
                )  # +1 because 0-indexed
                full_shape = (max_timepoints, img.shape[0], img.shape[1], img.shape[2])
            else:
                raise AssertionError("Unexpected image shape")

            arrays, _ = self._get_or_create_zarr_arrays(
                fov_id, full_shape, img.dtype, "image"
            )

            # Write single timepoint
            if len(img.shape) == 3:  # (C, Y, X)
                write_single_timepoint(
                    arrays=arrays,
                    data=img,
                    t=timestep,
                    scaler=self.zarr_scaler,
                    axes="tcyx",
                )

        elif (
            folder in [seg["name"] for seg in self.segmentators]
            or folder == "particles"
        ):
            # Store as segmentation labels
            max_timepoints = (
                metadata.get("max_timestep", 100) + 1
            )  # +1 because 0-indexed

            # Label images are typically 2D (Y, X), not 3D (C, Y, X)
            if len(img.shape) == 2:  # (Y, X)
                full_shape = (max_timepoints, img.shape[0], img.shape[1])  # (T, Y, X)
            elif (
                len(img.shape) == 3
            ):  # (C, Y, X) - shouldn't happen for labels, but handle it
                full_shape = (max_timepoints, img.shape[1], img.shape[2])  # (T, Y, X)
            else:
                raise AssertionError(f"Unexpected label image shape: {img.shape}")

            arrays, _ = self._get_or_create_zarr_arrays(
                fov_id, full_shape, img.dtype, "labels", name=folder
            )

            # Write single label timepoint
            write_single_label_timepoint(
                arrays=arrays,
                label_data=img,
                t=timestep,
                scaler=self.zarr_scaler,
                axes="tyx",
            )

        else:
            # For other data types, fall back to creating individual tifffiles
            fname = metadata["fname"]
            tifffile.imwrite(
                os.path.join(path, folder, fname + ".tiff"),
                img,
                compression="zlib",
                compressionargs={"level": 5},
            )
