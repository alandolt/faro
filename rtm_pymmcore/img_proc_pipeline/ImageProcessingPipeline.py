"""Standard image processing pipeline implementation."""

import os
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import tifffile
from useq import MDAEvent

import rtm_pymmcore.segmentation.base_segmentation as base_segmentation
import rtm_pymmcore.stimulation.base_stimulation as base_stimulation
import rtm_pymmcore.tracking.abstract_tracker as abstract_tracker
import rtm_pymmcore.feature_extraction.abstract_fe as abstract_fe
from rtm_pymmcore.data_structures import Fov, ImgType
from rtm_pymmcore.utils import labels_to_particles, create_folders

from .base_image_processing_pipeline import BaseImageProcessingPipeline


class ImageProcessingPipeline(BaseImageProcessingPipeline):

    def __init__(
        self,
        storage_path: str,
        segmentators: List[base_segmentation.Segmentator] = None,
        feature_extractor: abstract_fe.FeatureExtractor = None,
        stimulator: base_stimulation.Stim = None,
        tracker: abstract_tracker.Tracker = None,
        feature_extractor_optocheck: abstract_fe.FeatureExtractor = None,
    ):
        """
        Standard image processing pipeline that stores images as TIFF files.

        This pipeline processes images through segmentation, feature extraction,
        tracking, and optional stimulation, storing all intermediate results
        as TIFF files in organized folder structures.

        Args:
            storage_path: Path where results will be stored
            segmentators: List of segmentation algorithms
            feature_extractor: Feature extraction algorithm
            stimulator: Stimulation algorithm
            tracker: Cell tracking algorithm
            feature_extractor_optocheck: Feature extractor for optogenetic checks
        """

        self.segmentators = segmentators
        self.feature_extractor = feature_extractor
        self.stimulator = stimulator
        self.tracker = tracker
        self.feature_extractor_optocheck = feature_extractor_optocheck
        self.storage_path = storage_path

        folders = ["raw", "tracks"]

        if self.stimulator is not None:
            folders.extend(["stim_mask", "stim"])

        if self.tracker is not None:
            folders.append("particles")

        if self.feature_extractor is not None:
            if hasattr(self.feature_extractor, "extra_folders"):
                folders.extend(self.feature_extractor.extra_folders)

        if self.segmentators:
            for seg in self.segmentators:
                if isinstance(seg, dict) and "name" in seg:
                    folders.append(seg["name"])
                elif hasattr(seg, "name"):
                    folders.append(seg.name)

        if self.feature_extractor_optocheck is not None:
            folders.append("optocheck")
            if hasattr(self.feature_extractor_optocheck, "extra_folders"):
                folders.extend(self.feature_extractor_optocheck.extra_folders)
        create_folders(self.storage_path, folders)

    def store_img(
        self, img: np.ndarray, metadata: Dict[str, Any], path: str, folder: str
    ) -> None:
        """
        Store an image as a compressed TIFF file.

        Args:
            img: Image array to store
            metadata: Metadata containing filename information
            path: Base storage path
            folder: Specific folder within the path
        """
        fname = metadata["fname"]
        tifffile.imwrite(
            os.path.join(path, folder, fname + ".tiff"),
            img,
            compression="zlib",
            compressionargs={"level": 5},
        )

    def run(self, img: np.ndarray, event: MDAEvent) -> Dict[str, Any]:
        """
        Run the complete image processing pipeline.

        Args:
            img: Input image array
            event: MDAEvent containing metadata

        Returns:
            Dictionary containing pipeline results
        """
        metadata = event.metadata
        fov_obj: Fov = metadata["fov_object"]

        # Get previous tracking data
        if metadata["timestep"] > 0:
            df_old = fov_obj.tracks_queue.get(block=True, timeout=360)
        else:
            df_old = pd.DataFrame()

        # Handle optogenetic check images
        if metadata["img_type"] == ImgType.IMG_OPTOCHECK:
            n_optocheck_channels = len(metadata["optocheck_channels"])
            n_channels = len(metadata["channels"])
            img_optocheck = img[n_channels:]
            img = img[:n_channels]

        shape_img = (img.shape[-2], img.shape[-1])

        # Process segmentation
        if self.segmentators is not None:
            segmentation_results = {}
            for seg in self.segmentators:
                segmentation_results[seg["name"]] = seg["class"].segment(
                    img[seg["use_channel"], :, :]
                )

        # Process stimulation if enabled
        if metadata["stim"]:
            stim_mask, labels_stim = self.stimulator.get_stim_mask(
                segmentation_results, metadata, img
            )
            fov_obj.stim_mask_queue.put_nowait(stim_mask)

        # Process feature extraction

        if self.feature_extractor is None:
            df_new = pd.DataFrame([metadata])
            df_new = pd.concat([df_old, df_new], ignore_index=True)
            masks_for_fe = None
        else:
            df_new, masks_for_fe = self.feature_extractor.extract_features(
                segmentation_results, img
            )

            for key, value in metadata.items():
                if isinstance(value, (list, tuple)):
                    df_new[key] = pd.Series([value] * len(df_new))
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        df_new[subkey] = [subvalue] * len(df_new)
                else:
                    df_new[key] = value

        # Process tracking
        if self.tracker is not None:
            df_tracked = self.tracker.track_cells(df_old, df_new, metadata)
        else:
            df_tracked = pd.concat([df_old, df_new], ignore_index=True)

        # Handle optogenetic check feature extraction
        if metadata["img_type"] == ImgType.IMG_OPTOCHECK:
            if (
                self.feature_extractor_optocheck is not None
                and self.tracker is not None
            ):
                df_tracked = self.feature_extractor_optocheck.extract_features(
                    segmentation_results, img_optocheck, df_tracked
                )

        # Store updated tracking data
        fov_obj.tracks_queue.put(df_tracked)

        # Prepare tracking data for storage
        if not df_tracked.empty:
            df_tracked = df_tracked.drop(
                columns=["fov_object", "img_type", "last_channel"], errors="ignore"
            )

        # Convert data types for efficiency
        df_datatypes = {
            "timestep": np.uint32,
            "particle": np.uint32,
            "label": np.uint32,
            "time": np.float32,
            "fov": np.uint16,
            "stim_exposure": np.float32,
        }

        existing_columns = {
            col: dtype
            for col, dtype in df_datatypes.items()
            if col in df_tracked.columns
        }

        try:
            df_tracked = df_tracked.astype(existing_columns)
        except ValueError as e:
            print(f"Error in converting datatypes: {e}")
            print("df_tracked columns:", df_tracked.columns.tolist())

        # Store tracking data
        df_tracked.to_parquet(
            os.path.join(self.storage_path, "tracks", f"{metadata['fname']}.parquet")
        )

        # Store stimulation masks
        if self.stimulator is not None:
            if metadata["stim"]:
                self.store_img(stim_mask, metadata, self.storage_path, "stim_mask")
            else:
                self.store_img(
                    np.zeros(shape_img, np.uint8),
                    metadata,
                    self.storage_path,
                    "stim_mask",
                )
                self.store_img(
                    np.zeros(shape_img, np.uint8), metadata, self.storage_path, "stim"
                )

        # Store feature extraction masks
        if masks_for_fe is not None:
            for mask_fe in masks_for_fe:
                for key, value in mask_fe.items():
                    self.store_img(value, metadata, self.storage_path, key)

        # Store segmentation results
        if self.segmentators is not None:
            if self.tracker is None:
                for key, value in segmentation_results.items():
                    self.store_img(value, metadata, self.storage_path, key)
            else:
                for (key, value), segmentator in zip(
                    segmentation_results.items(), self.segmentators
                ):
                    if segmentator.get("save_tracked", False):
                        tracked_label = labels_to_particles(value, df_tracked)
                        self.store_img(
                            tracked_label, metadata, self.storage_path, "particles"
                        )
                        self.store_img(value, metadata, self.storage_path, key)
                    else:
                        self.store_img(value, metadata, self.storage_path, key)

        ## store raws image:
        if metadata["img_type"] == ImgType.IMG_RAW:
            self.store_img(img, metadata, self.storage_path, "raw")
        elif metadata["img_type"] == ImgType.IMG_OPTOCHECK:
            self.store_img(
                img[: len(metadata["channels"])], metadata, self.storage_path, "raw"
            )
            self.store_img(img, metadata, self.storage_path, "optocheck")

        # Cleanup: remove previous tracking file
        if metadata["timestep"] > 0:
            fname_previous = f'{str(fov_obj.index).zfill(3)}_{str(metadata["timestep"]-1).zfill(5)}.parquet'
            previous_file = os.path.join(self.storage_path, "tracks", fname_previous)
            if os.path.exists(previous_file):
                os.remove(previous_file)

        return {"result": "STOP"}
