"""Post-experiment image processing pipeline implementation."""

import os
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from tifffile import tifffile

import rtm_pymmcore.segmentation.base_segmentation as base_segmentation
import rtm_pymmcore.stimulation.base_stimulation as base_stimulation
import rtm_pymmcore.tracking.abstract_tracker as abstract_tracker
import rtm_pymmcore.feature_extraction.abstract_fe as abstract_fe
from rtm_pymmcore.data_structures import Fov, ImgType
from rtm_pymmcore.utils import labels_to_particles, create_folders

from concurrent.futures import ThreadPoolExecutor, as_completed


from .base_image_processing_pipeline import BaseImageProcessingPipeline


class ImageProcessingPipeline_postExperiment(BaseImageProcessingPipeline):
    def __init__(
        self,
        img_storage_path: str,
        out_path: str,
        df_acquire: pd.DataFrame,
        segmentators: List[base_segmentation.Segmentator] = None,
        feature_extractor: abstract_fe.FeatureExtractor = None,
        tracker: abstract_tracker.Tracker = None,
        feature_extractor_optocheck: abstract_fe.FeatureExtractor = None,
        use_old_segmentations: bool = False,
        n_jobs: int = 2,
    ):
        self.segmentators = segmentators
        self.feature_extractor = feature_extractor
        self.tracker = tracker
        self.df_acquire = df_acquire
        self.feature_extractor_optocheck = feature_extractor_optocheck
        self.storage_path = out_path
        self.img_storage_path = img_storage_path
        self.use_old_segmentations = use_old_segmentations
        self.n_jobs = n_jobs

        folders = ["raw", "tracks"]
        if self.tracker is not None:
            folders.append("particles")
        if self.feature_extractor is not None:
            if hasattr(self.feature_extractor, "extra_folders"):
                folders.extend(self.feature_extractor.extra_folders)
        if self.segmentators is not None:
            for seg in self.segmentators:
                folders.append(seg["name"])
        if feature_extractor_optocheck is not None:
            folders.append("optocheck")
            if hasattr(feature_extractor_optocheck, "extra_folders"):
                folders.extend(feature_extractor_optocheck.extra_folders)
        create_folders(self.storage_path, folders)

    def run(self):
        unique_fovs = self.df_acquire["fov"].unique()
        max_workers = min(self.n_jobs, len(unique_fovs))  # Limit number of threads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.run_on_fov, fov_id): fov_id
                for fov_id in unique_fovs
            }
            for future in as_completed(futures):
                fov_id = futures[future]
                try:
                    future.result()
                    print(f"Finished processing FOV {fov_id}")
                except Exception as e:
                    print(f"Error processing FOV {fov_id}: {str(e)}")
        print("Finished processing all FOVs.")

    def run_on_fov(self, fov_id) -> dict:
        df = self.df_acquire.query("fov == @fov_id").copy()
        df_old = pd.DataFrame()

        fov_obj = Fov(0)
        df.loc[:, "fov_object"] = fov_obj

        for index, row in df.iterrows():
            img = tifffile.imread(
                os.path.join(self.img_storage_path, "raw", row["fname"] + ".tiff")
            )
            metadata = row.to_dict()
            shape_img = (img.shape[-2], img.shape[-1])
            segmentation_results = {}
            if self.use_old_segmentations:
                for seg in self.segmentators:
                    segmentation_results[seg["name"]] = tifffile.imread(
                        os.path.join(
                            self.img_storage_path, seg["name"], row["fname"] + ".tiff"
                        )
                    )
            else:
                for seg in self.segmentators:
                    segmentation_results[seg["name"]] = seg["class"].segment(
                        img[seg["use_channel"], :, :]
                    )

            if self.feature_extractor is not None:
                df_new, masks_for_fe = self.feature_extractor.extract_features(
                    segmentation_results, img
                )
                for key, value in metadata.items():
                    if isinstance(value, (list, tuple, np.ndarray)):
                        df_new[key] = pd.Series([value] * len(df_new))
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            df_new[subkey] = [subvalue] * len(df_new)
                    else:
                        df_new[key] = value

            if self.tracker is not None:
                df_tracked = self.tracker.track_cells(df_old, df_new, metadata)
            else:
                df_tracked = pd.concat([df_old, df_new], ignore_index=True)
            df_old = df_tracked

            if masks_for_fe is not None:
                for mask_fe in masks_for_fe:
                    for key, value in mask_fe.items():
                        self.store_img(value, metadata, self.storage_path, key)

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
                        if not self.use_old_segmentations:
                            self.store_img(value, metadata, self.storage_path, key)
                    else:
                        if not self.use_old_segmentations:
                            self.store_img(value, metadata, self.storage_path, key)

        df_tracked.drop(columns=["fov_object"], inplace=True)
        df_tracked.to_parquet(
            os.path.join(self.storage_path, "tracks", f"{metadata['fname']}.parquet")
        )

        return df_tracked

    def concat_fovs(self):

        fovs_i_list = os.listdir(os.path.join(self.storage_path, "tracks"))
        fovs_i_list.sort()
        dfs = []

        for fov_i in fovs_i_list:

            track_file = os.path.join(self.storage_path, "tracks", fov_i)
            df = pd.read_parquet(track_file)
            dfs.append(df)

        pd.concat(dfs).to_parquet(os.path.join(self.storage_path, "exp_data.parquet"))

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
