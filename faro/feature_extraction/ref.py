import numpy as np
from skimage.measure import label

import skimage
from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table
import pandas as pd
from .base_ref import FeatureExtractorRef


class RefFE(FeatureExtractorRef):

    def extract_features(self, segmentation_results, image, df_tracked, metadata):
        # Measure each label, then give it the particle ID of the newest
        # frame. Not with a particle image: particle 0 would be background.
        labels = np.expand_dims(segmentation_results[self.used_mask], 0)
        table = skimage.measure.regionprops_table(
            labels, image, properties=["label", "mean_intensity"]
        )
        table = pd.DataFrame.from_dict(table)
        table = table.rename({"mean_intensity": "ref_mean_intensity"}, axis=1)
        newest = df_tracked[df_tracked["timestep"] == df_tracked["timestep"].max()]
        table = table.merge(newest[["label", "particle"]], on="label")
        table = table.drop(columns="label")

        if not self.multi_timepoint:
            df_tracked = df_tracked.merge(table, on=["particle"])
        elif self.multi_timepoint:
            if "ref_mean_intensity" not in df_tracked.columns:
                df_tracked["ref_mean_intensity"] = np.nan
            if "fov_timestep" in metadata:
                timestep_name_variable = "fov_timestep"
            else:
                timestep_name_variable = "timestep"

            table[timestep_name_variable] = metadata[timestep_name_variable]
            table["fov"] = metadata["fov"]
            # Update only matching particle/timestep/fov combinations
            mask = (
                (df_tracked["particle"].isin(table["particle"]))
                & (
                    df_tracked[timestep_name_variable]
                    == metadata[timestep_name_variable]
                )
                & (df_tracked["fov"] == metadata["fov"])
            )
            df_tracked.loc[mask, "ref_mean_intensity"] = df_tracked.loc[
                mask, "particle"
            ].map(table.set_index("particle")["ref_mean_intensity"])

        return df_tracked


# Backwards-compat alias
OptoCheckFE = RefFE
