"""Output writer and source format of the reanalysis.

ImageProcessingPipeline_postExperiment with an OmeZarrWriter hard-links the
raw arrays of the source store into the output store, so it needs an
OME-Zarr source. For a TIFF source, use a TiffWriter.
"""

from __future__ import annotations

import os

import pytest

from faro.core.controller import Controller
from faro.core.conversion import load_events_json
from faro.core.data_structures import SegmentationMethod
from faro.core.pipeline_post import ImageProcessingPipeline_postExperiment
from faro.core.writers import OmeZarrWriter, TiffWriter
from faro.feature_extraction.simple import SimpleFE
from faro.segmentation.base import OtsuSegmentator
from faro.tracking.trackpy import TrackerTrackpy

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import CircleScene, make_events, make_pipeline


def test_omezarr_output_needs_an_omezarr_source(tmp_dir):
    """A TIFF source with an OmeZarrWriter fails at once with a clear error."""
    src = os.path.join(tmp_dir, "src")
    pipeline = make_pipeline(src, tracker=TrackerTrackpy(search_range=50))
    ctrl = Controller(FakeMicroscope(CircleScene()), pipeline, writer=TiffWriter(src))
    try:
        ctrl.run_experiment(make_events(2), validate=False).wait()
    finally:
        ctrl.finish_experiment()

    out = os.path.join(tmp_dir, "out")
    with pytest.raises(ValueError, match="TiffWriter"):
        ImageProcessingPipeline_postExperiment(
            img_storage_path=src,
            out_path=out,
            events=load_events_json(src),
            segmentators=[SegmentationMethod("labels", OtsuSegmentator(), 0, False)],
            feature_extractor=SimpleFE("labels"),
            tracker=TrackerTrackpy(search_range=50),
            n_jobs=1,
            writer=OmeZarrWriter(out),
        )


class ChannelSpyFE(SimpleFE):
    """SimpleFE that records the channel count of each image."""

    def __init__(self):
        super().__init__("labels")
        self.n_channels = []

    def extract_features(self, labels, image, df_tracked=None, metadata=None):
        self.n_channels.append(image.shape[0])
        return super().extract_features(labels, image, df_tracked, metadata)


def test_zarr_source_gives_the_imaging_channels_only(tmp_dir):
    """The zarr store also holds the stim readout, which the live pipeline
    never got. The reanalysis must read the imaging channels only.
    """
    src = os.path.join(tmp_dir, "src")
    pipeline = make_pipeline(src, tracker=TrackerTrackpy(search_range=50))
    ctrl = Controller(
        FakeMicroscope(CircleScene()), pipeline, writer=OmeZarrWriter(src)
    )
    try:
        ctrl.run_experiment(make_events(3, stim_frames=(1,)), validate=False).wait()
    finally:
        ctrl.finish_experiment()

    out = os.path.join(tmp_dir, "out")
    spy = ChannelSpyFE()
    ImageProcessingPipeline_postExperiment(
        img_storage_path=src,
        out_path=out,
        events=load_events_json(src),
        segmentators=[SegmentationMethod("labels", OtsuSegmentator(), 0, False)],
        feature_extractor=spy,
        tracker=TrackerTrackpy(search_range=50),
        n_jobs=1,
        writer=TiffWriter(out),
    ).run()
    assert spy.n_channels == [1, 1, 1]
