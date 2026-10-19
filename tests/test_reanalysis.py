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
