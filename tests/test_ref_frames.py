"""Reference frames in the pipeline.

A reference frame is stored in two parts: its imaging channels as "raw" and
its reference channels as "ref". When the pipeline is overloaded, a frame is
reloaded from storage later; the reload must give the pipeline both parts.
ControllerSimulated replays stored frames and must do the same.

A ref phase (``rtm_metadata={"img_type": ImgType.IMG_REF}`` without
``ref_channels``, see README) has no separate reference channels: its imaging
channels are the reference. It is stored whole as "ref".
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest
from useq import MDAEvent

from faro.core.controller import Analyzer, Controller, ControllerSimulated
from faro.core.conversion import load_events_json, save_events_json
from faro.core.data_structures import Channel, ImgType, RTMEvent, RTMSequence, combine
from faro.core.writers import OmeZarrWriter, TiffWriter
from faro.feature_extraction.base_ref import FeatureExtractorRef
from faro.tracking.trackpy import TrackerTrackpy

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import CircleScene, make_circle_image, make_pipeline

H, W = make_circle_image().shape
REF_VALUE = 7  # pixel value of the reference channel in RefScene


def _metadata(img_type):
    return {
        "fov": 0,
        "timestep": 0,
        "fname": "000_00000",
        "img_type": img_type,
        "channels": ["phase-contrast", "fitc"],
        "ref_channels": ["ref"],
        "stim": False,
    }


class SpyRefFE(FeatureExtractorRef):
    def __init__(self):
        super().__init__("labels")
        self.calls = 0
        self.images = []

    def extract_features(self, segmentation_results, image, df_tracked, metadata):
        self.calls += 1
        self.images.append(np.asarray(image))
        return df_tracked


class RefScene(CircleScene):
    """CircleScene whose reference channel ("stim-405") is a flat image."""

    def render(self, event):
        if event.channel is not None and event.channel.config == "stim-405":
            return np.full((H, W), REF_VALUE, dtype=np.uint16)
        return super().render(event)


def _inline_ref_events():
    """Imaging at t=0 and t=1, and one reference channel at t=1."""
    ref = (Channel(config="stim-405", exposure=10),)
    return [
        RTMEvent(
            index={"t": t, "p": 0},
            channels=(Channel(config="phase-contrast", exposure=10),),
            ref_channels=ref if t == 1 else (),
            metadata={},
        )
        for t in range(2)
    ]


def _ref_phase_events():
    """Two imaging timepoints, then the README ref phase."""
    experiment = RTMSequence(
        time_plan={"interval": 0, "loops": 2},
        stage_positions=[(0, 0, 0)],
        channels=[{"config": "phase-contrast", "exposure": 10}],
    )
    ref_phase = RTMSequence(
        time_plan={"interval": 0, "loops": 1},
        stage_positions=[(0, 0, 0)],
        channels=[{"config": "phase-contrast", "exposure": 10}],
        rtm_metadata={"img_type": ImgType.IMG_REF},
    )
    return combine(experiment, ref_phase, axis="t")


class TestReloadFrame:
    def test_reference_frame_gets_its_reference_channels_back(self, tmp_dir):
        writer = TiffWriter(tmp_dir)
        raw = np.full((2, H, W), 1, dtype=np.uint16)
        ref = np.full((1, H, W), 7, dtype=np.uint16)
        meta = _metadata(ImgType.IMG_REF)
        writer.write(raw, meta, "raw")
        writer.write(ref, meta, "ref")

        analyzer = Analyzer(writer=writer)
        try:
            img = analyzer._reload_frame(meta)
            assert img.shape == (3, H, W)  # imaging channels, then reference
            np.testing.assert_array_equal(img[:2], raw)
            np.testing.assert_array_equal(img[2], ref[0])
            # A normal frame is only its imaging channels.
            raw_meta = {**meta, "img_type": ImgType.IMG_RAW}
            assert analyzer._reload_frame(raw_meta).shape == (2, H, W)
        finally:
            analyzer.shutdown(wait=False)

    @pytest.mark.parametrize("writer_cls", [TiffWriter, OmeZarrWriter])
    def test_ref_phase_frame_is_reloaded_from_ref(self, tmp_dir, writer_cls):
        """A ref phase frame is stored whole as "ref", not as "raw"."""
        analyzer = Analyzer(writer=writer_cls(tmp_dir))
        try:
            meta = {
                **_metadata(ImgType.IMG_REF),
                "channels": ["phase-contrast"],
                "ref_channels": [],
            }
            img = make_circle_image()[np.newaxis]
            analyzer._do_store(img, meta, "raw")
            np.testing.assert_array_equal(analyzer._reload_frame(meta), img)
        finally:
            analyzer.shutdown(wait=False)


class TestReferenceChannelsMissing:
    def test_pipeline_skips_the_reference_features(self, tmp_dir, caplog):
        """The imaging channels must not be measured as the reference."""
        pipeline = make_pipeline(tmp_dir, tracker=TrackerTrackpy(search_range=50))
        spy = SpyRefFE()
        pipeline.feature_extractor_ref = spy
        analyzer = Analyzer(pipeline, writer=TiffWriter(tmp_dir))
        try:
            meta = _metadata(ImgType.IMG_REF)
            meta["channels"] = ["phase-contrast"]
            event = MDAEvent(index={"t": 0, "p": 0}, metadata=meta)
            only_imaging = make_circle_image()[np.newaxis]  # no reference channel
            with caplog.at_level(logging.WARNING, logger="faro.core.pipeline"):
                pipeline.run(img=only_imaging, event=event)
            assert spy.calls == 0
            assert "has no reference channels" in caplog.text
        finally:
            analyzer.shutdown(wait=False)


class TestRefPhase:
    @pytest.mark.parametrize("with_writer", [False, True], ids=["no-writer", "writer"])
    def test_ref_phase_is_measured(self, tmp_dir, caplog, with_writer):
        """The README's ref phase, chained with combine(), gets ref features."""
        pipeline = make_pipeline(tmp_dir, tracker=TrackerTrackpy(search_range=50))
        spy = SpyRefFE()
        pipeline.feature_extractor_ref = spy
        writer = TiffWriter(tmp_dir) if with_writer else None
        ctrl = Controller(FakeMicroscope(CircleScene()), pipeline, writer=writer)
        with caplog.at_level(logging.WARNING, logger="faro.core.pipeline"):
            try:
                ctrl.run_experiment(_ref_phase_events(), validate=False).wait()
            finally:
                ctrl.finish_experiment()
        assert ctrl.background_errors == []
        assert spy.calls == 1
        assert "has no reference channels" not in caplog.text

    def test_events_json_keeps_img_type(self, tmp_dir):
        """events.json stores img_type by name and gives the enum back."""
        save_events_json(tmp_dir, _ref_phase_events())
        events = load_events_json(tmp_dir)
        assert events[-1].metadata["img_type"] is ImgType.IMG_REF
        assert "img_type" not in events[0].metadata


def _replay(project_path, storage_path):
    """Replay a stored run with ControllerSimulated; return the ref FE spy."""
    pipeline = make_pipeline(storage_path, tracker=TrackerTrackpy(search_range=50))
    spy = SpyRefFE()
    pipeline.feature_extractor_ref = spy
    ctrl = ControllerSimulated(FakeMicroscope(CircleScene()), pipeline, project_path)
    try:
        ctrl.run_experiment(_inline_ref_events(), validate=False).wait()
    finally:
        ctrl.finish_experiment()
    assert ctrl.background_errors == []
    return spy


class TestReplay:
    """ControllerSimulated gives the pipeline [imaging | ref], as the live run did."""

    @pytest.mark.parametrize("writer_cls", [TiffWriter, OmeZarrWriter])
    def test_replay_measures_the_reference_channel(self, tmp_dir, writer_cls):
        live_path = os.path.join(tmp_dir, "live")
        pipeline = make_pipeline(live_path, tracker=TrackerTrackpy(search_range=50))
        ctrl = Controller(
            FakeMicroscope(RefScene()), pipeline, writer=writer_cls(live_path)
        )
        try:
            ctrl.run_experiment(_inline_ref_events(), validate=False).wait()
        finally:
            ctrl.finish_experiment()

        spy = _replay(live_path, os.path.join(tmp_dir, "replay"))
        assert spy.calls == 1
        assert np.all(spy.images[0] == REF_VALUE)

    def test_replay_of_data_stored_before_the_split(self, tmp_dir):
        """Before 2026-05, ref/ held the whole [imaging | ref] stack."""
        old_path = os.path.join(tmp_dir, "old")
        circle = make_circle_image()
        writer = TiffWriter(old_path)
        writer.write(circle[np.newaxis], {"fname": "000_00000"}, "raw")
        stack = np.stack([circle, np.full_like(circle, REF_VALUE)])
        writer.write(stack, {"fname": "000_00001"}, "ref")

        spy = _replay(old_path, os.path.join(tmp_dir, "replay"))
        assert spy.calls == 1
        assert np.all(spy.images[0] == REF_VALUE)
