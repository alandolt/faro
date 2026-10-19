"""Reference frames in the pipeline.

A reference frame is stored in two parts: its imaging channels as "raw" and
its reference channels as "ref". When the pipeline is overloaded, a frame is
reloaded from storage later; the reload must give the pipeline both parts.
ControllerSimulated replays stored frames and must do the same, and so must
the reanalysis (ImageProcessingPipeline_postExperiment).

A ref phase (``rtm_metadata={"img_type": ImgType.IMG_REF}`` without
``ref_channels``, see README) has no separate reference channels: its imaging
channels are the reference. It is stored whole as "ref".
"""

from __future__ import annotations

import logging
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from useq import MDAEvent

from faro.core.controller import Analyzer, Controller, ControllerSimulated
from faro.core.conversion import (
    convert_tiff_to_omezarr,
    events_to_df,
    load_events_json,
    save_events_json,
)
from faro.core.data_structures import (
    Channel,
    ImgType,
    RTMEvent,
    RTMSequence,
    SegmentationMethod,
    combine,
)
from faro.core.pipeline_post import ImageProcessingPipeline_postExperiment
from faro.core.writers import OmeZarrWriter, TiffWriter
from faro.feature_extraction.base_ref import FeatureExtractorRef
from faro.feature_extraction.ref import RefFE
from faro.feature_extraction.simple import SimpleFE
from faro.segmentation.base import OtsuSegmentator
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


def _inline_ref_events(n_timepoints=2):
    """Imaging at each timepoint, and one reference channel at t=1."""
    ref = (Channel(config="stim-405", exposure=10),)
    return [
        RTMEvent(
            index={"t": t, "p": 0},
            channels=(Channel(config="phase-contrast", exposure=10),),
            ref_channels=ref if t == 1 else (),
            metadata={},
        )
        for t in range(n_timepoints)
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


class TestRefFE:
    def test_particle_0_is_measured(self):
        """trackpy and motile give particle 0 to a cell; it must get a value."""
        labels = np.zeros((4, 4), dtype=np.uint16)
        labels[0, 0], labels[3, 3] = 1, 2
        ref = np.zeros((1, 4, 4), dtype=np.uint16)
        ref[0, 0, 0], ref[0, 3, 3] = 10, 20
        # The newest timestep (1) has other label -> particle pairs than t=0.
        df = pd.DataFrame(
            {
                "timestep": [0, 0, 1, 1],
                "fov": 0,
                "fname": ["000_00000"] * 2 + ["000_00001"] * 2,
                "label": [1, 2, 1, 2],
                "particle": [0, 1, 1, 0],
            }
        )
        out = RefFE("labels").extract_features(
            {"labels": labels}, ref, df, {"timestep": 1, "fov": 0}
        )
        newest = out[out["timestep"] == 1].set_index("particle")
        assert newest["ref_mean_intensity"].to_dict() == {0: 20.0, 1: 10.0}
        assert out.loc[out["timestep"] == 0, "ref_mean_intensity"].isna().all()


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

    @pytest.mark.parametrize("writer_cls", [TiffWriter, OmeZarrWriter])
    def test_stack_is_reloaded_from_raw(self, tmp_dir, writer_cls):
        """With as many imaging as reference channels, ref/ cannot tell this
        stack from a reference image, so it is stored as raw."""
        writer = writer_cls(tmp_dir)
        if writer_cls is OmeZarrWriter:
            writer.init_stream(["Pos0"], ["phase-contrast"], H, W, n_timepoints=1)
        analyzer = Analyzer(writer=writer)
        try:
            meta = {**_metadata(ImgType.IMG_REF), "channels": ["phase-contrast"]}
            img = make_circle_image()[np.newaxis]
            analyzer._do_store(img, meta, "raw")
            np.testing.assert_array_equal(analyzer._reload_frame(meta), img)
        finally:
            analyzer.shutdown(wait=False)

    def test_replay_tracks_the_frame(self, tmp_dir):
        live_path = os.path.join(tmp_dir, "live")
        analyzer = Analyzer(writer=TiffWriter(live_path))
        try:
            for t, img_type in enumerate([ImgType.IMG_RAW, ImgType.IMG_REF]):
                meta = {**_metadata(img_type), "channels": ["phase-contrast"]}
                meta.update(timestep=t, fname=f"000_{t:05d}")
                analyzer._do_store(make_circle_image()[np.newaxis], meta, "raw")
        finally:
            analyzer.shutdown(wait=True)
        replay_path = os.path.join(tmp_dir, "replay")
        assert _replay(live_path, replay_path).calls == 0
        tracks = pd.read_parquet(
            os.path.join(replay_path, "tracks", "0_latest.parquet")
        )
        assert sorted(tracks.timestep.unique()) == [0, 1]


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


def _run_live(path, writer_cls, events, scene):
    """Run events on the fake microscope and store them with writer_cls."""
    pipeline = make_pipeline(path, tracker=TrackerTrackpy(search_range=50))
    ctrl = Controller(FakeMicroscope(scene), pipeline, writer=writer_cls(path))
    try:
        ctrl.run_experiment(events, validate=False).wait()
    finally:
        ctrl.finish_experiment()


def _reanalyse(live_path, out_path, writer_cls, ref_fe=None, df_acquire=None):
    """Reanalyse a stored run as the reanalysis template does; return the ref FE.

    The ref FE is ref_fe, or a SpyRefFE if ref_fe is None. With df_acquire,
    the reanalysis reads it instead of events.json.
    """
    spy = SpyRefFE() if ref_fe is None else ref_fe
    post = ImageProcessingPipeline_postExperiment(
        img_storage_path=live_path,
        out_path=out_path,
        events=None if df_acquire is not None else load_events_json(live_path),
        df_acquire=df_acquire,
        segmentators=[SegmentationMethod("labels", OtsuSegmentator(), 0, False)],
        feature_extractor=SimpleFE("labels"),
        tracker=TrackerTrackpy(search_range=50),
        feature_extractor_ref=spy,
        n_jobs=1,  # n_jobs > 1 only prints the error of a FOV
        writer=writer_cls(out_path),
    )
    post.run()
    return spy


class TestReanalysis:
    @pytest.mark.parametrize("writer_cls", [TiffWriter, OmeZarrWriter])
    def test_ref_phase_is_measured(self, tmp_dir, writer_cls):
        """A ref phase frame is read from ref/, where the live run stored it."""
        live_path = os.path.join(tmp_dir, "live")
        _run_live(live_path, writer_cls, _ref_phase_events(), CircleScene())
        spy = _reanalyse(live_path, os.path.join(tmp_dir, "out"), writer_cls)
        assert spy.calls == 1

    @pytest.mark.parametrize("writer_cls", [TiffWriter, OmeZarrWriter])
    def test_inline_reference_channel_is_measured(self, tmp_dir, writer_cls):
        """A frame with ref_channels gets the reference features of ref/."""
        live_path = os.path.join(tmp_dir, "live")
        _run_live(live_path, writer_cls, _inline_ref_events(), RefScene())
        spy = _reanalyse(live_path, os.path.join(tmp_dir, "out"), writer_cls)
        assert spy.calls == 1
        assert np.all(spy.images[0] == REF_VALUE)

    def test_df_acquire_from_parquet_is_measured(self, tmp_dir):
        """ref_channels read back from parquet are arrays, not lists."""
        live_path = os.path.join(tmp_dir, "live")
        _run_live(live_path, TiffWriter, _inline_ref_events(), RefScene())
        parquet = os.path.join(tmp_dir, "acquire.parquet")
        events_to_df(load_events_json(live_path)).to_parquet(parquet)
        df_acquire = pd.read_parquet(parquet)
        out_path = os.path.join(tmp_dir, "out")
        spy = _reanalyse(live_path, out_path, TiffWriter, df_acquire=df_acquire)
        assert spy.calls == 1

    @pytest.mark.parametrize("writer_cls", [TiffWriter, OmeZarrWriter])
    @pytest.mark.parametrize(
        "events", [_inline_ref_events, _ref_phase_events], ids=["inline", "phase"]
    )
    def test_missing_ref_image_keeps_the_frame_results(
        self, tmp_dir, events, writer_cls
    ):
        """Without ref/, a reference frame keeps its labels, only no ref
        features. A ref phase frame has no other image and is skipped.
        """
        live_path = os.path.join(tmp_dir, "live")
        _run_live(live_path, writer_cls, events(), RefScene())
        shutil.rmtree(os.path.join(live_path, "ref"))
        out_path = os.path.join(tmp_dir, "out")
        spy = _reanalyse(live_path, out_path, TiffWriter)
        assert spy.calls == 0
        assert sorted(os.listdir(os.path.join(out_path, "labels"))) == [
            "000_00000.tiff",
            "000_00001.tiff",
        ]

    @pytest.mark.parametrize("writer_cls", [TiffWriter, OmeZarrWriter])
    @pytest.mark.parametrize(
        "events", [_inline_ref_events, _ref_phase_events], ids=["inline", "phase"]
    )
    def test_empty_fov_keeps_its_tracks(self, tmp_dir, events, writer_cls):
        """A FOV without cells gets no ref features, as live, but its tracks."""
        live_path = os.path.join(tmp_dir, "live")
        _run_live(live_path, writer_cls, events(), CircleScene(blank_frames={0, 1, 2}))
        out_path = os.path.join(tmp_dir, "out")
        _reanalyse(live_path, out_path, writer_cls, ref_fe=RefFE("labels"))
        assert os.listdir(os.path.join(out_path, "tracks")) == ["0_latest.parquet"]

    @pytest.mark.parametrize("source", ["tiff", "zarr"])
    @pytest.mark.parametrize("n_timepoints", [2, 3], ids=["ref-last", "ref-middle"])
    def test_ref_holds_the_whole_stack_of_old_data(self, tmp_dir, source, n_timepoints):
        """Before 2026-05, ref/ held the whole [imaging | ref] stack, and raw
        had no frame for it. A zarr store converted from such data has an
        empty frame there, or no frame when it is the last timepoint.
        """
        old_path = os.path.join(tmp_dir, "old")
        circle = make_circle_image()
        writer = TiffWriter(old_path)
        for t in range(n_timepoints):
            if t != 1:
                writer.write(circle[np.newaxis], {"fname": f"000_{t:05d}"}, "raw")
        stack = np.stack([circle, np.full_like(circle, REF_VALUE)])
        writer.write(stack, {"fname": "000_00001"}, "ref")
        if source == "zarr":
            converted_path = os.path.join(tmp_dir, "converted")
            convert_tiff_to_omezarr(old_path, converted_path, verbose=False)
            old_path = converted_path
        save_events_json(old_path, _inline_ref_events(n_timepoints))

        out_path = os.path.join(tmp_dir, "out")
        spy = _reanalyse(old_path, out_path, TiffWriter)
        assert spy.calls == 1
        assert spy.images[0].shape == (1, H, W)  # the reference channel only
        assert np.all(spy.images[0] == REF_VALUE)
        # t=1 is segmented on the imaging channel from ref/
        tracks = pd.read_parquet(os.path.join(out_path, "tracks", "0_latest.parquet"))
        n_cells = tracks.groupby("timestep").size()
        assert n_cells.get(1, 0) == n_cells[0] > 0
