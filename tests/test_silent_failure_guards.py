"""Guards against runs that fail without anyone noticing.

On 2026-08-07 a 46-hour plate stored nothing and reported success: the
notebook assigned ``ctrl.pipeline`` and ``ctrl.writer``, which Python accepted
as new attributes that the Controller never read.
"""

from __future__ import annotations

import os
import time

import pytest

import faro.core.controller as controller_module
from faro.core.controller import KEEP_ERRORS_PER_KIND, Analyzer, Controller
from faro.core.writers import TiffWriter
from faro.tracking.trackpy import TrackerTrackpy

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import CircleScene, make_events, make_pipeline


class TestControllerAttributes:
    def test_pipeline_and_writer_are_readable(self, tmp_dir):
        pipeline = make_pipeline(tmp_dir, tracker=TrackerTrackpy(search_range=50))
        writer = TiffWriter(tmp_dir)
        ctrl = Controller(FakeMicroscope(CircleScene()), pipeline, writer=writer)
        assert ctrl.pipeline is pipeline
        assert ctrl.writer is writer

    def test_assigning_pipeline_or_writer_raises(self, tmp_dir):
        ctrl = Controller(FakeMicroscope(CircleScene()), None)
        pipeline = make_pipeline(tmp_dir, tracker=TrackerTrackpy(search_range=50))
        with pytest.raises(AttributeError, match="new Controller"):
            ctrl.pipeline = pipeline
        with pytest.raises(AttributeError, match="new Controller"):
            ctrl.writer = TiffWriter(tmp_dir)
        assert ctrl.pipeline is None and ctrl.writer is None


class TestRunThatStoresNothing:
    def test_run_without_writer_and_pipeline_is_refused(self):
        ctrl = Controller(FakeMicroscope(CircleScene()), None)
        with pytest.raises(RuntimeError, match="store nothing"):
            ctrl.run_experiment(make_events(2), validate=False)
        with pytest.raises(RuntimeError, match="store nothing"):
            ctrl.load_experiment(make_events(2), validate=False)

    def test_run_with_only_a_writer_is_allowed(self, tmp_dir):
        ctrl = Controller(
            FakeMicroscope(CircleScene()), None, writer=TiffWriter(tmp_dir)
        )
        ctrl.run_experiment(make_events(2), validate=False).wait()
        ctrl.finish_experiment()
        assert ctrl.background_errors == []
        assert sorted(os.listdir(os.path.join(tmp_dir, "raw"))) == [
            "000_00000.tiff",
            "000_00001.tiff",
        ]


class TestErrorCapping:
    def test_only_the_first_errors_of_a_kind_are_kept(self):
        analyzer = Analyzer()  # no pipeline, no writer
        try:
            for i in range(250):
                try:
                    raise ValueError(f"bug {i}")
                except ValueError as e:
                    analyzer._record_background_error("pipeline", e)
            try:
                raise KeyError("other")
            except KeyError as e:
                analyzer._record_background_error("storage", e)

            kept = [(e.source, e.exc_type) for e in analyzer.background_errors]
            assert kept.count(("pipeline", "ValueError")) == KEEP_ERRORS_PER_KIND
            assert kept.count(("storage", "KeyError")) == 1
            assert analyzer.error_summary() == {
                ("pipeline", "ValueError"): 250,
                ("storage", "KeyError"): 1,
            }
        finally:
            analyzer.shutdown(wait=False)

    def test_controller_collects_the_counts(self, tmp_dir, monkeypatch):
        # 8 failing frames, with a lower limit than the default.
        monkeypatch.setattr(controller_module, "KEEP_ERRORS_PER_KIND", 5)
        pipeline = make_pipeline(tmp_dir, tracker=TrackerTrackpy(search_range=50))

        def broken_run(*args, **kwargs):
            raise ValueError("pipeline bug")

        monkeypatch.setattr(pipeline, "run", broken_run)
        ctrl = Controller(FakeMicroscope(CircleScene()), pipeline)
        ctrl.run_experiment(make_events(8), validate=False).wait()
        ctrl.finish_experiment()
        assert ctrl.error_summary()[("pipeline", "ValueError")] == 8
        assert len(ctrl.background_errors) == 5


class TestWaitIdle:
    def test_waits_for_a_frame_that_is_still_being_stored(self, tmp_dir, monkeypatch):
        """wait_idle returned while the storage worker still wrote the last frame.

        The queue was empty, but the frame was not handed to the pipeline yet,
        so a caller could read the tracks before the last frame was in them.
        """
        store = Analyzer._do_store

        def slow_store(self, *args, **kwargs):
            time.sleep(0.3)
            store(self, *args, **kwargs)

        monkeypatch.setattr(Analyzer, "_do_store", slow_store)
        pipeline = make_pipeline(tmp_dir, tracker=TrackerTrackpy(search_range=50))
        ctrl = Controller(FakeMicroscope(CircleScene()), pipeline)
        ctrl.run_experiment(make_events(3), validate=False).wait()
        assert ctrl._analyzer.wait_idle(timeout=10)
        assert ctrl._analyzer.stored_images == 3
        ctrl.finish_experiment()
