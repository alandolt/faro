"""Guards against runs that fail without anyone noticing.

On 2026-08-07 a 46-hour plate stored nothing and reported success: the
notebook assigned ``ctrl.pipeline`` and ``ctrl.writer``, which Python accepted
as new attributes that the Controller never read.
"""

from __future__ import annotations

import pytest

from faro.core.controller import Controller
from faro.core.writers import TiffWriter
from faro.tracking.trackpy import TrackerTrackpy

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import CircleScene, make_pipeline


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
