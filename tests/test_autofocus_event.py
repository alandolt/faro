"""USE_AUTOFOCUS_EVENT: a HardwareAutofocus event at each new stage position.

The Controller adds the autofocus event before the events of an RTMEvent
when the stage goes to a new XY position, and once at the start of the run.
An RTMSequence ignores useq's autofocus_plan, which would otherwise acquire
a channel twice.
"""

from __future__ import annotations

import pytest
import useq
from useq import HardwareAutofocus

from faro.core.controller import Controller
from faro.core.data_structures import Channel, RTMEvent, RTMSequence
from faro.tracking.trackpy import TrackerTrackpy

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import CircleScene, make_pipeline


def _events(positions: list[tuple[int, float, float]]) -> list[RTMEvent]:
    """One RTMEvent per (t, x, y). The p index follows the XY position."""
    p_of: dict[tuple[float, float], int] = {}
    events = []
    for i, (t, x, y) in enumerate(positions):
        p = p_of.setdefault((x, y), len(p_of))
        events.append(
            RTMEvent(
                index={"t": t, "p": p},
                channels=(Channel(config="phase-contrast", exposure=10),),
                x_pos=x,
                y_pos=y,
                min_start_time=i * 0.01,
                metadata={},
            )
        )
    return events


def _run(tmp_path, events, *, use_af: bool) -> tuple[list, int]:
    """Run ``events``; return the started events as (action, t, p, x, y)
    and the number of frames."""
    mic = FakeMicroscope(CircleScene())
    mic.USE_AUTOFOCUS_EVENT = use_af
    started = []
    mic.mmc.mda.events.eventStarted.connect(
        lambda ev: started.append(
            (
                type(ev.action).__name__,
                ev.index.get("t"),
                ev.index.get("p"),
                ev.x_pos,
                ev.y_pos,
            )
        )
    )
    frames = []
    mic.mmc.mda.events.frameReady.connect(lambda img, ev, *_: frames.append(ev))

    pipeline = make_pipeline(
        str(tmp_path),
        tracker=TrackerTrackpy(search_range=50, memory=3),
        with_stim=False,
    )
    ctrl = Controller(mic, pipeline)
    handle = ctrl.run_experiment(events, validate=False)
    handle.wait()
    ctrl._analyzer.wait_idle()
    ctrl._analyzer.shutdown(wait=True)
    return started, len(frames)


def _af(started):
    return [s for s in started if s[0] == "HardwareAutofocus"]


def test_no_autofocus_when_flag_off(tmp_path):
    events = _events([(0, 0, 0), (0, 100, 0), (1, 0, 0), (1, 100, 0)])
    started, n_frames = _run(tmp_path, events, use_af=False)
    assert _af(started) == []
    assert n_frames == 4


def test_autofocus_at_each_position_change(tmp_path):
    # Two positions, two timepoints, position-first order: the stage moves
    # before every event.
    events = _events([(0, 0, 0), (0, 100, 0), (1, 0, 0), (1, 100, 0)])
    started, n_frames = _run(tmp_path, events, use_af=True)

    assert [s[1:] for s in _af(started)] == [
        (0, 0, 0, 0),
        (0, 1, 100, 0),
        (1, 0, 0, 0),
        (1, 1, 100, 0),
    ]
    # Each autofocus comes right before the image at its position.
    for i, s in enumerate(started):
        if s[0] == "HardwareAutofocus":
            nxt = started[i + 1]
            assert nxt[0] != "HardwareAutofocus"
            assert nxt[1:3] == s[1:3]
    # The autofocus gives no frame.
    assert n_frames == 4


def test_autofocus_once_when_the_stage_stays(tmp_path):
    events = _events([(0, 5, 5), (1, 5, 5), (2, 5, 5)])
    started, n_frames = _run(tmp_path, events, use_af=True)
    assert [s[1:] for s in _af(started)] == [(0, 0, 5, 5)]
    assert started[0][0] == "HardwareAutofocus"
    assert n_frames == 3


def test_autofocus_once_without_stage_positions(tmp_path):
    events = [
        RTMEvent(
            index={"t": t, "p": 0},
            channels=(Channel(config="phase-contrast", exposure=10),),
            metadata={},
        )
        for t in range(3)
    ]
    started, _ = _run(tmp_path, events, use_af=True)
    assert len(_af(started)) == 1
    assert started[0][0] == "HardwareAutofocus"


def test_rtm_sequence_ignores_autofocus_plan():
    kw = dict(
        time_plan={"interval": 1, "loops": 2},
        stage_positions=[(0, 0, 0), (10, 10, 0)],
        channels=[{"config": "DAPI", "exposure": 10}],
    )
    plain = list(RTMSequence(**kw))
    with pytest.warns(UserWarning, match="USE_AUTOFOCUS_EVENT"):
        seq = RTMSequence(
            **kw,
            autofocus_plan=useq.AxesBasedAF(autofocus_motor_offset=50, axes=("t", "p")),
        )
    with_af = list(seq)
    # Before the fix, the autofocus event added a second DAPI per (t, p).
    assert [[c.config for c in e.channels] for e in with_af] == [["DAPI"]] * 4
    assert [e.index for e in with_af] == [e.index for e in plain]
    assert all(not isinstance(e.action, HardwareAutofocus) for e in with_af)
