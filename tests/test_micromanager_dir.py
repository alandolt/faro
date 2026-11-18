"""Which Micro-Manager folder a scope uses (``PyMMCoreMicroscope.micromanager_dir``)."""

from __future__ import annotations

import pytest

from faro.microscope.pymmcore import PyMMCoreMicroscope


class Scope(PyMMCoreMicroscope):
    MICROMANAGER_PATH = r"C:\Program Files\Micro-Manager-2.0_api74"


@pytest.fixture
def scope(monkeypatch):
    monkeypatch.delenv("MICROMANAGER_PATH", raising=False)
    return Scope()


def test_class_folder_last(scope):
    assert scope.micromanager_dir() == Scope.MICROMANAGER_PATH


def test_environment_variable_beats_class_folder(scope, monkeypatch):
    monkeypatch.setenv("MICROMANAGER_PATH", r"D:\bundle\Micro-Manager-2.0")
    assert scope.micromanager_dir() == r"D:\bundle\Micro-Manager-2.0"


def test_folder_by_hand_beats_everything(scope, monkeypatch):
    monkeypatch.setenv("MICROMANAGER_PATH", r"D:\bundle\Micro-Manager-2.0")
    assert scope.micromanager_dir(r"E:\MM-test") == r"E:\MM-test"


def test_empty_values_are_skipped(scope, monkeypatch):
    monkeypatch.setenv("MICROMANAGER_PATH", "")
    assert scope.micromanager_dir("") == Scope.MICROMANAGER_PATH
