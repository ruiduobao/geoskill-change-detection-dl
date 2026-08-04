"""Pytest fixtures for change-detection-dl tests."""
import importlib.util
import os
import sys
import pytest

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(SKILL_DIR, "geoskill-change-detection-dl.py")


def load_module():
    spec = importlib.util.spec_from_file_location("change_detection_dl", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["change_detection_dl"] = module
    spec.loader.exec_module(module)
    return module


mod = load_module()


@pytest.fixture
def skill_dir():
    return SKILL_DIR


@pytest.fixture
def script_path():
    return SCRIPT
