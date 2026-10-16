"""Tests for scripts/check_reproducibility.py

Behavior:
- If no artifacts: exit 0 (skip, safe for historical state)
- If artifacts exist: integrity + provenance checks
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import check_reproducibility as cr  # noqa: E402


def test_parse_readme_averages():
    text = """| Avg faithfulness | **56.7%** | 46.7% ⚠️ |"""
    got = cr.parse_readme_averages(text)
    assert got['m_readme'] == '56.7'
    assert got['p_readme'] == '46.7'


def test_parse_readme_distractor_avg():
    text = """| Distractor | **8.3%** | Normal context |"""
    assert cr.parse_readme_distractor_avg(text) == '8.3'
