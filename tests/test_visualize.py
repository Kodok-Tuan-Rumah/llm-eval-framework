import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reports.visualize import (
    _load_faithfulness_threshold,
    load_latest,
    load_latest_summary,
    chart_clean,
    chart_distractor,
)

def test_load_faithfulness_threshold():
    thresh = _load_faithfulness_threshold()
    assert isinstance(thresh, float)
    assert 0.0 <= thresh <= 1.0

def test_load_latest_nonexistent():
    res = load_latest("nonexistent-model", "clean")
    assert res == []

def test_load_latest_summary_nonexistent():
    res = load_latest_summary("nonexistent-model", "clean")
    assert res == {}

@patch("reports.visualize.plt")
def test_chart_clean(mock_plt):
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    results = [
        {"case_id": "clean_001", "faithfulness_score": 1.0},
        {"case_id": "clean_002", "faithfulness_score": 0.5},
    ]

    chart_clean(results, model_name="test-model")
    mock_plt.subplots.assert_called_once()
    mock_plt.savefig.assert_called_once()
    mock_plt.close.assert_called_once()

@patch("reports.visualize.plt")
def test_chart_distractor(mock_plt):
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    results = [
        {"case_id": "dist_001", "faithfulness_score": 1.0},
    ]

    chart_distractor(results, model_name="test-model")
    mock_plt.subplots.assert_called_once()
    mock_plt.savefig.assert_called_once()
    mock_plt.close.assert_called_once()
