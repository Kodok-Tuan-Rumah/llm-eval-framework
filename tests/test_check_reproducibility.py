import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_reproducibility as cr


def test_parse_readme_averages():
    text = """| Avg faithfulness | **56.7%** | 46.7% [WARNING] |"""
    got = cr.parse_readme_averages(text)
    assert got["m_readme"] == "56.7"
    assert got["p_readme"] == "46.7"


def test_parse_readme_distractor_avg():
    text = """| Distractor | **8.3%** | Normal context |"""
    assert cr.parse_readme_distractor_avg(text) == "8.3"
