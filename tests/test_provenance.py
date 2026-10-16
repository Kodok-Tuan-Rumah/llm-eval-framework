"""Tests for pipeline/provenance.py"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import provenance  # noqa: E402


def test_sha256_file_none():
    assert provenance.sha256_file(None) is None
    assert provenance.sha256_file('/no/such/file.json') is None


def test_config_snapshot_has_thresholds():
    cs = provenance.config_snapshot()
    assert 'thresholds' in cs
    assert 'models' in cs
    assert 'inference' in cs


def test_build_provenance_keys():
    p = provenance.build_provenance(
        evaluator_version='1.2.2',
        dataset_name='clean',
        dataset_path=Path(__file__).resolve().parents[1] / 'data' / 'dataset' / 'clean_cases.json',
    )
    for k in ('evaluator_version', 'git_commit', 'git_dirty', 'config_snapshot', 'dataset', 'generated_at'):
        assert k in p
    assert p['dataset']['name'] == 'clean'
    assert isinstance(p['dataset']['sha256'], str) or p['dataset']['sha256'] is None
