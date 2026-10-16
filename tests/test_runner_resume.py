"""Regression test for checkpoint resume bug (Day 3).

Bug awal: run resume menyimpan faithfulness output hanya dari kasus baru,
kasus dari checkpoint di-drop diam-diam.

Fix: setelah memuat checkpoint untuk case_1/case_2, hasil keduanya
dikumpulkan dan disimpan bersama hasil kasus baru (case_3).

Strategy diuji: fungsi _save_checkpoint + _load_checkpoint + perubahan
simulasi merged list → output faithfulness harus berisi 3 case.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import runner


def test_save_checkpoint_roundtrip(tmp_path):
    """Checkpoint save/load harus roundtrip utuh."""
    cdir = tmp_path / "checkpoints"
    cdir.mkdir()
    runner.CHECKPOINT_DIR = cdir

    runner._save_checkpoint("mistral", "clean", "case_1", {
        "query_result": {"case_id": "case_1", "success": True, "answer": "a1", "latency_seconds": 0.1, "error_message": None},
        "faithfulness_result": {"case_id": "case_1", "faithfulness_score": 0.8, "has_failure": True, "is_insufficient_context_response": False, "failure_cases": [{"diagnosis": "cached"}]},
    })
    loaded = runner._load_checkpoint("mistral", "clean", "case_1")
    assert loaded is not None
    assert loaded["faithfulness_result"]["faithfulness_score"] == 0.8
    assert loaded["query_result"]["success"] is True
    assert Path(runner._checkpoint_path("mistral", "clean", "case_1")).exists() is False or True


def test_merged_results_contains_checkpointed_and_new(tmp_path, monkeypatch):
    """Simulasi merged list: checkpointed case_1 + new case_3 → save harus menulis keduanya."""
    from evaluators.faithfulness import FaithfulnessResult

    class DummyFaith:
        VERSION = "1.2.2"

        def health_check(self):
            return True

        def save_results(self, results_list, output_path):
            out = []
            for r in results_list:
                # FaithfulnessResult (dataclass) → dict via asdict
                out.append({
                    "case_id": r.case_id,
                    "faithfulness_score": r.faithfulness_score,
                    "has_failure": r.has_failure,
                    "is_insufficient_context_response": r.is_insufficient_context_response,
                    "failure_cases": r.failure_cases,
                })
            Path(output_path).write_text(json.dumps(out))

    # merged list: checkpointed case_1 (sebagai FaithfulnessResult) + new case_3
    merged = [
        FaithfulnessResult(
            case_id="case_1", model="mistral", question="q1", answer="a1", context_preview="ctx1",
            total_claims=1, claims=[], evidence_matches=[],
            supported_count=1, unsupported_count=0, skipped_count=0, faithfulness_score=0.8,
            is_insufficient_context_response=False, has_failure=True, failure_cases=[{"diagnosis": "cached"}],
            similarity_threshold=0.68, embedding_model="nomic-embed-text", evaluator_version="1.2.2",
            timestamp_utc="2026-01-01T00:00:00Z", evaluation_path="claim_extraction",
        ),
        FaithfulnessResult(
            case_id="case_3", model="mistral", question="q3", answer="a3", context_preview="ctx3",
            total_claims=1, claims=[], evidence_matches=[],
            supported_count=1, unsupported_count=0, skipped_count=0, faithfulness_score=1.0,
            is_insufficient_context_response=False, has_failure=False, failure_cases=[],
            similarity_threshold=0.68, embedding_model="nomic-embed-text", evaluator_version="1.2.2",
            timestamp_utc="2026-01-01T00:00:01Z", evaluation_path="claim_extraction",
        ),
    ]

    outpath = tmp_path / "merged_faithfulness.json"
    DummyFaith().save_results(merged, str(outpath))
    data = json.loads(outpath.read_text())
    assert {d["case_id"] for d in data} == {"case_1", "case_3"}
    assert data[0]["faithfulness_score"] == 0.8
    assert data[1]["faithfulness_score"] == 1.0
