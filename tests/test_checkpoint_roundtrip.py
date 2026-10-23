"""Checkpoint serialization must preserve every field (E2).

Regression: checkpoint save/load intentionally dropped provenance and nested
data, so resuming a run rewrote thresholds/timestamps or produced
``FaithfulnessResult`` objects with ``model=""`` and empty claims/evidence.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import runner
from pipeline.model_client import QueryOutput
from evaluators.faithfulness import FaithfulnessResult


CLAIMS = [
    {
        "claim_id": "clean_001_c1",
        "text": "PATCH updates a resource.",
        "source_answer": "PATCH updates a resource.",
        "claim_type": "factual",
    },
    {
        "claim_id": "clean_001_c2",
        "text": "PUT replaces a resource.",
        "source_answer": "PUT replaces a resource.",
        "claim_type": "factual",
    },
]
EVIDENCE_MATCHES = [
    {
        "claim_id": "clean_001_c1",
        "best_match_in_context": "PATCH is used to update a resource.",
        "similarity_score": 0.9123,
        "threshold": 0.51,
        "supported": True,
    },
    {
        "claim_id": "clean_001_c2",
        "best_match_in_context": "PUT is used to replace a resource.",
        "similarity_score": 0.8871,
        "threshold": 0.51,
        "supported": True,
    },
]
FAILURE_CASES = [
    {
        "claim_id": "clean_001_c3",
        "unsupported_claim": "DELETE archives a resource.",
        "best_match_in_context": "DELETE removes a resource.",
        "similarity_score": 0.42,
        "threshold": 0.51,
        "technical_conflict": False,
        "diagnosis": "UNSUPPORTED",
    }
]


def _make_faithfulness_result() -> FaithfulnessResult:
    """Fully populated result with non-default values in every field."""
    return FaithfulnessResult(
        case_id="clean_001",
        model="mistral:7b",
        question="Which HTTP method updates a resource?",
        answer="PATCH updates a resource. PUT replaces a resource.",
        context_preview="PATCH is used to update a resource. PUT is used to replace a resource.",
        total_claims=3,
        claims=list(CLAIMS),
        evidence_matches=list(EVIDENCE_MATCHES),
        supported_count=2,
        unsupported_count=1,
        skipped_count=0,
        faithfulness_score=2 / 3,
        is_insufficient_context_response=False,
        has_failure=True,
        failure_cases=list(FAILURE_CASES),
        similarity_threshold=0.51,
        embedding_model="nomic-embed-text-v1.5",
        evaluator_version="9.9.9",
        timestamp_utc="2020-01-02T03:04:05+00:00",
        evaluation_path="claim_extraction",
    )


def _make_query_output() -> QueryOutput:
    return QueryOutput(
        case_id="clean_001",
        model="mistral:7b",
        question="Which HTTP method updates a resource?",
        context_length_chars=1234,
        answer="PATCH updates a resource.",
        latency_seconds=1.234,
        timestamp_utc="2020-01-02T03:04:05+00:00",
        success=True,
        error_message=None,
    )


@pytest.fixture
def checkpoint_dir(tmp_path, monkeypatch):
    cdir = tmp_path / "checkpoints"
    monkeypatch.setattr(runner, "CHECKPOINT_DIR", cdir)
    return cdir


def test_faithfulness_roundtrip_preserves_all_fields():
    original = _make_faithfulness_result()
    payload = runner._faithfulness_to_dict(original)

    restored = runner._checkpoint_to_faithfulness_result(
        payload,
        # Current-run values must not replace stored provenance.
        similarity_threshold=0.99,
        embedding_model="current-embedding-model",
        evaluator_version="current-evaluator-version",
    )

    assert asdict(restored) == asdict(original)
    assert restored.model == "mistral:7b"
    assert restored.claims == CLAIMS
    assert restored.evidence_matches == EVIDENCE_MATCHES


def test_query_output_roundtrip_preserves_provenance():
    original = _make_query_output()

    payload = runner._query_output_to_dict(original)
    restored = runner._dict_to_query_output(payload)

    assert asdict(restored) == asdict(original)
    assert restored.model == "mistral:7b"
    assert restored.question == "Which HTTP method updates a resource?"
    assert restored.context_length_chars == 1234
    assert restored.timestamp_utc == "2020-01-02T03:04:05+00:00"


def test_resume_does_not_rewrite_threshold_or_timestamp(checkpoint_dir):
    stored = _make_faithfulness_result()
    runner._save_checkpoint(
        "mistral",
        "clean",
        "clean_001",
        {
            "case_id": "clean_001",
            "model": "mistral",
            "dataset": "clean",
            "timestamp_utc": "2020-01-02T03:04:05+00:00",
            "query_result": runner._query_output_to_dict(_make_query_output()),
            "faithfulness_result": runner._faithfulness_to_dict(stored),
        },
    )

    case = {
        "id": "clean_001",
        "question": "Which HTTP method updates a resource?",
        "context": "PATCH is used to update a resource.",
    }
    resumed = runner.process_single_case(
        case=case,
        model_name="mistral",
        dataset_name="clean",
        similarity_threshold=0.99,
        resume=True,
        client=None,
        evaluator=None,
        embedding_model="current-embedding-model",
        evaluator_version="current-evaluator-version",
    )

    faith = resumed["faithfulness_result"]
    assert faith.similarity_threshold == stored.similarity_threshold == 0.51
    assert faith.embedding_model == stored.embedding_model == "nomic-embed-text-v1.5"
    assert faith.evaluator_version == stored.evaluator_version == "9.9.9"
    assert faith.timestamp_utc == stored.timestamp_utc == "2020-01-02T03:04:05+00:00"

    query = resumed["query_result"]
    assert query.model == "mistral:7b"
    assert query.question == "Which HTTP method updates a resource?"
    assert query.context_length_chars == 1234
    assert query.timestamp_utc == "2020-01-02T03:04:05+00:00"


def test_schema_2_checkpoint_without_nested_fields_is_ignored_and_rerun(
    checkpoint_dir, monkeypatch
):
    """E1-era schema-2 checkpoints must be rejected, not resumed with defaults."""
    path = runner._checkpoint_path("mistral", "clean", "clean_001")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "case_id": "clean_001",
                "model": "mistral",
                "dataset": "clean",
                "timestamp_utc": "2020-01-02T03:04:05+00:00",
                # E1-era query_result had no model/question/context/timestamp.
                "query_result": {
                    "case_id": "clean_001",
                    "success": True,
                    "answer": "PATCH updates a resource.",
                    "latency_seconds": 1.0,
                    "error_message": None,
                },
                # E1-era faithfulness_result had required fields only: no
                # context_preview, claims, evidence_matches, or skipped_count.
                "faithfulness_result": {
                    "case_id": "clean_001",
                    "model": "mistral",
                    "question": "Which HTTP method updates a resource?",
                    "answer": "PATCH updates a resource.",
                    "faithfulness_score": 0.8,
                    "supported_count": 2,
                    "total_claims": 2,
                    "similarity_threshold": 0.51,
                    "embedding_model": "nomic-embed-text",
                    "evaluator_version": "9.9.9",
                    "timestamp_utc": "2020-01-02T03:04:05+00:00",
                    "has_failure": False,
                    "is_insufficient_context_response": False,
                    "failure_cases": [],
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = runner._load_checkpoint("mistral", "clean", "clean_001")
    assert loaded is None, "schema-2 checkpoint must be ignored after the E2 format change"

    case = {
        "id": "clean_001",
        "question": "Which HTTP method updates a resource?",
        "context": "PATCH is used to update a resource.",
    }
    monkeypatch.setattr(runner, "_query_with_retry", lambda *a, **k: _make_query_output())

    class DummyEvaluator:
        def evaluate(self, case, model_answer, model_name):
            return _make_faithfulness_result()

    rerun = runner.process_single_case(
        case=case,
        model_name="mistral",
        dataset_name="clean",
        similarity_threshold=0.51,
        resume=True,
        client=None,
        evaluator=DummyEvaluator(),
        embedding_model="nomic-embed-text-v1.5",
        evaluator_version="9.9.9",
    )

    assert rerun["cached"] is False, "schema-2 checkpoint must trigger a re-run"
    assert rerun["faithfulness_result"].claims == CLAIMS
    assert rerun["faithfulness_result"].evidence_matches == EVIDENCE_MATCHES


def test_claims_and_evidence_matches_survive_resume(checkpoint_dir):
    stored = _make_faithfulness_result()
    runner._save_checkpoint(
        "mistral",
        "clean",
        "clean_001",
        {
            "case_id": "clean_001",
            "model": "mistral",
            "dataset": "clean",
            "timestamp_utc": "2020-01-02T03:04:05+00:00",
            "query_result": runner._query_output_to_dict(_make_query_output()),
            "faithfulness_result": runner._faithfulness_to_dict(stored),
        },
    )

    case = {
        "id": "clean_001",
        "question": "Which HTTP method updates a resource?",
        "context": "PATCH is used to update a resource.",
    }
    resumed = runner.process_single_case(
        case=case,
        model_name="mistral",
        dataset_name="clean",
        similarity_threshold=0.99,
        resume=True,
        client=None,
        evaluator=None,
        embedding_model="current-embedding-model",
        evaluator_version="current-evaluator-version",
    )

    faith = resumed["faithfulness_result"]
    assert faith.claims == CLAIMS
    assert faith.evidence_matches == EVIDENCE_MATCHES
    assert faith.context_preview == stored.context_preview
    assert faith.answer == stored.answer
    assert faith.skipped_count == stored.skipped_count
    assert faith.evaluation_path == stored.evaluation_path
    assert faith.failure_cases == FAILURE_CASES
