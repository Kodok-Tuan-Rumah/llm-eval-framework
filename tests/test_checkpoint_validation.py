import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluators.faithfulness import FaithfulnessResult
from pipeline import runner


def _complete_faithfulness(case_id: str = "case_1", score: float = 0.8) -> dict:
    return {
        "case_id": case_id,
        "model": "mistral",
        "question": f"q-{case_id}",
        "answer": "a",
        "faithfulness_score": score,
        "supported_count": 1,
        "total_claims": 1,
        "similarity_threshold": 0.68,
        "embedding_model": "nomic-embed-text",
        "evaluator_version": "1.2.2",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "unsupported_count": 0,
        "has_failure": False,
        "is_insufficient_context_response": False,
        "failure_cases": [],
    }


def _complete_checkpoint(case_id: str = "case_1", score: float = 0.8) -> dict:
    return {
        "schema_version": runner.CHECKPOINT_SCHEMA_VERSION,
        "case_id": case_id,
        "model": "mistral",
        "dataset": "clean",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "query_result": {
            "case_id": case_id,
            "success": True,
            "answer": "a",
            "latency_seconds": 0.1,
            "error_message": None,
        },
        "faithfulness_result": _complete_faithfulness(case_id, score),
    }


def _write_raw(path: Path, payload) -> None:
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def checkpoint_dir(tmp_path, monkeypatch):
    cdir = tmp_path / "checkpoints"
    monkeypatch.setattr(runner, "CHECKPOINT_DIR", cdir)
    return cdir


def test_null_faithfulness_result_is_not_complete(checkpoint_dir):
    data = _complete_checkpoint()
    data["faithfulness_result"] = None

    assert runner._checkpoint_is_complete(data) is False

    _write_raw(runner._checkpoint_path("mistral", "clean", "case_1"), data)
    assert runner._load_checkpoint("mistral", "clean", "case_1") is None


def test_old_schema_checkpoint_is_ignored(checkpoint_dir):
    data = _complete_checkpoint()
    data["schema_version"] = runner.CHECKPOINT_SCHEMA_VERSION - 1

    assert runner._checkpoint_is_complete(data) is False

    _write_raw(runner._checkpoint_path("mistral", "clean", "case_1"), data)
    assert runner._load_checkpoint("mistral", "clean", "case_1") is None


def test_missing_required_field_is_ignored(checkpoint_dir):
    data = _complete_checkpoint()
    del data["faithfulness_result"]["supported_count"]

    assert runner._checkpoint_is_complete(data) is False

    _write_raw(runner._checkpoint_path("mistral", "clean", "case_1"), data)
    assert runner._load_checkpoint("mistral", "clean", "case_1") is None


def test_corrupt_json_is_ignored(checkpoint_dir):
    path = runner._checkpoint_path("mistral", "clean", "case_1")
    _write_raw(path, '{"case_id": "case_1", "faithfulness_result": ')

    assert runner._load_checkpoint("mistral", "clean", "case_1") is None


def test_complete_checkpoint_is_reused(checkpoint_dir):
    runner._save_checkpoint("mistral", "clean", "case_1", _complete_checkpoint())

    loaded = runner._load_checkpoint("mistral", "clean", "case_1")

    assert loaded is not None
    assert loaded["schema_version"] == runner.CHECKPOINT_SCHEMA_VERSION
    assert loaded["faithfulness_result"]["case_id"] == "case_1"

    result = runner._checkpoint_to_faithfulness_result(
        loaded["faithfulness_result"],
        similarity_threshold=0.68,
        embedding_model="nomic-embed-text",
        evaluator_version="1.2.2",
    )
    assert isinstance(result, FaithfulnessResult)
    assert result.case_id == "case_1"
    assert result.faithfulness_score == 0.8


def test_checkpoint_to_result_none_raises_checkpoint_invalid():
    with pytest.raises(runner.CheckpointInvalidError):
        runner._checkpoint_to_faithfulness_result(
            None,
            similarity_threshold=0.68,
            embedding_model="nomic-embed-text",
            evaluator_version="1.2.2",
        )

    with pytest.raises(runner.CheckpointInvalidError):
        runner._checkpoint_to_faithfulness_result(
            {},
            similarity_threshold=0.68,
            embedding_model="nomic-embed-text",
            evaluator_version="1.2.2",
        )


def test_resume_after_null_checkpoint_reruns_case(tmp_path, monkeypatch):
    """Integration: checkpoint null tidak boleh membuat resume crash / skip."""
    results = tmp_path / "results"
    checkpoints = tmp_path / "checkpoints"
    results.mkdir()
    checkpoints.mkdir()
    monkeypatch.setattr(runner, "REPORTS_DIR", results)
    monkeypatch.setattr(runner, "CHECKPOINT_DIR", checkpoints)

    cases = [
        {"id": "case_1", "question": "q1", "context": "ctx1"},
        {"id": "case_2", "question": "q2", "context": "ctx2"},
    ]
    monkeypatch.setattr(runner, "load_dataset", lambda *a, **k: cases)

    evaluated = []

    def fake_query_template(client, case, case_type):
        from pipeline.model_client import QueryOutput

        return QueryOutput(
            case_id=case["id"],
            model="mistral",
            question=case["question"],
            context_length_chars=len(case["context"]),
            answer="ans",
            latency_seconds=0.1,
            timestamp_utc="2026-01-01T00:00:00Z",
            success=True,
        )

    class DummyClient:
        def __init__(self, *a, **kw):
            self.model = "mistral"
            self.temperature = 0.0
            self.max_tokens = 4096
            self._endpoint = "http://localhost:11434/api/chat"
            self.timeout_seconds = 900

        def health_check(self):
            return True

    class DummyFaith:
        VERSION = "1.2.2"

        def __init__(self, similarity_threshold=None):
            self.similarity_threshold = similarity_threshold

        def health_check(self):
            return True

        def evaluate(self, case, model_answer, model_name):
            evaluated.append(case["id"])
            return FaithfulnessResult(
                case_id=case["id"],
                model=model_name,
                question=case["question"],
                answer=model_answer,
                context_preview=case["context"],
                total_claims=1,
                claims=[],
                evidence_matches=[],
                supported_count=1,
                unsupported_count=0,
                skipped_count=0,
                faithfulness_score=0.7,
                is_insufficient_context_response=False,
                has_failure=False,
                failure_cases=[],
                similarity_threshold=self.similarity_threshold,
                embedding_model="nomic-embed-text",
                evaluator_version=self.VERSION,
                timestamp_utc="2026-01-01T00:00:00Z",
                evaluation_path="claim_extraction",
            )

        def save_results(self, results_list, output_path):
            out = [
                {
                    "case_id": r.case_id,
                    "faithfulness_score": r.faithfulness_score,
                    "has_failure": r.has_failure,
                    "is_insufficient_context_response": r.is_insufficient_context_response,
                    "failure_cases": r.failure_cases,
                }
                for r in results_list
            ]
            Path(output_path).write_text(json.dumps(out), encoding="utf-8")

    # Checkpoint lama/rusak: schema current tapi faithfulness_result null.
    runner._save_checkpoint(
        "mistral",
        "clean",
        "case_1",
        {
            "case_id": "case_1",
            "model": "mistral",
            "dataset": "clean",
            "timestamp_utc": "2026-01-01T00:00:00Z",
            "query_result": {
                "case_id": "case_1",
                "success": True,
                "answer": "a1",
                "latency_seconds": 0.1,
                "error_message": None,
            },
            "faithfulness_result": None,
        },
    )

    monkeypatch.setattr(runner, "_query_with_template", fake_query_template)
    monkeypatch.setattr(runner, "OllamaClient", DummyClient)
    monkeypatch.setattr(runner, "FaithfulnessEvaluator", DummyFaith)

    runner.run_evaluation(model_name="mistral", dataset_name="clean", resume=True)

    assert "case_1" in evaluated, "null checkpoint must trigger re-run"
    assert "case_2" in evaluated

    files = sorted(results.glob("mistral_clean_*_faithfulness.json"))
    assert files, "output file not created"
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    by_id = {d["case_id"]: d for d in data}
    assert set(by_id) == {"case_1", "case_2"}
    assert by_id["case_1"]["faithfulness_score"] == 0.7

    # Checkpoint hasil run ulang harus lengkap: run berikutnya me-reuse, bukan rerun.
    evaluated_after_first_run = list(evaluated)
    runner.run_evaluation(model_name="mistral", dataset_name="clean", resume=True)
    assert evaluated == evaluated_after_first_run, "complete checkpoints must be reused"
