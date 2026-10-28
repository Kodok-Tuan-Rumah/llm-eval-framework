import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluators.faithfulness import FaithfulnessResult
from pipeline import runner


def _make_faithfulness_result(
    case_id: str, score: float, has_failure: bool = False
) -> FaithfulnessResult:
    return FaithfulnessResult(
        case_id=case_id,
        model="mistral",
        question=f"q-{case_id}",
        answer="a",
        context_preview="ctx",
        total_claims=1,
        claims=[],
        evidence_matches=[],
        supported_count=1 if score > 0 else 0,
        unsupported_count=1 if score > 0 else 0,
        skipped_count=0,
        faithfulness_score=score,
        is_insufficient_context_response=False,
        has_failure=has_failure,
        failure_cases=[{"diagnosis": "x"}] if has_failure else [],
        similarity_threshold=0.68,
        embedding_model="nomic-embed-text",
        evaluator_version="1.2.2",
        timestamp_utc="2026-01-01T00:00:00Z",
        evaluation_path="claim_extraction",
    )


def _complete_faithfulness_dict(
    case_id: str,
    score: float,
    has_failure: bool = False,
    failure_cases: list | None = None,
) -> dict:
    """Checkpoint faithfulness payload yang lolos completeness gate."""
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
        "has_failure": has_failure,
        "is_insufficient_context_response": False,
        "failure_cases": failure_cases or [],
    }


def test_checkpoint_roundtrip(tmp_path):
    cdir = tmp_path / "checkpoints"
    cdir.mkdir()
    runner.CHECKPOINT_DIR = cdir

    runner._save_checkpoint(
        "mistral",
        "clean",
        "case_1",
        {
            "query_result": {
                "case_id": "case_1",
                "success": True,
                "answer": "a1",
                "latency_seconds": 0.1,
                "error_message": None,
            },
            "faithfulness_result": _complete_faithfulness_dict(
                "case_1", 0.8, has_failure=True, failure_cases=[{"diagnosis": "cached"}]
            ),
        },
    )
    loaded = runner._load_checkpoint("mistral", "clean", "case_1")
    assert loaded is not None
    assert loaded["faithfulness_result"]["faithfulness_score"] == 0.8
    assert loaded["query_result"]["success"] is True


def test_merged_save_contains_checkpointed_and_new(tmp_path):
    outpath = tmp_path / "merged_faithfulness.json"

    class DummyFaith:
        VERSION = "1.2.2"

        def health_check(self):
            return True

        def save_results(self, results_list, output_path):
            out = []
            for r in results_list:
                out.append(
                    {
                        "case_id": r.case_id,
                        "faithfulness_score": r.faithfulness_score,
                        "has_failure": r.has_failure,
                        "is_insufficient_context_response": r.is_insufficient_context_response,
                        "failure_cases": r.failure_cases,
                    }
                )
            Path(output_path).write_text(json.dumps(out))

    merged = [
        _make_faithfulness_result("case_1", 0.8, has_failure=True),
        _make_faithfulness_result("case_3", 1.0),
    ]
    DummyFaith().save_results(merged, str(outpath))
    data = json.loads(outpath.read_text())
    assert {d["case_id"] for d in data} == {"case_1", "case_3"}
    assert data[0]["faithfulness_score"] == 0.8
    assert data[1]["faithfulness_score"] == 1.0


def test_run_evaluation_resume_merges_checkpointed_and_new(tmp_path, monkeypatch):
    """Integration: resume run dengan checkpoint → output faithfulness berisi semua case."""
    results = tmp_path / "results"
    checkpoints = tmp_path / "checkpoints"
    results.mkdir()
    checkpoints.mkdir()
    monkeypatch.setattr(runner, "REPORTS_DIR", results)
    monkeypatch.setattr(runner, "CHECKPOINT_DIR", checkpoints)

    cases = [
        {"id": "case_1", "question": "q1", "context": "ctx1"},
        {"id": "case_2", "question": "q2", "context": "ctx2"},
        {"id": "case_3", "question": "q3", "context": "ctx3"},
    ]
    monkeypatch.setattr(runner, "load_dataset", lambda *a, **k: cases)

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
            return _make_faithfulness_result(case["id"], 1.0)

        def save_results(self, results_list, output_path):
            out = []
            for r in results_list:
                out.append(
                    {
                        "case_id": r.case_id,
                        "faithfulness_score": r.faithfulness_score,
                        "has_failure": r.has_failure,
                        "is_insufficient_context_response": r.is_insufficient_context_response,
                        "failure_cases": r.failure_cases,
                    }
                )
            Path(output_path).write_text(json.dumps(out))

    # checkpoint untuk case_1 & case_2
    runner._save_checkpoint(
        "mistral",
        "clean",
        "case_1",
        {
            "query_result": {
                "case_id": "case_1",
                "success": True,
                "answer": "a1",
                "latency_seconds": 0.1,
                "error_message": None,
            },
            "faithfulness_result": _complete_faithfulness_dict(
                "case_1", 0.8, has_failure=True, failure_cases=[{"diagnosis": "cached"}]
            ),
        },
    )
    runner._save_checkpoint(
        "mistral",
        "clean",
        "case_2",
        {
            "query_result": {
                "case_id": "case_2",
                "success": True,
                "answer": "a2",
                "latency_seconds": 0.1,
                "error_message": None,
            },
            "faithfulness_result": _complete_faithfulness_dict(
                "case_2",
                0.9,
                has_failure=True,
                failure_cases=[{"diagnosis": "cached2"}],
            ),
        },
    )

    monkeypatch.setattr(runner, "_query_with_template", fake_query_template)
    monkeypatch.setattr(runner, "OllamaClient", DummyClient)
    monkeypatch.setattr(runner, "FaithfulnessEvaluator", DummyFaith)

    runner.run_evaluation(model_name="mistral", dataset_name="clean", resume=True)

    files = sorted(results.glob("mistral_clean_*_faithfulness.json"))
    assert files, "output file not created"
    data = json.loads(files[-1].read_text())
    case_ids = {d["case_id"] for d in data}
    assert case_ids == {"case_1", "case_2", "case_3"}, (
        f"expected 3 cases, got {sorted(case_ids)}"
    )
