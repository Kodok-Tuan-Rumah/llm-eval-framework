import os
import sys
from itertools import cycle
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api as api_module
from evaluators.faithfulness import FaithfulnessResult


def _fake_result(**overrides):
    base = dict(
        case_id="api_case",
        model="test",
        question="Which method?",
        answer="PATCH",
        context_preview="ctx",
        total_claims=3,
        claims=[],
        evidence_matches=[],
        supported_count=3,
        unsupported_count=0,
        skipped_count=0,
        faithfulness_score=1.0,
        is_insufficient_context_response=False,
        has_failure=False,
        failure_cases=[],
        similarity_threshold=0.68,
        embedding_model="nomic-embed-text",
        evaluator_version="1.2.2",
        timestamp_utc="2026-01-01T00:00:00+00:00",
    )
    base.update(overrides)
    return FaithfulnessResult(**base)


@pytest.fixture
def stub_evaluator(monkeypatch):
    def _install(result):
        evaluator = MagicMock()
        evaluator.evaluate.return_value = result
        monkeypatch.setattr(api_module, "evaluator", evaluator)
        return evaluator

    return _install


class TestResponseSchema:
    def test_evaluator_version_is_part_of_response(self):
        assert "evaluator_version" in api_module.EvalResponse.model_fields


class TestEvaluateEndpoint:
    def test_returns_evaluator_version(self, stub_evaluator):
        stub_evaluator(_fake_result())
        response = api_module.evaluate(
            api_module.EvalRequest(
                answer="PATCH", context="Use PATCH.", question="Which method?"
            )
        )
        assert response.evaluator_version == "1.2.2"
        assert response.faithfulness_score == 1.0
        assert response.supported_claims == 3
        assert response.total_claims == 3
        assert response.has_failure is False
        assert response.failure_mode == "none"

    def test_failure_mode_unsupported_claims(self, stub_evaluator):
        stub_evaluator(
            _fake_result(
                faithfulness_score=0.0,
                has_failure=True,
                supported_count=0,
                unsupported_count=3,
            )
        )
        response = api_module.evaluate(
            api_module.EvalRequest(answer="bad answer", context="ctx", question="q")
        )
        assert response.failure_mode == "unsupported_claims"

    def test_failure_mode_insufficient_context(self, stub_evaluator):
        stub_evaluator(
            _fake_result(
                faithfulness_score=0.0,
                has_failure=True,
                is_insufficient_context_response=True,
            )
        )
        response = api_module.evaluate(
            api_module.EvalRequest(
                answer="INSUFFICIENT_CONTEXT", context="ctx", question="q"
            )
        )
        assert response.failure_mode == "insufficient_context"

    def test_empty_answer_rejected(self, stub_evaluator):
        stub_evaluator(_fake_result())
        with pytest.raises(HTTPException) as exc:
            api_module.evaluate(
                api_module.EvalRequest(answer="   ", context="ctx", question="q")
            )
        assert exc.value.status_code == 400

    def test_empty_context_rejected(self, stub_evaluator):
        stub_evaluator(_fake_result())
        with pytest.raises(HTTPException) as exc:
            api_module.evaluate(
                api_module.EvalRequest(answer="PATCH", context="", question="q")
            )
        assert exc.value.status_code == 400


class TestHealthEndpoint:
    def test_health(self):
        assert api_module.health() == {"status": "ok", "service": "llm-eval-framework"}


class TestReadyEndpoint:
    def test_ready_when_evaluator_and_embedding_are_ready(self, stub_evaluator):
        evaluator = stub_evaluator(_fake_result())
        evaluator.health_check.return_value = True

        assert api_module.ready() == {
            "status": "ready",
            "service": "llm-eval-framework",
            "check": "readiness",
        }

    def test_not_ready_when_embedding_dependency_is_down(self, stub_evaluator):
        evaluator = stub_evaluator(_fake_result())
        evaluator.health_check.return_value = False

        with pytest.raises(HTTPException) as exc:
            api_module.ready()
        assert exc.value.status_code == 503

    def test_not_ready_when_evaluator_is_not_initialized(self, monkeypatch):
        monkeypatch.setattr(api_module, "evaluator", None)

        with pytest.raises(HTTPException) as exc:
            api_module.ready()
        assert exc.value.status_code == 503


class TestMetricsEndpoint:
    def test_metrics_exposes_prometheus_text(self):
        response = api_module.metrics()
        assert response.media_type == api_module.CONTENT_TYPE_LATEST
        body = response.body.decode()
        assert "http_request_duration_seconds" in body
        assert "eval_requests_total" in body


class TestEvaluateBatchEndpoint:
    def _install_batch(self, results, monkeypatch):
        evaluator = MagicMock()
        evaluator.evaluate.side_effect = cycle(results)
        monkeypatch.setattr(api_module, "evaluator", evaluator)
        return evaluator

    def test_returns_results_and_summary_shape(self, stub_evaluator, monkeypatch):
        results = [
            _fake_result(
                case_id="case_0", faithfulness_score=1.0, supported_count=3,
                unsupported_count=0, has_failure=False,
            ),
            _fake_result(
                case_id="case_1", faithfulness_score=0.5, supported_count=1,
                unsupported_count=1, has_failure=True,
            ),
            _fake_result(
                case_id="case_2", faithfulness_score=0.0,
                is_insufficient_context_response=True, supported_count=0,
                unsupported_count=0, has_failure=True,
            ),
        ]
        evaluator = self._install_batch(results, monkeypatch)

        response = api_module.evaluate_batch(
            api_module.EvalBatchRequest(
                items=[
                    api_module.EvalBatchItem(claim="c0", context="ctx0"),
                    api_module.EvalBatchItem(claim="c1", context="ctx1"),
                    api_module.EvalBatchItem(claim="c2", context="ctx2"),
                ]
            )
        )

        assert len(response.results) == 3
        assert response.results[0].claim_index == 0
        assert response.results[0].faithfulness_score == 1.0
        assert response.results[0].failure_mode == "none"
        assert response.results[0].supported_claims == 3
        assert response.results[0].total_claims == 3

        assert response.results[1].claim_index == 1
        assert response.results[1].faithfulness_score == 0.5
        assert response.results[1].failure_mode == "unsupported_claims"

        assert response.results[2].claim_index == 2
        assert response.results[2].failure_mode == "insufficient_context"

        summary = response.summary
        assert abs(summary.avg_faithfulness - 0.5) < 1e-9
        assert abs(summary.supported_pct - round(1 / 3, 4)) < 1e-9
        assert abs(summary.partial_pct - round(1 / 3, 4)) < 1e-9
        assert abs(summary.unsupported_pct - round(1 / 3, 4)) < 1e-9
        assert evaluator.evaluate.call_count == 3

    def test_batch_is_deterministic(self, stub_evaluator, monkeypatch):
        results = [
            _fake_result(
                case_id="case_0", faithfulness_score=1.0, supported_count=3,
                unsupported_count=0, has_failure=False,
            ),
            _fake_result(
                case_id="case_1", faithfulness_score=0.5, supported_count=1,
                unsupported_count=1, has_failure=True,
            ),
        ]
        self._install_batch(results, monkeypatch)

        req = api_module.EvalBatchRequest(
            items=[
                api_module.EvalBatchItem(claim="c0", context="ctx0"),
                api_module.EvalBatchItem(claim="c1", context="ctx1"),
            ]
        )

        first = api_module.evaluate_batch(req)
        second = api_module.evaluate_batch(req)

        assert len(first.results) == len(second.results) == 2
        for a, b in zip(first.results, second.results):
            assert a.claim_index == b.claim_index
            assert a.faithfulness_score == b.faithfulness_score
            assert a.failure_mode == b.failure_mode
            assert a.supported_claims == b.supported_claims
            assert a.total_claims == b.total_claims

        assert first.summary.avg_faithfulness == second.summary.avg_faithfulness
        assert first.summary.supported_pct == second.summary.supported_pct
        assert first.summary.partial_pct == second.summary.partial_pct
        assert first.summary.unsupported_pct == second.summary.unsupported_pct

    def test_empty_items_rejected(self, stub_evaluator, monkeypatch):
        self._install_batch([], monkeypatch)
        with pytest.raises(HTTPException) as exc:
            api_module.evaluate_batch(api_module.EvalBatchRequest(items=[]))
        assert exc.value.status_code == 400

    def test_empty_claim_rejected(self, stub_evaluator, monkeypatch):
        self._install_batch([], monkeypatch)
        with pytest.raises(HTTPException) as exc:
            api_module.evaluate_batch(
                api_module.EvalBatchRequest(
                    items=[api_module.EvalBatchItem(claim="   ", context="ctx")]
                )
            )
        assert exc.value.status_code == 400

    def test_empty_context_rejected(self, stub_evaluator, monkeypatch):
        self._install_batch([], monkeypatch)
        with pytest.raises(HTTPException) as exc:
            api_module.evaluate_batch(
                api_module.EvalBatchRequest(
                    items=[api_module.EvalBatchItem(claim="c", context="")]
                )
            )
        assert exc.value.status_code == 400
