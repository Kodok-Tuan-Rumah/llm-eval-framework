import os
import sys
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
