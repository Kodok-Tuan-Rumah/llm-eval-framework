import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators.retrieval_dependency import (
    RetrievalDependencyEvaluator,
    RetrievalDependencyResult,
    ConditionResult,
    DependencyComparison,
    DEFAULT_CHANGE_THRESHOLD,
)
from evaluators.faithfulness import OllamaEmbedder

@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=OllamaEmbedder)

    def fake_embed(text: str) -> np.ndarray:
        np.random.seed(hash(text[:50]) % (2**32))
        v = np.random.rand(768).astype(np.float32)
        return v / np.linalg.norm(v)

    embedder.embed.side_effect = fake_embed
    embedder.embed_batch.side_effect = lambda texts: np.stack([fake_embed(t) for t in texts])
    embedder.health_check.return_value = True
    embedder.model = "nomic-embed-text-mock"
    return embedder

@pytest.fixture
def evaluator(mock_embedder):
    ev = RetrievalDependencyEvaluator.__new__(RetrievalDependencyEvaluator)
    ev.change_threshold = DEFAULT_CHANGE_THRESHOLD
    ev.embedding_model = "mock"
    ev.embedder = mock_embedder
    return ev

def test_evaluator_init(mock_embedder):
    with patch("evaluators.retrieval_dependency.OllamaEmbedder", return_value=mock_embedder):
        ev = RetrievalDependencyEvaluator(change_threshold=0.7)
        assert ev.change_threshold == 0.7
        assert ev.health_check() is True

def test_evaluate_from_answers(evaluator):
    res = evaluator.evaluate_from_answers(
        case_id="dep_001",
        question="What is FastAPI?",
        full_answer="FastAPI is a modern web framework for Python.",
        no_context_answer="I don't know based on general knowledge.",
        wrong_answer="Django is a full-stack framework.",
        model_name="test-model",
        full_context="FastAPI is a modern web framework...",
        wrong_context="Django is a full-stack framework...",
    )

    assert isinstance(res, RetrievalDependencyResult)
    assert res.case_id == "dep_001"
    assert res.model == "test-model"
    assert len(res.conditions) == 3
    assert len(res.comparisons) == 2
    assert 0.0 <= res.full_vs_no_context_similarity <= 1.0
    assert 0.0 <= res.full_vs_wrong_context_similarity <= 1.0
    assert isinstance(res.is_context_dependent, bool)
    assert isinstance(res.has_failure, bool)
    assert isinstance(res.failure_flags, list)

def test_evaluate_from_answers_no_change(mock_embedder):
    # Same embedding for all answers to test independence / failure
    mock_embedder.embed_batch.return_value = np.ones((3, 768), dtype=np.float32)
    mock_embedder.embed_batch.return_value /= np.linalg.norm(mock_embedder.embed_batch.return_value, axis=1, keepdims=True)

    ev = RetrievalDependencyEvaluator.__new__(RetrievalDependencyEvaluator)
    ev.change_threshold = 0.85
    ev.embedding_model = "mock"
    ev.embedder = mock_embedder

    res = ev.evaluate_from_answers(
        case_id="dep_002",
        question="What is FastAPI?",
        full_answer="Same answer",
        no_context_answer="Same answer",
        wrong_answer="Same answer",
        model_name="test-model",
    )

    # When similarity is 1.0 (>= threshold), context_independence_flag is True, meaning answer didn't change when context removed!
    assert res.full_vs_no_context_similarity == 1.0
    assert res.is_context_dependent is False
    assert res.has_failure is True
    flags = [f["flag"] for f in res.failure_flags]
    assert "CONTEXT_INDEPENDENT_NO_CONTEXT" in flags
    assert "CONTEXT_INDEPENDENT_WRONG_CONTEXT" in flags
