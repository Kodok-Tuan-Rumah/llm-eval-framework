# LLM Eval Framework — Project Goals & Integration Spec

> **Repo:** `llm-eval-framework`  
> **Status:** Active Evaluation & Diagnostic Engine  
> **Last Updated:** 2026-10-21  

---

## 1. System Objective

Provide a rigorous, reproducible, and deterministic **evaluation and diagnostic toolkit** for scientific RAG systems. This framework measures the factual fidelity (*faithfulness*), logical consistency, answer sensitivity, and retrieval dependency of generated answers against their retrieved evidence.

It acts as an independent verification layer in the monorepo workspace, ensuring that claims made by the RAG engine are strictly supported by the indexed corpus.

---

## 2. Workspace Product Boundaries

| Component | Responsibility | Inputs | Outputs |
|---|---|---|---|
| **Evaluators (`evaluators/`)** | Core mathematical and heuristic evaluation logic (Faithfulness, Consistency, Sensitivity, Retrieval Dependency). | Question, Answer, Contexts (Evidence) | Scores (`0.0` to `1.0`), Failure Modes, Supported Claims count |
| **Pipeline (`pipeline/`)** | Execution runner, configuration loading, prompt templates, and provenance tracking. | Dataset records, YAML configs | Evaluation runs, logs, and checkpoints |
| **FastAPI Wrapper (`api.py`)** | Lightweight HTTP service exposing evaluation endpoints (`/evaluate`, `/health`) for downstream orchestrators like `mcp-gateway`. | JSON request (`answer`, `context`, `question`) | Evaluation result envelope (`faithfulness_score`, `has_failure`, `supported_claims`) |
| **Reporting (`reports/`)** | Visualization, chart generation, and markdown/HTML analysis report renderers. | Evaluation logs & metrics JSON | Executive summary reports, reliability diagrams |

---

## 3. Canonical Evaluation Contracts

### 3.1 Evaluation Request Contract (`POST /evaluate`)
```json
{
  "question": "What is the attention mechanism?",
  "answer": "Attention allows models to focus on relevant parts of the input.",
  "context": "The attention mechanism computes weights for input tokens..."
}
```

### 3.2 Evaluation Response Contract
```json
{
  "faithfulness_score": 0.95,
  "has_failure": false,
  "failure_mode": "none",
  "supported_claims": 2,
  "total_claims": 2,
  "evaluator_version": "v1.0.0"
}
```

* **Rule:** Evaluation failure must return a graceful diagnostic payload (`unavailable` or `error`), never crashing the caller or fabricating a false positive score.

---

## 4. Current Roadmap & Next Steps

1. **Test Suite Verification:** Run all unit and integration tests under `tests/` to guarantee baseline stability.
2. **Gateway Integration Alignment:** Verify that `mcp-gateway` successfully consumes the `/evaluate` endpoint during end-to-end runs.
3. **Metric Calibration:** Use evaluation logs to calibrate thresholds for hallucination detection and abstention triggers in `rag-research`.
