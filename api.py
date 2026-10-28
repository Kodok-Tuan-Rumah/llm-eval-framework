import logging
import time
import uuid
from contextlib import asynccontextmanager

from evaluators.faithfulness import FaithfulnessEvaluator
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

logger = logging.getLogger(__name__)
evaluator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global evaluator
    logger.info("Initializing FaithfulnessEvaluator...")
    evaluator = FaithfulnessEvaluator()
    logger.info("FaithfulnessEvaluator ready")
    yield


app = FastAPI(title="LLM Eval Service", version="1.0.0", lifespan=lifespan)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60, 120, 300),
)
EVAL_REQUESTS_TOTAL = Counter(
    "eval_requests_total",
    "Completed evaluations by failure mode",
    ["failure_mode", "has_failure"],
)


@app.middleware("http")
async def observe_http_requests(request: Request, call_next):
    started = time.perf_counter()
    try:
        return await call_next(request)
    finally:
        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            path=request.url.path,
        ).observe(time.perf_counter() - started)


class EvalRequest(BaseModel):
    answer: str
    context: str
    question: str


class EvalResponse(BaseModel):
    faithfulness_score: float
    failure_mode: str
    has_failure: bool
    supported_claims: int
    total_claims: int
    evaluator_version: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "llm-eval-framework"}


@app.get("/ready")
def ready():
    if evaluator is None:
        raise HTTPException(
            status_code=503, detail="FaithfulnessEvaluator not initialized"
        )
    if not evaluator.health_check(timeout=2.0):
        raise HTTPException(
            status_code=503, detail="Ollama embedding dependency unavailable"
        )
    return {"status": "ready", "service": "llm-eval-framework", "check": "readiness"}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: EvalRequest):
    if evaluator is None:
        raise HTTPException(
            status_code=503,
            detail="FaithfulnessEvaluator belum diinisialisasi. Tunggu startup selesai atau coba lagi dalam beberapa saat.",
        )
    if not req.answer.strip():
        raise HTTPException(status_code=400, detail="answer cannot be empty")
    if not req.context.strip():
        raise HTTPException(status_code=400, detail="context cannot be empty")

    case = {
        "id": str(uuid.uuid4()),
        "context": req.context,
        "question": req.question,
    }

    try:
        result = evaluator.evaluate(case=case, model_answer=req.answer)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"evaluation failed: {e}")

    failure_mode = "none"
    if result.has_failure:
        failure_mode = "unsupported_claims"
    if result.is_insufficient_context_response:
        failure_mode = "insufficient_context"

    EVAL_REQUESTS_TOTAL.labels(
        failure_mode=failure_mode,
        has_failure=str(result.has_failure).lower(),
    ).inc()

    return EvalResponse(
        faithfulness_score=result.faithfulness_score,
        failure_mode=failure_mode,
        has_failure=result.has_failure,
        supported_claims=result.supported_count,
        total_claims=result.total_claims,
        evaluator_version=result.evaluator_version,
    )
