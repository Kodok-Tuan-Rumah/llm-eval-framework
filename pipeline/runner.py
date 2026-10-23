import json
import argparse
import logging
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.model_client import OllamaClient, save_query_results, QueryOutput
from pipeline.prompt_templates import PromptTemplate, TEMPLATE_VERSION
from pipeline.config_loader import config
from pipeline.provenance import build_provenance
from evaluators.faithfulness import FaithfulnessEvaluator, FaithfulnessResult
from evaluators.consistency import ConsistencyEvaluator

logger = logging.getLogger("runner")
DATA_DIR    = PROJECT_ROOT / "data" / "dataset"
REPORTS_DIR = PROJECT_ROOT / config["pipeline"]["results_dir"]
CHECKPOINT_DIR = PROJECT_ROOT / config["pipeline"]["checkpoint_dir"]

DATASET_FILES = {
    "clean":      DATA_DIR / "clean_cases.json",
    "distractor": DATA_DIR / "distractor_cases.json",
    "paraphrase": DATA_DIR / "paraphrase_cases.json",
}

# checkpoint schema — bump saat format berubah agar checkpoint lama diabaikan
# v3: simpan seluruh field FaithfulnessResult + provenance QueryOutput (E2)
CHECKPOINT_SCHEMA_VERSION = 3

REQUIRED_FAITHFULNESS_FIELDS = {
    "case_id", "model", "question", "answer",
    "faithfulness_score", "supported_count", "total_claims",
    "similarity_threshold", "embedding_model", "evaluator_version",
    "timestamp_utc",
}


class CheckpointInvalidError(ValueError):
    """Checkpoint tidak lengkap / tidak bisa dikonversi menjadi hasil evaluasi."""

# dataset loader
def load_dataset(dataset_name: str, limit: int = None) -> list[dict]:
    if dataset_name == "all":
        cases = []
        for name, path in DATASET_FILES.items():
            if path.exists():
                loaded = _load_json_cases(path, name)
                cases.extend(loaded)
                logger.info(f"Loaded {len(loaded)} cases from {name}")
            else:
                logger.warning(f"Dataset file not found: {path}")
        return cases[:limit] if limit else cases

    path = DATASET_FILES.get(dataset_name)
    if not path:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Valid options: {list(DATASET_FILES.keys()) + ['all']}"
        )
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Pastikan file ada di: {DATA_DIR}"
        )

    cases = _load_json_cases(path, dataset_name)
    return cases[:limit] if limit else cases

def _load_json_cases(path: Path, dataset_name: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    for case in cases:
        missing = {"id", "context"} - set(case.keys())
        if missing:
            raise ValueError(
                f"Case '{case.get('id', 'unknown')}' di dataset '{dataset_name}' "
                f"missing required fields: {missing}"
            )
        if "question" not in case and "variants" not in case:
            raise ValueError(
                f"Case '{case.get('id', 'unknown')}' di dataset '{dataset_name}' "
                f"harus punya field 'question' (faithfulness) atau 'variants' (consistency)"
            )
        for variant in case.get("variants", []):
            missing_variant = {"label", "question"} - set(variant.keys())
            if missing_variant:
                raise ValueError(
                    f"Case '{case.get('id', 'unknown')}' di dataset '{dataset_name}' "
                    f"variant missing required fields: {missing_variant}"
                )
    return cases

# checkpoint helpers — resume capability
def _checkpoint_path(model: str, dataset: str, case_id: str) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace(":", "-")
    return CHECKPOINT_DIR / f"{safe_model}__{dataset}__{case_id}.json"

def _checkpoint_is_complete(data: dict) -> bool:
    """True hanya untuk checkpoint schema terkini dengan faithfulness lengkap."""
    if not isinstance(data, dict):
        return False
    if data.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return False
    faithfulness_result = data.get("faithfulness_result")
    if not isinstance(faithfulness_result, dict) or not faithfulness_result:
        return False
    return REQUIRED_FAITHFULNESS_FIELDS.issubset(faithfulness_result)

def _load_checkpoint(model: str, dataset: str, case_id: str) -> dict | None:
    path = _checkpoint_path(model, dataset, case_id)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not _checkpoint_is_complete(data):
            logger.warning(
                "Incomplete/stale checkpoint ignored | case_id=%s | schema=%s",
                case_id, data.get("schema_version") if isinstance(data, dict) else None,
            )
            return None
        logger.debug(f"Checkpoint loaded | case_id={case_id}")
        return data
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Corrupt checkpoint ignored | case_id={case_id} | {e}")
        return None

def _save_checkpoint(model: str, dataset: str, case_id: str, result: dict) -> None:
    path = _checkpoint_path(model, dataset, case_id)
    payload = {**result, "schema_version": CHECKPOINT_SCHEMA_VERSION}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.debug(f"Checkpoint saved | case_id={case_id} | path={path}")

def _clear_checkpoints(model: str, dataset: str, case_ids: list[str]) -> None:
    cleared = 0
    for case_id in case_ids:
        path = _checkpoint_path(model, dataset, case_id)
        if path.exists():
            path.unlink()
            cleared += 1
    if cleared:
        logger.info(f"Cleared {cleared} existing checkpoints (--no-resume)")

# retry-aware query — dengan exponential backoff
def _query_with_retry(
    client: OllamaClient,
    case: dict,
    case_type: str,
    max_retries: int = None,
    backoff_base: int = None,
) -> QueryOutput:
    max_retries  = max_retries  or config["inference"]["max_retries"]
    backoff_base = backoff_base or config["inference"]["retry_backoff_base"]
    case_id      = case["id"]
    last_error   = None

    for attempt in range(max_retries):
        try:
            result = _query_with_template(client, case, case_type)
            if result.success:
                if attempt > 0:
                    logger.info(f"Retry succeeded | case_id={case_id} | attempt={attempt + 1}")
                return result
            raise RuntimeError(result.error_message or "Query returned success=False")

        except Exception as e:
            last_error = e
            wait = backoff_base ** attempt  # 1s, 2s, 4s
            logger.warning(
                f"Query attempt {attempt + 1}/{max_retries} failed | "
                f"case_id={case_id} | error={e}"
            )
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
                
    logger.error(
        f"All {max_retries} attempts failed | case_id={case_id} | "
        f"last_error={last_error}"
    )
    return QueryOutput(
        case_id=case_id,
        model=client.model,
        question=case["question"],
        context_length_chars=len(case["context"]),
        answer="",
        latency_seconds=0.0,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        success=False,
        error_message=f"All {max_retries} retries failed. Last: {last_error}",
    )

# Main evaluation pipeline
def run_evaluation(
    model_name: str,
    dataset_name: str,
    limit: int = None,
    similarity_threshold: float = None,
    resume: bool = True,
    dry_run: bool = False,
) -> dict:
    if dataset_name == "paraphrase":
        # Paraphrase dievaluasi sebagai dimensi konsistensi, bukan faithfulness.
        return run_consistency_evaluation(
            model_name=model_name,
            limit=limit,
            dry_run=dry_run,
        )

    similarity_threshold = similarity_threshold or config["thresholds"]["faithfulness_evidence"]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info(
        f"Starting evaluation | run_id={run_id} | model={model_name} | "
        f"dataset={dataset_name} | limit={limit} | resume={resume} | dry_run={dry_run}"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = REPORTS_DIR / f"{model_name.replace(':', '-')}_{dataset_name}_{run_id}"

    # Load dataset
    logger.info("Loading dataset...")
    cases = load_dataset(dataset_name, limit=limit)
    logger.info(f"Loaded {len(cases)} cases")

    # Case paraphrase punya 'variants' (bukan 'question') dan dievaluasi lewat
    # run_consistency_evaluation; pipeline faithfulness hanya clean + distractor.
    paraphrase_cases = [c for c in cases if "variants" in c]
    if paraphrase_cases:
        cases = [c for c in cases if "variants" not in c]
        logger.info(
            f"{len(paraphrase_cases)} paraphrase cases di-skip dari run faithfulness — "
            f"evaluasi terpisah: --dataset paraphrase"
        )

    if not cases:
        logger.error("No cases to evaluate.")
        sys.exit(1)

    if dry_run:
        print(f"\n[DRY RUN] Dataset valid: {len(cases)} cases")
        if paraphrase_cases:
            print(f"[DRY RUN] {len(paraphrase_cases)} paraphrase cases di-skip "
                  f"(dievaluasi via --dataset paraphrase)")
        print(f"[DRY RUN] Model: {model_name} | Threshold: {similarity_threshold}")
        for case in cases:
            print(f"  {case['id']} — {case['question'][:60]}")
        print("\n[DRY RUN] Selesai. Tidak ada inference yang dijalankan.")
        return {"dry_run": True, "cases": len(cases), "skipped_paraphrase": len(paraphrase_cases)}

    client = OllamaClient(model=model_name, temperature=0.0)
    faithfulness_evaluator = FaithfulnessEvaluator(
        similarity_threshold=similarity_threshold
    )

    # health checks
    logger.info("Running health checks...")
    if not client.health_check():
        logger.error(f"Model '{model_name}' tidak tersedia. Jalankan: ollama pull {model_name}")
        sys.exit(1)
    if not faithfulness_evaluator.health_check():
        logger.error("nomic-embed-text tidak tersedia. Jalankan: ollama pull nomic-embed-text")
        sys.exit(1)
    logger.info("All health checks passed.\n")

def run_evaluation(
    model_name: str,
    dataset_name: str,
    limit: int = None,
    similarity_threshold: float = None,
    resume: bool = True,
    dry_run: bool = False,
    workers: int = 3,
) -> dict:
    if dataset_name == "paraphrase":
        # Paraphrase dievaluasi sebagai dimensi konsistensi, bukan faithfulness.
        return run_consistency_evaluation(
            model_name=model_name,
            limit=limit,
            dry_run=dry_run,
        )

    similarity_threshold = similarity_threshold or config["thresholds"]["faithfulness_evidence"]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info(
        f"Starting evaluation | run_id={run_id} | model={model_name} | "
        f"dataset={dataset_name} | limit={limit} | resume={resume} | "
        f"dry_run={dry_run} | workers={workers}"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = REPORTS_DIR / f"{model_name.replace(':', '-')}_{dataset_name}_{run_id}"

    # Load dataset
    logger.info("Loading dataset...")
    cases = load_dataset(dataset_name, limit=limit)
    logger.info(f"Loaded {len(cases)} cases")

    # Case paraphrase punya 'variants' (bukan 'question') dan dievaluasi lewat
    # run_consistency_evaluation; pipeline faithfulness hanya clean + distractor.
    paraphrase_cases = [c for c in cases if "variants" in c]
    if paraphrase_cases:
        cases = [c for c in cases if "variants" not in c]
        logger.info(
            f"{len(paraphrase_cases)} paraphrase cases di-skip dari run faithfulness — "
            f"evaluasi terpisah: --dataset paraphrase"
        )

    if not cases:
        logger.error("No cases to evaluate.")
        sys.exit(1)

    if dry_run:
        print(f"\n[DRY RUN] Dataset valid: {len(cases)} cases")
        if paraphrase_cases:
            print(f"[DRY RUN] {len(paraphrase_cases)} paraphrase cases di-skip "
                  f"(dievaluasi via --dataset paraphrase)")
        print(f"[DRY RUN] Model: {model_name} | Threshold: {similarity_threshold}")
        print(f"[DRY RUN] Workers: {workers}")
        for case in cases:
            print(f"  {case['id']} — {case['question'][:60]}")
        print("\n[DRY RUN] Selesai. Tidak ada inference yang dijalankan.")
        return {"dry_run": True, "cases": len(cases), "skipped_paraphrase": len(paraphrase_cases)}

    client = OllamaClient(model=model_name, temperature=0.0)
    faithfulness_evaluator = FaithfulnessEvaluator(
        similarity_threshold=similarity_threshold
    )

    # health checks
    logger.info("Running health checks...")
    if not client.health_check():
        logger.error(f"Model '{model_name}' tidak tersedia. Jalankan: ollama pull {model_name}")
        sys.exit(1)
    if not faithfulness_evaluator.health_check():
        logger.error("nomic-embed-text tidak tersedia. Jalankan: ollama pull nomic-embed-text")
        sys.exit(1)
    logger.info("All health checks passed.\n")

    case_ids = [c["id"] for c in cases]
    if not resume:
        _clear_checkpoints(model_name, dataset_name, case_ids)

    # Cek checkpoint: pisahkan case yang sudah selesai dari yang belum
    already_done = [cid for cid in case_ids if _load_checkpoint(model_name, dataset_name, cid)]
    pending_cases = [c for c in cases if c["id"] not in already_done]

    if already_done and resume:
        logger.info(
            f"Resume mode: {len(already_done)}/{len(cases)} cases already done, skipping."
        )

    if not pending_cases:
        logger.info("Semua case sudah selesai. Tidak ada yang perlu dijalankan ulang.")
        # Muat semua hasil dari checkpoint untuk summary
        query_results = []
        faithfulness_results = []
        for cid in case_ids:
            cached = _load_checkpoint(model_name, dataset_name, cid)
            query_results.append(_dict_to_query_output(cached.get("query_result", {})))
            faith_result_obj = _checkpoint_to_faithfulness_result(
                cached.get("faithfulness_result", {}),
                similarity_threshold=similarity_threshold,
                embedding_model=config["models"]["embedding"],
                evaluator_version=FaithfulnessEvaluator.VERSION,
            )
            faithfulness_results.append(faith_result_obj)
        total_elapsed = 0.0
    else:
        logger.info(
            f"Starting inference | {len(pending_cases)} pending | "
            f"{len(already_done)} cached | model={model_name} | workers={workers}\n"
        )

        query_results = []
        faithfulness_results = []
        total = len(cases)
        run_start = time.time()

        # Muat hasil dari checkpoint yang sudah ada
        for cid in already_done:
            cached = _load_checkpoint(model_name, dataset_name, cid)
            query_results.append(_dict_to_query_output(cached.get("query_result", {})))
            faith_result_obj = _checkpoint_to_faithfulness_result(
                cached.get("faithfulness_result", {}),
                similarity_threshold=similarity_threshold,
                embedding_model=config["models"]["embedding"],
                evaluator_version=FaithfulnessEvaluator.VERSION,
            )
            faithfulness_results.append(faith_result_obj)

        # Proses pending cases secara parallel
        total_pending = len(pending_cases)
        completed_count = len(already_done)
        lock = threading.Lock()

        # Tempat penyimpanan hasil sesuai indeks untuk menjaga urutan
        pending_query_results = [None] * total_pending
        pending_faith_results = [None] * total_pending

        def process_case_task(case, idx):
            return idx, process_single_case(
                case=case,
                model_name=model_name,
                dataset_name=dataset_name,
                similarity_threshold=similarity_threshold,
                resume=resume,
                client=client,
                evaluator=faithfulness_evaluator,
                embedding_model=config["models"]["embedding"],
                evaluator_version=FaithfulnessEvaluator.VERSION,
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_case_task, case, i): i
                for i, case in enumerate(pending_cases)
            }

            for future in as_completed(futures):
                idx, result = future.result()
                pending_query_results[idx] = result["query_result"]
                pending_faith_results[idx] = result["faithfulness_result"]

                with lock:
                    completed_count += 1
                    elapsed_total = round(time.time() - run_start, 1)
                    done_pending = completed_count - len(already_done)
                    remaining = total_pending - done_pending
                    if done_pending > 0:
                        avg_time = elapsed_total / done_pending
                        eta = remaining * avg_time / workers
                    else:
                        eta = 0
                    logger.info(
                        f"Progress: {completed_count}/{total} cases | "
                        f"ETA={eta//60:.0f}m {eta%60:.0f}s remaining"
                    )

        # Extend hasil dalam urutan case
        query_results.extend(pending_query_results)
        for r in pending_faith_results:
            if r is not None:
                faithfulness_results.append(r)

        total_elapsed = round(time.time() - run_start, 1)

    # save raw query results
    query_output_path = str(output_prefix) + "_queries.json"
    save_query_results(query_results, query_output_path)

    # save faithfulness results — merge checkpointed + newly computed (resume-safe)
    all_faith_results = list(faithfulness_results)
    faith_output_path = str(output_prefix) + "_faithfulness.json"
    faithfulness_evaluator.save_results(all_faith_results, faith_output_path)

    # build & save summary
    summary = _build_summary(
        run_id=run_id,
        model_name=model_name,
        dataset_name=dataset_name,
        cases=cases,
        query_results=query_results,
        faithfulness_results=faithfulness_results,
        total_elapsed=total_elapsed,
        similarity_threshold=similarity_threshold,
    )
    summary_path = str(output_prefix) + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _print_summary(summary, faithfulness_results)
    logger.info(f"Run complete | outputs saved to {REPORTS_DIR}")
    return summary

# checkpoint proxy — lightweight wrapper untuk cached results
class _CheckpointFaithfulness:
    def __init__(self, data: dict):
        self._data = data or {}
        
    @property
    def faithfulness_score(self) -> float:
        return self._data.get("faithfulness_score", 0.0)

    @property
    def has_failure(self) -> bool:
        return self._data.get("has_failure", False)

    @property
    def is_insufficient_context_response(self) -> bool:
        return self._data.get("is_insufficient_context_response", False)

    @property
    def case_id(self) -> str:
        return self._data.get("case_id", "unknown")

    @property
    def failure_cases(self) -> list:
        return self._data.get("failure_cases", [])

    @property
    def question(self) -> str:
        return self._data.get("question", "")


def _checkpoint_to_faithfulness_result(
    data: dict | None,
    similarity_threshold: float,
    embedding_model: str,
    evaluator_version: str,
) -> FaithfulnessResult:
    """Konversi checkpoint dict → FaithfulnessResult (dataclass) untuk merge resume-safe."""
    if not isinstance(data, dict) or not data:
        raise CheckpointInvalidError("faithfulness_result is missing")
    missing = REQUIRED_FAITHFULNESS_FIELDS - set(data)
    if missing:
        raise CheckpointInvalidError(f"faithfulness_result missing fields: {sorted(missing)}")

    # Provenance tersimpan menang atas nilai run sekarang. Fallback ke argumen
    # run hanya untuk checkpoint lama yang belum menyimpan field tersebut.
    stored_threshold = data.get("similarity_threshold")
    stored_embedding = data.get("embedding_model")
    stored_version = data.get("evaluator_version")
    stored_timestamp = data.get("timestamp_utc")

    return FaithfulnessResult(
        case_id=data.get("case_id", "unknown"),
        model=data.get("model", ""),
        question=data.get("question", ""),
        answer=data.get("answer", ""),
        context_preview=data.get("context_preview", ""),
        total_claims=data.get("total_claims", 0) or 0,
        claims=data.get("claims", []) or [],
        evidence_matches=data.get("evidence_matches", []) or [],
        supported_count=data.get("supported_count", 0) or 0,
        unsupported_count=data.get("unsupported_count", 0) or 0,
        skipped_count=data.get("skipped_count", 0) or 0,
        faithfulness_score=data.get("faithfulness_score", 0.0) or 0.0,
        is_insufficient_context_response=bool(data.get("is_insufficient_context_response", False)),
        has_failure=bool(data.get("has_failure", False)),
        failure_cases=data.get("failure_cases", []) or [],
        similarity_threshold=(
            stored_threshold if stored_threshold is not None else similarity_threshold
        ),
        embedding_model=(
            stored_embedding if stored_embedding is not None else embedding_model
        ),
        evaluator_version=(
            stored_version if stored_version is not None else evaluator_version
        ),
        timestamp_utc=(
            stored_timestamp
            if stored_timestamp is not None
            else datetime.now(timezone.utc).isoformat()
        ),
        evaluation_path=data.get("evaluation_path", "claim_extraction"),
    )


def _query_output_to_dict(result: QueryOutput) -> dict:
    """Serialisasi QueryOutput untuk checkpoint — semua provenance ikut disimpan."""
    return {
        "case_id": result.case_id,
        "model": result.model,
        "question": result.question,
        "context_length_chars": result.context_length_chars,
        "answer": result.answer,
        "latency_seconds": result.latency_seconds,
        "timestamp_utc": result.timestamp_utc,
        "success": result.success,
        "error_message": result.error_message,
    }


def _dict_to_query_output(d: dict) -> QueryOutput:
    return QueryOutput(
        case_id=d.get("case_id", "unknown"),
        model=d.get("model", ""),
        question=d.get("question", ""),
        context_length_chars=d.get("context_length_chars", 0),
        answer=d.get("answer", ""),
        latency_seconds=d.get("latency_seconds", 0.0),
        timestamp_utc=d.get("timestamp_utc", datetime.now(timezone.utc).isoformat()),
        success=d.get("success", False),
        error_message=d.get("error_message"),
    )


def _faithfulness_to_dict(result) -> dict:
    return {
        "case_id": result.case_id,
        "model": result.model,
        "question": result.question,
        "answer": result.answer,
        "context_preview": result.context_preview,
        "total_claims": result.total_claims,
        "claims": result.claims,
        "evidence_matches": result.evidence_matches,
        "supported_count": result.supported_count,
        "unsupported_count": result.unsupported_count,
        "skipped_count": result.skipped_count,
        "faithfulness_score": result.faithfulness_score,
        "is_insufficient_context_response": result.is_insufficient_context_response,
        "has_failure": result.has_failure,
        "failure_cases": result.failure_cases,
        "similarity_threshold": result.similarity_threshold,
        "embedding_model": result.embedding_model,
        "evaluator_version": result.evaluator_version,
        "timestamp_utc": result.timestamp_utc,
        "evaluation_path": result.evaluation_path,
    }

# Query helpers
def _query_with_template(client: OllamaClient, case: dict, case_type: str) -> QueryOutput:
    import requests as _requests

    case_id = case["id"]

    if case_type == "distractor":
        distractor_type = case.get("distractor_type", "unknown")
        prompt = PromptTemplate.distractor_qa(
            context=case["context"],
            question=case["question"],
            distractor_type=distractor_type,
        )
    else:
        prompt = PromptTemplate.strict_qa(
            context=case["context"],
            question=case["question"],
        )

    payload = {
        "model": client.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": client.temperature,
            "num_predict": client.max_tokens,
        },
    }

    start = time.time()
    try:
        response = _requests.post(
            client._endpoint,
            json=payload,
            timeout=client.timeout_seconds,
        )
        response.raise_for_status()
        latency = round(time.time() - start, 3)
        raw     = response.json()
        answer  = raw.get("message", {}).get("content", "").strip()

        logger.info(
            f"Query OK | case_id={case_id} | latency={latency}s | "
            f"answer_length={len(answer)} chars"
        )
        return QueryOutput(
            case_id=case_id,
            model=client.model,
            question=case["question"],
            context_length_chars=len(case["context"]),
            answer=answer,
            latency_seconds=latency,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            success=bool(answer),
        )

    except Exception as e:
        latency = round(time.time() - start, 3)
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"Query FAILED | case_id={case_id} | {error_msg}")
        return QueryOutput(
            case_id=case_id,
            model=client.model,
            question=case["question"],
            context_length_chars=len(case["context"]),
            answer="",
            latency_seconds=latency,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            success=False,
            error_message=error_msg,
        )

def _detect_case_type(case: dict) -> str:
    # Gunakan field eksplisit "type" dari dataset; fallback ke deteksi berdasarkan ID
    # untuk backward compatibility dengan dataset lama.
    case_type = case.get("type", "")
    if case_type:
        return case_type
    case_id = case.get("id", "")
    if "distractor" in case_id:
        return "distractor"
    if "paraphrase" in case_id:
        return "paraphrase"
    return "clean"


def process_single_case(
    case: dict,
    model_name: str,
    dataset_name: str,
    similarity_threshold: float,
    resume: bool,
    client: OllamaClient,
    evaluator: FaithfulnessEvaluator,
    embedding_model: str,
    evaluator_version: str,
) -> dict:
    """Proses satu case: cek checkpoint, query, evaluasi, simpan checkpoint.
    
    Thread-safe: setiap case menulis ke file checkpoint sendiri.
    """
    case_id = case["id"]
    case_type = _detect_case_type(case)

    # Cek checkpoint dulu
    if resume:
        cached = _load_checkpoint(model_name, dataset_name, case_id)
        if cached:
            logger.info(f"  [{case_id}] SKIPPED (checkpoint exists)")
            return {
                "case_id": case_id,
                "query_result": _dict_to_query_output(cached.get("query_result", {})),
                "faithfulness_result": _checkpoint_to_faithfulness_result(
                    cached.get("faithfulness_result", {}),
                    similarity_threshold=similarity_threshold,
                    embedding_model=embedding_model,
                    evaluator_version=evaluator_version,
                ),
                "cached": True,
                "elapsed": 0.0,
            }

    # Jalankan query + evaluasi
    logger.info(f"  [{case_id}] Running")
    case_start = time.time()
    query_result = _query_with_retry(client, case, case_type)
    faith_result = None

    if query_result.success and query_result.answer:
        faith_result = evaluator.evaluate(
            case=case,
            model_answer=query_result.answer,
            model_name=model_name,
        )
    else:
        logger.warning(
            f"  [{case_id}] Skipping faithfulness eval | "
            f"reason={'query failed' if not query_result.success else 'empty answer'}"
        )

    # Simpan checkpoint (tiap case ke file sendiri, tidak ada konflik thread)
    faith_result_dict = _faithfulness_to_dict(faith_result) if faith_result else None
    checkpoint_data = {
        "case_id": case_id,
        "model": model_name,
        "dataset": dataset_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "query_result": _query_output_to_dict(query_result),
        "faithfulness_result": faith_result_dict,
    }
    _save_checkpoint(model_name, dataset_name, case_id, checkpoint_data)

    elapsed = round(time.time() - case_start, 1)
    logger.info(f"  [{case_id}] Done | elapsed={elapsed}s")

    return {
        "case_id": case_id,
        "query_result": query_result,
        "faithfulness_result": faith_result,
        "cached": False,
        "elapsed": elapsed,
    }


# consistency runner — paraphrase dataset (multi-variant cases)
def run_consistency_evaluation(
    model_name: str,
    limit: int = None,
    dry_run: bool = False,
) -> dict:
    """Evaluasi dimensi konsistensi pada dataset paraphrase.

    Setiap case punya beberapa varian pertanyaan; semua varian ditanyakan ke
    model, lalu ConsistencyEvaluator mengukur stabilitas jawaban antar varian.
    Tidak memakai checkpoint resume (dataset kecil — 6 cases).
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info(
        f"Starting consistency evaluation | run_id={run_id} | model={model_name} | "
        f"dataset=paraphrase | limit={limit} | dry_run={dry_run} | checkpoint=off"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = REPORTS_DIR / f"{model_name.replace(':', '-')}_paraphrase_{run_id}"

    cases = load_dataset("paraphrase", limit=limit)
    total_variants = sum(len(c.get("variants", [])) for c in cases)
    logger.info(f"Loaded {len(cases)} cases ({total_variants} variants)")
    if not cases:
        logger.error("No cases to evaluate.")
        sys.exit(1)

    if dry_run:
        print(f"\n[DRY RUN] Dataset valid: {len(cases)} cases ({total_variants} variants)")
        print(f"[DRY RUN] Model: {model_name} | Evaluation: consistency")
        for case in cases:
            labels = ", ".join(v["label"] for v in case.get("variants", []))
            print(f"  {case['id']} — {len(case.get('variants', []))} variants ({labels})")
        print("\n[DRY RUN] Selesai. Tidak ada inference yang dijalankan.")
        return {"dry_run": True, "cases": len(cases), "variants": total_variants}

    client = OllamaClient(model=model_name, temperature=0.0)
    consistency_evaluator = ConsistencyEvaluator()

    # health checks
    logger.info("Running health checks...")
    if not client.health_check():
        logger.error(f"Model '{model_name}' tidak tersedia. Jalankan: ollama pull {model_name}")
        sys.exit(1)
    if not consistency_evaluator.health_check():
        logger.error("nomic-embed-text tidak tersedia. Jalankan: ollama pull nomic-embed-text")
        sys.exit(1)
    logger.info("All health checks passed.\n")

    results = consistency_evaluator.run_and_evaluate(
        cases=cases,
        client=client,
        model_name=model_name,
    )

    results_path = str(output_prefix) + "_consistency.json"
    consistency_evaluator.save_results(results, results_path)

    scores = [r.consistency_score for r in results]
    summary = {
        "run_id": run_id,
        "model": model_name,
        "dataset": "paraphrase",
        "dimension": "consistency",
        "template_version": TEMPLATE_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "semantic_threshold": consistency_evaluator.semantic_threshold,
            "rouge_threshold": consistency_evaluator.rouge_threshold,
            "embedding_model": consistency_evaluator.embedding_model,
        },
        "inference": {
            "total_cases": len(cases),
            "total_variants": total_variants,
        },
        "consistency": {
            "evaluated": len(results),
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0,
            "fully_consistent": sum(1 for r in results if r.is_fully_consistent),
            "cases_with_failure": sum(1 for r in results if r.has_failure),
        },
        "failure_case_ids": [r.case_id for r in results if r.has_failure],
        "results_file": Path(results_path).name,
        "provenance": build_provenance(
            evaluator_version=consistency_evaluator.VERSION,
            dataset_name="paraphrase",
            dataset_path=DATASET_FILES.get("paraphrase"),
        ),
    }
    summary_path = str(output_prefix) + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    consistency_evaluator.print_summary(results)
    logger.info(f"Consistency run complete | outputs saved to {REPORTS_DIR}")
    return summary

# summary helpers
def _build_summary(
    run_id, model_name, dataset_name, cases,
    query_results, faithfulness_results,
    total_elapsed, similarity_threshold,
) -> dict:
    success_queries = [r for r in query_results if getattr(r, "success", False)]
    insufficient = [
        r for r in faithfulness_results
        if r.is_insufficient_context_response
    ]
    failures    = [r for r in faithfulness_results if r.has_failure]
    faith_scores = [r.faithfulness_score for r in faithfulness_results]

    return {
        "run_id": run_id,
        "model": model_name,
        "dataset": dataset_name,
        "template_version": TEMPLATE_VERSION,
        "evaluator_version": FaithfulnessEvaluator.VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "similarity_threshold": similarity_threshold,
            "embedding_model": config["models"]["embedding"],
        },
        "inference": {
            "total_cases": len(cases),
            "successful_queries": len(success_queries),
            "failed_queries": len(query_results) - len(success_queries),
            "total_elapsed_seconds": total_elapsed,
            "avg_latency_seconds": round(
                sum(getattr(r, "latency_seconds", 0) for r in query_results) / len(query_results), 1
            ) if query_results else 0,
        },
        "faithfulness": {
            "evaluated": len(faithfulness_results),
            "avg_score": round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else 0,
            "perfect_score_count": sum(1 for s in faith_scores if s == 1.0),
            "cases_with_failure": len(failures),
            "insufficient_context_responses": len(insufficient),
            "failure_rate": round(
                len(failures) / len(faithfulness_results), 4
            ) if faithfulness_results else 0,
        },
        "failure_case_ids": [r.case_id for r in failures],
        "provenance": build_provenance(
            evaluator_version=FaithfulnessEvaluator.VERSION,
            dataset_name=dataset_name,
            dataset_path=DATASET_FILES.get(dataset_name) if dataset_name in DATASET_FILES else None,
        ),
    }

def _print_summary(summary: dict, faithfulness_results: list) -> None:
    f   = summary["faithfulness"]
    inf = summary["inference"]
    print("\n" + "=" * 60)
    print("EVALUATION RUN COMPLETE")
    print("=" * 60)
    print(f"Run ID      : {summary['run_id']}")
    print(f"Model       : {summary['model']}")
    print(f"Dataset     : {summary['dataset']}")
    print(f"Total time  : {inf['total_elapsed_seconds']:.0f}s "
          f"({inf['total_elapsed_seconds']/60:.1f} min)")
    print()
    print("INFERENCE")
    print(f"  Total cases      : {inf['total_cases']}")
    print(f"  Successful       : {inf['successful_queries']}")
    print(f"  Failed           : {inf['failed_queries']}")
    print(f"  Avg latency      : {inf['avg_latency_seconds']}s")
    print()
    print("FAITHFULNESS")
    print(f"  Evaluated        : {f['evaluated']}")
    print(f"  Avg score        : {f['avg_score']} ({f['avg_score']*100:.1f}%)")
    print(f"  Perfect (1.0)    : {f['perfect_score_count']}")
    print(f"  Cases w/ failure : {f['cases_with_failure']}")
    print(f"  Failure rate     : {f['failure_rate']*100:.1f}%")
    print(f"  INSUFFICIENT_CTX : {f['insufficient_context_responses']}")

    if summary["failure_case_ids"]:
        print()
        print("FAILURE CASES:")
        for case_id in summary["failure_case_ids"]:
            result = next(
                (r for r in faithfulness_results if r.case_id == case_id), None
            )
            if result:
                print(f"\n  {case_id} (score={result.faithfulness_score})")
                for fc in result.failure_cases:
                    print(f"     Claim    : {fc.get('unsupported_claim', '')[:80]}")
                    print(f"     Score    : {fc.get('similarity_score', 'N/A')}")
                    if fc.get("technical_conflict"):
                        print(f"     Technical conflict detected")
                    print(f"     Diagnosis: {fc.get('diagnosis', '')}")
    print("\n" + "=" * 60)

# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Framework — Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run — validasi dataset tanpa inference
  python pipeline/runner.py --model phi3:mini --dataset clean --dry-run

  # Development: 3 cases pertama
  python pipeline/runner.py --model phi3:mini --dataset clean --limit 3

  # Full eval dengan resume (default)
  python pipeline/runner.py --model mistral --dataset clean

  # Force re-run dari awal (ignore checkpoint)
  python pipeline/runner.py --model mistral --dataset clean --no-resume

  # Background process (progress tersimpan, bisa resume jika crash)
  nohup python pipeline/runner.py --model mistral --dataset clean > logs/run.log 2>&1 &
        """
    )
    parser.add_argument("--model", type=str, default="phi3:mini",
        help="Nama model Ollama (default: phi3:mini)")
    parser.add_argument("--dataset", type=str, default="clean",
        choices=["clean", "distractor", "paraphrase", "all"],
        help="Dataset yang dievaluasi (default: clean)")
    parser.add_argument("--limit", type=int, default=None,
        help="Batasi jumlah cases (default: semua)")
    parser.add_argument("--threshold", type=float, default=None,
        help=f"Similarity threshold (default: dari config.yaml = "
             f"{config['thresholds']['faithfulness_evidence']})")
    parser.add_argument("--no-resume", action="store_true",
        help="Ignore checkpoint, re-run semua cases dari awal")
    parser.add_argument("--dry-run", action="store_true",
        help="Validasi dataset dan config tanpa inference")
    parser.add_argument("--workers", type=int, default=3,
        help="Jumlah worker parallel untuk inference (default: 3)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 60)
    print("LLM EVALUATION FRAMEWORK")
    print("=" * 60)
    print(f"Model     : {args.model}")
    print(f"Dataset   : {args.dataset}")
    print(f"Limit     : {args.limit or 'all cases'}")
    threshold_display = args.threshold or config["thresholds"]["faithfulness_evidence"]
    print(f"Threshold : {threshold_display}")
    print(f"Resume : {not args.no_resume}")
    print(f"Dry run   : {args.dry_run}")
    print(f"Workers   : {args.workers}")
    print("=" * 60 + "\n")
    
    run_evaluation(
        model_name=args.model,
        dataset_name=args.dataset,
        limit=args.limit,
        similarity_threshold=args.threshold,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        workers=args.workers,
    )
    