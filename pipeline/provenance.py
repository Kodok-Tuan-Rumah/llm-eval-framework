import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pipeline.config_loader import config

PROJECT_ROOT = Path(__file__).parent.parent

# Snapshot konfigurasi yang relevan untuk mereproduksi sebuah hasil.
_SNAPSHOT_SECTIONS = ("thresholds", "models", "inference")


def sha256_file(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    digest = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: str | Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root or PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def git_dirty(root: str | Path | None = None) -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root or PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:
        pass
    return None


def config_snapshot() -> dict:
    return {section: dict(config.get(section, {})) for section in _SNAPSHOT_SECTIONS}


def build_provenance(
    evaluator_version: str,
    dataset_name: str | None = None,
    dataset_path: str | Path | None = None,
    extra: dict | None = None,
) -> dict:
    provenance = {
        "evaluator_version": evaluator_version,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "config_snapshot": config_snapshot(),
        "dataset": {
            "name": dataset_name,
            "sha256": sha256_file(dataset_path),
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }
    if extra:
        provenance.update(extra)
    return provenance
