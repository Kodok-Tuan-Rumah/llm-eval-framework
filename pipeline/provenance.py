"""Provenance metadata untuk setiap artefak hasil evaluasi.

Tujuan
──────
Setiap angka yang dihasilkan runner harus membawa konteks yang cukup untuk
menjawab: "hasil ini dibuat oleh kode, konfigurasi, dan dataset versi apa?"

Tanpa ini, hasil dari threshold 0.75 bisa tertukar dengan hasil threshold 0.68,
dan tidak ada yang bisa membuktikan angka mana yang berlaku — persis masalah
yang ditemukan pada audit laporan Februari 2026.
"""

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config_loader import config

PROJECT_ROOT = Path(__file__).parent.parent

# Snapshot konfigurasi yang relevan untuk mereproduksi sebuah hasil.
_SNAPSHOT_SECTIONS = ("thresholds", "models", "inference")


def sha256_file(path: str | Path | None) -> str | None:
    """SHA-256 sebuah file, atau None kalau path tidak ada/None."""
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
    """Short commit SHA dari working tree, atau None kalau bukan git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root or PROJECT_ROOT),
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def git_dirty(root: str | Path | None = None) -> bool | None:
    """True kalau ada perubahan belum di-commit (hasil mungkin tidak reproducible)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root or PROJECT_ROOT),
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:
        pass
    return None


def config_snapshot() -> dict:
    """Salinan seluruh nilai konfigurasi yang memengaruhi hasil."""
    return {section: dict(config.get(section, {})) for section in _SNAPSHOT_SECTIONS}


def build_provenance(
    evaluator_version: str,
    dataset_name: str | None = None,
    dataset_path: str | Path | None = None,
    extra: dict | None = None,
) -> dict:
    """Bangun blok provenance lengkap untuk satu artefak hasil.

    Parameters
    ----------
    evaluator_version : str
        Versi evaluator yang menghasilkan angka (mis. FaithfulnessEvaluator.VERSION).
    dataset_name, dataset_path :
        Nama dataset dan path file yang di-hash (untuk deteksi data berubah).
    extra :
        Metadata tambahan opsional (mis. threshold efektif run ini).
    """
    provenance = {
        "evaluator_version": evaluator_version,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "config_snapshot": config_snapshot(),
        "dataset": {
            "name": dataset_name,
            "sha256": sha256_file(dataset_path),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        provenance.update(extra)
    return provenance
