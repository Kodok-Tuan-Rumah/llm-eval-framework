"""Regression tests for atomic checkpoint writes (E3).

Bug: ``_save_checkpoint`` menulis langsung ke path target. Proses yang mati di
tengah penulisan meninggalkan file JSON parsial, dan resume berikutnya
menganggap checkpoint itu corrupt — kehilangan hasil yang sudah ada. Penulisan
harus lewat file ``.tmp`` lalu ``os.replace`` agar target selalu berisi JSON
yang utuh (lama atau baru, tidak pernah setengah).
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import runner


PAYLOAD = {
    "case_id": "case_1",
    "model": "mistral",
    "dataset": "clean",
    "timestamp_utc": "2026-01-01T00:00:00Z",
    "faithfulness_result": {"case_id": "case_1", "faithfulness_score": 0.8},
}


@pytest.fixture
def checkpoint_dir(tmp_path, monkeypatch):
    cdir = tmp_path / "checkpoints"
    monkeypatch.setattr(runner, "CHECKPOINT_DIR", cdir)
    return cdir


def test_write_uses_replace(checkpoint_dir, monkeypatch):
    """Checkpoint harus ditulis ke ``.tmp`` lalu di-rename via ``os.replace``."""
    calls = []
    real_replace = os.replace

    def spy_replace(src, dst):
        src, dst = Path(src), Path(dst)
        calls.append(
            {
                "src": src,
                "dst": dst,
                # Snapshot isi sumber tepat sebelum replace: bukti file sudah
                # lengkap & ter-flush sebelum target ditimpa.
                "src_payload": json.loads(src.read_text(encoding="utf-8")),
                "dst_existed_before_replace": dst.exists(),
            }
        )
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)

    runner._save_checkpoint("mistral", "clean", "case_1", PAYLOAD)

    path = runner._checkpoint_path("mistral", "clean", "case_1")
    assert len(calls) == 1, "checkpoint save must rename exactly once via os.replace"
    call = calls[0]
    assert call["src"] == path.with_suffix(path.suffix + ".tmp")
    assert call["src"].name.endswith(".json.tmp")
    assert call["src"] != path
    assert call["dst"] == path
    # Target belum pernah ditulis langsung: rename adalah publikasi pertama.
    assert call["dst_existed_before_replace"] is False
    assert call["src_payload"]["case_id"] == "case_1"
    assert call["src_payload"]["schema_version"] == runner.CHECKPOINT_SCHEMA_VERSION

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["case_id"] == "case_1"
    assert saved["schema_version"] == runner.CHECKPOINT_SCHEMA_VERSION


def test_failed_write_leaves_previous_checkpoint_intact(checkpoint_dir):
    """Gagal serialisasi tidak boleh merusak checkpoint lama."""
    runner._save_checkpoint("mistral", "clean", "case_1", PAYLOAD)
    path = runner._checkpoint_path("mistral", "clean", "case_1")
    previous_bytes = path.read_bytes()

    bad_payload = {
        "case_id": "case_1",
        "unserializable": object(),  # json.dump raises TypeError after partial output
    }
    with pytest.raises(TypeError):
        runner._save_checkpoint("mistral", "clean", "case_1", bad_payload)

    assert path.exists()
    assert path.read_bytes() == previous_bytes
    assert json.loads(path.read_text(encoding="utf-8"))["case_id"] == "case_1"
    # Penulisan gagal tidak boleh meninggalkan sisa file sementara.
    assert list(checkpoint_dir.glob("*.tmp")) == []


def test_no_tmp_file_remains_after_success(checkpoint_dir):
    """Sukses: hanya file checkpoint final, tanpa artefak ``.tmp``."""
    runner._save_checkpoint("mistral", "clean", "case_1", PAYLOAD)

    path = runner._checkpoint_path("mistral", "clean", "case_1")
    assert path.exists()
    assert list(checkpoint_dir.iterdir()) == [path]
    assert list(checkpoint_dir.glob("*.tmp")) == []
    assert list(checkpoint_dir.glob("*.json.tmp")) == []

    # Format dan stamp schema tidak berubah karena implementasi atomik.
    text = path.read_text(encoding="utf-8")
    assert text == json.dumps(
        {**PAYLOAD, "schema_version": runner.CHECKPOINT_SCHEMA_VERSION},
        indent=2,
        ensure_ascii=False,
    )
