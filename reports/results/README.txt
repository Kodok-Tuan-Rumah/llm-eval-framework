# reports/results/

Folder ini berisi artefak hasil evaluasi yang di-commit (summary, faithfulness, consistency).

Aturan:
- Hanya file JSON hasil (*_summary.json, *_faithfulness.json, *_consistency.json) yang boleh di-commit.
- Setiap artefak wajib memiliki blok `provenance` (evaluator_version, git_commit, git_dirty, config_snapshot, dataset.sha256, generated_at).
- Jangan commit log besar, checkpoints, atau file sementara.
- Jika artefak ada di sini, maka `scripts/check_reproducibility.py` akan melakukan cross-check ketat (angka README vs artefak, integrity provenance, kesamaan threshold/config vs kode saat ini).
- Jika artefak tidak ada, guard akan melewati (skip) — policy ini menjaga angka historis tetap terdokumentasi tanpa memaksa re-run.
