import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.runner import (
    _load_json_cases,
    load_dataset,
    run_consistency_evaluation,
    run_evaluation,
)


class TestLoadDataset:
    def test_clean_dataset_has_questions(self):
        cases = load_dataset("clean")
        assert len(cases) == 10
        assert all("question" in c for c in cases)

    def test_paraphrase_dataset_loads_with_variants(self):
        cases = load_dataset("paraphrase")
        assert len(cases) == 6
        for case in cases:
            assert "variants" in case
            assert len(case["variants"]) >= 2
            for variant in case["variants"]:
                assert variant["label"]
                assert variant["question"]

    def test_all_dataset_includes_paraphrase(self):
        cases = load_dataset("all")
        assert any("variants" in c for c in cases)
        assert any("question" in c for c in cases)

    def test_unknown_dataset_rejected(self):
        with pytest.raises(ValueError):
            load_dataset("does-not-exist")


class TestLoadJsonCasesValidation:
    def _write(self, tmp_path, payload):
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_case_without_question_or_variants_rejected(self, tmp_path):
        path = self._write(tmp_path, {"cases": [{"id": "x", "context": "ctx"}]})
        with pytest.raises(ValueError, match="variants"):
            _load_json_cases(path, "test")

    def test_case_missing_id_rejected(self, tmp_path):
        path = self._write(tmp_path, {"cases": [{"context": "ctx", "question": "q"}]})
        with pytest.raises(ValueError, match="missing required fields"):
            _load_json_cases(path, "test")

    def test_variant_missing_question_rejected(self, tmp_path):
        path = self._write(
            tmp_path,
            {
                "cases": [
                    {
                        "id": "x",
                        "context": "ctx",
                        "variants": [{"label": "original"}],
                    }
                ]
            },
        )
        with pytest.raises(ValueError, match="variant missing required fields"):
            _load_json_cases(path, "test")


class TestDryRuns:
    def test_paraphrase_dry_run_reports_variants(self, capsys):
        summary = run_consistency_evaluation(model_name="test-model", dry_run=True)
        assert summary["dry_run"] is True
        assert summary["cases"] == 6
        assert summary["variants"] >= 12
        captured = capsys.readouterr().out
        assert "consistency" in captured

    def test_paraphrase_dataset_delegates_to_consistency(self):
        summary = run_evaluation(
            model_name="test-model", dataset_name="paraphrase", dry_run=True
        )
        assert summary["dry_run"] is True
        assert summary["cases"] == 6
        assert "variants" in summary

    def test_all_dry_run_skips_paraphrase(self, capsys):
        summary = run_evaluation(
            model_name="test-model", dataset_name="all", dry_run=True
        )
        assert summary["dry_run"] is True
        assert summary["skipped_paraphrase"] == 6
        assert summary["cases"] == 15  # 10 clean + 5 distractor
        captured = capsys.readouterr().out
        assert "di-skip" in captured
