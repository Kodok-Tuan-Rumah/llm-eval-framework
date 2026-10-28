import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "reports"))

import generate_report as gr
import visualize as viz


# ── Helpers ───────────────────────────────────────────────────────────────────
def _case(case_id: str, score: float, insufficient: bool = False) -> dict:
    return {
        "case_id": case_id,
        "faithfulness_score": score,
        "has_failure": score != 1.0,
        "is_insufficient_context_response": insufficient,
        "failure_cases": [] if score == 1.0 else [{"diagnosis": "synthetic failure"}],
    }


def _write_run(
    results_dir: Path,
    model: str,
    dataset: str,
    run_id: str,
    cases: list,
    avg_latency: float | None = None,
) -> None:
    safe = model.replace(":", "-")
    (results_dir / f"{safe}_{dataset}_{run_id}_faithfulness.json").write_text(
        json.dumps(cases)
    )
    if avg_latency is not None:
        (results_dir / f"{safe}_{dataset}_{run_id}_summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "evaluator_version": "test-9.9.9",
                    "inference": {"avg_latency_seconds": avg_latency},
                }
            )
        )


@pytest.fixture
def synthetic_results(tmp_path, monkeypatch):
    results = tmp_path / "results"
    charts = tmp_path / "charts"
    results.mkdir()
    charts.mkdir()
    monkeypatch.setattr(gr, "RESULTS_DIR", results)
    monkeypatch.setattr(gr, "CHARTS_DIR", charts)

    # mistral / clean → avg 50% (1.0 + 0.0)
    _write_run(
        results,
        "mistral",
        "clean",
        "20260101_000000",
        [_case("clean_001", 1.0), _case("clean_002", 0.0)],
        avg_latency=100.0,
    )
    # mistral / distractor → avg 0%
    _write_run(
        results,
        "mistral",
        "distractor",
        "20260101_000001",
        [_case("distractor_001", 0.0)],
    )
    # phi3:mini / clean → avg 25% (1.0 + 0.0 + 0.0 + 0.0)
    _write_run(
        results,
        "phi3:mini",
        "clean",
        "20260101_000002",
        [
            _case("clean_001", 1.0),
            _case("clean_002", 0.0),
            _case("clean_003", 0.0),
            _case("clean_004", 0.0),
        ],
        avg_latency=50.0,
    )
    return results, charts


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestFailLoud:
    def test_missing_artifacts_raises_instead_of_fabricating(
        self, tmp_path, monkeypatch
    ):
        empty = tmp_path / "results"
        empty.mkdir()
        monkeypatch.setattr(gr, "RESULTS_DIR", empty)

        with pytest.raises(SystemExit) as exc:
            gr.generate_report(str(tmp_path / "report.html"))

        assert "tidak lengkap" in str(exc.value)
        assert not (tmp_path / "report.html").exists()

    def test_hardcoded_fallback_data_removed(self):
        for name in (
            "MISTRAL_CLEAN_DATA",
            "MISTRAL_DISTRACTOR_DATA",
            "PHI3_CLEAN_DATA",
        ):
            assert not hasattr(gr, name), f"{name} masih ada — fallback belum dihapus"


class TestAggregatesDerivedFromArtifacts:
    def test_report_numbers_come_from_artifacts(self, synthetic_results, tmp_path):
        out = tmp_path / "report.html"
        gr.generate_report(str(out))
        html = out.read_text()
        assert "50.0%" in html
        assert "25.0%" in html
        assert "56.7%" not in html
        assert "46.7%" not in html

    def test_evaluator_version_read_from_artifact(self, synthetic_results, tmp_path):
        out = tmp_path / "report.html"
        gr.generate_report(str(out))
        html = out.read_text()
        assert "test-9.9.9" in html

    def test_missing_summary_does_not_crash(
        self, synthetic_results, tmp_path, monkeypatch
    ):
        (synthetic_results[0] / "mistral_clean_20260101_000000_summary.json").unlink()
        out = tmp_path / "report.html"
        gr.generate_report(str(out))
        assert out.exists()


class TestThresholdsSingleSourceOfTruth:
    def test_thresholds_table_matches_config(self):
        cfg = gr.load_config()
        rows = {r["name"]: r["value"] for r in gr.build_thresholds(cfg)}
        assert rows["Faithfulness evidence matching"] == str(
            cfg["thresholds"]["faithfulness_evidence"]
        )

    def test_visualizer_threshold_matches_config(self):
        cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
        assert viz.FAITH_THRESHOLD == cfg["thresholds"]["faithfulness_evidence"]
