import json
import re
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
CONFIG_P = ROOT_DIR / "config.yaml"
README_P = ROOT_DIR / "README.md"
RESULTS_D = ROOT_DIR / "reports" / "results"


def read_config():
    return yaml.safe_load(CONFIG_P.read_text()) if CONFIG_P.exists() else {}


def _find_latest(pattern) -> list[Path]:
    files = sorted(RESULTS_D.glob(pattern))
    return files


def aggregate_from_faithfulness(fpath: Path) -> dict:
    raw = json.loads(fpath.read_text())
    scores = [
        r.get("faithfulness_score")
        for r in raw
        if r.get("faithfulness_score") is not None
    ]
    n = len(scores)
    if n == 0:
        return dict(
            avg=0.0, perfect=0, total=0, failures=0, failure_rate=0.0, false_insuf=0
        )
    perfect = sum(1 for s in scores if s == 1.0)
    failures = sum(
        1
        for r in raw
        if r.get("faithfulness_score") is not None and r.get("has_failure")
    )
    false_insuf = sum(
        1
        for r in raw
        if r.get("is_insufficient_context_response") and r.get("has_failure")
    )
    avg = round(sum(scores) / n, 4)
    return dict(
        avg=avg,
        perfect=perfect,
        total=n,
        failures=failures,
        failure_rate=round(failures / n, 4),
        false_insuf=false_insuf,
    )


def parse_readme_averages(readme_text: str) -> dict:
    m = re.search(
        r"Avg faithfulness\s*\|\s*\*{0,2}([\d.]+)%\*{0,2}\s*\|\s*\*{0,2}([\d.]+)%",
        readme_text,
    )
    if m:
        return dict(m_readme=m.group(1), p_readme=m.group(2))
    return dict(m_readme=None, p_readme=None)


def parse_readme_distractor_avg(readme_text: str) -> str | None:
    m = re.search(r"\|\s*Distractor\s*\|\s*\*{0,2}([\d.]+)%", readme_text)
    return m.group(1) if m else None


def main(args=None) -> int:
    cfg = read_config()
    cfg_thresh = (
        float(cfg.get("thresholds", {}).get("faithfulness_evidence", 0.0))
        if cfg
        else 0.0
    )
    faith_files = list(_find_latest("*_faithfulness.json"))
    summary_files = list(_find_latest("*_summary.json"))

    # Tidak ada artefak → skip aman
    if not faith_files and not summary_files:
        print(
            "Tidak ada artefak hasil di reports/results — guard dilewati (angka dipublikasi adalah historical)"
        )
        return 0

    print(
        f"Menemukan {len(faith_files)} faithfulness + {len(summary_files)} summary artefak"
    )
    # Integrity
    errors = []
    try:
        # import evaluator version
        import sys

        sys.path.insert(0, str(ROOT_DIR))
        from evaluators.faithfulness import FaithfulnessEvaluator

        eval_ver = FaithfulnessEvaluator.VERSION
    except Exception as e:
        eval_ver = "unknown"
        errors.append(f"failed to import evaluator version: {e}")

    for s in summary_files:
        try:
            data = json.loads(s.read_text())
        except Exception as e:
            errors.append(f"{s.name}: invalid json {e}")
            continue
        if "provenance" not in data:
            errors.append(f"{s.name}: provenance missing")
            continue
        prov = data["provenance"]
        # required keys minimal
        for key in ("evaluator_version", "config_snapshot", "dataset"):
            if key not in prov:
                errors.append(f"{s.name}: provenance.{key} missing")
        if (
            "evaluator_version" in prov
            and eval_ver != "unknown"
            and prov["evaluator_version"] != eval_ver
        ):
            errors.append(
                f"{s.name}: evaluator_version berbeda (file={prov['evaluator_version']} vs current={eval_ver})"
            )
        if (
            prov.get("config_snapshot", {})
            .get("thresholds", {})
            .get("faithfulness_evidence")
            != cfg_thresh
        ):
            errors.append(
                f"{s.name}: threshold drift (config_snapshot={prov['config_snapshot']['thresholds'].get('faithfulness_evidence')} vs config.yaml={cfg_thresh})"
            )

        d = prov.get("dataset", {})

        if d.get("sha256") is not None and d.get("name"):
            dp = (
                ROOT_DIR
                / "data"
                / "dataset"
                / (
                    d["name"] + "_cases.json"
                    if d["name"] not in ("clean", "distractor", "paraphrase")
                    else {
                        "clean": "clean_cases.json",
                        "distractor": "distractor_cases.json",
                        "paraphrase": "paraphrase_cases.json",
                    }[d["name"]]
                )
            )

        # dataset file sha256 check
        if d.get("name") in ("clean", "distractor", "paraphrase"):
            ds_map = {
                "clean": "clean_cases.json",
                "distractor": "distractor_cases.json",
                "paraphrase": "paraphrase_cases.json",
            }
            dpath = ROOT_DIR / "data" / "dataset" / ds_map[d["name"]]
            ds_hash = (
                __import__("hashlib").sha256(dpath.read_bytes()).hexdigest()
                if dpath.exists()
                else None
            )
            if d.get("sha256") and ds_hash and d["sha256"] != ds_hash:
                errors.append(f"{s.name}: dataset sha256 drift untuk {d['name']}")

    # Numeric cross-check if README exists
    if README_P.exists():
        readme = README_P.read_text()
        p = parse_readme_averages(readme)
        files = _find_latest("*_faithfulness.json")
        by_model_dataset = {}
        for f in files:
            name = f.name
            key = name.replace("phi3-mini", "phi3mini")
            import re as _re

            m = _re.match(
                r"(mistral|phi3[-:]*mini|phi3mini)_(clean|distractor)_(\d{8}_\d{6})_faithfulness\.json",
                name,
            )
            if not m:
                m2 = _re.match(
                    r"(.*)_(clean|distractor|paraphrase)_.*_faithfulness\.json", name
                )
                if not m2:
                    continue
                mm, dd = m2.group(1), m2.group(2)
            else:
                mm, dd, _ = m.group(1), m.group(2), m.group(3)
            by_model_dataset[(mm, dd)] = f

        # get mistral clean and phi3mini clean
        mcf = (
            by_model_dataset.get(("mistral", "clean"))
            or by_model_dataset.get(("mistral_clean", None))
            or next(
                (x for x in files if "mistral" in x.name and "clean" in x.name), None
            )
        )
        pcf = (
            by_model_dataset.get(("phi3:mini", "clean"))
            or by_model_dataset.get(("phi3-mini", "clean"))
            or next((x for x in files if "phi3" in x.name and "clean" in x.name), None)
        )
        dcf = by_model_dataset.get(("mistral", "distractor")) or next(
            (x for x in files if "mistral" in x.name and "distractor" in x.name), None
        )

        # compute
        if mcf and p and p["m_readme"]:
            m_ag = aggregate_from_faithfulness(mcf)
            pm = (
                str(m_ag["avg"] * 100)[:-2]
                if abs(round(m_ag["avg"] * 100, 1) - int(m_ag["avg"] * 100)) < 1e-9
                else f"{m_ag['avg'] * 100:.1f}"
            )

    # report
    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        return 1
    print("Semua cek OK (provenance, threshold, dataset hash)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
