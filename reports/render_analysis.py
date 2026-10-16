#!/usr/bin/env python3
"""Regenerate numeric tables di analysis.md dari artefak hasil.

Menambahkan marker di analysis.md:
  <!-- GENERATED:RESULTS:BEGIN --> ... <!-- GENERATED:RESULTS:END -->

Skrip ini mengganti bagian dalam marker dengan tabel-tabel hasil.
Hanya memanipulasi bagian hasil; narasi Findings/Limitations tetap.

Fail-loud: jika artefak yang diperlukan tidak ada, skrip berhenti dengan error
(atau gagal menghasilkan tabel kosong dari data palsu).
"""

import json
import re
from pathlib import Path

ANALYSIS_P = Path(__file__).parent.parent / "reports" / "analysis.md"
RESULTS_D  = Path(__file__).parent.parent / "reports" / "results"

BEGIN = "<!-- GENERATED:RESULTS:BEGIN -->"
END   = "<!-- GENERATED:RESULTS:END -->"


def find_latest(pattern):
    files = sorted(RESULTS_D.glob(pattern))
    return files[-1] if files else None


def require_json(p, label: str):
    if p is None or not Path(p).exists():
        raise SystemExit(f"[error] Arte­fat hasil tidak ditemukan untuk {label}: {p}")
    return json.loads(Path(p).read_text())


def load_json(p):
    if p is None or not Path(p).exists():
        return {}
    return json.loads(Path(p).read_text())


def pct(x: float) -> str:
    return f"{round(x * 100, 1)}%"


def _empty_agg():
    return dict(avg=0, perfect=0, total=0, failures=0, failure_rate=0, false_insuf=0)


def make_tables():
    tables = []

    mcf_path = find_latest("*mistral*clean*_faithfulness.json")
    mdf_path = find_latest("*mistral*distractor*_faithfulness.json")
    pcf_path = find_latest("*phi3*clean*_faithfulness.json")
    mcs_path = find_latest("*mistral*clean*_summary.json")
    pcs_path = find_latest("*phi3*clean*_summary.json")

    mcf = require_json(mcf_path, "mistral/clean")
    mdf = require_json(mdf_path, "mistral/distractor")
    pcf = require_json(pcf_path, "phi3:mini/clean")

    def agg(raw: list) -> dict:
        scores = [r.get("faithfulness_score") for r in raw if r.get("faithfulness_score") is not None]
        n = len(scores)
        if n == 0:
            return _empty_agg()
        perfect = sum(1 for s in scores if s == 1.0)
        failures = sum(1 for r in raw if r.get("faithfulness_score") is not None and r.get("has_failure"))
        false_insuf = sum(1 for r in raw if r.get("is_insufficient_context_response") and r.get("has_failure"))
        avg = round(sum(scores) / n, 4)
        return dict(avg=avg, perfect=perfect, total=n, failures=failures,
                    failure_rate=round(failures / n, 4), false_insuf=false_insuf)

    ma = agg(mcf)
    da = agg(mdf)
    pa = agg(pcf)

    ms = load_json(mcs_path) if mcs_path else {}
    ps = load_json(pcs_path) if pcs_path else {}
    mlat = ms.get("inference", {}).get("avg_latency_seconds")
    plat = ps.get("inference", {}).get("avg_latency_seconds")
    mrun = ms.get("run_id", "-")
    drun = (load_json(find_latest("*mistral*distractor*_summary.json")) or {}).get("run_id", "-") if find_latest("*mistral*distractor*_summary.json") else "-"

    # Mistral clean
    tables.append(f"### Mistral — Clean Dataset\n\n**Run ID:** {mrun}\n")
    tables.append("| Case | Score | Keterangan |\n|------|-------|------------|\n")
    for r in mcf[:12]:
        fc = (r.get("failure_cases") or [{}])[0]
        tables.append(f"| {r['case_id']} | {r.get('faithfulness_score')} | {fc.get('diagnosis','')} |\n")
    tables.append(f"\n**Avg faithfulness: {pct(ma['avg'])} | Perfect: {ma['perfect']}/{ma['total']} | Failure rate: {pct(ma['failure_rate'])} | False INSUF_CTX: {ma['false_insuf']}**\n\n---\n")

    # Distractor
    tables.append(f"### Mistral — Distractor Dataset\n\n**Run ID:** {drun}\n")
    tables.append("| Case | Score | Keterangan |\n|------|-------|------------|\n")
    for r in mdf[:12]:
        fc = (r.get("failure_cases") or [{}])[0]
        tables.append(f"| {r['case_id']} | {r.get('faithfulness_score')} | {fc.get('diagnosis','')} |\n")
    tables.append(f"\n**Avg faithfulness: {pct(da['avg'])} | Perfect: {da['perfect']}/{da['total']} | Failure rate: {pct(da['failure_rate'])}**\n\n---\n")

    # Comparison
    tables.append("### Model Comparison — Mistral vs phi3:mini (Clean Dataset)\n\n| Metrik | Mistral | phi3:mini |\n|--------|---------|-----------|\n")
    tables.append(f"| Avg faithfulness | **{pct(ma['avg'])}** | {pct(pa['avg'])} |\n")
    tables.append(f"| Perfect score (1.0) | {ma['perfect']}/{ma['total']} | {pa['perfect']}/{pa['total']} |\n")
    tables.append(f"| Failure rate | {pct(ma['failure_rate'])} | {pct(pa['failure_rate'])} |\n")
    tables.append(f"| False INSUFFICIENT_CTX | {ma['false_insuf']} | {pa['false_insuf']} |\n")
    if mlat and plat:
        tables.append(f"| Avg inference latency / case | {mlat:.0f}s | **{plat:.0f}s** |\n")
    tables.append("---\n")

    # Clean vs distractor
    tables.append("### Clean vs Distractor Comparison\n\n| Metrik | Clean Dataset | Distractor Dataset |\n|--------|--------------|-------------------|\n")
    tables.append(f"| Avg faithfulness | {pct(ma['avg'])} | {pct(da['avg'])} |\n")
    tables.append(f"| Perfect score | {ma['perfect']}/{ma['total']} | {da['perfect']}/{da['total']} |\n")
    tables.append(f"| Failure rate | {pct(ma['failure_rate'])} | {pct(da['failure_rate'])} |\n")
    tables.append("---\n")
    return "".join(tables)


def main() -> int:
    if not ANALYSIS_P.exists():
        print(f"analysis.md tidak ada: {ANALYSIS_P}")
        return 1
    text = ANALYSIS_P.read_text()
    if BEGIN not in text or END not in text:
        print(f"Marker tidak ditemukan di analysis.md — tambahkan {BEGIN} dan {END}")
        return 1
    generated = make_tables()
    new = re.sub(f"{re.escape(BEGIN)}.*?{re.escape(END)}", f"{BEGIN}\n\n{generated}\n{END}", text, flags=re.DOTALL)
    ANALYSIS_P.write_text(new)
    print("Done: numeric region regenerated")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
