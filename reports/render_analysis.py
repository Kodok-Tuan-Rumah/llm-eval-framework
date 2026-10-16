#!/usr/bin/env python3
"""Regenerate numeric tables di analysis.md dari artefak hasil.

Menambahkan marker di analysis.md:
  <!-- GENERATED:RESULTS:BEGIN --> ... <!-- GENERATED:RESULTS:END -->

Skrip ini mengganti bagian dalam marker dengan tabel-tabel hasil.
Hanya memanipulasi bagian hasil; narasi Findings/Limitations tetap.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ANALYSIS_P = Path(__file__).parent.parent / "reports" / "analysis.md"
RESULTS_D  = Path(__file__).parent.parent / "reports" / "results"
CONFIG_P   = Path(__file__).parent.parent / "config.yaml"

BEGIN = "<!-- GENERATED:RESULTS:BEGIN -->"
END   = "<!-- GENERATED:RESULTS:END -->"


def find_latest(pattern):
    files = sorted(RESULTS_D.glob(pattern))
    return files[-1] if files else None


def load_json(p):
    return json.loads(Path(p).read_text()) if p and Path(p).exists() else {}


def pct(x: float) -> str:
    return f"{round(x*100, 1)}%"


def make_tables():
    tables = []

    mcf = find_latest("*mistral*clean*_faithfulness.json")
    mdf = find_latest("*mistral*distractor*_faithfulness.json")
    pcf = find_latest("*phi3*clean*_faithfulness.json")
    mcs = find_latest("*mistral*clean*_summary.json")
    pcs = find_latest("*phi3*clean*_summary.json")

    def agg(p):
        raw = load_json(p)
        if not raw:
            return dict(avg=0,perfect=0,total=0,failures=0,failure_rate=0,false_insuf=0)
        scores = [r.get('faithfulness_score') for r in raw if r.get('faithfulness_score') is not None]
        n = len(scores)
        if n == 0: return dict(avg=0,perfect=0,total=0,failures=0,failure_rate=0,false_insuf=0)
        perfect = sum(1 for s in scores if s == 1.0)
        failures = sum(1 for r in raw if r.get('faithfulness_score') is not None and r.get('has_failure'))
        false_insuf = sum(1 for r in raw if r.get('is_insufficient_context_response') and r.get('has_failure'))
        avg = round(sum(scores)/n,4)
        return dict(avg=avg,perfect=perfect,total=n,failures=failures,failure_rate=round(failures/n,4),false_insuf=false_insuf)

    ma, da, pa = agg(mcf), agg(mdf), agg(pcf)
    ms, ps = load_json(mcs), load_json(pcs)
    mlat = ms.get('inference',{}).get('avg_latency_seconds')
    plat = ps.get('inference',{}).get('avg_latency_seconds')
    mrun = ms.get('run_id','-')
    drun = (find_latest('*mistral*distractor*_summary.json') and load_json(find_latest('*mistral*distractor*_summary.json'))).get('run_id','-') if find_latest('*mistral*distractor*_summary.json') else '-'

    # Mistral clean
    tables.append(f"### Mistral — Clean Dataset\n\n**Run ID:** {mrun}\n")
    # per-case table
    tables.append("| Case | Score | Keterangan |\n|------|-------|------------|\n")
    for r in load_json(mcf)[:12]:
        fc = (r.get('failure_cases') or [{}])[0]
        tables.append(f"| {r['case_id']} | {r.get('faithfulness_score')} | {fc.get('diagnosis','')} |\n")
    tables.append(f"\n**Avg faithfulness: {pct(ma['avg'])} | Perfect: {ma['perfect']}/{ma['total']} | Failure rate: {pct(ma['failure_rate'])} | False INSUF_CTX: {ma['false_insuf']}**\n\n---\n")

    # Distractor
    tables.append(f"### Mistral — Distractor Dataset\n\n**Run ID:** {drun}\n")
    tables.append("| Case | Score | Keterangan |\n|------|-------|------------|\n")
    for r in load_json(mdf)[:12]:
        fc = (r.get('failure_cases') or [{}])[0]
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
