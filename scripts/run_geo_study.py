#!/usr/bin/env python3
"""GEO-Exact study driver: runs every (case, method, trial) in a fresh process,
appends JSON lines to results/geo_study.jsonl (resumable), then writes
results/GEO_STUDY.md with the comparison tables.

Usage:
  python3 run_geo_study.py run        # run everything not yet in the jsonl
  python3 run_geo_study.py summary    # rebuild GEO_STUDY.md from the jsonl
  python3 run_geo_study.py run --only bp_be   # restrict to designs containing text
"""
from __future__ import annotations

import json
import statistics as stx
import subprocess
import sys
from math import exp, log
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
JSONL = RESULTS / "geo_study.jsonl"
REPORT = RESULTS / "GEO_STUDY.md"

TRAIN = ["ariane133", "bp", "bp_be", "swerv_wrapper"]
TEST_A = ["ariane136", "bp_fe", "bp_multi"]
TEST_B = ["aes", "dynamic_node", "ibex", "jpeg", "swerv"]

CASES = ([("Train", d, p, "tcad25") for d in TRAIN for p in (4.0, 6.0)]
         + [("Test A", d, p, "tcad25") for d in TEST_A for p in (4.0, 6.0)]
         + [("Scalability", "bp_quad", p, "tcad25") for p in (4.0, 6.0)]
         + [("Test B", d, 2.0, "cadb1021") for d in TEST_B])

TRIALS = {"sw": 3, "sw_indexed": 3, "dghl": 5, "geo": 5, "knn8": 3, "knn16": 3, "knn32": 3}
TRIALS_BPQUAD = {"sw": 1, "sw_indexed": 1, "dghl": 3, "geo": 3, "knn8": 1, "knn16": 1, "knn32": 1}
METHODS = ["sw", "sw_indexed", "dghl", "geo"]          # main comparison table
KNN = ["knn8", "knn16", "knn32"]                        # review 3.2 baseline table
RUN_METHODS = METHODS + KNN
TIMEOUT_S = 900


def load_records() -> list[dict]:
    if not JSONL.exists():
        return []
    out = []
    for line in JSONL.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            out.append(json.loads(line))
    return out


def key(rec: dict) -> tuple:
    return (rec["design"], float(rec["pitch"]), rec["method"])


def run(only: str | None) -> None:
    RESULTS.mkdir(exist_ok=True)
    done: dict[tuple, int] = {}
    for r in load_records():
        done[key(r)] = done.get(key(r), 0) + 1
    for split, design, pitch, source in CASES:
        if only and only not in design:
            continue
        trials = TRIALS_BPQUAD if design == "bp_quad" else TRIALS
        for method in RUN_METHODS:
            have = done.get((design, pitch, method), 0)
            for t in range(have, trials[method]):
                cmd = [sys.executable, str(HERE / "run_one.py"), design, str(pitch), source, method]
                print(f"[{split}] {design}@{pitch:g} {method} trial {t + 1}/{trials[method]}", flush=True)
                try:
                    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    rec = {"design": design, "pitch": pitch, "source": source, "method": method,
                           "error": f"timeout {TIMEOUT_S}s"}
                else:
                    lines = [l for l in cp.stdout.splitlines() if l.startswith("{")]
                    if cp.returncode != 0 or not lines:
                        rec = {"design": design, "pitch": pitch, "source": source, "method": method,
                               "error": (cp.stderr or cp.stdout)[-2000:]}
                    else:
                        rec = json.loads(lines[-1])
                rec["split"] = split
                with JSONL.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                if "error" in rec:
                    print("   ERROR:", rec["error"].splitlines()[-1] if rec["error"] else "?", flush=True)
                    break


def geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x > 0]
    return exp(sum(log(x) for x in xs) / len(xs)) if xs else float("nan")


def summary() -> None:
    recs = load_records()
    by: dict[tuple, list[dict]] = {}
    errors = []
    for r in recs:
        if "error" in r:
            errors.append(r)
            continue
        by.setdefault(key(r), []).append(r)

    def cell(design, pitch, method):
        rs = by.get((design, pitch, method), [])
        if not rs:
            return None
        totals = {round(r["total_recomputed"], 6) for r in rs}
        rts = [r["runtime_s"] for r in rs]
        return {
            "total": rs[0]["total_recomputed"],
            "deterministic": len(totals) == 1,
            "injective": all(r["injective"] for r in rs),
            "rt": stx.mean(rts), "rt_sd": stx.stdev(rts) if len(rts) > 1 else 0.0,
            "trials": len(rs), "rss_mb": max(r["maxrss_kb"] for r in rs) / 1024.0,
            "max_disp": rs[0]["max_disp"], "p95": rs[0]["p95_disp"], "n": rs[0]["n"], "m": rs[0]["m"],
        }

    L = ["# GEO-Exact study: SW-level accuracy at DGHL speed", "",
         "Displacement in microns (total L1). Runtime = mean of trials, legalizer call only, "
         "each trial in a fresh process. `vs SW` = (method / SW) - 1; negative is better than SW.", ""]
    L += ["| Split | Design | Pitch | n | SW disp | SW rt (s) | SW-idx rt (s) | DGHL vs SW | DGHL rt (s) "
          "| GEO vs SW | GEO rt (s) | GEO speedup vs DGHL | GEO speedup vs SW-idx | checks |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    agg: dict[str, dict[str, list]] = {}
    for split, design, pitch, _ in CASES:
        sw, swi, dg, ge = (cell(design, pitch, m) for m in METHODS)
        if not (sw and dg and ge):
            L.append(f"| {split} | {design} | {pitch:g} | – | missing data | | | | | | | | | |")
            continue
        checks = []
        if swi and abs(swi["total"] - sw["total"]) > 1e-6:
            checks.append(f"SW-idx≠SW ({swi['total'] - sw['total']:+.2f})")
        for nm, c in (("sw", sw), ("swi", swi), ("dghl", dg), ("geo", ge)):
            if c and not c["injective"]:
                checks.append(f"{nm} NOT injective")
            if c and not c["deterministic"]:
                checks.append(f"{nm} nondeterministic")
        if ge["total"] > sw["total"] + 1e-6:
            checks.append("GEO>SW (should be impossible)")
        if ge["total"] > dg["total"] + 1e-6:
            checks.append("GEO>DGHL (should be impossible)")
        dvs = dg["total"] / sw["total"] - 1
        gvs = ge["total"] / sw["total"] - 1
        sp_d = dg["rt"] / ge["rt"]
        sp_s = (swi["rt"] / ge["rt"]) if swi else float("nan")
        a = agg.setdefault(split, {"dvs": [], "gvs": [], "sp_d": [], "sp_s": [], "sp_sw": []})
        a["dvs"].append(dvs); a["gvs"].append(gvs); a["sp_d"].append(sp_d)
        a["sp_s"].append(sp_s); a["sp_sw"].append(sw["rt"] / ge["rt"])
        L.append(f"| {split} | {design} | {pitch:g} | {sw['n']:,} | {sw['total']:,.2f} | {sw['rt']:.3f} | "
                 f"{(swi['rt'] if swi else float('nan')):.3f} | {100 * dvs:+.3f}% | {dg['rt']:.3f} ± {dg['rt_sd']:.3f} | "
                 f"**{100 * gvs:+.3f}%** | **{ge['rt']:.3f} ± {ge['rt_sd']:.3f}** | {sp_d:.2f}× | {sp_s:.2f}× | "
                 f"{'; '.join(checks) if checks else 'ok'} |")
    L += ["", "## Aggregates (mean of per-case ratios; speedups are geometric means)", "",
          "| Split | cases | DGHL vs SW | GEO vs SW | GEO speedup vs DGHL | GEO speedup vs SW-idx | GEO speedup vs SW |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for split in ["Train", "Test A", "Scalability", "Test B"]:
        a = agg.get(split)
        if not a:
            continue
        L.append(f"| {split} | {len(a['dvs'])} | {100 * stx.mean(a['dvs']):+.3f}% | {100 * stx.mean(a['gvs']):+.3f}% | "
                 f"{geomean(a['sp_d']):.2f}× | {geomean(a['sp_s']):.2f}× | {geomean(a['sp_sw']):.2f}× |")
    L += ["", "## Review 3.2 baseline: k-NN sparse exact (SciPy LAPJVsp), vs SW and GEO", "",
          "`k used` > k means the k-NN graph had no legal full matching and k was doubled globally.", "",
          "| Design | Pitch | " + " | ".join(f"{m} vs SW | {m} rt (s) | {m} k used" for m in KNN)
          + " | GEO vs SW | GEO rt (s) |",
          "|---|---:|" + "---:|---:|---:|" * len(KNN) + "---:|---:|"]
    knn_agg: dict[str, list] = {m: [] for m in KNN}
    for split, design, pitch, _ in CASES:
        sw = cell(design, pitch, "sw")
        ge = cell(design, pitch, "geo")
        if not (sw and ge):
            continue
        parts = []
        for m in KNN:
            c = cell(design, pitch, m)
            rs = by.get((design, pitch, m), [])
            if c:
                vs = c["total"] / sw["total"] - 1
                knn_agg[m].append((vs, c["rt"] / ge["rt"]))
                kused = rs[0].get("stats", {}).get("k_used", "?")
                parts.append(f"{100 * vs:+.3f}% | {c['rt']:.3f} | {kused}")
            else:
                parts.append("failed | – | –")
        L.append(f"| {design} | {pitch:g} | " + " | ".join(parts)
                 + f" | {100 * (ge['total'] / sw['total'] - 1):+.3f}% | {ge['rt']:.3f} |")
    L += ["", "| Method | cases run | mean vs SW | GEO speedup over it (geo. mean) |", "|---|---:|---:|---:|"]
    for m in KNN:
        xs = knn_agg[m]
        if xs:
            L.append(f"| {m} | {len(xs)} | {100 * stx.mean(x[0] for x in xs):+.3f}% | {geomean([x[1] for x in xs]):.2f}× |")
    L += ["", "## Tail displacement (single trial; microns)", "",
          "| Design | Pitch | SW max | SW p95 | DGHL max | DGHL p95 | GEO max | GEO p95 | GEO peak RSS (MB) | DGHL peak RSS (MB) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split, design, pitch, _ in CASES:
        sw, _, dg, ge = (cell(design, pitch, m) for m in METHODS)
        if sw and dg and ge:
            L.append(f"| {design} | {pitch:g} | {sw['max_disp']:.2f} | {sw['p95']:.2f} | {dg['max_disp']:.2f} | "
                     f"{dg['p95']:.2f} | {ge['max_disp']:.2f} | {ge['p95']:.2f} | {ge['rss_mb']:.0f} | {dg['rss_mb']:.0f} |")
    if errors:
        L += ["", "## Errors", ""]
        for e in errors:
            L.append(f"- {e['design']}@{e['pitch']} {e['method']}: {str(e['error']).splitlines()[-1][:200]}")
    REPORT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    if cmd == "run":
        run(only)
        summary()
    elif cmd == "summary":
        summary()
    else:
        raise SystemExit(__doc__)
