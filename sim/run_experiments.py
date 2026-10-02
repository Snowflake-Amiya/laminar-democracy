"""
run_experiments.py -- Runs experiments E1-E6 for the TDD white paper.

E1: Decision quality by system x domain (DD / T2 / T3 / SORT), v = 0.7
E2: Welfare vs test validity v in {0.1, 0.3, 0.5, 0.7, 0.9}
E3: Tier-3 pool size sweep (Hong-Page selection penalty) + T2 cutoff steel-man
E4: Referendum signature thresholds (tier-specific agenda access)
E5: Capture resistance (misinformation vs expert-bribery, with/without veto)
E6: Status-acquisition incentive dynamics (competence growth)

Outputs CSVs to results/ and a machine-readable summary.json with the
2-tier vs 3-tier verdict.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_core import (Params, evaluate_systems, run_referendum_experiment,
                      capture_prob_basic, incentive_dynamics, phi, norm_ppf,
                      tier_masks, make_population)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

SEED = 20261003
SYSTEMS = ("DD", "T2", "T3", "SORT")
MIX = (0.45, 0.35, 0.20)


def welf(acc):
    return sum(MIX[i] * acc[i] for i in range(3))


def csv_write(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"  wrote {os.path.basename(path)} ({len(rows)} rows)")


# ----------------------------------------------------------------------
def e1(p, rng, reps=300):
    print("E1: decision quality by system x domain (v=0.70) ...")
    acc = {s: {d: [] for d in "ABC"} for s in SYSTEMS}
    vot = {s: {d: [] for d in "ABC"} for s in SYSTEMS}
    for r in range(reps):
        out = evaluate_systems(p, rng, p.validity)
        for s in SYSTEMS:
            for d in "ABC":
                a, v = out[s][d]
                acc[s][d].append(a)
                vot[s][d].append(v)
    rows = []
    for s in SYSTEMS:
        for d in "ABC":
            a = np.array(acc[s][d])
            rows.append([s, d, round(a.mean(), 4), round(a.std(ddof=1) /
                       np.sqrt(len(a)), 4), round(np.mean(vot[s][d]), 1)])
    csv_write(os.path.join(RES, "e1_accuracy.csv"),
              ["system", "domain", "accuracy", "se", "mean_voters"], rows)
    summary = {s: {d: float(np.mean(acc[s][d])) for d in "ABC"}
               for s in SYSTEMS}
    for s in SYSTEMS:
        summary[s]["welfare"] = welf([summary[s]["A"], summary[s]["B"],
                                      summary[s]["C"]])
    return summary


# ----------------------------------------------------------------------
def e2(p, rng, reps=240):
    print("E2: welfare vs test validity sweep ...")
    vs = [0.1, 0.3, 0.5, 0.7, 0.9]
    rows = []
    curves = {s: {v: [] for v in vs} for s in SYSTEMS}
    curvesC = {s: {v: [] for v in vs} for s in SYSTEMS}
    for v in vs:
        for r in range(reps):
            out = evaluate_systems(p, rng, v)
            for s in SYSTEMS:
                curves[s][v].append(welf([out[s]["A"][0], out[s]["B"][0],
                                          out[s]["C"][0]]))
                curvesC[s][v].append(out[s]["C"][0])
        for s in SYSTEMS:
            rows.append([v, s,
                         round(np.mean(curves[s][v]), 4),
                         round(np.std(curves[s][v], ddof=1) /
                               np.sqrt(reps), 4),
                         round(np.mean(curvesC[s][v]), 4)])
    csv_write(os.path.join(RES, "e2_welfare.csv"),
              ["validity", "system", "welfare", "se", "acc_C"], rows)
    return {s: {str(v): float(np.mean(curves[s][v])) for v in vs}
            for s in SYSTEMS}, \
           {s: {str(v): float(np.mean(curvesC[s][v])) for v in vs}
            for s in SYSTEMS}


# ----------------------------------------------------------------------
def e3(p, rng, reps=240):
    print("E3: tier-3 pool size sweep + two-tier cutoff steel-man ...")
    q3s = [0.02, 0.05, 0.10, 0.20, 0.30]
    vs = [0.3, 0.7]
    betas = [0.0, 0.30, 0.60, 1.00]
    rows = []
    resC = {}
    for v in vs:
        for beta in betas:
            for q3 in q3s:
                accs = []
                for r in range(reps):
                    out = evaluate_systems(p, rng, v, systems=("T3",),
                                           q3=q3, beta=beta)
                    accs.append(out["T3"]["C"][0])
                resC[(v, beta, q3)] = float(np.mean(accs))
                rows.append(["T3", v, beta, q3,
                             round(resC[(v, beta, q3)], 4), ""])
    # steel-man the 2-tier design: sweep its single status cutoff,
    # recording BOTH B and C accuracy (a single pool faces a trade-off)
    q2s = [0.10, 0.20, 0.30]
    for v in vs:
        for beta in [0.0, 0.30]:
            for q2 in q2s:
                accsB, accsC = [], []
                for r in range(reps):
                    out = evaluate_systems(p, rng, v, systems=("T2",),
                                           q2=q2, beta=beta)
                    accsB.append(out["T2"]["B"][0])
                    accsC.append(out["T2"]["C"][0])
                resC[("T2", v, beta, q2)] = float(np.mean(accsC))
                resC[("T2B", v, beta, q2)] = float(np.mean(accsB))
                rows.append(["T2", v, beta, q2,
                             round(resC[("T2", v, beta, q2)], 4),
                             round(resC[("T2B", v, beta, q2)], 4)])
    csv_write(os.path.join(RES, "e3_poolsize.csv"),
              ["system", "validity", "beta", "cutoff", "acc_C", "acc_B"],
              rows)
    return resC


# ----------------------------------------------------------------------
def e4(p, rng):
    print("E4: referendum signature thresholds ...")
    rows = []
    for (y, z, tag) in [(2.0, 2.0, "raw y=2 z=2"),
                        (2.0, 5.0, "adjusted y=2 z=5")]:
        for r in run_referendum_experiment(p, rng, y=y, z=z):
            r2 = dict(r)
            r2["parameterization"] = tag
            rows.append(r2)
    csv_write(os.path.join(RES, "e4_referendum.csv"),
              ["parameterization", "path", "pool_size", "pool_engagement",
               "threshold_signatures", "threshold_poolshare",
               "success_rate", "median_days"],
              [[r["parameterization"], r["path"], r["pool_size"],
                r["pool_engagement"], r["threshold_signatures"],
                r["threshold_poolshare"], round(r["success_rate"], 3),
                round(r["median_days"], 1)] for r in rows])
    return rows


# ----------------------------------------------------------------------
def e5(p, rng, n_pops=25, n_reps_per_pop=200):
    print("E5: capture resistance ...")
    scenarios = []
    # (label, electorate, delta_mis, bribe_share)
    for dname, delta in [("weak", 0.3), ("strong", 0.8)]:
        scenarios.append((f"basic referendum - {dname} misinformation",
                          "all", delta, 0.0))
        scenarios.append((f"tier-2 path - {dname} misinformation",
                          "t2", delta, 0.0))
        scenarios.append((f"tier-3 path - {dname} misinformation",
                          "t3", delta, 0.0))
    # expert-bribery attacks: buy 5% / 15% of the tier-3 pool
    scenarios.append(("tier-3 path - bribe 5% of pool", "t3", 0.0, 0.05))
    scenarios.append(("tier-3 path - bribe 15% of pool", "t3", 0.0, 0.15))
    scenarios.append(("tier-2 path - bribe 15% of pool", "t2", 0.0, 0.15))

    rows = []
    for label, elect, delta, bribe in scenarios:
        p_cap = capture_prob_basic(rng, p, delta, n_pops=n_pops,
                                   n_reps_per_pop=n_reps_per_pop,
                                   electorate=elect, bribe_share=bribe,
                                   veto=False)
        p_cap_v = capture_prob_basic(rng, p, delta, n_pops=n_pops,
                                     n_reps_per_pop=n_reps_per_pop,
                                     electorate=elect, bribe_share=bribe,
                                     veto=True)
        rows.append([label, elect, delta, bribe,
                     round(p_cap, 4), round(p_cap_v, 4)])
    csv_write(os.path.join(RES, "e5_capture.csv"),
              ["scenario", "electorate", "delta_mis", "bribe_share",
               "p_capture", "p_capture_with_veto"], rows)
    return rows


# ----------------------------------------------------------------------
def e6(p, rng, periods=12):
    print("E6: status-acquisition incentive dynamics ...")
    tdd = incentive_dynamics(p, rng, periods=periods)
    # DD baseline: no status incentive -> competence static
    shiftB = np.sqrt(2.0) * norm_ppf(p.target_acc[1])
    shiftC = np.sqrt(2.0) * norm_ppf(p.target_acc[2])
    theta = rng.normal(0, 1, p.n)
    compB0 = float(phi(shiftB + theta).mean())
    compC0 = float(phi(shiftC + theta).mean())
    rows = []
    for i, t in enumerate(tdd["periods"]):
        rows.append([t, "TDD", round(tdd["share_t2"][i], 4),
                     round(tdd["share_t3"][i], 4),
                     round(tdd["mean_competence_B"][i], 4),
                     round(tdd["mean_competence_C"][i], 4)])
        rows.append([t, "DD (no incentive)", round(p.q2, 4),
                     round(p.q3, 4), round(compB0, 4), round(compC0, 4)])
    csv_write(os.path.join(RES, "e6_dynamics.csv"),
              ["period", "system", "share_t2", "share_t3",
               "mean_competence_B", "mean_competence_C"], rows)
    return tdd, compB0, compC0


# ----------------------------------------------------------------------
def verdict(e1s, e2w, e2c, e3c):
    """Compare T3 vs T2 per the pre-registered rule: prefer T3 only if it is
    not dominated by T2 across the plausible validity range (v >= 0.5)."""
    adv = {}
    for v in ["0.3", "0.5", "0.7", "0.9"]:
        adv[v] = e2w["T3"][v] - e2w["T2"][v]
    cadv = {v: e2c["T3"][v] - e2c["T2"][v] for v in adv}
    best_t2 = {}
    for v in [0.3, 0.7]:
        for beta in [0.0, 0.30]:
            best_t2[(v, beta)] = max(e3c.get(("T2", v, beta, q), -1)
                                     for q in [0.10, 0.20, 0.30])
    notes = {
        "welfare_advantage_T3_minus_T2": {k: round(x, 4) for k, x in adv.items()},
        "C_domain_advantage_T3_minus_T2": {k: round(x, 4)
                                           for k, x in cadv.items()},
        "best_steelmanned_T2_C_accuracy": {f"v={v},beta={b}": round(x, 4)
                                           for (v, b), x in best_t2.items()},
        "T3_C_accuracy_v07_beta03_q010": round(e3c.get((0.7, 0.30, 0.10), -1), 4),
        "T3_C_accuracy_v07_beta10_q010": round(e3c.get((0.7, 1.00, 0.10), -1), 4),
        "T3_C_accuracy_v03_beta03_q010": round(e3c.get((0.3, 0.30, 0.10), -1), 4),
    }
    ok = all(adv[v] > 0 for v in ["0.5", "0.7", "0.9"])
    notes["rule_prefer_T3"] = bool(ok)
    return notes


def main():
    t0 = time.time()
    p = Params()
    rng = np.random.default_rng(SEED)

    e1s = e1(p, rng)
    e2w, e2c = e2(p, rng)
    e3c = e3(p, rng)
    e4r = e4(p, rng)
    e5r = e5(p, rng)
    e6r, compB0, compC0 = e6(p, rng)

    v = verdict(e1s, e2w, e2c, e3c)

    summary = {
        "seed": SEED,
        "params": {
            "n": p.n, "target_acc": p.target_acc, "sigma_pop": p.sigma_pop,
            "issue_mix": p.issue_mix, "validity": p.validity,
            "q2": p.q2, "q3": p.q3, "beta_groupthink": p.beta_groupthink,
            "assembly_size": p.assembly_size,
        },
        "e1_accuracy": e1s,
        "e2_welfare": e2w,
        "e2_accuracy_C": e2c,
        "verdict": v,
    }
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {time.time() - t0:.1f}s -> {RES}")

    print("\n=== Headline numbers ===")
    for s in SYSTEMS:
        a = e1s[s]
        print(f"{s:5s}  A={a['A']:.3f}  B={a['B']:.3f}  C={a['C']:.3f}  "
              f"welfare={a['welfare']:.3f}")
    print("\n=== Verdict (T3 vs T2) ===")
    print(json.dumps(v, indent=2))


if __name__ == "__main__":
    main()
