"""
verify_results.py -- CI gate: assert that a fresh run of run_experiments.py
reproduces the published reference results.

The simulation is fully seed-deterministic (seed 20261003) and was verified
bit-exact on numpy 2.1.3 / scipy 1.14.1. Cross-platform BLAS differences can
in principle perturb floating-point sums, so we assert with a 1e-6 absolute
tolerance rather than bit equality. Any drift beyond that is a real finding:
either a dependency broke behavioral compatibility, or the model changed.

Reference values below are the ones reported in the white paper.
Exit code 0 = reproduced; 1 = drift detected.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "results", "summary.json")

TOL = 1e-6

REFERENCE = {
    "seed": 20261003,
    "e1_welfare": {
        "DD": 0.824775,
        "T2": 0.9885833333333334,
        "T3": 0.9960833333333333,
        "SORT": 0.909375,
    },
    "e1_accuracy_domain_C": {
        "DD": 0.536375,
        "T2": 0.954375,
        "T3": 0.9922499999999999,
        "SORT": 0.7154166666666667,
    },
    "e2_welfare_T3_at_v0.7": 0.9958541666666666,
    "verdict_rule_prefer_T3": True,
    "verdict_welfare_advantage_T3_minus_T2_at_v0.3": 0.0153,
}


def close(a, b, what):
    if abs(a - b) > TOL:
        print(f"FAIL  {what}: got {a!r}, expected {b!r} (|drift| > {TOL})")
        return False
    print(f"  ok  {what}: {a:.6f}")
    return True


def main():
    with open(SUMMARY) as f:
        s = json.load(f)

    ok = True

    if s.get("seed") != REFERENCE["seed"]:
        print(f"FAIL  seed: got {s.get('seed')!r}, expected {REFERENCE['seed']!r}")
        ok = False
    else:
        print("  ok  seed: 20261003")

    for sysname, expected in REFERENCE["e1_welfare"].items():
        ok = close(s["e1_accuracy"][sysname]["welfare"], expected,
                   f"E1 welfare [{sysname}]") and ok

    for sysname, expected in REFERENCE["e1_accuracy_domain_C"].items():
        ok = close(s["e1_accuracy"][sysname]["C"], expected,
                   f"E1 domain-C accuracy [{sysname}]") and ok

    ok = close(s["e2_welfare"]["T3"]["0.7"],
               REFERENCE["e2_welfare_T3_at_v0.7"],
               "E2 welfare T3 @ v=0.7") and ok

    ok = close(s["verdict"]["welfare_advantage_T3_minus_T2"]["0.3"],
               REFERENCE["verdict_welfare_advantage_T3_minus_T2_at_v0.3"],
               "E2 welfare advantage T3-T2 @ v=0.3") and ok

    if s["verdict"]["rule_prefer_T3"] is not REFERENCE["verdict_rule_prefer_T3"]:
        print("FAIL  pre-registered verdict flipped: rule_prefer_T3 is False")
        ok = False
    else:
        print("  ok  pre-registered verdict holds: rule_prefer_T3 = True")

    print()
    if ok:
        print("REPRODUCED: all reference results match within 1e-6.")
        return 0
    print("DRIFT DETECTED: fresh run does not match published results.")
    print("If the model intentionally changed, update REFERENCE here and in "
          "the paper, and note it in the release changelog.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
