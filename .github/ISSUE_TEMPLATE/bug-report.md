---
name: Simulation bug report
about: Something in the code doesn't do what it should
title: "[Bug] "
labels: bug
---

**Observed behavior:**

**Expected behavior (what the paper/docs claim):**

**Environment:**

- Python version:
- numpy / scipy / matplotlib versions (`pip list | grep -E "numpy|scipy|matplotlib"`):
- OS:

**Steps to reproduce (exact commands):**

```bash
python sim/run_experiments.py
```

**Output / traceback:**

```
...
```

**Note on determinism:** the model is seed-fixed (20261003). If
`sim/verify_results.py` reports drift on your machine, include the drift
report. Cross-platform numerical drift is a finding, not a nuisance, and we
track it.
