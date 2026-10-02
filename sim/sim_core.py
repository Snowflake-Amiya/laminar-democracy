"""
sim_core.py -- Agent-based simulation core for Tiered Direct Democracy (TDD)
with adaptive epistemic friction.

Model summary
-------------
Population of N agents. Each agent i has:
  theta_i      : latent civic competence ~ N(0,1)
  eng_i        : turnout / signing propensity, positively correlated with theta
  score_i      : certification-test score = sqrt(v)*theta_i + sqrt(1-v)*eta_i
                 where v = test validity (squared correlation with true ability)

Issues belong to three domains:
  A (basic civic)     : decided by everyone in all systems
  B (intermediate)    : decided by everyone (DD) / tier-2 electorate (T3) / status pool (T2)
  C (highly technical): decided by everyone (DD) / tier-3 electorate (T3) / status pool (T2)

Vote correctness uses a latent-signal model:
  signal_ij = shift_d + theta_i + sigma_pop_d*u_j + sigma_group*w_j + sigma_id_d*eps_ij
  agent votes correctly iff signal_ij > 0.
  u_j  : population-wide correlated error ("misinformation wave"), shared by all voters
  w_j  : within-group correlated error ("groupthink"), active only for selected electorates
  sigma_group = beta * (1 - pool_share)   [Hong-Page style selection penalty]

Referendums: tier-specific signature thresholds collected in a fixed time window.

Systems compared:
  DD   : pure direct democracy (everyone votes on everything)
  T2   : two-tier bifurcation (single status pool votes on B and C)
  T3   : three-tier bifurcation (tier-2 votes on B, tier-3 votes on C)
  SORT : standing citizens' assembly (random sample of 500) decides everything
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass

DOMAINS = ("A", "B", "C")

# ----------------------------------------------------------------------
# Small math helpers
# ----------------------------------------------------------------------
from scipy.stats import norm as _norm

_ERF = np.vectorize(math.erf)


def phi(x):
    """Standard normal CDF, elementwise."""
    return _norm.cdf(x)


def norm_ppf(p):
    """Standard normal inverse CDF."""
    return float(_norm.ppf(p))


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
@dataclass
class Params:
    n: int = 2000                    # population size
    # mean individual correctness of the median citizen, by domain
    target_acc: tuple = (0.60, 0.54, 0.47)
    # population-level correlated-error strength ("misinformation waves")
    sigma_pop: tuple = (0.15, 0.40, 0.50)
    # agenda mix: share of issues in each domain
    issue_mix: tuple = (0.45, 0.35, 0.20)
    # engagement (turnout propensity)
    engagement_mean: float = 0.55
    engagement_sd: float = 0.32
    engagement_theta_corr: float = 0.30
    # certification test
    validity: float = 0.70           # v in [0,1]
    q2: float = 0.30                 # tier-2 pool share (top of score dist)
    q3: float = 0.10                 # tier-3 pool share
    # elite groupthink strength
    beta_groupthink: float = 0.30
    # assembly (SORT) size, deliberation correlation and briefing uplift
    assembly_size: int = 500
    assembly_sigma: float = 0.08
    assembly_uplift: float = 0.40    # competence gain from briefing (Fishkin)
    # issues per replication
    m_issues: int = 200


# ----------------------------------------------------------------------
# Population and tiers
# ----------------------------------------------------------------------
def make_population(rng: np.random.Generator, p: Params, validity: float):
    """Sample a population; return theta, engagement, and test scores."""
    theta = rng.normal(0.0, 1.0, p.n)
    rho = p.engagement_theta_corr
    z = rng.normal(0.0, 1.0, p.n)
    eng = np.clip(p.engagement_mean + p.engagement_sd * (rho * theta +
                math.sqrt(1 - rho * rho) * z), 0.05, 0.98)
    eta = rng.normal(0.0, 1.0, p.n)
    score = math.sqrt(validity) * theta + math.sqrt(1.0 - validity) * eta
    return theta, eng, score


def tier_masks(score: np.ndarray, p: Params):
    """Top-q2 / top-q3 masks from test scores."""
    c2 = np.quantile(score, 1.0 - p.q2)
    c3 = np.quantile(score, 1.0 - p.q3)
    return score >= c2, score >= c3


# ----------------------------------------------------------------------
# Issue generation (paired across systems within a replication)
# ----------------------------------------------------------------------
def make_issue_shocks(rng: np.random.Generator, p: Params):
    """Per-domain issue counts and shared shocks (u = pop shock, w = group shock)."""
    counts = {}
    remain = p.m_issues
    for d, share in zip(DOMAINS, p.issue_mix):
        counts[d] = max(1, int(round(share * p.m_issues)))
    u = {}
    w = {}
    for idx, d in enumerate(DOMAINS):
        m = counts[d]
        u[d] = p.sigma_pop[idx] * rng.normal(0.0, 1.0, m)
        w[d] = rng.normal(0.0, 1.0, m)
    eps = {d: rng.normal(0.0, 1.0, (p.n, counts[d])) for d in DOMAINS}
    shifts = tuple(math.sqrt(2.0) * norm_ppf(t) for t in p.target_acc)
    return counts, shifts, u, w, eps


# ----------------------------------------------------------------------
# Voting on one domain for one electorate
# ----------------------------------------------------------------------
def decide_domain_sigma(theta, eng, electorate, shift_arr, u_arr, w_arr, eps,
                        sigma_pop, sigma_group, rng, turnout=True):
    n = theta.shape[0]
    m = shift_arr.shape[0]
    sigma_id = math.sqrt(max(0.0, 1.0 - sigma_pop ** 2))
    sig = (shift_arr[None, :] + theta[:, None] + u_arr[None, :]
           + sigma_group * w_arr[None, :] + sigma_id * eps)
    correct = sig > 0.0
    if turnout:
        to = rng.random((n, m)) < eng[:, None]
    else:
        to = np.ones((n, m), dtype=bool)
    voting = to & electorate[:, None]
    n_voters = voting.sum(axis=0)
    n_correct = (correct & voting).sum(axis=0)
    dec = np.where(n_correct > n_voters / 2.0, 1.0,
                   np.where(n_correct == n_voters / 2.0, 0.5, 0.0))
    dec = np.where(n_voters == 0, 0.5, dec)
    return float(dec.mean()), float(n_voters.mean())


# ----------------------------------------------------------------------
# System evaluation for one replication (paired shocks)
# ----------------------------------------------------------------------
def evaluate_systems(p: Params, rng: np.random.Generator, validity: float,
                     systems=("DD", "T2", "T3", "SORT"),
                     q2=None, q3=None, beta=None, with_assembly=True):
    """Run one replication; return {system: {domain: (acc, voters)}}.

    All systems share the same population, issues, and shock draws (paired).
    """
    q2 = p.q2 if q2 is None else q2
    q3 = p.q3 if q3 is None else q3
    beta = p.beta_groupthink if beta is None else beta

    theta, eng, score = make_population(rng, p, validity)
    t2, t3 = tier_masks(score, p if (q2 is None and q3 is None) else
                        _patched(p, q2, q3))
    assembly = np.zeros(p.n, dtype=bool)
    assembly[rng.choice(p.n, size=p.assembly_size, replace=False)] = True

    counts, shifts, u, w, eps = make_issue_shocks(rng, p)
    everyone = np.ones(p.n, dtype=bool)

    out = {}
    for sysname in systems:
        dom_acc = {}
        for idx, d in enumerate(DOMAINS):
            m = counts[d]
            if sysname == "DD":
                elect, s_group = everyone, 0.0
            elif sysname == "T2":
                elect = everyone if d == "A" else t2
                s_group = 0.0 if d == "A" else beta * (1.0 - q2)
            elif sysname == "T3":
                if d == "A":
                    elect, s_group = everyone, 0.0
                elif d == "B":
                    elect, s_group = t2, beta * (1.0 - q2)
                else:
                    elect, s_group = t3, beta * (1.0 - q3)
            elif sysname == "SORT":
                if with_assembly:
                    elect = assembly
                    s_group = p.assembly_sigma
                    theta_use = theta + p.assembly_uplift * assembly.astype(float)
                    eng_use = np.where(assembly, 0.92, eng)
                else:
                    elect, s_group = everyone, 0.0
                    theta_use, eng_use = theta, eng
                acc, voters = decide_domain_sigma(
                    theta_use, eng_use, elect,
                    np.full(m, shifts[idx]), u[d], w[d], eps[d],
                    p.sigma_pop[idx], s_group, rng)
                dom_acc[d] = (acc, voters)
                continue
            else:
                raise ValueError(sysname)
            acc, voters = decide_domain_sigma(
                theta, eng, elect, np.full(m, shifts[idx]), u[d], w[d], eps[d],
                p.sigma_pop[idx], s_group, rng)
            dom_acc[d] = (acc, voters)
        out[sysname] = dom_acc
    return out


def _patched(p: Params, q2, q3):
    import copy
    p2 = copy.copy(p)
    p2.q2, p2.q3 = q2, q3
    return p2


# ----------------------------------------------------------------------
# E4: Referendum signature collection (Poisson first-passage)
# ----------------------------------------------------------------------
def referendum_first_passage(rng, pool_size, pool_eng, support, threshold,
                             sign_rate, window_days):
    """Simulate one initiative: returns (success, days_to_threshold).

    Signatures arrive as a homogeneous Poisson process with rate
    pool_size * support * pool_eng * sign_rate per day.
    Time to the k-th signature ~ Gamma(k, 1/rate).
    """
    rate = pool_size * support * pool_eng * sign_rate
    if rate <= 0 or threshold <= 0:
        return False, float(window_days)
    days = rng.gamma(shape=max(1, int(threshold)), scale=1.0 / rate)
    return bool(days <= window_days), float(days)


def run_referendum_experiment(p, rng, x=0.05, y=2.0, z=2.0,
                              support=0.15, sign_rate=0.010,
                              window_days=180, n_init=4000):
    """Tier-specific signature thresholds; returns per-path metrics.

    Paths:
      basic : pool = all citizens,   threshold = x*N
      tier2 : pool = tier-2 members, threshold = x*N / y
      tier3 : pool = tier-3 members, threshold = x*N / z
    """
    theta, eng, score = make_population(rng, p, p.validity)
    t2, t3 = tier_masks(score, p)
    pools = {
        "basic (all citizens)": (p.n, float(eng.mean())),
        "tier-2 members": (int(t2.sum()), float(eng[t2].mean())),
        "tier-3 members": (int(t3.sum()), float(eng[t3].mean())),
    }
    thresholds = {
        "basic (all citizens)": x * p.n,
        "tier-2 members": x * p.n / y,
        "tier-3 members": x * p.n / z,
    }
    rows = []
    for path, (pool, peng) in pools.items():
        thr = thresholds[path]
        succ = np.empty(n_init)
        days = np.empty(n_init)
        for i in range(n_init):
            s, d = referendum_first_passage(rng, pool, peng, support, thr,
                                            sign_rate, window_days)
            succ[i] = s
            days[i] = min(d, window_days)
        rows.append({
            "path": path,
            "pool_size": pool,
            "pool_engagement": round(peng, 3),
            "threshold_signatures": round(thr, 1),
            "threshold_poolshare": round(thr / pool, 3),
            "success_rate": float(succ.mean()),
            "median_days": float(np.median(days)),
        })
    return rows


# ----------------------------------------------------------------------
# E5: Capture resistance
# ----------------------------------------------------------------------
def _capture_single_pop(rng, p, delta_mis, n_reps, electorate, bribe_share,
                        veto, veto_threshold, validity):
    """One population draw: P(bad measure passes) for an attack routed
    through a given electorate. See capture_prob_basic for semantics."""
    n = p.n
    theta = rng.normal(0, 1, n)
    rho = p.engagement_theta_corr
    z = rng.normal(0, 1, n)
    eng = np.clip(p.engagement_mean + p.engagement_sd *
                  (rho * theta + math.sqrt(1 - rho * rho) * z), 0.05, 0.98)
    eta = rng.normal(0, 1, n)
    score = math.sqrt(validity) * theta + math.sqrt(1 - validity) * eta

    if electorate == "all":
        elect = np.ones(n, dtype=bool)
        s_group = 0.0
    elif electorate == "t2":
        elect = score >= np.quantile(score, 1 - p.q2)
        s_group = p.beta_groupthink * (1 - p.q2)
    elif electorate == "t3":
        elect = score >= np.quantile(score, 1 - p.q3)
        s_group = p.beta_groupthink * (1 - p.q3)
    else:
        raise ValueError(electorate)

    # signals on the (technical, domain-C-like) bad measure
    shift_c = math.sqrt(2.0) * norm_ppf(p.target_acc[2])
    u = p.sigma_pop[2] * rng.normal(0, 1, (1, n_reps))          # shared
    eps = math.sqrt(1 - p.sigma_pop[2] ** 2) * rng.normal(0, 1, (n, n_reps))
    sig = shift_c + theta[:, None] + u + eps
    if delta_mis > 0:
        sig = sig - delta_mis * (1.0 - phi(theta))[:, None]
    if s_group > 0:
        sig = sig + s_group * rng.normal(0, 1, (1, n_reps))     # shared
    correct = sig > 0.0

    to = rng.random((n, n_reps)) < np.where(elect, eng, 0.0)[:, None]
    voting = to & elect[:, None]
    n_voters = voting.sum(axis=0)
    n_correct = (correct & voting).sum(axis=0)

    if bribe_share > 0:
        # bribed members vote wrong: remove their correct votes
        bribed = elect & (rng.random(n) < bribe_share)
        n_bribed_votes = (to & bribed[:, None]).sum(axis=0)
        n_correct = np.maximum(0, n_correct - n_bribed_votes)

    passed = (n_correct * 2.0 < n_voters) & (n_voters > 0)

    if not veto:
        return float(passed.mean())

    # veto referendum: general electorate judges "overturn the bad law"
    # judgment is domain-A-like (easier), but the attacker spends the same
    # misinformation against overturning.
    shift_a = math.sqrt(2.0) * norm_ppf(p.target_acc[0])
    vu = p.sigma_pop[0] * rng.normal(0, 1, (1, n_reps))         # shared
    veps = math.sqrt(1 - p.sigma_pop[0] ** 2) * rng.normal(0, 1, (n, n_reps))
    vsig = shift_a + theta[:, None] + vu + veps
    if delta_mis > 0:
        vsig = vsig - delta_mis * (1.0 - phi(theta))[:, None]
    vcorrect = vsig > 0.0
    vto = rng.random((n, n_reps)) < eng[:, None]
    v_voters = vto.sum(axis=0)
    v_correct = (vcorrect & vto).sum(axis=0)
    upheld = v_correct / np.maximum(v_voters, 1) > veto_threshold
    # capture survives only if measure passed AND veto failed
    return float((passed & ~upheld).mean())


def capture_prob_basic(rng, p, delta_mis, n_pops=25, n_reps_per_pop=200,
                       electorate="all", bribe_share=0.0, veto=False,
                       veto_threshold=0.55, validity=None):
    """P(bad measure passes), marginalized over population draws.

    electorate: "all" (basic referendum), "t2" or "t3" (tier path).

    General-electorate attack: misinformation shifts each voter's signal down
    by delta_mis * (1 - Phi(theta_i))  (low-competence voters hit hardest).
    Tier attack: a bribed share of the tier votes wrong deterministically.
    Optional democratic veto: general supermajority can overturn.
    Shared per-ballot shocks model correlated errors (misinformation waves,
    groupthink); idiosyncratic noise is per-voter. Averaging over n_pops
    population draws matters because under strong misinformation the veto
    share sits near the threshold, so single-draw estimates are unstable.
    """
    validity = p.validity if validity is None else validity
    probs = [_capture_single_pop(rng, p, delta_mis, n_reps_per_pop,
                                  electorate, bribe_share, veto,
                                  veto_threshold, validity)
             for _ in range(n_pops)]
    return float(np.mean(probs))


# ----------------------------------------------------------------------
# E6: Status-acquisition incentive dynamics
# ----------------------------------------------------------------------
def incentive_dynamics(p, rng, periods=12, validity=0.70,
                       status_value=1.0, learn_gain=0.50, learn_decay=0.12,
                       learn_cap=0.90, cost=(0.0, 0.15, 0.40),
                       logit_temp=0.25, criterion_referenced=True):
    """Track competence growth and status-pool expansion over periods.

    Citizens choose study effort e in {0, 0.5, 1} each period.
    EU(e) = status_value * Phi(sqrt(v)*(theta_eff) + learn_gain*e - cutoff)
            - cost(e)   + logit noise.
    Cumulative study raises effective ability up to learn_cap.
    Cutoffs are criterion-referenced (fixed at initial percentile), so the
    status pool can grow as the population invests -- the system is not
    zero-sum.
    """
    n = p.n
    theta = rng.normal(0, 1, n)
    cum_effort = np.zeros(n)
    theta_eff = theta.copy()
    cutoff2 = norm_ppf(1 - p.q2)
    cutoff3 = norm_ppf(1 - p.q3)
    efforts = np.array([0.0, 0.5, 1.0])
    costs = np.array(cost)

    share2, share3 = [], []
    compB, compC = [], []
    shiftB = math.sqrt(2.0) * norm_ppf(p.target_acc[1])
    shiftC = math.sqrt(2.0) * norm_ppf(p.target_acc[2])

    for t in range(periods):
        score = math.sqrt(validity) * theta_eff + \
            math.sqrt(1 - validity) * rng.normal(0, 1, n)
        t2 = score >= cutoff2
        t3 = score >= cutoff3
        share2.append(float(t2.mean()))
        share3.append(float(t3.mean()))
        compB.append(float(phi(shiftB + theta_eff).mean()))
        compC.append(float(phi(shiftC + theta_eff).mean()))

        # effort choice for next period (softmax over EU)
        eu = np.empty((n, 3))
        for k, e in enumerate(efforts):
            pass_score = phi(np.sqrt(validity) * theta_eff +
                             learn_gain * e - cutoff2)
            eu[:, k] = status_value * pass_score - costs[k]
        noise = rng.logistic(0, logit_temp, (n, 3))
        choice = np.argmax(eu + noise, axis=1)
        cum_effort += efforts[choice]
        theta_eff = theta + np.minimum(learn_cap, learn_decay * cum_effort)

    return {
        "periods": list(range(periods)),
        "share_t2": share2, "share_t3": share3,
        "mean_competence_B": compB, "mean_competence_C": compC,
    }
