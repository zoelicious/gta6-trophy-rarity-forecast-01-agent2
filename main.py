#!/usr/bin/env python3
"""Hierarchical shrunk Theil-Sen per-holder-count trophy forecaster.

Entry for make.fun gta6-trophy-rarity-forecast-01 (agent 2).

Model (fundamentally different from a closed-form dilution M x alpha model):
work in COUNTS, not rates.  For each trophy i, holders at the target follow a
POWER LAW in the tracked-player count plus incumbent progress,

    C_i(T) = C_i0 * (N(T) / N0) ** beta_i * exp(h_i * H_days),

where beta_i is the trophy's catch-up exponent (how fast its holders grow
when the tracked base grows) and h_i its incumbent-progress rate -- the
residual holder growth after the N-driven part is removed, so the player
flood is never counted twice.  Both are
estimated per trophy with a robust Theil-Sen slope on the case history and
then shrunk hard toward a class prior (hierarchical shrinkage): with a flat
history the data say nothing, so the class prior rules; with a moving
history the data take over.

N(T) itself comes from a stale-page renewal model: Exophase refreshes its
counts in jumps hours apart, so the page goes stale and then jumps.  From
the staleness s (hours since N last changed) and a uniform refresh-interval
prior, we get P(a refresh lands before the target); the refresh branch
catches the tracked count up by a backlog factor J that is large on launch
night (many players bought/played but are not tracked yet) and decays as
days pass since release.  The no-refresh branch keeps N flat (plus damped
continuous growth).  Quantiles are read from the resulting discrete
scenario mixture convolved with per-trophy count noise -- never a point
forecast -- so q05 leans low, q95 leans high and the band is wide, as the
logit pinball geometry demands (q05 too high and q95 too low each cost
19x the opposite error).

An aggregate guard keeps implied trophies-per-player inside [0.80, 1.80].

Stdlib only, deterministic, no network, no wall-clock use.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------- constants

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
LO, HI = 0.01, 99.99
LOGIT_LO = math.log(LO / (100.0 - LO))
LOGIT_HI = math.log(HI / (100.0 - HI))

# Refresh-interval prior (hours). The kit documents the page unchanged
# for ~4.2 h on launch night (23:48Z probe through the 04:00Z read) and
# refreshed by the 04:30Z read, with refreshes "hours apart": the one
# interval ever observed is ~4.2-4.7 h, so the prior is centred on it
# with +/-0.4 h of slack -- deliberately NOT a point estimate, since a
# single observation cannot pin the interval.
REFRESH_A, REFRESH_B = 4.0, 4.8
# Backlog catch-up rate on launch night (per hour) and its decay (days).
# BACKLOG_G0 is set so the backlog factor over the observed ~4.5 h stale
# window reproduces the kit-documented first refresh (x6.06 players):
# ln(6.06) / (4.5 h * exp(-0.125/0.75)) ~= 0.47.
BACKLOG_G0, BACKLOG_TAU_DAYS = 0.47, 0.75
BACKLOG_HOURS_CAP = 6.0
# Continuous log-N growth: prior per day, 50% damping of the fitted slope.
G_PRIOR_DAY, G_DAMP = 0.14, 0.5
G_MIN_DAY, G_MAX_DAY = 0.0, 0.6
# Catch-up exponent floor and play-time scale. beta_class is the
# catch-up exponent once players have had days to play (final horizon);
# on launch night every player is a beginner, so class differences have
# not emerged yet and beta collapses toward BETA_FLOOR, which is the
# launch-night newcomer rate (~6-7% of the incumbent rate, research
# alpha ~ 0.07) expressed as a power-law exponent. k grows with the
# play time available by the target (target minus release date).
BETA_FLOOR, PLAY_TAU_H = 0.15, 36.0
# Theil-Sen shrinkage: pseudo-count of pairs.
SHRINK_K = 8.0
# Predictive spread knobs (calibration, frozen before scoring).
SIGMA_BETA, SIGMA_H_DAY, SIGMA_BASE = 0.10, 0.10, 0.15
SIGMA_LN_J = 0.25
J_GRID = (1.0 / 1.6, 1.0 / 1.2, 1.0, 1.2, 1.6)
# Aggregate guard on trophies per player (safety net only: the observed
# launch-night value is 1.48, the post-refresh value 1.11, so the band
# must admit both; it catches pathological beta blow-ups, nothing else).
R_GUARD_LO, R_GUARD_HI = 0.80, 1.80


# ------------------------------------------------------------------- helpers

def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(p):
    p = clamp(p, LO, HI) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    l = clamp(l, LOGIT_LO, LOGIT_HI)
    return 100.0 / (1.0 + math.exp(-l))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def round4(x):
    """Round to 4 decimals, half away from zero (matches JS Math.round)."""
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(repr(x)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP))


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def theil_sen(xs, ys, lo, hi):
    """Median pairwise slope, clipped. xs/ys equal-length lists."""
    n = len(xs)
    if n < 2:
        return None
    slopes = []
    for a in range(n):
        for b in range(a + 1, n):
            dx = xs[b] - xs[a]
            if abs(dx) < 1e-9:
                continue
            slopes.append((ys[b] - ys[a]) / dx)
    if not slopes:
        return None
    return clamp(median(slopes), lo, hi)


def subsample(items, max_n):
    """Deterministic evenly-spaced subsample keeping first and last."""
    n = len(items)
    if n <= max_n:
        return list(items)
    idx = sorted(set(
        [0, n - 1] + [round(i * (n - 1) / (max_n - 1)) for i in range(max_n)]))
    return [items[i] for i in idx]


# ------------------------------------------------------------- classification

def classify(trophy):
    """Class prior (beta, h_per_day) from name/description/tier keywords.

    beta = catch-up exponent of holders vs tracked players (1 = holders
    scale with the player base, 0 = holders frozen while players flood in).
    h = incumbent daily log-growth of holders once the page is fresh.
    """
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    text = name + " " + desc
    tier = trophy.get("tier") or ""

    # Tutorial: everybody has it from the first mission.
    if "gone fishing" in text or "just like paradise" in name:
        return ("tutorial", 1.0, 0.0)
    # Story chapters.
    if "complete chapter 1" in text:
        return ("story-ch1", 0.55, 0.30)
    if "complete chapter 2" in text:
        return ("story-ch2", 0.35, 0.25)
    if "complete chapter 3" in text:
        return ("story-ch3", 0.25, 0.20)
    if "complete chapter 4" in text:
        return ("story-ch4", 0.20, 0.18)
    if "final mission" in text and ("complete the final" in text
                                    or "gotta know when" in name):
        return ("story-final", 0.20, 0.15)
    # Platinum / 100% / heavy grind: practically unreachable in week one.
    if tier == "platinum" or "100% completion" in text \
            or "250 stunts" in text or "gold medal in all" in text \
            or "drone race at every" in text \
            or "100 locations" in text or "50 challenges" in text \
            or "3 collectible card sets" in text \
            or "meditate at every spot" in text \
            or "8 mission strands" in text:
        return ("grind", 0.03, 0.01)
    # Mission strands / named-mission completions.
    if "complete" in text and "mission" in text:
        return ("mission", 0.15, 0.15)
    # Early / easy pickups found while playing.
    if any(k in text for k in (
            "fishing challenge", "arcade game", "discover 100 vehicles",
            "try each type of activity", "4 star wanted level",
            "rob 3 shops", "combat focus", "unscheduled departure",
            "kinda hot in these rhinos", "jewelry store",
            "armored truck", "cocktail with a view", "bowling alley",
            "melee drive-by", "spend $100,000", "earn $250,000")):
        return ("easy", 0.25, 0.12)
    # Moderate collections / exploration.
    if any(k in text for k in (
            "unique animal", "explore all of leonida", "bank vaults",
            "illegal operation", "vehicle showroom", "garage properties",
            "register 10 vehicles", "20 different races")):
        return ("moderate", 0.10, 0.06)
    # Anything else: vague one-off moments, default to low catch-up.
    return ("other", 0.08, 0.05)


# ------------------------------------------------------------------- forecast

def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_min = float(case.get("horizon_minutes") or 0)
    horizon_h = horizon_min / 60.0
    horizon_d = horizon_min / 1440.0
    at = parse_time(case["at"])
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z") \
        if case.get("game", {}).get("release_date") else at
    days_since_release = max(0.0, (at - release).total_seconds() / 86400.0)
    # Play time available by the target (drives how far class
    # differences in beta have emerged: see BETA_FLOOR).
    target = parse_time(case["target"]) if case.get("target") else at
    t_play_h = max(0.0, (target - release).total_seconds() / 3600.0)
    k_play = 1.0 - math.exp(-t_play_h / PLAY_TAU_H)

    n_tr = len(trophies)
    last = history[-1] if history else None
    if last is None:
        return [{"id": t["id"], "q": [0.01, 0.01, 0.01, 0.01, 0.01]}
                for t in trophies]

    n0 = float(last["players_tracked"])
    rates0 = last["rates"]
    counts0 = []
    for r in rates0:
        counts0.append(round((r or 0.0) / 100.0 * n0) if r is not None else 0)

    # ---- reconstruct counts per read; detect steps in log N (CUSUM-ish:
    # a single-read jump in ln N above 0.15 is a coverage event, not growth)
    reads = []
    for h in history:
        n = float(h["players_tracked"])
        cs = [round((r or 0.0) / 100.0 * n) if r is not None else None
              for r in h["rates"]]
        reads.append((parse_time(h["at"]), n, cs))
    step_idx = []
    for k in range(1, len(reads)):
        if abs(math.log(reads[k][1] / reads[k - 1][1])) > 0.15:
            step_idx.append(k)
    win_start = step_idx[-1] if step_idx else 0
    window = reads[win_start:]

    # ---- staleness: hours since N last changed (or since first read)
    last_change_t = reads[0][0]
    for k in range(1, len(reads)):
        if reads[k][1] != reads[k - 1][1]:
            last_change_t = reads[k][0]
    staleness_h = max(0.0, (at - last_change_t).total_seconds() / 3600.0)

    # ---- continuous log-N growth on the post-step window, damped to prior
    g_day = G_PRIOR_DAY
    if len(window) >= 2:
        t0 = window[0][0]
        xs = [(r[0] - t0).total_seconds() / 86400.0 for r in window]
        ys = [math.log(max(r[1], 1.0)) for r in window]
        n_w = len(xs)
        mx, my = sum(xs) / n_w, sum(ys) / n_w
        den = sum((x - mx) ** 2 for x in xs)
        if den > 1e-12:
            slope = sum((x - mx) * (y - my)
                        for x, y in zip(xs, ys)) / den
            g_day = clamp(G_DAMP * slope + (1.0 - G_DAMP) * G_PRIOR_DAY,
                          G_MIN_DAY, G_MAX_DAY)

    # ---- refresh probability from the uniform interval prior, conditioned
    # on surviving `staleness_h` hours without a refresh; if the page is
    # staler than the prior's whole support, a refresh is overdue (P = 1)
    def interval_cdf(x):
        return clamp((x - REFRESH_A) / (REFRESH_B - REFRESH_A), 0.0, 1.0)

    if staleness_h >= REFRESH_B:
        p_refresh = 1.0
    else:
        p_refresh = (interval_cdf(staleness_h + horizon_h)
                     - interval_cdf(staleness_h)) / max(
                         1e-9, 1.0 - interval_cdf(staleness_h))
    p_refresh = clamp(p_refresh, 0.0, 1.0)

    # ---- backlog catch-up factor J on the refresh branch. The refresh
    # catches the count up only to the refresh instant, which is the
    # interval draw itself: use its conditional mean given it falls in
    # the window (not the target time -- the page freezes at the refresh).
    lo_i = max(staleness_h, REFRESH_A)
    hi_i = min(staleness_h + horizon_h, REFRESH_B)
    if hi_i <= lo_i:
        t_refresh = staleness_h + 0.5 * horizon_h
    else:
        t_refresh = 0.5 * (lo_i + hi_i)
    g_backlog = BACKLOG_G0 * math.exp(-days_since_release / BACKLOG_TAU_DAYS)
    t_backlog = min(t_refresh, BACKLOG_HOURS_CAP)
    j_med = clamp(math.exp(g_backlog * t_backlog), 1.0, 8.0)

    # ---- N(T) scenario grid: (multiplier M, weight)
    m_growth = math.exp(g_day * horizon_d)
    scenarios = [(m_growth, 1.0 - p_refresh)]
    # lognormal weights over the J grid (mu = ln j_med, sigma = SIGMA_LN_J)
    j_w = []
    for f in J_GRID:
        z = math.log(f) / SIGMA_LN_J
        j_w.append(math.exp(-0.5 * z * z))
    j_tot = sum(j_w)
    for f, w in zip(J_GRID, j_w):
        scenarios.append((m_growth * j_med * f, p_refresh * w / j_tot))
    # drop zero-weight scenarios, renormalise
    scenarios = [(m, w) for m, w in scenarios if w > 1e-9]
    w_tot = sum(w for _, w in scenarios)
    scenarios = [(m, w / w_tot) for m, w in scenarios]

    # ---- per-trophy beta / h: Theil-Sen on subsampled reads, shrunk to
    # the class prior (flat history -> no signal -> prior rules)
    sub = subsample(reads, 48)
    log_n = [math.log(max(r[1], 1.0)) for r in sub]
    t_days = [(r[0] - sub[0][0]).total_seconds() / 86400.0 for r in sub]
    # pairs with real N movement (for beta) and their count
    beta_pairs = [(a, b) for a in range(len(sub)) for b in range(a + 1, len(sub))
                  if abs(log_n[b] - log_n[a]) > 0.02]

    classes = [classify(t) for t in trophies]
    betas, hs = [], []
    for i in range(n_tr):
        _cls, beta_prior, h_prior = classes[i]
        # Collapse the class prior toward the launch-night floor by the
        # play time available (tutorial exempt: beta = 1 always).
        if _cls == "tutorial":
            beta_prior_eff = 1.0
        else:
            beta_prior_eff = (BETA_FLOOR
                              + (beta_prior - BETA_FLOOR) * k_play)
        log_c = [math.log(max((r[2][i] if r[2][i] is not None else 0) + 0.5, 0.5))
                 for r in sub]
        beta_ts = None
        if len(beta_pairs) >= 3:
            slopes = []
            for a, b in beta_pairs:
                slopes.append((log_c[b] - log_c[a]) / (log_n[b] - log_n[a]))
            beta_ts = clamp(median(slopes), -0.2, 1.2)
        w_beta = len(beta_pairs) / (len(beta_pairs) + SHRINK_K) \
            if beta_ts is not None else 0.0
        betas.append(w_beta * beta_ts + (1.0 - w_beta) * beta_prior_eff
                     if beta_ts is not None else beta_prior_eff)
    # Second pass: incumbent progress h is the RESIDUAL holder growth
    # after removing the N-driven part (log C - beta * log N). Estimating
    # h from raw log C would double-count the player flood -- for the
    # tutorial trophy (C = N by definition) it would even invent
    # explosive growth and trip the aggregate guard.
    for i in range(n_tr):
        _cls, _beta_prior, h_prior = classes[i]
        log_c = [math.log(max((r[2][i] if r[2][i] is not None else 0) + 0.5, 0.5))
                 for r in sub]
        resid = [lc - betas[i] * ln for lc, ln in zip(log_c, log_n)]
        h_ts = theil_sen(t_days, resid, -0.5, 2.0)
        n_pairs_h = len(t_days) * (len(t_days) - 1) // 2
        w_h = n_pairs_h / (n_pairs_h + SHRINK_K) if h_ts is not None else 0.0
        hs.append(w_h * h_ts + (1.0 - w_h) * h_prior
                  if h_ts is not None else h_prior)

    # ---- scenario means (counts) per trophy, aggregate-guarded PER
    # scenario: sum of counts in scenario k must imply a sane
    # trophies-per-player against that scenario's own N_k.
    mus_by_scen = []  # mus_by_scen[k][i]
    for m, _w in scenarios:
        row = []
        for i in range(n_tr):
            c0 = counts0[i]
            mu = max(c0, 0) * (m ** betas[i]) * math.exp(hs[i] * horizon_d)
            if c0 == 0:
                mu = max(mu, 0.3 * (m ** 0.1))
            row.append(mu)
        mus_by_scen.append(row)
    for k, (m, _w) in enumerate(scenarios):
        n_t = n0 * m
        tot = sum(mus_by_scen[k])
        r = tot / n_t if n_t > 0 else 0.0
        if r > R_GUARD_HI:
            f = R_GUARD_HI / r
            mus_by_scen[k] = [mu * f for mu in mus_by_scen[k]]
        elif 0.0 < r < R_GUARD_LO:
            f = R_GUARD_LO / r
            mus_by_scen[k] = [mu * f for mu in mus_by_scen[k]]

    # ---- predictive mixture per trophy -> quantiles on a logit grid
    grid = [LOGIT_LO + i * 0.01
            for i in range(int((LOGIT_HI - LOGIT_LO) / 0.01) + 1)]
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    rows = []
    for i, tr in enumerate(trophies):
        c0 = counts0[i]
        cls, beta_prior, _h = classes[i]
        beta = betas[i]
        h = hs[i]
        # tutorial trophy: pinned at 100% by construction (beta = 1 keeps
        # C/N = 1 in every scenario); emit the ceiling band directly.
        if cls == "tutorial":
            rows.append({"id": tr["id"],
                         "q": [99.0, 99.9, 99.99, 99.99, 99.99]})
            continue
        # mixture CDF on the logit grid
        cdf = [0.0] * len(grid)
        for k, (m, w) in enumerate(scenarios):
            mu = mus_by_scen[k][i]
            n_t = n0 * m
            p_med = clamp(100.0 * mu / n_t, LO, HI)
            mu_logit = logit_pct(p_med)
            var = (1.0 / max(mu, 0.5)
                   + (SIGMA_BETA * abs(math.log(max(m, 1e-9)))) ** 2
                   + (SIGMA_H_DAY * horizon_d) ** 2
                   + SIGMA_BASE ** 2)
            sd = math.sqrt(var)
            for gi, x in enumerate(grid):
                cdf[gi] += w * norm_cdf((x - mu_logit) / sd)
        qs = []
        for lv in levels:
            # first grid point whose CDF reaches lv (linear interp)
            j = 0
            while j < len(grid) - 1 and cdf[j] < lv:
                j += 1
            if j == 0:
                x = grid[0]
            else:
                c0v, c1v = cdf[j - 1], cdf[j]
                frac = 0.0 if c1v <= c0v else (lv - c0v) / (c1v - c0v)
                x = grid[j - 1] + frac * (grid[j] - grid[j - 1])
            qs.append(round4(inv_logit_pct(x)))
        # enforce monotone non-decreasing
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tr["id"], "q": qs})
    return rows


def fallback_rows(case):
    """Persistence + wide spread; never crash, never emit an invalid row."""
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_d = float(case.get("horizon_minutes") or 0) / 1440.0
    sd = 0.75 * math.sqrt(max(horizon_d, 1.0 / 24.0))
    rates = history[-1]["rates"] if history else [None] * len(trophies)
    rows = []
    for tr, r in zip(trophies, rates):
        r = 1.0 if r is None else clamp(r, LO, HI)
        base = logit_pct(r)
        qs = [round4(inv_logit_pct(base + z * sd)) for z in Z]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tr["id"], "q": qs})
    return rows


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        try:
            rows = forecast_case(case)
        except Exception:
            rows = fallback_rows(case)
        out = {"trophies": rows}
        with open(os.path.join(out_dir, case_id + ".out"), "w",
                  encoding="utf-8") as fh:
            json.dump(out, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
