"""Simulated accounts with known time ratios, for validating tvaft.TimeVaryingAFT (used by TV_AFT_validation.ipynb).

Three covariates, Weibull baseline shaped like the TFA data (shape ~0.45):
    x1  fixed trait, standard normal                                   true time ratio 0.70
    x2  0 -> 1 at a random time (e.g. starts posting NSFW), mean 5 d   true time ratio 0.40
    x3  log(1 + actions so far); actions arrive during the first 30 d  true time ratio 1.25
Event times are generated exactly by running each subject's clock Psi(t) until it reaches a Weibull threshold.

Time is in days. Library module, reads and writes no files:
    episodes, subjects = simulate(n, np.random.default_rng(seed))
    TimeVaryingAFT('weibull').fit(episodes, subjects, ['x1', 'x2', 'x3'])   # estimates should recover BETA_TRUE
"""
import numpy as np
import pandas as pd

TRUE_TIME_RATIOS = np.array([0.70, 0.40, 1.25])
BETA_TRUE = np.log(TRUE_TIME_RATIOS)
MU_TRUE, SIGMA_TRUE = np.log(15), 2.2       # baseline log-scale location (median clock ~15 * log(2)^2.2 d) and scale
                                            # (Weibull shape = 1 / 2.2 ~ 0.45)


def simulate(n, rng, admin=2000.0):
    """Returns (episodes, subjects) in the TimeVaryingAFT.fit format; subjects not accused by `admin` are censored.

    n: number of subjects. rng: numpy Generator. admin: administrative censoring time in days.
    episodes: id, start, stop, x1, x2, x3 (days); subjects: indexed by id with 'event' (1 = event, 0 = censored).
    """
    rows, subj = [], []
    for i in range(n):
        x1 = rng.normal()
        s = rng.exponential(5.0)                    # day x2 switches from 0 to 1
        rate = rng.gamma(1.0, 1.0)                  # subject's activity rate (actions per day)
        acts = np.cumsum(rng.exponential(1 / rate, size=200))   # Poisson-process action times, kept only before day 30
        acts = acts[acts < 30]
        cuts = np.unique(np.r_[0.0, s, acts])       # episode boundaries: wherever a covariate can change
        cuts = cuts[cuts < admin]
        # baseline threshold: if E ~ Exp(1), log E is standard minimum extreme value, so U is Weibull with log-scale
        # location MU_TRUE and scale SIGMA_TRUE (the 'weibull' baseline in tvaft)
        U = np.exp(MU_TRUE + SIGMA_TRUE * np.log(rng.exponential()))
        psi, t_event, ep = 0.0, None, []
        for j, a in enumerate(cuts):
            b = cuts[j + 1] if j + 1 < len(cuts) else admin
            x = np.array([x1, float(a >= s), np.log1p((acts <= a).sum())])   # x3 counts actions up to and including a
            speed = np.exp(-x @ BETA_TRUE)          # clock speed in this episode
            if psi + speed * (b - a) >= U:          # clock reaches U inside [a, b): solve linearly for the event time
                t_event = a + (U - psi) / speed
                ep.append((i, a, t_event, *x))
                break
            psi += speed * (b - a)
            ep.append((i, a, b, *x))
        rows += ep
        subj.append((i, int(t_event is not None)))   # no event before admin -> censored at admin
    episodes = pd.DataFrame(rows, columns=['id', 'start', 'stop', 'x1', 'x2', 'x3'])
    subjects = pd.DataFrame(subj, columns=['id', 'event']).set_index('id')
    return episodes, subjects
