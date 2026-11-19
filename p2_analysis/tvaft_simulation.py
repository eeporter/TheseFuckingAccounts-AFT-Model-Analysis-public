"""Simulated accounts with known time ratios, for validating tvaft.TimeVaryingAFT (used by TV_AFT_validation.ipynb).

Three covariates, Weibull baseline shaped like the TFA data (shape ~0.45):
    x1  fixed trait, standard normal                                   true time ratio 0.70
    x2  0 -> 1 at a random time (e.g. starts posting NSFW), mean 5 d   true time ratio 0.40
    x3  log(1 + actions so far); actions arrive during the first 30 d  true time ratio 1.25
Event times are generated exactly by running each subject's clock Psi(t) until it reaches a Weibull threshold.
"""
import numpy as np
import pandas as pd

TRUE_TIME_RATIOS = np.array([0.70, 0.40, 1.25])
BETA_TRUE = np.log(TRUE_TIME_RATIOS)
MU_TRUE, SIGMA_TRUE = np.log(15), 2.2


def simulate(n, rng, admin=2000.0):
    """Returns (episodes, subjects) in the TimeVaryingAFT.fit format; subjects not accused by `admin` are censored."""
    rows, subj = [], []
    for i in range(n):
        x1 = rng.normal()
        s = rng.exponential(5.0)
        rate = rng.gamma(1.0, 1.0)
        acts = np.cumsum(rng.exponential(1 / rate, size=200))
        acts = acts[acts < 30]
        cuts = np.unique(np.r_[0.0, s, acts])
        cuts = cuts[cuts < admin]
        U = np.exp(MU_TRUE + SIGMA_TRUE * np.log(rng.exponential()))
        psi, t_event, ep = 0.0, None, []
        for j, a in enumerate(cuts):
            b = cuts[j + 1] if j + 1 < len(cuts) else admin
            x = np.array([x1, float(a >= s), np.log1p((acts <= a).sum())])
            speed = np.exp(-x @ BETA_TRUE)
            if psi + speed * (b - a) >= U:
                t_event = a + (U - psi) / speed
                ep.append((i, a, t_event, *x))
                break
            psi += speed * (b - a)
            ep.append((i, a, b, *x))
        rows += ep
        subj.append((i, int(t_event is not None)))
    episodes = pd.DataFrame(rows, columns=['id', 'start', 'stop', 'x1', 'x2', 'x3'])
    subjects = pd.DataFrame(subj, columns=['id', 'event']).set_index('id')
    return episodes, subjects
