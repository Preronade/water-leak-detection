"""Phase 3: detectors. Every detector returns a boolean Series (True = alarm state).
`episodes()` turns that into alert timestamps."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

NIGHT = (2, 4)   # hours, minimum-night-flow window


def step_minutes(s):
    return (s.index[1] - s.index[0]).total_seconds() / 60.0


def load(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df["GAP"] = df["main"] - df[["A", "B", "C"]].sum(axis=1)     # mass balance
    return df


def persist(flag, n):
    """True only once `flag` has been True for n consecutive samples (the persistence check)."""
    return flag.astype(int).rolling(int(n)).min().fillna(0).astype(bool)


def episodes(flag, min_gap_h=6):
    """Rising edges of a flag, merged if closer than min_gap_h -> list of alert timestamps."""
    edges = flag.index[flag & ~flag.shift(1, fill_value=False)]
    out = []
    for t in edges:
        if not out or (t - out[-1]) >= pd.Timedelta(hours=min_gap_h):
            out.append(t)
    return out


# 1) continuous-flow rule -------------------------------------------------------------
def continuous_flow(s, hours=4):
    """Alarm if flow never touched zero for `hours` hours."""
    return persist(s > 0, hours * 60 / step_minutes(s))


# 2) minimum night flow ---------------------------------------------------------------
def night_stats(s):
    w = s[(s.index.hour >= NIGHT[0]) & (s.index.hour < NIGHT[1])]
    return w.groupby(w.index.normalize()).quantile(0.1)      # robust "minimum" per night


def night_threshold(train_s, margin=0.04):
    return float(night_stats(train_s).max() + margin)


def night_flow_flag(s, thr):
    n = night_stats(s)
    hit = n[n > thr]
    flag = pd.Series(False, index=s.index)
    flag.loc[hit.index + pd.Timedelta(hours=NIGHT[1])] = True   # decided at end of night window
    return flag


# 3) statistical baseline: hour-of-week mean + EWMA of residual ----------------------
class HowBaseline:
    def __init__(self, k=4.0, halflife_h=6, persist_h=2):
        self.k, self.hl, self.ph = k, halflife_h, persist_h

    @staticmethod
    def _how(i):
        return i.dayofweek * 24 + i.hour

    def fit(self, s):
        h = s.resample("1h").mean()
        how = self._how(h.index)
        self.mu = h.groupby(how).mean()
        n = h.groupby(how).count().mean()
        res = (h - self.mu.reindex(how).values) * np.sqrt(n / (n - 1))   # undo in-sample shrinkage
        self.thr = self.k * res.ewm(halflife=self.hl).mean().std()
        return self

    def flag(self, s):
        h = s.resample("1h").mean()
        res = h - self.mu.reindex(self._how(h.index)).values
        ew = res.ewm(halflife=self.hl).mean()
        return persist(ew > self.thr, self.ph)


# 4) ML: Isolation Forest on 6h window features --------------------------------------
def features(s):
    w = int(360 / step_minutes(s))
    r = s.rolling(w)
    f = pd.DataFrame({"mean": r.mean(), "std": r.std(), "min": r.min(),
                      "zero_frac": (s == 0).astype(float).rolling(w).mean()}).dropna()
    f = f.resample("1h").last().dropna()
    f["hs"], f["hc"] = np.sin(2 * np.pi * f.index.hour / 24), np.cos(2 * np.pi * f.index.hour / 24)
    f["wk"] = (f.index.dayofweek >= 5).astype(int)
    return f


class IFDetector:
    def __init__(self, q=0.999, persist_h=3):
        self.q, self.ph = q, persist_h

    def fit(self, s):
        X = features(s)
        self.m = IsolationForest(n_estimators=200, random_state=0).fit(X)
        self.thr = np.quantile(-self.m.score_samples(X), self.q)
        return self

    def flag(self, s):
        X = features(s)
        return persist(pd.Series(-self.m.score_samples(X) > self.thr, index=X.index), self.ph)
