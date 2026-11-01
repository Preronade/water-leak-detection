"""Phase 1: synthetic 3-zone building, 1-minute flow data, with injected leaks and non-leak events.

Run:  python -m sim.simulate
Out:  data/train.csv (clean), data/test.csv (leaks + events), data/labels.csv, data/events.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

START = pd.Timestamp("2026-01-05 00:00")      # a Monday
TRAIN_DAYS, TEST_DAYS = 56, 56
ZONES = ["A", "B", "C"]                        # A residential, B commercial, C mixed
DEADBAND = 0.05                                # L/min: sensor reads 0 below this (Hall sensors are coarse)
NOISE = 0.03                                   # 3% multiplicative sensor noise

# (zone, size L/h, kind). "MAIN" = leak in unmetered pipe between main meter and zone meters
PLAN = [("A", 30, "step"), ("B", 6, "step"), ("C", 12, "step"), ("MAIN", 60, "step"),
        ("A", 3, "step"), ("B", 120, "step"), ("C", 30, "grow"), ("A", 12, "grow"),
        ("B", 30, "step"), ("MAIN", 12, "step"), ("C", 6, "step"), ("A", 60, "step")]
LEAK_DAYS = 3
# Legit high usage that is NOT a leak (tests false alarms): (zone, day index in test, start hour, hours, L/min)
EVENTS = [("B", 51, 10, 5, 20), ("A", 53, 9, 5, 15)]


def _g(h, mu, sd):
    return np.exp(-0.5 * ((h - mu) / sd) ** 2)


def usage_prob(h, dow, zone):
    """Probability that some fixture is running in a given minute."""
    wk = dow >= 5
    if zone == "A":
        p = 0.02 + 0.45 * _g(h, 7.5, 1.2) + 0.55 * _g(h, 20, 1.8) + 0.12 * _g(h, 13, 2.5)
        p = np.where(wk, p * 1.15 + 0.1 * _g(h, 11, 2.5), p)
    elif zone == "B":
        p = 0.01 + 0.8 * ((h >= 8) & (h < 18)) * (0.8 + 0.2 * _g(h, 13, 3))
        p = np.where(wk, 0.01 + 0.08 * _g(h, 12, 3), p)
    else:
        p = 0.015 + 0.3 * _g(h, 8, 1.5) + 0.35 * _g(h, 18.5, 2) + 0.2 * _g(h, 12.5, 2)
    return np.clip(p, 0, 0.95)


def _read(x, rng):
    y = x * (1 + rng.normal(0, NOISE, len(x)))
    y[y < DEADBAND] = 0.0
    return np.clip(y, 0, None)


def generate(days, seed, t0=START, leaks=(), events=()):
    rng = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range(t0, periods=n, freq="min")
    h = idx.hour.values + idx.minute.values / 60
    dow, day = idx.dayofweek.values, np.arange(n) // 1440
    true = {}
    for z in ZONES:
        scale = rng.lognormal(0, 0.1, days)[day]
        on = rng.random(n) < np.clip(usage_prob(h, dow, z) * scale, 0, 0.95)
        true[z] = on * rng.lognormal(np.log(4.5), 0.5, n)
    main_extra = np.zeros(n)
    for z, d, hr, dur, lpm in events:
        a = d * 1440 + hr * 60
        true[z][a:a + dur * 60] += lpm
    for lk in leaks:
        a, b = int(lk["start_min"]), int(lk["end_min"])
        prof = np.full(b - a, lk["size_lph"] / 60.0)
        if lk["kind"] == "grow":
            ramp = min(2 * 1440, b - a)
            prof[:ramp] *= np.linspace(0, 1, ramp)
        (main_extra if lk["zone"] == "MAIN" else true[lk["zone"]])[a:b] += prof
    main_true = sum(true.values()) + main_extra
    df = pd.DataFrame({z: _read(true[z], rng) for z in ZONES}, index=idx)
    df["main"] = _read(main_true, rng)
    df.index.name = "ts"
    return df


def main():
    out = Path("data")
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    train = generate(TRAIN_DAYS, seed=1)
    t0 = START + pd.Timedelta(days=TRAIN_DAYS)
    leaks = []
    for i, (zone, size, kind) in enumerate(PLAN):
        s = (4 * i + 1 + rng.uniform(0.25, 0.8)) * 1440
        leaks.append(dict(leak_id=i + 1, zone=zone, size_lph=size, kind=kind,
                          start_min=int(s), end_min=int(s + LEAK_DAYS * 1440)))
    test = generate(TEST_DAYS, seed=2, t0=t0, leaks=leaks, events=EVENTS)
    labels = pd.DataFrame(leaks)
    labels["start"] = t0 + pd.to_timedelta(labels.start_min, unit="min")
    labels["end"] = t0 + pd.to_timedelta(labels.end_min, unit="min")
    labels = labels[["leak_id", "zone", "size_lph", "kind", "start", "end"]]
    ev = pd.DataFrame([dict(zone=z, start=t0 + pd.Timedelta(days=d, hours=hr),
                            end=t0 + pd.Timedelta(days=d, hours=hr + dur), lpm=l)
                       for z, d, hr, dur, l in EVENTS])
    train.round(4).to_csv(out / "train.csv")
    test.round(4).to_csv(out / "test.csv")
    labels.to_csv(out / "labels.csv", index=False)
    ev.to_csv(out / "events.csv", index=False)
    print(f"train {train.shape}, test {test.shape}, {len(labels)} leaks, {len(ev)} non-leak events")


if __name__ == "__main__":
    main()
