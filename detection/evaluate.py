"""Phase 7: run every detector on the test set, score against labels.
Run:  python -m detection.evaluate
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .rules import (load, continuous_flow, night_threshold, night_flow_flag,
                    HowBaseline, IFDetector, episodes)
from .localize import culprit_by_night, label_of

ZONES = ["A", "B", "C"]
DAY = pd.Timedelta(days=1)


def merge(ts, gap_h=6):
    """Merge alerts from different methods that are closer than gap_h into one."""
    out = []
    for t in ts:
        if not out or t - out[-1] >= pd.Timedelta(hours=gap_h):
            out.append(t)
    return out


def run_all(train, test):
    thr = {c: night_threshold(train[c]) for c in ZONES + ["GAP"]}
    alerts = {}                                           # method -> column -> [timestamps]
    def put(m, c, flag): alerts.setdefault(m, {})[c] = episodes(flag)
    for c in ZONES + ["GAP"]:
        put("night_flow", c, night_flow_flag(test[c], thr[c]))
    for z in ZONES:
        put("continuous", z, continuous_flow(test[z]))
        put("stat_ewma", z, HowBaseline().fit(train[z]).flag(test[z]))
        put("iso_forest", z, IFDetector().fit(train[z]).flag(test[z]))
    cols = ZONES + ["GAP"]
    alerts["combined"] = {c: merge(sorted(t for m in ("night_flow", "continuous", "stat_ewma", "iso_forest")
                                          for t in alerts[m].get(c, []))) for c in cols}
    return alerts, thr


def score(alerts, labels):
    rows, fa_rows = [], []
    for m, by_col in alerts.items():
        for _, L in labels.iterrows():
            col = "GAP" if L.zone == "MAIN" else L.zone
            hit = [t for t in by_col.get(col, []) if L.start <= t <= L.end + DAY]
            rows.append(dict(method=m, leak_id=L.leak_id, zone=L.zone, size_lph=L.size_lph,
                             detected=bool(hit),
                             latency_h=(hit[0] - L.start).total_seconds() / 3600 if hit else np.nan))
        for col, al in by_col.items():
            for t in al:
                ok = any((label_of(col) == L.zone) and L.start <= t <= L.end + DAY for _, L in labels.iterrows())
                if not ok:
                    fa_rows.append(dict(method=m, col=col, ts=t))
    return pd.DataFrame(rows), pd.DataFrame(fa_rows)


def main():
    train, test = load("data/train.csv"), load("data/test.csv")
    labels = pd.read_csv("data/labels.csv", parse_dates=["start", "end"])
    days = len(test) / 1440
    alerts, thr = run_all(train, test)
    json.dump({k: v for k, v in thr.items() if k in ZONES}, open("data/baseline.json", "w"), indent=2)

    res, fa = score(alerts, labels)
    methods = list(alerts)
    summ = []
    for m in methods:
        r = res[res.method == m]
        nfa = int((fa.method == m).sum()) if len(fa) else 0
        det_sizes = r[r.detected].size_lph
        summ.append(dict(method=m, recall=f"{r.detected.sum()}/{len(r)}",
                         median_latency_h=round(r.latency_h.median(), 1),
                         false_alarms=nfa, fa_per_month=round(nfa / (days / 30), 2),
                         smallest_detected_lph=det_sizes.min() if len(det_sizes) else None))
    summ = pd.DataFrame(summ)
    by_size = res.pivot_table(index="size_lph", columns="method", values="detected", aggfunc="mean")[methods]

    cul = culprit_by_night(test, thr)
    ok = []
    for _, L in labels.iterrows():
        c = cul.get(L.end.normalize(), np.nan)
        ok.append(isinstance(c, str) and label_of(c) == L.zone)

    pd.set_option("display.width", 200)
    print("\n== Summary (test set: %d days, %d leaks) ==" % (days, len(labels)))
    print(summ.to_string(index=False))
    print("\n== Fraction of leaks detected, by size (L/h) ==")
    print(by_size.round(2).to_string())
    print("\n== Median latency (h) by leak size ==")
    print(res.pivot_table(index="size_lph", columns="method", values="latency_h", aggfunc="median")[methods].round(1).to_string())
    print(f"\nLocalization accuracy (night-flow excess -> zone or MAIN): {sum(ok)}/{len(ok)}")
    if len(fa):
        print("\nFalse alarms:\n", fa.to_string(index=False))

    summ.to_csv("results/summary.csv", index=False)
    by_size.to_csv("results/detection_by_size.csv")
    res.to_csv("results/per_leak.csv", index=False)
    plt.figure(figsize=(7, 4))
    for m in methods:
        plt.plot(by_size.index, by_size[m], marker="o", label=m)
    plt.xscale("log"); plt.xlabel("leak size (L/h)"); plt.ylabel("fraction detected")
    plt.title("Detection rate vs leak size (simulated)"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig("results/detection_vs_size.png", dpi=150)


if __name__ == "__main__":
    main()
