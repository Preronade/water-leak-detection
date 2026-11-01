"""Phase 2: run the same detectors on an external CSV (BattLeDIM, LeakDB, your own meter export).
Run:  python -m detection.run_external --csv FILE --time-col Timestamp --value-col p227 --sep ";" --decimal "," --train-frac 0.5
Check the real column names / separators in the files you download; they differ per dataset.
"""
import argparse
import pandas as pd
from .rules import night_threshold, night_flow_flag, HowBaseline, IFDetector, episodes


def load_series(path, time_col, value_col, sep=",", decimal="."):
    df = pd.read_csv(path, sep=sep, decimal=decimal, parse_dates=[time_col], index_col=time_col)
    return df[value_col].astype(float).sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--time-col", required=True)
    ap.add_argument("--value-col", required=True); ap.add_argument("--sep", default=",")
    ap.add_argument("--decimal", default="."); ap.add_argument("--train-frac", type=float, default=0.5)
    a = ap.parse_args()
    s = load_series(a.csv, a.time_col, a.value_col, a.sep, a.decimal)
    cut = int(len(s) * a.train_frac)                      # time-ordered split, never shuffled
    train, test = s.iloc[:cut], s.iloc[cut:]
    print("train", train.index[0], "->", train.index[-1], "| test", test.index[0], "->", test.index[-1])
    thr = night_threshold(train, margin=0.05 * max(train.mean(), 1e-9))   # real meters rarely hit zero: scale margin
    print("night_flow alerts:", episodes(night_flow_flag(test, thr)))
    print("stat_ewma alerts :", episodes(HowBaseline().fit(train).flag(test)))
    print("iso_forest alerts:", episodes(IFDetector().fit(train).flag(test)))
    print("Now compare these timestamps with the dataset's leak start/end file.")


if __name__ == "__main__":
    main()
