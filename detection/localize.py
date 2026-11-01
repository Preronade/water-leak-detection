"""Phase 4: localization by night-flow excess per zone + mass-balance gap."""
import pandas as pd
from .rules import night_stats

COLS = ["A", "B", "C", "GAP"]     # GAP = main meter - sum of zone meters (leak in unmetered pipe)


def culprit_by_night(df, thr):
    """For every night: the column with the largest excess over its threshold, else NaN."""
    ex = pd.DataFrame({c: night_stats(df[c]) - thr[c] for c in COLS})
    best = ex.idxmax(axis=1)
    return best.where(ex.max(axis=1) > 0)


def label_of(col):
    return "MAIN" if col == "GAP" else col
