"""Streaming detector (no MQTT/DB dependencies, so it can be unit-tested)."""
from collections import deque
from datetime import datetime, timedelta

CONT_MIN = 240            # minutes of non-zero flow that triggers the continuous-flow rule
COOLDOWN = timedelta(hours=24)   # one alert per zone per rule per day while a leak persists


class StreamDetector:
    def __init__(self, night_thr, cost_per_kl=1.0):
        self.thr = night_thr                                  # {"A": x, "B": y, "C": z}
        self.cost = cost_per_kl
        self.buf = {z: deque(maxlen=400) for z in night_thr}  # (ts, flow)
        self.last_alert = {}                                  # (zone, kind) -> ts
        self.last_night = {}                                  # zone -> date already checked

    def _severity(self, lph):
        return "high" if lph >= 60 else "medium" if lph >= 12 else "low"

    def _alert(self, zone, kind, ts, lph):
        key = (zone, kind)
        if key in self.last_alert and ts - self.last_alert[key] < COOLDOWN:
            return None
        self.last_alert[key] = ts
        return dict(zone=zone, kind=kind, ts=ts.isoformat(), est_lph=round(lph, 1),
                    severity=self._severity(lph), litres_per_day=round(lph * 24),
                    cost_per_day=round(lph * 24 / 1000 * self.cost, 2))

    def update(self, zone, ts, flow):
        """Feed one reading. Returns an alert dict or None."""
        if zone not in self.buf:
            return None
        b = self.buf[zone]
        b.append((ts, flow))
        # rule 1: continuous flow
        if len(b) >= CONT_MIN:
            last = [f for _, f in list(b)[-CONT_MIN:]]
            if min(last) > 0:
                a = self._alert(zone, "continuous_flow", ts, min(last) * 60)
                if a:
                    return a
        # rule 2: minimum night flow, evaluated once at 04:00
        if ts.hour == 4 and self.last_night.get(zone) != ts.date():
            self.last_night[zone] = ts.date()
            w = sorted(f for t, f in b if t.date() == ts.date() and 2 <= t.hour < 4)
            if len(w) > 60:
                q = w[int(0.1 * len(w))]
                if q > self.thr[zone]:
                    return self._alert(zone, "night_flow", ts, q * 60)
        return None
