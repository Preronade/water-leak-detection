"""Phase 5: replay a CSV into MQTT as if it were live sensors.
Run:  python -m publisher.publish --start-day 0 --speed 1440     (1440 = one simulated day per minute)
"""
import argparse, json, time
import pandas as pd
import paho.mqtt.client as mqtt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/test.csv")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--site", default="site1")
    ap.add_argument("--speed", type=float, default=1440, help="simulated minutes per real minute")
    ap.add_argument("--start-day", type=int, default=0, help="skip this many days of the CSV")
    a = ap.parse_args()

    df = pd.read_csv(a.csv, index_col=0, parse_dates=True).iloc[a.start_day * 1440:]
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    c.connect(a.host, 1883)
    c.loop_start()
    delay = 60.0 / a.speed
    for ts, row in df.iterrows():
        for z in ("A", "B", "C", "main"):
            c.publish(f"{a.site}/{z}/flow",
                      json.dumps({"zone": z, "flow_lpm": round(float(row[z]), 3), "ts": ts.isoformat()}))
        time.sleep(delay)


if __name__ == "__main__":
    main()
