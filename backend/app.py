"""Phase 5: MQTT -> InfluxDB + detection + Telegram + FastAPI.
Run:  uvicorn backend.app:app --port 8000
Env:  INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, MQTT_HOST
"""
import json, os
from contextlib import asynccontextmanager
from datetime import datetime
import requests
import paho.mqtt.client as mqtt
from fastapi import FastAPI, HTTPException
from influxdb_client import InfluxDBClient, Point
from .detector import StreamDetector

thr = json.load(open(os.getenv("BASELINE", "data/baseline.json")))
det = StreamDetector(thr, cost_per_kl=float(os.getenv("COST_PER_KL", "1.0")))
ALERTS = []

influx = InfluxDBClient(url=os.getenv("INFLUX_URL", "http://localhost:8086"),
                        token=os.getenv("INFLUX_TOKEN", "mytoken"), org=os.getenv("INFLUX_ORG", "leak"))
writer = influx.write_api()           # batching by default
BUCKET = os.getenv("INFLUX_BUCKET", "flow")


def telegram(text):
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("[ALERT]", text)
        return
    requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data={"chat_id": chat, "text": text}, timeout=10)


def on_message(client, userdata, msg):
    d = json.loads(msg.payload)
    zone, flow, ts = d["zone"], float(d["flow_lpm"]), datetime.fromisoformat(d["ts"])
    writer.write(BUCKET, record=Point("flow").tag("zone", zone).field("flow_lpm", flow).field("sim_ts", d["ts"]))
    a = det.update(zone, ts, flow)
    if a:
        a["id"] = len(ALERTS) + 1
        a["false_alarm"] = False
        ALERTS.append(a)
        writer.write(BUCKET, record=Point("alert").tag("zone", zone).tag("kind", a["kind"])
                     .field("est_lph", a["est_lph"]).field("severity", a["severity"]))
        telegram(f"Possible leak in zone {zone} ({a['kind']}, {a['severity']}). "
                 f"~{a['est_lph']} L/h = {a['litres_per_day']} L/day, ~{a['cost_per_day']}/day. Sim time {a['ts']}")


@asynccontextmanager
async def lifespan(app):
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    c.on_message = on_message
    c.connect(os.getenv("MQTT_HOST", "localhost"), 1883)
    c.subscribe("site1/+/flow")
    c.loop_start()
    yield
    c.loop_stop()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True, "alerts": len(ALERTS)}


@app.get("/alerts")
def alerts():
    return ALERTS


@app.post("/alerts/{alert_id}/false_alarm")
def false_alarm(alert_id: int):
    """User marks an alert as normal usage. Use these to tune thresholds / add a suppression window."""
    for a in ALERTS:
        if a["id"] == alert_id:
            a["false_alarm"] = True
            return a
    raise HTTPException(404, "no such alert")
