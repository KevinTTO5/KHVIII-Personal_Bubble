# cyclops_main.py
# Unified control loop for Cyclops system
# Handles: Firestore ingestion, ADK agent reasoning, ultrasonic sensor alerts, and FCM push notifications

import os
import json
import time
import asyncio
import queue
import threading
from typing import Any, Dict, List

import firebase_admin
from firebase_admin import credentials, firestore, messaging

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
import serial  # for ultrasonic sensor (Arduino)

# --------------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------------
SERVICE_ACCOUNT = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service-account.json")
DEVICE_TOPIC = os.getenv("CYCLOPS_TOPIC", "user_1")
ARDUINO_PORT = os.getenv("CYCLOPS_ARDUINO_PORT", "/dev/tty.usbmodem1101")  # adjust
ARDUINO_BAUD = int(os.getenv("CYCLOPS_ARDUINO_BAUD", "9600"))
THRESHOLD_M = float(os.getenv("CYCLOPS_THRESHOLD_M", "2.0"))
ARDUINO_COOLDOWN_S = int(os.getenv("CYCLOPS_ULTRA_COOLDOWN_S", "8"))

REAPPEAR_WINDOW_S = 30
DETECT_PERIOD_S = 5
ENTITY_COOLDOWN_S = 10

CONSEC_THRESHOLD_S = 30.0
TOTAL_TIME_THRESHOLD_S = 120.0
REAPPEAR_THRESHOLD = 3

COL_INTERVALS = "intervals"
COL_ENTITIES = "entities"

# --------------------------------------------------------------------------------------
# FIREBASE INIT
# --------------------------------------------------------------------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# --------------------------------------------------------------------------------------
# ALERT UTILS (FCM)
# --------------------------------------------------------------------------------------
def send_alert_to_topic(topic: str, msg: str, severity: str = "MEDIUM", cooldown_s: int = 5):
    """Sends push notification via Firebase Cloud Messaging."""
    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title=f"{severity} Alert!",
                body=msg,
            ),
            topic=topic,
        )
        response = messaging.send(message)
        print(f"📲 FCM alert sent: {severity} - {msg}")
        return response
    except Exception as e:
        print("⚠️ FCM send failed:", e)
        return None

# --------------------------------------------------------------------------------------
# ADK DETECTIVE AGENT
# --------------------------------------------------------------------------------------
detective_agent = LlmAgent(
    name="detective_agent",
    model="models/gemini-2.5-flash",
    instruction="""
You are Cyclops Detective. You receive a JSON object:
{"entities":[{"entity_id": "e_3", "consec_time_s": 12.3, "total_time_s": 55.0,
              "recent_hits_ts":[<epoch seconds>...], "true_reappear_count": 1,
              "interval_appear_count": 7}, ...]}

Use these FIXED rules:
- Suspicious if consec_time_s >= 30.0
- OR if (appearances in last ~30s minus 1) >= 3
- OR if total_time_s >= 120.0

Return STRICT JSON Lines (one per entity):
{"entity_id":"e_1","verdict":"SUSPICIOUS|NOT_SUSPICIOUS","reason":"<short>"}

If insufficient info, default to NOT_SUSPICIOUS.
"""
)

# --------------------------------------------------------------------------------------
# FIRESTORE INGESTION (intervals -> entities)
# --------------------------------------------------------------------------------------
interval_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=500)

def _entity_doc_id(pid: int) -> str:
    return f"e_{int(pid)}"

@firestore.transactional
def _update_entity_tx(tx, ref, *, add_seconds: float, interval_id: int, now_s: int):
    """Transactional update per entity."""
    snap = ref.get(transaction=tx)
    if snap.exists:
        cur = snap.to_dict() or {}
        last_idx = cur.get("last_interval_id", -1)
        continuous = (last_idx == interval_id - 1)
        new_consec = (cur.get("consec_time_s", 0.0) + add_seconds) if continuous else float(add_seconds)
        hits: List[int] = cur.get("recent_hits_ts", []) or []
        hits = [t for t in hits if t >= now_s - REAPPEAR_WINDOW_S]
        hits.append(now_s)
        if len(hits) > 30:
            hits = hits[-30:]
        updates = {
            "entity_id": ref.id,
            "last_seen_ts": now_s,
            "last_interval_id": interval_id,
            "consec_time_s": float(new_consec),
            "total_time_s": firestore.Increment(float(add_seconds)),
            "interval_appear_count": firestore.Increment(1),
            "recent_hits_ts": hits
        }
        if not continuous and cur.get("interval_appear_count", 0) > 0:
            updates["true_reappear_count"] = firestore.Increment(1)
        tx.update(ref, updates)
    else:
        tx.set(ref, {
            "entity_id": ref.id,
            "last_seen_ts": now_s,
            "last_interval_id": interval_id,
            "consec_time_s": float(add_seconds),
            "total_time_s": float(add_seconds),
            "interval_appear_count": 1,
            "true_reappear_count": 0,
            "recent_hits_ts": [now_s]
        })

def _on_intervals_snapshot(col_snapshot, changes, read_time):
    for ch in changes:
        if ch.type.name == "ADDED":
            interval_q.put(ch.document.to_dict() or {})

def start_interval_listener():
    db.collection(COL_INTERVALS).on_snapshot(_on_intervals_snapshot)
    print(f"👂 Listening to '{COL_INTERVALS}/' …")

async def rollup_worker():
    """Consume intervals → update entities."""
    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, interval_q.get)
        try:
            interval_id = int(item.get("interval_id"))
            now_s = int(time.time())
            for p in item.get("people_data", []):
                t = float(p.get("time", 0.0))
                if t <= 0:
                    continue
                eid = _entity_doc_id(int(p.get("id")))
                tx = db.transaction()
                _update_entity_tx(
                    tx,
                    db.collection(COL_ENTITIES).document(eid),
                    add_seconds=t,
                    interval_id=interval_id,
                    now_s=now_s
                )
        finally:
            interval_q.task_done()

# --------------------------------------------------------------------------------------
# DETECTIVE LOOP (ADK agent)
# --------------------------------------------------------------------------------------
_last_alert_at: Dict[str, float] = {}

def fetch_active_entities(window_s: int = 40, limit: int = 100) -> List[Dict[str, Any]]:
    cutoff = time.time() - window_s
    q = (db.collection(COL_ENTITIES)
         .where("last_seen_ts", ">=", cutoff)
         .order_by("last_seen_ts", direction=firestore.Query.DESCENDING)
         .limit(limit))
    return [{**d.to_dict(), "doc_id": d.id} for d in q.stream()]

def build_prompt_json(entities: List[Dict[str, Any]]) -> str:
    items = []
    for e in entities:
        items.append({
            "entity_id": e.get("entity_id") or e.get("doc_id"),
            "consec_time_s": float(e.get("consec_time_s", 0.0)),
            "total_time_s": float(e.get("total_time_s", 0.0)),
            "recent_hits_ts": [int(t) for t in e.get("recent_hits_ts", [])],
            "true_reappear_count": int(e.get("true_reappear_count", 0)),
            "interval_appear_count": int(e.get("interval_appear_count", 0)),
        })
    return json.dumps({"entities": items}, separators=(",", ":"))

def parse_jsonl(text: str) -> List[Dict[str, Any]]:
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            out.append(json.loads(s))
        except json.JSONDecodeError:
            continue
    return out

async def run_detective_once(runner: InMemoryRunner, entities: List[Dict[str, Any]]):
    if not entities:
        return []
    prompt = build_prompt_json(entities)
    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    buf: List[str] = []
    async for ev in runner.run_async(
        user_id="cyclops_sys",
        session_id="main",
        new_message=content
    ):
        if getattr(ev, "content", None):
            for p in ev.content.parts or []:
                if getattr(p, "text", None):
                    buf.append(p.text)
    text = "".join(buf)
    return parse_jsonl(text)

def maybe_alert(verdicts: List[Dict[str, Any]]):
    now = time.time()
    for v in verdicts:
        if (v.get("verdict") or "").upper() != "SUSPICIOUS":
            continue
        eid = v.get("entity_id", "unknown")
        if now - _last_alert_at.get(eid, 0) < ENTITY_COOLDOWN_S:
            continue
        reason = v.get("reason", "Detected suspicious pattern")
        msg = f"High alert! {reason} (entity: {eid})"
        send_alert_to_topic(DEVICE_TOPIC, msg, severity="HIGH", cooldown_s=8)
        _last_alert_at[eid] = now
        print("🚨 suspicious entity:", eid, reason)

async def detective_loop():
    runner = InMemoryRunner(agent=detective_agent, app_name="cyclops")
    while True:
        ents = fetch_active_entities(window_s=REAPPEAR_WINDOW_S + 10, limit=100)
        verdicts = await run_detective_once(runner, ents)
        if verdicts:
            maybe_alert(verdicts)
        await asyncio.sleep(DETECT_PERIOD_S)

# --------------------------------------------------------------------------------------
# ULTRASONIC SENSOR LOOP (pySerial)
# --------------------------------------------------------------------------------------
def ultrasonic_worker():
    """Continuously read Arduino ultrasonic sensor values."""
    print("📡 Starting Ultrasonic Listener…")
    try:
        ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
    except Exception as e:
        print("⚠️ Could not open serial port:", e)
        return

    last_alert = 0
    while True:
        try:
            line = ser.readline().decode("utf-8").strip()
            if not line:
                continue
            try:
                distance = float(line)
            except ValueError:
                continue
            now = time.time()
            if distance <= THRESHOLD_M and now - last_alert > ARDUINO_COOLDOWN_S:
                msg = f"Object detected within {distance:.2f} meters!"
                send_alert_to_topic(DEVICE_TOPIC, msg, severity="CRITICAL")
                print("🚨 Ultrasonic Trigger:", msg)
                last_alert = now
        except Exception as e:
            print("⚠️ Ultrasonic read failed:", e)
            time.sleep(1)

# --------------------------------------------------------------------------------------
# MAIN ENTRYPOINT
# --------------------------------------------------------------------------------------
async def main():
    start_interval_listener()

    # run ultrasonic listener in a separate thread
    threading.Thread(target=ultrasonic_worker, daemon=True).start()

    await asyncio.gather(
        rollup_worker(),   # Firestore ingestion
        detective_loop(),  # ADK agent reasoning
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 Cyclops shutting down")
