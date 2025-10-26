# orchestrator_adk_detective.py
# End-to-end: Firestore intervals -> entities rollup + ADK detective loop + FCM alerts.
from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
import asyncio
import queue
from typing import Any, Dict, List

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import FieldFilter

from google.genai import Client
from google.genai import types
from google.adk.sessions import InMemorySessionService

from tools_alert import send_alert_to_topic  # same dir; provides FCM sending

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
SERVICE_ACCOUNT = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service-account.json")
DEVICE_TOPIC = os.getenv("CYCLOPS_TOPIC", "user_1")   # Android app subscribes to this
REAPPEAR_WINDOW_S = int(os.getenv("REAPPEAR_WINDOW_S", "30"))
DETECT_PERIOD_S = int(os.getenv("DETECT_PERIOD_S", "5"))

# thresholds
CONSEC_THRESHOLD_S = float(os.getenv("CONSEC_THRESHOLD_S", "30.0"))
TOTAL_TIME_THRESHOLD_S = float(os.getenv("TOTAL_TIME_THRESHOLD_S", "120.0"))
REAPPEAR_THRESHOLD = int(os.getenv("REAPPEAR_THRESHOLD", "3"))

# entity alert cooldown (seconds)
ENTITY_COOLDOWN_S = int(os.getenv("ENTITY_COOLDOWN_S", "10"))

# Firestore collections
COL_PRESENCE_WINDOW = "presence_windows"
COL_ENTITIES = "entities"

GENAI_MODEL = os.getenv("CYCLOPS_MODEL", "gemini-2.5-flash")
genai_client = Client(api_key=os.getenv("GOOGLE_API_KEY"))

# --------------------------------------------------------------------------------------
# Firebase init
# --------------------------------------------------------------------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# --------------------------------------------------------------------------------------
# ADK Detective Agent (LLM)
# --------------------------------------------------------------------------------------
from google.genai import Client

genai_client = Client(api_key=os.getenv("GOOGLE_API_KEY"))

def run_detective_direct(entities):
    if not entities:
        return []

    prompt_json = build_prompt_json(entities)
    model = os.getenv("CYCLOPS_MODEL", "gemini-1.5-flash")

    system_instr = (
    "You are Cyclops Detective. You receive a JSON object in the following form:\n"
    '{"entities":[{"entity_id":"e_3","consec_time_s":12.3,"total_time_s":55.0,'
    '"recent_hits_ts":[1700000000,1700000050],'
    '"interval_appear_count":7}, ...]}\n\n"'
    "Follow these rules to analyze each entity:\n"
    "1. Ignore and completely skip any entity that appears to be noise:\n"
    "   - If an entity has more than 4 appearances within any 2-second window "
    "(based on recent_hits_ts), do not include it in the output at all.\n"
    "2. For all remaining entities, apply these fixed detection rules:\n"
    "   - Mark as SUSPICIOUS if consec_time_s >= 30.0\n"
    "   - Mark as SUSPICIOUS if (appearances in the last ~30 seconds minus 1) >= 3\n"
    "   - Mark as SUSPICIOUS if total_time_s >= 120.0\n"
    "   - Otherwise, mark as NOT_SUSPICIOUS.\n\n"
    "Return strict JSON Lines output (one line per analyzed entity), with no extra text:\n"
    '{"entity_id":"e_1","verdict":"SUSPICIOUS|NOT_SUSPICIOUS","reason":"<short>"}\n\n'
    "If an entity was ignored due to excessive duplicate detections, omit it entirely.\n"
    "If information is insufficient, default to NOT_SUSPICIOUS."
)


    # Use chats.create to interact with Gemini
    chat = genai_client.chats.create(model=model)
    response = chat.send_message(
        f"{system_instr}\n\nNow analyze:\n{prompt_json}"
    )

    # Extract plain text output
    text = response.text if hasattr(response, "text") else str(response)
    return parse_jsonl(text)


# --------------------------------------------------------------------------------------
# Interval listener -> rollup into entities
# --------------------------------------------------------------------------------------
interval_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=500)

def _entity_doc_id(pid: int) -> str:
    return f"e_{int(pid)}"

@firestore.transactional
def _update_entity_tx(tx, ref, *, add_seconds: float, interval_id: int, now_s: int):
    """Transactional updater for one entity record."""
    snap = ref.get(transaction=tx)
    if snap.exists:
        cur = snap.to_dict() or {}
        last_idx = cur.get("last_interval_id", -1)
        continuous = (last_idx == interval_id - 1)

        new_consec = (cur.get("consec_time_s", 0.0) + add_seconds) if continuous else float(add_seconds)

        # Keep a rolling window of recent hits (for reappearance rule)
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

        tx.update(ref, updates)
    else:
        tx.set(ref, {
            "entity_id": ref.id,
            "last_seen_ts": now_s,
            "last_interval_id": interval_id,
            "consec_time_s": float(add_seconds),
            "total_time_s": float(add_seconds),
            "interval_appear_count": 1,
            "recent_hits_ts": [now_s]
        })

def _on_intervals_snapshot(col_snapshot, changes, read_time):
    for ch in changes:
        if ch.type.name != "ADDED":
            continue
        interval_q.put(ch.document.to_dict() or {})

def start_interval_listener():
    db.collection(COL_PRESENCE_WINDOW).on_snapshot(_on_intervals_snapshot)
    print(f"👂 Listening to '{COL_PRESENCE_WINDOW}/' …")

async def rollup_worker():
    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, interval_q.get)
        try:
            interval_id = int(item.get("interval_id"))
            now_s = int(time.time())

            # Optional audit to a different collection ONLY:
            # db.collection(COL_AUDIT).add({**item, "server_ts": firestore.SERVER_TIMESTAMP, "origin": "orchestrator"})

            for p in item.get("people_data", []):
                t = float(p.get("time", 0.0))
                if t <= 0:
                    continue
                eid = f"e_{int(p.get('id'))}"
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
# ADK detective loop (every 5s)
# --------------------------------------------------------------------------------------
_last_alert_at: Dict[str, float] = {}

def fetch_active_entities(window_s: int = 40, limit: int = 100) -> List[Dict[str, Any]]:
    """Pull recently-seen entities to reduce read cost."""
    cutoff = time.time() - window_s
    return [
        {**d.to_dict(), "doc_id": d.id}
        for d in (db.collection(COL_ENTITIES)
                    .where(filter=FieldFilter("last_seen_ts", ">=", cutoff))
                    .order_by("last_seen_ts", direction=firestore.Query.DESCENDING)
                    .limit(limit)
                    .stream())
    ]

def build_prompt_json(entities: List[Dict[str, Any]]) -> str:
    """Pack entities into one compact JSON for the agent."""
    items = []
    for e in entities:
        items.append({
            "entity_id": e.get("entity_id") or e.get("doc_id"),
            "consec_time_s": float(e.get("consec_time_s", 0.0)),
            "total_time_s": float(e.get("total_time_s", 0.0)),
            "recent_hits_ts": [int(t) for t in e.get("recent_hits_ts", [])],
            "interval_appear_count": int(e.get("interval_appear_count", 0)),
        })
    return json.dumps({"entities": items}, separators=(",", ":"))

def parse_jsonl(text: str) -> List[Dict[str, Any]]:
    """Parse strict JSON Lines from the agent."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            out.append(json.loads(s))
        except json.JSONDecodeError:
            # ignore malformed lines
            continue
    return out


def maybe_alert(verdicts: List[Dict[str, Any]]):
    now = time.time()

    # Custom message mapping based on the reason string
    def interpret_reason(reason: str) -> str:
        reason = reason.lower()
        if "consec" in reason or "consecutive" in reason:
            return "High Alert! Someone has been behind you for more than 30 consecutive seconds!"
        elif "total" in reason or "120" in reason or "2 minute" in reason:
            return "Alert! Someone has been following you for a total of 2 minutes or more!"
        else:
            return "High Alert! Suspicious behavior detected!"

    for v in verdicts:
        verdict = (v.get("verdict") or "").upper()
        if verdict != "SUSPICIOUS":
            continue

        eid = v.get("entity_id", "unknown")
        if now - _last_alert_at.get(eid, 0) < ENTITY_COOLDOWN_S:
            continue

        reason = v.get("reason", "")
        msg = interpret_reason(reason)
        res = send_alert_to_topic(DEVICE_TOPIC, msg, severity="HIGH", cooldown_s=8)
        _last_alert_at[eid] = now

        print(f"🚨 alert: {eid} | {msg} | Raw reason: {reason} | Result: {res}")

async def detective_loop_adk():
    while True:
        ents = fetch_active_entities(window_s=REAPPEAR_WINDOW_S + 10, limit=100)
        verdicts = run_detective_direct(ents)   # <-- direct call, no async, no sessions
        if verdicts:
            maybe_alert(verdicts)
        await asyncio.sleep(DETECT_PERIOD_S)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
async def main():
    start_interval_listener()
    await asyncio.gather(
        rollup_worker(),       # intervals -> entities
        detective_loop_adk(),  # ADK reasoning + alerts
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("bye")
