# seed_sample_data.py
# Creates 10 intervals with realistic data + updates entities summaries.
# presence_windows uses your exact keys; entities holds rolling stats.

import os, time, random, datetime
import firebase_admin
from firebase_admin import credentials, firestore

# -------- Firebase init --------
KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service-account.json")
if not firebase_admin._apps:
    cred = credentials.Certificate(KEY_PATH)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# -------- Helpers --------
def to_dt(iso_z: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(iso_z.replace("Z", "+00:00"))

def iso_z(dt: datetime.datetime) -> str:
    return dt.replace(tzinfo=datetime.timezone.utc).isoformat().replace("+00:00", "Z")

# --- transactional entity updater (adds/resets consecutive time, increments totals) ---
@firestore.transactional
def _update_entity_tx(tx, ref, *, add_seconds: float, current_interval_id: int,
                      mark_true_reappear: bool):
    snap = ref.get(transaction=tx)
    now_s = int(time.time())
    if snap.exists:
        cur = snap.to_dict() or {}
        last_idx = cur.get("last_interval_id", -1)
        continuous = (last_idx == current_interval_id - 1)
        new_consec = (cur.get("consec_time_s", 0.0) + add_seconds) if continuous else float(add_seconds)
        updates = {
            "entity_id": ref.id,
            "last_seen_ts": now_s,
            "last_interval_id": current_interval_id,
            "consec_time_s": float(new_consec),
            "total_time_s": firestore.Increment(float(add_seconds)),
            "interval_appear_count": firestore.Increment(1),
        }
        if mark_true_reappear and not continuous:
            updates["true_reappear_count"] = firestore.Increment(1)
        tx.update(ref, updates)
    else:
        tx.set(ref, {
            "entity_id": ref.id,
            "last_seen_ts": now_s,
            "last_interval_id": current_interval_id,
            "consec_time_s": float(add_seconds),
            "total_time_s": float(add_seconds),
            "interval_appear_count": 1,
            "true_reappear_count": 1 if mark_true_reappear else 0
        })

def write_presence_window_exact(doc: dict):
    # Convert ISO times to Firestore Timestamps but keep your original field names
    win = dict(doc)
    win["start_time"] = to_dt(doc["start_time"])
    win["end_time"]   = to_dt(doc["end_time"])
    db.collection("presence_windows").add(win)

# -------- Generate 10 realistic intervals --------
def seed():
    # Simulated people: 1,2,3 with different behavior
    #  - id 1: mostly continuous presence
    #  - id 2: disappears for a couple intervals, then reappears (true reappear)
    #  - id 3: sporadic (on/off)
    start = datetime.datetime.utcnow()
    interval_len = 5  # seconds

    # track reappearance_counter per id (how many intervals they've appeared in)
    appear_counts = {1: 0, 2: 0, 3: 0}
    # track whether present in previous interval (for true reappear)
    prev_present = {1: False, 2: False, 3: False}

    for i in range(1, 11):  # interval_id: 1..10
        t0 = start + datetime.timedelta(seconds=(i-1)*interval_len)
        t1 = t0 + datetime.timedelta(seconds=interval_len)

        # Decide who appears this interval
        present = []
        # id 1: appears in almost all intervals
        if random.random() < 0.9:
            present.append(1)
        # id 2: absent in intervals 4-5, reappears after (true reappear)
        if i not in (4, 5):
            if random.random() < 0.75:
                present.append(2)
        # id 3: 50% chance each interval
        if random.random() < 0.5:
            present.append(3)

        # Build people_data with realistic "time in interval"
        people_data = []
        for pid in sorted(set(present)):
            appear_counts[pid] += 1
            # If they were present last interval, their time is high; first frame back might be shorter
            base = 3.5 if prev_present[pid] else 2.0
            dur = max(0.6, min(5.0, random.gauss(base, 0.8)))  # clamp 0.6..5.0
            people_data.append({
                "id": pid,
                "time": round(dur, 2),
                "reappearance_counter": appear_counts[pid]
            })

        # Write to presence_windows with your exact schema
        interval_doc = {
            "interval_id": i,
            "start_time": iso_z(t0),
            "end_time":   iso_z(t1),
            "people_data": people_data
        }
        write_presence_window_exact(interval_doc)

        # Update entities summaries per person present this interval
        for p in people_data:
            eid = f"e_{p['id']}"
            mark_true_reappear = (prev_present[p["id"]] is False and appear_counts[p["id"]] > 1)
            # special case: force true reappear for id 2 when coming back after gap
            if p["id"] == 2 and i == 6:
                mark_true_reappear = True
            tx = db.transaction()
            _update_entity_tx(tx, db.collection("entities").document(eid),
                              add_seconds=float(p["time"]),
                              current_interval_id=i,
                              mark_true_reappear=mark_true_reappear)

        # Update prev_present flags for next loop
        now_present = {1: False, 2: False, 3: False}
        for pid in present:
            now_present[pid] = True
        prev_present = now_present

        print(f"interval {i}: wrote {len(people_data)} people")

    print("\n✅ Seed complete. Check Firestore:")
    print("• presence_windows → 10 interval docs")
    print("• entities → docs e_1, e_2, e_3 with consec_time_s / total_time_s / counts")

if __name__ == "__main__":
    seed()
