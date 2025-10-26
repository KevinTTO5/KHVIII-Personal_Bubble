import math
import time
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import serial
from scipy.spatial.distance import cosine
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pickle
import os
import json
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore, messaging
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# -------- CONFIGURATION --------
SER = None          # serial handle
SER_PORT = "/dev/tty.usbmodem101"   # set your COM port here or leave to auto-probe
_last_line = ""     # last serial line

last_alert_time = 0

LAST_REAL_R = None
LAST_REAL_TS = 0.0
MAX_STALE_SEC = 1.0
alpha_r = 0.4
ema_r = None
use_sim = False
sim_r = 1.8

PERSON_DATA_FILE = "person_database.pkl"

# ------------ Your geometry and other low-level functions here (normalize, etc.) ------------
# Include them as needed for your scoring if you like.

class ImprovedPersonTracker:
    def __init__(self, interval_duration=5.0):
        # Device selection (CUDA > MPS > CPU) and fp16 toggle
        try:
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.use_half = True
                print("using cuda and fp16 weights")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
                self.use_half = False
            else:
                self.device = 'cpu'
                self.use_half = False
                print("using cpu, no quantization")
        except Exception:
            self.device = 'cpu'
            self.use_half = False

        print("Loading YOLOv11 model...")
        self.yolo_model = YOLO('yolo11s.pt')
        print("loaded yolo with cuda")
        # Move model to device and enable fp16 when safe
        try:
            self.yolo_model.to(self.device)
            self.use_half = (self.device == 'cuda')

            if self.use_half:
                major, minor = torch.cuda.get_device_capability()
                if major < 7:
                    self.use_half = False
        except Exception as e:
            print(f"Warning: could not place YOLO on {self.device}: {e}")
            self.device = 'cpu'
            self.use_half = False
        
        print("Initializing DeepSORT...")
        self.tracker = DeepSort(
            max_age=20,
            n_init=3,
            max_iou_distance=0.5,
            max_cosine_distance=0.25,
            nn_budget=None,
            embedder="mobilenet",
            half=False,
            bgr=True,
            embedder_gpu=(self.device == 'cuda'),
        )
        if (self.device == 'cuda'):
            print("deepsort vision embedder is using cuda")
        else:
            print("deepsort vision embedder is not using cuda")
        
        # Database
        self.person_database = {}
        self.temp_to_perm_id = {}
        self.next_perm_id = 1
        self.max_features_per_person = 30  # Fewer features keep identities distinct
        self.reidentification_threshold = 0.45
        self.strict_accept_similarity = 0.65  # Extra-strict gate for re-ID acceptance
        self.min_detection_confidence = 0.5
        self.people_seen_together = set()
        
        
        # Interval tracking
        self.interval_duration = interval_duration
        self.interval_start_time = time.time()
        self.interval_id = 1
        self.current_interval_data = {}
        self.completed_intervals = []
        
        self.load_database()
        
        # Initialize Firestore (absolute creds path)
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate('/Users/iaddchehaeb/Documents/GitHub/KHVIII-Personal_Bubble/service-account.json') #CHANGE THIS DEPENDING ON YOUR SYSTEM
                firebase_admin.initialize_app(cred)
            self.db = firestore.client()
        except Exception as e:
            print(f"Error initializing Firestore: {e}")
            self.db = None
        
    def load_database(self):
        if os.path.exists(PERSON_DATA_FILE):
            try:
                with open(PERSON_DATA_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.person_database = data['database']
                    self.temp_to_perm_id = data.get('mappings', {})
                    self.next_perm_id = data['next_id']
                    self.people_seen_together = data.get('seen_together', set())
                print(f"✓ Loaded database with {len(self.person_database)} people")
            except Exception as e:
                print(f"Error loading database: {e}")
    
    def save_database(self):
        try:
            with open(PERSON_DATA_FILE, 'wb') as f:
                pickle.dump({
                    'database': self.person_database,
                    'mappings': self.temp_to_perm_id,
                    'next_id': self.next_perm_id,
                    'seen_together': self.people_seen_together,
                }, f)
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def detect_people(self, frame):
        """Detect people using YOLOv11"""
        results = self.yolo_model(
            frame,
            classes=[0],
            verbose=False,
            device=self.device,
            half=self.use_half,
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                if confidence > self.min_detection_confidence:
                    w = x2 - x1
                    h = y2 - y1
                    
                    if w > 20 and h > 40:
                        detections.append(([x1, y1, w, h], confidence, 'person'))
        
        return detections
    
    def find_matching_person(self, feature_vector, current_people_in_frame):
        """Strict appearance-only matching using cosine similarity (best-of set)."""
        if not self.person_database or feature_vector is None:
            return None, 0.0
        
        best_match_id = None
        best_similarity = 0.0
        
        for person_id, stored_features in self.person_database.items():
            if not stored_features:
                continue
            if person_id in current_people_in_frame:
                continue
            
            max_similarity_for_person = 0.0
            for stored_feature in stored_features:
                try:
                    similarity = 1 - cosine(feature_vector, stored_feature)
                except Exception:
                    continue
                if similarity > max_similarity_for_person:
                    max_similarity_for_person = similarity
            
            if max_similarity_for_person > best_similarity:
                best_similarity = max_similarity_for_person
                best_match_id = person_id
        
        if best_similarity > self.reidentification_threshold:
            return best_match_id, best_similarity
        return None, 0.0
    
    def add_feature_to_database(self, person_id, feature_vector):
        """Add appearance feature to database"""
        if feature_vector is None:
            return
            
        if person_id not in self.person_database:
            self.person_database[person_id] = []
        
        self.person_database[person_id].append(feature_vector)
        
        if len(self.person_database[person_id]) > self.max_features_per_person:
            self.person_database[person_id] = self.person_database[person_id][-self.max_features_per_person:]
    
    def check_spatial_conflict(self, new_bbox, existing_bboxes, threshold=0.3):
        """Check if new detection overlaps too much with existing ones"""
        if not existing_bboxes:
            return False
            
        x1, y1, x2, y2 = new_bbox
        area1 = (x2 - x1) * (y2 - y1)
        
        for existing_bbox in existing_bboxes:
            ex1, ey1, ex2, ey2 = existing_bbox
            
            ix1 = max(x1, ex1)
            iy1 = max(y1, ey1)
            ix2 = min(x2, ex2)
            iy2 = min(y2, ey2)
            
            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                area2 = (ex2 - ex1) * (ey2 - ey1)
                
                iou = intersection / (area1 + area2 - intersection)
                if iou > threshold:
                    return True
        
        return False
    
    def update_interval_tracking(self, tracked_people, frame_duration):
        """Update interval tracking with proper reappearance detection"""
        people_in_frame = set(person['id'] for person in tracked_people if person['id'] > 0)
        
        for person in tracked_people:
            person_id = person['id']
            
            if person_id not in self.current_interval_data:
                self.current_interval_data[person_id] = {
                    'time': 0.0,
                    'reappearance_counter': 1,
                    'currently_visible': True
                }
            else:
                if not self.current_interval_data[person_id]['currently_visible']:
                    self.current_interval_data[person_id]['reappearance_counter'] += 1
                
                self.current_interval_data[person_id]['currently_visible'] = True
            
            self.current_interval_data[person_id]['time'] += frame_duration
        
        for person_id in self.current_interval_data:
            if person_id not in people_in_frame:
                self.current_interval_data[person_id]['currently_visible'] = False
    
    def check_and_finalize_interval(self):
        """Check if interval should end and finalize it"""
        current_time = time.time()
        elapsed = current_time - self.interval_start_time
        
        if elapsed >= self.interval_duration:
            self.finalize_interval(elapsed)
            return True
        return False
    
    def finalize_interval(self, actual_duration):
        """Create complete interval JSON and reset"""
        end_time = time.time()
        
        interval_json = {
            "interval_id": self.interval_id,
            "start_time": datetime.fromtimestamp(self.interval_start_time).isoformat() + "Z",
            "end_time": datetime.fromtimestamp(end_time).isoformat() + "Z",
            "duration": round(actual_duration, 2),
            "people_data": []
        }
        
        for person_id, data in self.current_interval_data.items():
            person_json = {
                "id": person_id,
                "time": round(min(data['time'], self.interval_duration), 2),
                "reappearance_counter": data['reappearance_counter']
            }
            interval_json["people_data"].append(person_json)
        
        interval_json["people_data"].sort(key=lambda x: x['id'])
        self.completed_intervals.append(interval_json)
        
        # Write interval to Firestore
        if getattr(self, 'db', None) is not None:
            try:
                self.db.collection('presence_windows').add(interval_json)
            except Exception as e:
                print(f"Error writing interval to Firestore: {e}")
        
        # Verbose printing
        print("\n" + "="*70)
        print(f"INTERVAL {self.interval_id} COMPLETED")
        print("="*70)
        print(json.dumps(interval_json, indent=2))
        
        if interval_json["people_data"]:
            print(f"\n📊 Summary: {len(interval_json['people_data'])} people detected")
            for person in interval_json["people_data"]:
                reapp_indicator = "🚨 CIRCLING" if person['reappearance_counter'] >= 3 else ""
                print(f"   Person {person['id']}: {person['time']}s visible, "
                      f"{person['reappearance_counter']} appearances {reapp_indicator}")
        else:
            print("\n📊 Summary: No people detected this interval")
        
        print("="*70 + "\n")
        
        self.interval_id += 1
        self.interval_start_time = time.time()
        self.current_interval_data = {}
    
    def track_people(self, frame, detections, frame_duration):
        """Track people with strict immediate ID assignment and spatial conflict check."""
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        tracked_people = []
        current_people_ids = []
        current_bboxes = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltrb()
            
            try:
                feature_vector = track.get_feature()
            except:
                feature_vector = None
            
            # New track? Assign immediately using strict appearance similarity
            if track_id not in self.temp_to_perm_id:
                # Avoid duplicate overlapping detections in same frame
                if self.check_spatial_conflict(bbox, current_bboxes, threshold=0.3):
                    continue
                
                matched_id, similarity = self.find_matching_person(feature_vector, current_people_in_frame=current_people_ids)
                if matched_id and similarity > self.strict_accept_similarity:
                    self.temp_to_perm_id[track_id] = matched_id
                    print(f"✓ Re-identified Person {matched_id} (similarity: {similarity:.3f})")
                else:
                    self.temp_to_perm_id[track_id] = self.next_perm_id
                    print(f"✓ New person: Person {self.next_perm_id}{'' if not matched_id else f' (similarity {similarity:.3f} too low)'}")
                    self.next_perm_id += 1
            
            perm_id = self.temp_to_perm_id[track_id]
            
            # Store feature for this person
            if feature_vector is not None:
                self.add_feature_to_database(perm_id, feature_vector)
            
            current_people_ids.append(perm_id)
            current_bboxes.append(bbox)
            
            tracked_people.append({
                'id': perm_id,
                'bbox': bbox,
            })
        
        # Record co-occurrences
        if len(current_people_ids) >= 2:
            for i in range(len(current_people_ids)):
                for j in range(i + 1, len(current_people_ids)):
                    pair = tuple(sorted([current_people_ids[i], current_people_ids[j]]))
                    self.people_seen_together.add(pair)
        
        self.update_interval_tracking(tracked_people, frame_duration)
        
        return tracked_people
    
    def draw_tracks(self, frame, tracked_people):
        """Draw bounding boxes with interval stats"""
        for person in tracked_people:
            bbox = person['bbox']
            person_id = person['id']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Unique color per person
            color_seed = person_id * 50
            person_color = (
                (color_seed * 67) % 256,
                (color_seed * 131) % 256,
                (color_seed * 199) % 256
            )
            label = f"Person {person_id}"
            
            if person_id in self.current_interval_data:
                interval_time = self.current_interval_data[person_id]['time']
                reappearances = self.current_interval_data[person_id]['reappearance_counter']
                sublabel = f"{interval_time:.1f}s | {reappearances} app"
                if reappearances >= 3:
                    sublabel += " 🚨"
            else:
                sublabel = "Tracking"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 2)
            
            label_bg_y = max(y1 - 50, 50)
            cv2.rectangle(frame, (x1, label_bg_y), (x1 + 220, y1), person_color, -1)
            
            cv2.putText(frame, label, (x1 + 5, y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, sublabel, (x1 + 5, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def reset(self):
        """Reset everything"""
        self.person_database = {}
        self.temp_to_perm_id = {}
        self.next_perm_id = 1
        self.people_seen_together = set()
        self.current_interval_data = {}
        self.completed_intervals = []
        self.interval_id = 1
        self.interval_start_time = time.time()
        if os.path.exists(PERSON_DATA_FILE):
            os.remove(PERSON_DATA_FILE)
        print("✓ Database and intervals reset")

# --------- Ultrasonic I/O ---------
def _open_serial_once():
    global SER, SER_PORT
    if serial is None:
        return
    if SER and SER.is_open:
        return
    ports_to_try = [SER_PORT] if SER_PORT else []
    ports_to_try += [f"COM{i}" for i in range(3, 12)]
    for p in ports_to_try:
        if not p:
            continue
        try:
            SER = serial.Serial(p, 9600, timeout=0.05)
            SER_PORT = p
            time.sleep(0.1)
            print(f"[Serial] Connected on {p}")
            return
        except Exception:
            continue

def get_ultrasonic_reading():
    global SER, _last_line
    try:
        _open_serial_once()
        if not SER or not SER.is_open:
            return None
        line = SER.readline().decode(errors='ignore').strip()
        if not line:
            return None
        _last_line = line
        m = re.search(r"([0-9]*\.?[0-9]+)", line)
        if not m:
            return None
        value = float(m.group(1))
        if "cm" in line.lower():
            return value / 100.0
        if " m" in line.lower():
            return value
        return value / 100.0
    except Exception:
        try:
            if SER:
                SER.close()
        except Exception:
            pass
        SER = None
        return None

# --------- ImprovedPersonTracker class here as you provided --------
# (Include your entire ImprovedPersonTracker class definition from your code with no changes)


def main():
    global LAST_REAL_R, LAST_REAL_TS, MAX_STALE_SEC, ema_r, use_sim, sim_r

    tracker = ImprovedPersonTracker(interval_duration=5.0)
    
    cap = cv2.VideoCapture(0)  # Remove second param if on Mac
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("🎥 Starting STRICT appearance-based tracking...")
    print(f"⏱️  Interval: {tracker.interval_duration}s")
    print("Controls: q=quit | r=reset | s=save | p=print intervals\n")
    
    frame_count = 0
    last_frame_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading frame")
                break
            
            current_time = time.time()
            frame_duration = current_time - last_frame_time
            last_frame_time = current_time
            
            frame_count += 1
            
            # Read ultrasonic sensor
            now = time.time()
            r_new = get_ultrasonic_reading()
            if r_new is not None:
                LAST_REAL_R = r_new
                LAST_REAL_TS = now
            
            if use_sim:
                r_meas = sim_r
            else:
                if LAST_REAL_R is not None and (now - LAST_REAL_TS) <= MAX_STALE_SEC:
                    r_meas = LAST_REAL_R
                else:
                    r_meas = ema_r if ema_r is not None else sim_r
            
            if ema_r is None:
                ema_r = r_meas
            else:
                ema_r = alpha_r * r_meas + (1 - alpha_r) * ema_r
            
            def alert_action(distance):
                global last_alert_time
                alert_cooldown_sec = 10
                current_time = time.time()
                if (current_time - last_alert_time) > alert_cooldown_sec:
                    TOPIC = "user_1"                 # if your app subscribed to "user_1"
                    DEVICE_TOKEN = None              # or paste a real FCM registration token string

                # ---- 3) build a high-priority DATA message (your service expects this) ----
                    msg = messaging.Message(
                        data={
                            "type": "ALERT",
                            "msg": f"Cyclops: alert — person {distance: .2f}m!"
                        },
                        android=messaging.AndroidConfig(priority="normal"),
                        topic=TOPIC if not DEVICE_TOKEN else None,
                        token=DEVICE_TOKEN if DEVICE_TOKEN else None,
                    )
                    resp = messaging.send(msg, app=firebase_admin.get_app())
                    print("✅ Sent. Message ID:", resp)
                    last_alert_time = current_time 
            
            # In your loop:
            if ema_r < 1.0:
                alert_action(ema_r)
            
            # Detect and track people
            detections = tracker.detect_people(frame)
            tracked_people = tracker.track_people(frame, detections, frame_duration)
            tracker.check_and_finalize_interval()
            frame = tracker.draw_tracks(frame, tracked_people)
            
            # Draw ultrasonic distance in top-left corner
            if ema_r is not None:
                dist_text = f"Ultrasonic dist: {ema_r:.2f} m"
                cv2.putText(frame, dist_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
            
            cv2.imshow('Strict Re-ID Tracker with Ultrasonic Distance', frame)
            
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('r'):
                tracker.reset()
            elif key == ord('s'):
                tracker.save_database()
                print(f"💾 Saved")
            elif key == ord('p'):
                print("\n" + "="*70)
                print("ALL COMPLETED INTERVALS")
                print("="*70)
                print(json.dumps(tracker.completed_intervals, indent=2))
                print("="*70 + "\n")
            elif key == ord('u'):
                use_sim = not use_sim
            elif key == ord('['):
                sim_r = max(0.3, sim_r - 0.05)
            elif key == ord(']'):
                sim_r = min(6.0, sim_r + 0.05)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
    
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("🧹 Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        if SER and SER.is_open:
            SER.close()
        print(f"\n✅ Complete: {len(tracker.person_database)} people, {len(tracker.completed_intervals)} intervals")


if __name__ == "__main__":
    main()
