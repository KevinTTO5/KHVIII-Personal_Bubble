"""
YOLOv11 + DeepSORT Person Tracker with Improved Person Differentiation
Focuses on correctly distinguishing between different people
"""

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pickle
import os
from scipy.spatial.distance import cosine

PERSON_DATA_FILE = "person_database.pkl"

class ImprovedPersonTracker:
    def __init__(self):
        print("Loading YOLOv11 model...")
        self.yolo_model = YOLO('yolo11n.pt')
        
        print("Initializing DeepSORT...")
        # Stricter matching to avoid confusing people
        self.tracker = DeepSort(
            max_age=20,
            n_init=3,  # Balanced confirmation
            max_iou_distance=0.5,  # STRICTER - boxes must overlap more to be same person
            max_cosine_distance=0.25,  # STRICTER - appearance must match closely
            nn_budget=None,
            embedder="mobilenet",
            half=False,
            bgr=True,
            embedder_gpu=False,
        )
        
        # Database
        self.person_database = {}
        self.temp_to_perm_id = {}
        self.next_perm_id = 1
        
        # Config - STRICTER thresholds to avoid mixing people
        self.max_features_per_person = 10  # Fewer features = more distinctive
        self.reidentification_threshold = 0.55 
        self.min_detection_confidence = 0.5  # Relaxed to detect people better
        
        # For tracking when people were last seen together
        self.people_seen_together = set()  # (id1, id2) pairs
        
        self.load_database()
        
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
        """Detect people using YOLOv11 - relaxed filtering"""
        results = self.yolo_model(frame, classes=[0], verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                # Simple validation - just confidence
                if confidence > self.min_detection_confidence:
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Very basic sanity check
                    if w > 20 and h > 40:  # Very minimal size requirement
                        detections.append(([x1, y1, w, h], confidence, 'person'))
        
        return detections
    
    def find_matching_person(self, feature_vector, current_people_in_frame):
        """
        Search database for matching person with STRICT matching
        Also checks if this person was seen with others currently in frame
        """
        if not self.person_database or feature_vector is None:
            return None, 0
        
        best_match_id = None
        best_similarity = 0
        
        for person_id, stored_features in self.person_database.items():
            if not stored_features:
                continue
            
            # Skip if this person is already in the current frame
            # (can't be two instances of same person)
            if person_id in current_people_in_frame:
                continue
                
            similarities = []
            for stored_feature in stored_features:
                try:
                    similarity = 1 - cosine(feature_vector, stored_feature)
                    similarities.append(similarity)
                except:
                    continue
            
            if similarities:
                # Use BEST match (not average) for stricter comparison
                max_similarity = max(similarities)
                
                if max_similarity > best_similarity:
                    best_similarity = max_similarity
                    best_match_id = person_id
        
        # STRICT threshold - only match if very confident
        if best_similarity > self.reidentification_threshold:
            return best_match_id, best_similarity
        
        return None, 0
    
    def add_feature_to_database(self, person_id, feature_vector):
        """Add appearance feature to database"""
        if feature_vector is None:
            return
            
        if person_id not in self.person_database:
            self.person_database[person_id] = []
        
        self.person_database[person_id].append(feature_vector)
        
        # Keep fewer features to maintain distinctiveness
        if len(self.person_database[person_id]) > self.max_features_per_person:
            self.person_database[person_id] = self.person_database[person_id][-self.max_features_per_person:]
    
    def check_spatial_conflict(self, new_bbox, existing_bboxes, threshold=0.3):
        """
        Check if new detection overlaps too much with existing ones
        Two people can't occupy the same space
        """
        x1, y1, x2, y2 = new_bbox
        area1 = (x2 - x1) * (y2 - y1)
        
        for existing_bbox in existing_bboxes:
            ex1, ey1, ex2, ey2 = existing_bbox
            
            # Calculate intersection
            ix1 = max(x1, ex1)
            iy1 = max(y1, ey1)
            ix2 = min(x2, ex2)
            iy2 = min(y2, ey2)
            
            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                area2 = (ex2 - ex1) * (ey2 - ey1)
                
                # If overlap is significant, reject
                iou = intersection / (area1 + area2 - intersection)
                if iou > threshold:
                    return True
        
        return False
    
    def track_people(self, frame, detections):
        """Track people with strict differentiation"""
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        tracked_people = []
        current_people_ids = []
        current_bboxes = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltrb()
            
            # Get feature
            try:
                feature_vector = track.get_feature()
            except:
                feature_vector = None
            
            # New track?
            if track_id not in self.temp_to_perm_id:
                # Check for spatial conflicts
                if self.check_spatial_conflict(bbox, current_bboxes, threshold=0.3):
                    print(f"⚠ Skipping overlapping detection (likely duplicate)")
                    continue
                
                # Try to match with existing person
                matched_id, similarity = self.find_matching_person(feature_vector, current_people_ids)
                
                if matched_id:
                    # Only accept if VERY confident
                    if similarity > 0.65:  # Extra strict for re-ID
                        self.temp_to_perm_id[track_id] = matched_id
                        print(f"✓ Re-identified Person {matched_id} (similarity: {similarity:.3f})")
                    else:
                        # Not confident enough - treat as new person
                        self.temp_to_perm_id[track_id] = self.next_perm_id
                        print(f"✓ New person: Person {self.next_perm_id} (similarity {similarity:.3f} too low)")
                        self.next_perm_id += 1
                else:
                    # New person
                    self.temp_to_perm_id[track_id] = self.next_perm_id
                    print(f"✓ New person: Person {self.next_perm_id}")
                    self.next_perm_id += 1
            
            perm_id = self.temp_to_perm_id[track_id]
            
            # Store feature only if high quality
            if feature_vector is not None:
                self.add_feature_to_database(perm_id, feature_vector)
            
            current_people_ids.append(perm_id)
            current_bboxes.append(bbox)
            
            tracked_people.append({
                'id': perm_id,
                'bbox': bbox,
            })
        
        # Record which people were seen together (helps prevent merging)
        if len(current_people_ids) >= 2:
            for i in range(len(current_people_ids)):
                for j in range(i + 1, len(current_people_ids)):
                    pair = tuple(sorted([current_people_ids[i], current_people_ids[j]]))
                    self.people_seen_together.add(pair)
        
        return tracked_people
    
    def draw_tracks(self, frame, tracked_people):
        """Draw bounding boxes and IDs"""
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
            
            # Draw thick box
            cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 3)
            
            # Draw label with feature count
            label = f"Person {person_id}"
            feature_count = len(self.person_database.get(person_id, []))
            sublabel = f"({feature_count} features)"
            
            # Main label background
            label_bg_y = max(y1 - 50, 50)
            cv2.rectangle(frame, (x1, label_bg_y), (x1 + 200, y1), person_color, -1)
            
            # Text
            cv2.putText(frame, label, (x1 + 5, y1 - 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, sublabel, (x1 + 5, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def reset(self):
        """Reset everything"""
        self.person_database = {}
        self.temp_to_perm_id = {}
        self.next_perm_id = 1
        self.people_seen_together = set()
        if os.path.exists(PERSON_DATA_FILE):
            os.remove(PERSON_DATA_FILE)
        print("✓ Database reset")


def main():
    tracker = ImprovedPersonTracker()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n=== Improved Person Differentiation Tracker ===")
    print("Features:")
    print("  • Stricter matching to avoid mixing people")
    print("  • Spatial conflict detection")
    print("  • Tracks who was seen together")
    print("  • Higher re-identification threshold")
    print("\nControls:")
    print("  'q' - Quit and save")
    print("  'r' - Reset all person IDs")
    print("  's' - Save database manually")
    print("  't' - Toggle strict mode (adjust threshold)")
    print("===============================================\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("error with cam")
            break
        
        frame_count += 1
        
        # Detect and track
        detections = tracker.detect_people(frame)
        tracked_people = tracker.track_people(frame, detections)
        
        # Draw
        frame = tracker.draw_tracks(frame, tracked_people)
        
        # Info overlay
        total = len(tracker.person_database)
        active = len(tracked_people)
        threshold = tracker.reidentification_threshold
        
        cv2.putText(frame, f"Total: {total} | Active: {active}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"ReID Threshold: {threshold:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Improved Tracker', frame)
        
        # Auto-save
        if frame_count % 150 == 0:
            tracker.save_database()
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset()
        elif key == ord('s'):
            tracker.save_database()
            print(f"✓ Saved ({total} people)")
        elif key == ord('t'):
            # Toggle between strict and very strict
            if tracker.reidentification_threshold == 0.65:
                tracker.reidentification_threshold = 0.75
                print("Switched to VERY STRICT mode (threshold: 0.75)")
            else:
                tracker.reidentification_threshold = 0.65
                print("Switched to STRICT mode (threshold: 0.65)")
    
    # Final save
    tracker.save_database()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Session complete: {len(tracker.person_database)} people tracked")


if __name__ == "__main__":
    main()