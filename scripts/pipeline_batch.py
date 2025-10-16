import cv2
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from ultralytics import YOLO
from ultralytics.engine.results import Boxes # <-- NEW IMPORT
import torch
import numpy as np 
import psycopg2
import yaml
import json
from pathlib import Path

# --- Global State Variables for Dynamic Interaction (Accessible by Mouse Callback) ---
WINDOW_NAME = 'Count_Vehicles'
drawing_line = False
temp_start_point = None
current_mouse_pos = None 
count_line_coords_list = [] 
selected_line_index = -1
frame_for_callback = None
class_names = {} 
# Variables to store the actual IDs for correct filtering and remapping
MOTORCYCLE_COCO_ID = None
NEW_MOTORCYCLE_ID = None
BEST_EXCLUDED_ID = None # ไม่ตัดคลาสใด ๆ โดยค่าเริ่มต้น

# ROI persistence state (will be used by main loop to persist to DB)
roi_dirty = False

# --- Helper Functions for Line Selection (Unchanged) ---

def is_point_near_line(point, line_coords, tolerance=7):
    """
    Checks if a given point (x, y) is close to the line segment defined by line_coords (x1, y1, x2, y2).
    """
    x, y = point
    x1, y1, x2, y2 = line_coords
    
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx*dx + dy*dy
    
    if len_sq == 0:
        dist_sq = (x - x1)**2 + (y - y1)**2
        return dist_sq <= tolerance**2

    t = ((x - x1) * dx + (y - y1) * dy) / len_sq
    
    if t < 0.0:
        closest_x, closest_y = x1, y1
    elif t > 1.0:
        closest_x, closest_y = x2, y2
    else:
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
    dist = np.sqrt((x - closest_x)**2 + (y - closest_y)**2)
    
    return dist <= tolerance

def handle_mouse_event(event, x, y, flags, param):
    """
    Mouse callback function to handle drawing and selecting lines dynamically.
    """
    global drawing_line, temp_start_point, count_line_coords_list, selected_line_index, current_mouse_pos

    current_mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        found_selection = False
        
        for i, line_coords in enumerate(count_line_coords_list):
            if is_point_near_line(clicked_point, line_coords):
                if selected_line_index == i:
                    selected_line_index = -1 
                    print(f"Line {i} deselected.")
                else:
                    selected_line_index = i
                    print(f"Line {i} selected for deletion. Press Backspace/Delete to remove.")
                
                found_selection = True
                drawing_line = False
                temp_start_point = None
                return
        
        if found_selection:
            return 

        selected_line_index = -1 

        if not drawing_line:
            temp_start_point = (x, y)
            drawing_line = True
            print("Start point set (P1). Click again for end point (P2).")
        else:
            temp_end_point = (x, y)
            line = (int(temp_start_point[0]), int(temp_start_point[1]), int(temp_end_point[0]), int(temp_end_point[1]))
            count_line_coords_list.append(line)
            # mark ROI as changed; main loop will persist to DB if enabled
            global roi_dirty
            roi_dirty = True
            drawing_line = False
            temp_start_point = None
            print(f"New line added: {line}")

# --- Counter Class (Unchanged) ---
class Counter:
    """
    Counts objects that cross a predefined line, regardless of direction.
    """
    def __init__(self, count_line_coords_list, roi_coords=None):
        self.count_line_coords_list = count_line_coords_list
        self.roi_coords = roi_coords
        self.tracking_history = {}
        self.class_counts = {}
        
    def _is_line_crossed(self, p1, p2, p3, p4):
        """
        Helper function to check if line p1p2 intersects with line p3p4.
        """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def update(self, detections):
        """
        Updates tracking and returns only detections that crossed any count line (once per track).
        """
        if not detections:
            return []
        crossed_detections = []

        for det in detections:
            # Ensure the detection has an ID (tracked object)
            if det.id is None:
                continue

            track_id = int(det.id.item())
            class_id = int(det.cls.item())
            
            x1, y1, x2, y2 = det.xyxy[0].int().tolist()
            current_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if track_id in self.tracking_history:
                previous_center = self.tracking_history[track_id]

                # Check if the object has crossed any of the counting lines
                for line_coords in self.count_line_coords_list:
                    p3 = (line_coords[0], line_coords[1])
                    p4 = (line_coords[2], line_coords[3])
                    
                    if self._is_line_crossed(previous_center, current_center, p3, p4):
                        if track_id not in self.class_counts:
                            self.class_counts[track_id] = class_id 
                            crossed_detections.append(det)
                            break 

            # Update the history for the next frame
            self.tracking_history[track_id] = current_center
        return crossed_detections

# --- Export and Utility Functions (Unchanged) ---

def rescale_coords(coords, original_width, original_height, target_width, target_height):
    """
    Rescales coordinates.
    """
    if coords is None:
        return None
    
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    
    new_coords = []
    for line_coords in coords:
        x1, y1, x2, y2 = line_coords
        new_x1 = int(x1 * scale_x)
        new_y1 = int(y1 * scale_y)
        new_x2 = int(x2 * scale_x)
        new_y2 = int(y2 * scale_y)
        new_coords.append((new_x1, new_y1, new_x2, new_y2))
    
    return new_coords

def export_to_csv(counts_by_interval, interval_timestamps, class_names, output_folder):
    """
    Exports the counting data to a single CSV file with real-time timestamps.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"class_counts_{timestamp}.csv"
    
    full_path = os.path.join(output_folder, output_filename)
    
    # Use unique column headers to avoid duplicate class names (e.g., motorcycle from two models)
    sorted_class_pairs = [(cid, class_names[cid]) for cid in sorted(class_names.keys())]
    data = {'Interval_Minutes': []}
    header_by_cid = {}
    for cid, name in sorted_class_pairs:
        header = f"{name} (id={cid})"
        header_by_cid[cid] = header
        data[header] = []
    
    for i, (start_time, end_time) in enumerate(interval_timestamps):
        start_dt = datetime.fromtimestamp(start_time)
        end_dt = datetime.fromtimestamp(end_time)
        data['Interval_Minutes'].append(f"{start_dt.strftime('%H:%M:%S')} - {end_dt.strftime('%H:%M:%S')}")
        
        class_counts = counts_by_interval.get(i, {})
        for class_id in sorted(class_names.keys()):
            header = header_by_cid[class_id]
            count = class_counts.get(class_id, 0)
            data[header].append(count)

    df = pd.DataFrame(data)
    df.to_csv(full_path, index=False)
    print(f"Class counting data exported to {full_path}")

# --- Database helpers (PostgreSQL) ---

def _load_db_config_from_yaml():
    """
    Load database configuration from configs/config.yaml
    """
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
        # when running as script from project root, fallback path
        if not os.path.exists(config_path):
            config_path = os.path.join('configs', 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return cfg.get('database', {})
    except Exception as e:
        print(f"❌ Cannot load DB config: {e}")
        return {}


def _get_db_connection():
    """
    Get a PostgreSQL connection. Prefer central connector; fallback to YAML config.
    """
    # Try central db_config.get_connection first to match working settings
    try:
        from database.db_config import get_connection as central_get_connection
        conn = central_get_connection()
        if conn:
            return conn
    except Exception as e:
        print(f"⚠️ Central DB connector unavailable, fallback to YAML: {e}")

    db_cfg = _load_db_config_from_yaml()
    if not db_cfg or not db_cfg.get('enabled', False):
        return None
    try:
        conn = psycopg2.connect(
            host=db_cfg.get('host', 'localhost'),
            user=db_cfg.get('user', 'postgres'),
            password=db_cfg.get('password', ''),
            database=db_cfg.get('database', 'vehicle_detection_db'),
            port=db_cfg.get('port', 5432)
        )
        return conn
    except Exception as e:
        print(f"❌ PostgreSQL connection error: {e}")
        return None


def _load_full_config():
    """
    Load full YAML config to access output settings (snapshots, etc.).
    """
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
        if not os.path.exists(config_path):
            config_path = os.path.join('configs', 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️ Cannot load full config: {e}")
        return {}


def _get_or_create_bin_15min(conn, start_dt: datetime, end_dt: datetime) -> int:
    """
    Get or create bins_15min row and return bin_15min_id
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bin_15min_id FROM bins_15min WHERE start_ts = %s AND end_ts = %s",
            (start_dt, end_dt)
        )
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute(
            "INSERT INTO bins_15min (start_ts, end_ts) VALUES (%s, %s) RETURNING bin_15min_id",
            (start_dt, end_dt)
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        return new_id


def _upsert_counts(conn, camera_id: int, bin_id: int, grouped_counts: dict):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO counts (
                bin_15min_id, camera_id,
                motorcycle_tuk_tuk, sedan_pickup_suv, van, minibus_bus, truck6_truck10_trailer
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (bin_15min_id, camera_id)
            DO UPDATE SET
                motorcycle_tuk_tuk = counts.motorcycle_tuk_tuk + EXCLUDED.motorcycle_tuk_tuk,
                sedan_pickup_suv = counts.sedan_pickup_suv + EXCLUDED.sedan_pickup_suv,
                van = counts.van + EXCLUDED.van,
                minibus_bus = counts.minibus_bus + EXCLUDED.minibus_bus,
                truck6_truck10_trailer = counts.truck6_truck10_trailer + EXCLUDED.truck6_truck10_trailer
            """,
            (
                bin_id, camera_id,
                grouped_counts.get('motorcycle_tuk_tuk', 0),
                grouped_counts.get('sedan_pickup_suv', 0),
                grouped_counts.get('van', 0),
                grouped_counts.get('minibus_bus', 0),
                grouped_counts.get('truck6_truck10_trailer', 0),
            )
        )
        conn.commit()


def process_video_and_count(video_path, model_path_best, model_path_yolo12xl, interval_minutes=2):
    """
    Processes a video file, performs object detection and tracking using two models,
    and counts objects crossing a line, allowing dynamic line management.
    
    UPDATED: Uses Class ID Remapping by creating a new Boxes object to fix the AttributeError.
    """
    global roi_dirty
    global count_line_coords_list, selected_line_index, drawing_line, temp_start_point, class_names, WINDOW_NAME, current_mouse_pos
    global MOTORCYCLE_COCO_ID, NEW_MOTORCYCLE_ID, BEST_EXCLUDED_ID

    # --- Model Loading and Class Name Merging ---
    print("กำลังโหลดโมเดล YOLO ทั้งสองตัว...")
    try:
        model_best = YOLO(model_path_best)
        model_yolo12xl = YOLO(model_path_yolo12xl)

        # 1. โหลดชื่อคลาสจาก model_best เป็นฐาน
        class_names.clear()
        class_names.update(model_best.names)
        
        # 2. ค้นหา Class ID จริงของ 'motorcycle' ในโมเดล yolo12x.pt (COCO ID คือ 3)
        YOLO12XL_MOTORCYCLE_NAME = 'motorcycle'
        MOTORCYCLE_COCO_ID = next((k for k, v in model_yolo12xl.names.items() if v == YOLO12XL_MOTORCYCLE_NAME), None)

        if MOTORCYCLE_COCO_ID is None:
            print(f"⚠️ คำเตือน: ไม่พบคลาส '{YOLO12XL_MOTORCYCLE_NAME}' ในโมเดล yolo12x.pt การตรวจจับมอเตอร์ไซค์จะไม่ทำงาน.")
        else:
            # 3. กำหนด Class ID ใหม่ที่ไม่ซ้ำกัน
            # หา ID สูงสุดที่ใช้ใน best.pt แล้วเพิ่ม 1
            max_id = max(class_names.keys()) if class_names else -1
            NEW_MOTORCYCLE_ID = max_id + 1
            
            # 4. เพิ่มคลาส motorcycle ด้วย ID ใหม่ลงใน class_names
            class_names[NEW_MOTORCYCLE_ID] = YOLO12XL_MOTORCYCLE_NAME
            print(f"✅ ดึงคลาส '{YOLO12XL_MOTORCYCLE_NAME}' (COCO ID {MOTORCYCLE_COCO_ID}) และทำแผนที่ ID ใหม่เป็น {NEW_MOTORCYCLE_ID} เพื่อหลีกเลี่ยงการชนกัน")
        
        print(f"รายการคลาสที่ใช้งานทั้งหมด: {class_names}")
        if BEST_EXCLUDED_ID is not None:
            excluded_class_name = model_best.names.get(BEST_EXCLUDED_ID, "Unknown/None")
            print(f"🚫 การตรวจจับจาก best.pt ที่มี ID {BEST_EXCLUDED_ID} ({excluded_class_name}) จะถูกตัดออก")
        else:
            print("✅ คลาส best.pt ครบ")

    except Exception as e:
        print(f"ข้อผิดพลาดในการโหลดโมเดล: {e}")
        return

    # --- Video Setup & Dynamic Interaction Initialization (Unchanged) ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ข้อผิดพลาด: ไม่สามารถเปิดไฟล์วิดีโอ {video_path} ได้")
        return

    ret, initial_frame = cap.read()
    if not ret:
        print("ข้อผิดพลาด: ไม่สามารถอ่านเฟรมแรกของวิดีโอได้")
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_height, frame_width, _ = initial_frame.shape
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) 
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse_event)
    
    SCREEN_WIDTH_ASSUMED = 1600
    SCREEN_HEIGHT_ASSUMED = 900
    center_x = (SCREEN_WIDTH_ASSUMED - frame_width) // 2
    center_y = (SCREEN_HEIGHT_ASSUMED - frame_height) // 2
    cv2.moveWindow(WINDOW_NAME, max(0, center_x), max(0, center_y))
    
    # --- Main Processing Loop ---
    
    counter = Counter(count_line_coords_list=count_line_coords_list)
    output_folder = '/app/output_data/CSV'
    db_conn = _get_db_connection()
    if db_conn:
        print("✅ DB saving enabled")
    else:
        print("⚠️ DB not configured or cannot connect; will skip DB saving")
    
    # Ensure camera exists in DB and get its id
    camera_id = 1
    if db_conn:
        try:
            def _get_or_create_camera(conn, name: str, source_path: str) -> int:
                with conn.cursor() as cur:
                    # Always insert a new camera row (do not check for duplicates)
                    cur.execute(
                        "INSERT INTO cameras (name, source_path, roi_json) VALUES (%s, %s, %s) RETURNING camera_id",
                        (name, source_path, None)
                    )
                    new_id = cur.fetchone()[0]
                    conn.commit()
                    return new_id

            cam_name = os.path.splitext(os.path.basename(str(video_path)))[0] if video_path is not None else 'Camera'
            cam_source = str(video_path)
            camera_id = _get_or_create_camera(db_conn, cam_name, cam_source)
            print(f"📷 Using camera_id={camera_id} for source '{cam_source}'")
        except Exception as e:
            print(f"⚠️ Could not ensure camera row: {e}. Falling back to camera_id=1")

    # Snapshot settings
    cfg_all = _load_full_config()
    output_cfg = cfg_all.get('output', {}) if isinstance(cfg_all, dict) else {}
    save_snapshots = bool(output_cfg.get('save_snapshots', False))
    snapshots_dir = output_cfg.get('snapshots_dir', 'snapshots')
    try:
        if save_snapshots:
            Path(snapshots_dir).mkdir(parents=True, exist_ok=True)
            print(f"📸 Snapshots enabled → {snapshots_dir}")
        else:
            print("📸 Snapshots disabled (set output.save_snapshots: true in configs/config.yaml)")
    except Exception as e:
        print(f"⚠️ Cannot prepare snapshots directory: {e}")
        save_snapshots = False
    start_time = time.time()
    last_interval_export_time = start_time
    counts_by_interval = {}
    last_total_counts = {class_id: 0 for class_id in class_names.keys()} 
    interval_timestamps = []

    # --- เพิ่มตัวแปรสำหรับ real-time count (reset ทุก 15 นาที) ---
    realtime_counts = {class_id: 0 for class_id in class_names.keys()}
    realtime_last_reset_time = start_time

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("Exit program...")
            break
            
        current_time = time.time()
        
        # --- Persist ROI to DB if changed ---
        if db_conn and roi_dirty:
            try:
                with db_conn.cursor() as cur:
                    roi_payload = {
                        'lines': [
                            { 'x1': l[0], 'y1': l[1], 'x2': l[2], 'y2': l[3], 'direction': 'any' }
                            for l in count_line_coords_list
                        ],
                        'roi_polygon': None
                    }
                    cur.execute(
                        "UPDATE cameras SET roi_json = %s WHERE camera_id = %s",
                        (json.dumps(roi_payload), camera_id)
                    )
                    db_conn.commit()
                    roi_dirty = False
            except Exception as e:
                print(f"⚠️ Failed to persist ROI to DB: {e}")

        # --- Interval Export Logic (Unchanged) ---
        if current_time - last_interval_export_time >= (interval_minutes * 60):
            print(f"กำลังส่งออกข้อมูลสำหรับการนับใน {interval_minutes} นาทีที่ผ่านมา...")
            
            per_interval_counts = {}
            total_counts = {}
            for class_id in class_names.keys():
                total_counts[class_id] = len([tid for tid, cid in counter.class_counts.items() if cid == class_id])

            for class_id, current_count in total_counts.items():
                previous_count = last_total_counts.get(class_id, 0)
                per_interval_counts[class_id] = current_count - previous_count
            
            interval_index = len(counts_by_interval)
            
            if any(per_interval_counts.values()):
                # CSV export (optional)
                temp_counts_by_interval = {interval_index: per_interval_counts}
                temp_interval_timestamps = [(last_interval_export_time, current_time)]
                export_to_csv(temp_counts_by_interval, temp_interval_timestamps, class_names, output_folder)
            
            last_total_counts = total_counts.copy()
            last_interval_export_time = current_time
        
        # --- Multi-Model Detection and Tracking (FIXED LOGIC) ---
        results_best = model_best.track(frame, persist=True, tracker='bytetrack.yaml', verbose=False)
        detections_best = results_best[0].boxes if results_best and results_best[0].boxes.id is not None else None
        
        results_yolo12xl = model_yolo12xl.track(frame, persist=True, tracker='bytetrack.yaml', verbose=False)
        detections_yolo12xl = results_yolo12xl[0].boxes if results_yolo12xl and results_yolo12xl[0].boxes.id is not None else None
        
        combined_detections = []
        
        # 1. ดึงการตรวจจับจาก best.pt (ไม่ตัดคลาสเมื่อ BEST_EXCLUDED_ID=None)
        if detections_best:
            for det in detections_best:
                # Add all classes from model_best EXCEPT the excluded ID (ถ้ามีการตั้งค่า)
                if BEST_EXCLUDED_ID is None or int(det.cls.item()) != BEST_EXCLUDED_ID:
                    combined_detections.append(det)
        
        # 2. ดึงการตรวจจับจาก yolo12x.pt (เฉพาะคลาส motorcycle) และเปลี่ยน ID
        if detections_yolo12xl and MOTORCYCLE_COCO_ID is not None and NEW_MOTORCYCLE_ID is not None:
            for det in detections_yolo12xl:
                current_cls_id = int(det.cls.item())
                
                # ถ้าเป็นมอเตอร์ไซค์ (ใช้ ID จริงจาก COCO)
                if current_cls_id == MOTORCYCLE_COCO_ID:
                    
                    # ตรวจสอบ ID ซ้ำ: ตรวจสอบว่า ID นี้ไม่ได้ถูกใช้โดยคลาสอื่นจาก best.pt แล้ว
                    is_duplicate_id = any(d.id == det.id for d in combined_detections)
                    
                    if not is_duplicate_id:
                        # --- FIXED: Use data.clone() and Boxes constructor ---
                        # 1. Create a deep copy of the underlying tensor data
                        det_data_copy = det.data.clone()

                        # 2. The class ID is the last element of the data tensor (index -1)
                        # Modify the class ID column of the copied tensor
                        det_data_copy[:, -1] = NEW_MOTORCYCLE_ID

                        # 3. Recreate a new Boxes object with the modified data and original shape
                        # This object now represents the motorcycle with the unique NEW_MOTORCYCLE_ID
                        det_modified = Boxes(det_data_copy, det.orig_shape)
                        
                        combined_detections.append(det_modified)

        # Update the counter with the combined detections
        updated_detections = counter.update(combined_detections)

        # Insert detection events into DB (only those that crossed the line)
        if db_conn and updated_detections:
            try:
                with db_conn.cursor() as cur:
                    now_ts = datetime.now()
                    # Get 15-min bin for now
                    start_dt_floor = now_ts.replace(minute=(now_ts.minute // 15) * 15, second=0, microsecond=0)
                    end_dt_aligned = start_dt_floor + timedelta(minutes=15)
                    bin_id = _get_or_create_bin_15min(db_conn, start_dt_floor, end_dt_aligned)

                    for det in updated_detections:
                        try:
                            x1, y1, x2, y2 = det.xyxy[0].int().tolist()
                            cls_id = int(det.cls.item())
                            vehicle_class = class_names.get(cls_id, 'unknown')
                            conf = float(det.conf.item()) if hasattr(det, 'conf') else 0.0
                            track_id = int(det.id.item()) if det.id is not None else -1
                            snapshot_path = None
                            # Save snapshot only for crossing events
                            if save_snapshots:
                                try:
                                    # guard boundaries
                                    h, w = frame.shape[:2]
                                    x1c = max(0, min(w, x1)); x2c = max(0, min(w, x2))
                                    y1c = max(0, min(h, y1)); y2c = max(0, min(h, y2))
                                    if x2c > x1c and y2c > y1c:
                                        crop = frame[y1c:y2c, x1c:x2c]
                                        ts_ms = int(time.time() * 1000)
                                        fname = f"snapshot_{camera_id}_{track_id}_{ts_ms}.jpg"
                                        fpath = os.path.join(snapshots_dir, fname)
                                        ok = cv2.imwrite(fpath, crop)
                                        if ok:
                                            snapshot_path = fpath
                                        else:
                                            print(f"⚠️ Failed to write snapshot: {fpath}")
                                except Exception as se:
                                    print(f"⚠️ Snapshot save error: {se}")
                            # direction could be enhanced based on line orientation; mark as 'crossed'
                            cur.execute(
                                """
                                INSERT INTO detections (
                                    ts, camera_id, bin_15min_id, track_id, vehicle_class, conf,
                                    x1, y1, x2, y2, direction, lane, snapshot_path
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    now_ts, camera_id, bin_id, track_id, vehicle_class, conf,
                                    x1, y1, x2, y2, 'crossed', None, snapshot_path
                                )
                            )

                            # --- เพิ่ม logic update count real-time ---
                            # เพิ่ม count class นี้ในตัวแปร realtime_counts
                            if cls_id in realtime_counts:
                                realtime_counts[cls_id] += 1
                            else:
                                realtime_counts[cls_id] = 1


                            # upsert (insert/update) count ใน table counts ทันที เฉพาะ group/class ที่เพิ่งผ่านเส้น
                            # (ใช้ bin_id เดียวกับ 15 นาทีนี้)
                            # --- map class id เป็น group key ---
                            group_key = None
                            class_name = class_names.get(cls_id, '').lower()
                            group_map = {
                                'motorcycle': 'motorcycle_tuk_tuk',
                                'tuk_tuk': 'motorcycle_tuk_tuk',
                                'sedan': 'sedan_pickup_suv',
                                'pickup_single': 'sedan_pickup_suv',
                                'pickup_double': 'sedan_pickup_suv',
                                'suv': 'sedan_pickup_suv',
                                'van': 'van',
                                'minibus': 'minibus_bus',
                                'bus': 'minibus_bus',
                                'truck6': 'truck6_truck10_trailer',
                                'truck10': 'truck6_truck10_trailer',
                                'trailer': 'truck6_truck10_trailer',
                            }
                            group_key = group_map.get(class_name)
                            if group_key:
                                grouped = {
                                    'motorcycle_tuk_tuk': 0,
                                    'sedan_pickup_suv': 0,
                                    'van': 0,
                                    'minibus_bus': 0,
                                    'truck6_truck10_trailer': 0,
                                }
                                grouped[group_key] = 1
                                try:
                                    _upsert_counts(db_conn, camera_id, bin_id, grouped)
                                except Exception as up_e:
                                    print(f"⚠️ Real-time upsert count error: {up_e}")

                        except Exception as ie:
                            # continue other rows
                            print(f"⚠️ Skip det insert error: {ie}")
                    db_conn.commit()
            except Exception as e:
                print(f"⚠️ Failed to insert detections: {e}")
        
        # --- Reset realtime_counts ทุก 15 นาที ---
        if current_time - realtime_last_reset_time >= 15 * 60:
            # reset ตัวแปรนับ
            realtime_counts = {class_id: 0 for class_id in class_names.keys()}
            realtime_last_reset_time = current_time

        # --- Display Logic (Uses the merged class_names) ---
        if combined_detections:
            for det in combined_detections:
                x1, y1, x2, y2 = det.xyxy[0].int().tolist()
                
                class_id = int(det.cls.item())
                main_class_name = class_names.get(class_id, "unknown")
                track_id = int(det.id.item())
                
                # กำหนดสีตามคลาสที่ถูก re-map เพื่อให้เห็นชัดเจน
                color = (0, 255, 0) # Green for normal
                if class_id == NEW_MOTORCYCLE_ID:
                    color = (255, 0, 255) # Magenta for motorcycle

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                display_text = f"ID:{track_id} C:{main_class_name}"
                cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        # --- Drawing and Display Counts (Unchanged) ---
        if count_line_coords_list:
            for i, coords in enumerate(count_line_coords_list):
                line_color = (255, 0, 0)
                line_thickness = 2
                if i == selected_line_index:
                    line_color = (0, 255, 255)
                    line_thickness = 4
                cv2.line(frame, (coords[0], coords[1]), (coords[2], coords[3]), line_color, line_thickness)
        
        if drawing_line and temp_start_point and current_mouse_pos:
            cv2.line(frame, temp_start_point, current_mouse_pos, (0, 165, 255), 1) 
            cv2.circle(frame, temp_start_point, 5, (0, 165, 255), -1) 
            cv2.putText(frame, "P1 Set (Click P2)", (temp_start_point[0] + 10, temp_start_point[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        y_offset = 30
        Font_Scale = 1
        Font_bold = 2
        
        total_counts = {}
        for class_id in class_names.keys():
            total_counts[class_id] = len([tid for tid, cid in counter.class_counts.items() if cid == class_id])

        for class_id in sorted(class_names.keys()):
            class_name = class_names.get(class_id, "Unknown")
            count = total_counts.get(class_id, 0)
            text = f"Count {class_name}: {count}"
            
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, Font_Scale, (255, 0, 255), Font_bold)
            y_offset += 30

        # total_count = sum(total_counts.values())
        # cv2.putText(frame, f"Total Count: {total_count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, Font_Scale, (0, 0, 255), Font_bold)
        
        cv2.imshow(WINDOW_NAME, frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 8 or key == 127: 
            if selected_line_index != -1 and selected_line_index < len(count_line_coords_list):
                deleted_line = count_line_coords_list.pop(selected_line_index)
                selected_line_index = -1 
                print(f"Line deleted: {deleted_line}")
                # mark ROI as changed
                roi_dirty = True
        
        if key == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
    # Final export
    per_interval_counts = {}
    total_counts = {}
    for class_id in class_names.keys():
        total_counts[class_id] = len([tid for tid, cid in counter.class_counts.items() if cid == class_id])
    
    for class_id, current_count in total_counts.items():
        previous_count = last_total_counts.get(class_id, 0)
        per_interval_counts[class_id] = current_count - previous_count
        
    if any(per_interval_counts.values()):
        interval_index = len(counts_by_interval)
        counts_by_interval[interval_index] = per_interval_counts
        interval_timestamps.append((last_interval_export_time, time.time()))
        
    if counts_by_interval:
        export_to_csv(counts_by_interval, interval_timestamps, class_names, output_folder)
    
    return None

if __name__ == '__main__':
    # ตั้งค่า path ของคุณ
    video_path = "E:/Luam/yolov11_2/test_vehicle_videos/v4.mp4" # webcam = 0
    your_model_path = '../weights/best.pt'
    yolo12xl_model_path = '../weights/yolo12x.pt'
    
    process_video_and_count(video_path, your_model_path, yolo12xl_model_path, interval_minutes=15)