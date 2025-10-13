#db_config.py

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import Error
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
DB_CONFIG = {
    "host": "localhost",
    "user": "postgres",
    "password": "YOUR_PASSWORD_HERE",
    "database": "vehicle_detection_db",
    "port": 5432
}

# --- DATABASE CONNECTION ---
def get_connection():
    """
    สร้างการเชื่อมต่อกับฐานข้อมูล PostgreSQL
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        if conn:
            logger.info("Can connect DB ✅") 
            return conn
    except Error as e:
        logger.error(f"❌ PostgreSQL connection error: {e}")
    return None

def init_db():
    """
    Initialize database and tables based on the final schema.
    """
    # 1. เชื่อมต่อโดยไม่ต้องระบุชื่อ database เพื่อสร้าง database ก่อน
    temp_config = DB_CONFIG.copy()
    temp_config.pop('database')
    
    try:
        conn = psycopg2.connect(**temp_config)
    except Error as e:
        logger.error(f"❌ Cannot connect to PostgreSQL Server: {e}")
        return

    conn.autocommit = True
    cursor = conn.cursor()
    
    # 2. สร้าง database ถ้าไม่มี
    database_name = DB_CONFIG['database']
    try:
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{database_name}'")
        exists = cursor.fetchone()
        if not exists:
            cursor.execute(f"CREATE DATABASE {database_name};")
            logger.info(f"✅ Database '{database_name}' created.")
        else:
            logger.info(f"✅ Database '{database_name}' already exists.")
    except Error as e:
        logger.error(f"❌ Error creating database: {e}")
        cursor.close()
        conn.close()
        return

    cursor.close()
    conn.close()
    
    # 3. เชื่อมต่อกับ database ที่สร้างแล้ว
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
    except Error as e:
        logger.error(f"❌ Cannot connect to database: {e}")
        return

    # --- ตาราง 1: bins_15min ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bins_15min (
        bin_15min_id SERIAL PRIMARY KEY,
        start_ts TIMESTAMP NOT NULL,
        end_ts TIMESTAMP NOT NULL
    );
    """)

    # --- ตาราง 2: cameras (ใช้ source_path แทน rtsp_url) ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cameras (
        camera_id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        source_path VARCHAR(512), -- สำหรับ Webcam ID ('0') หรือ Path วิดีโอ
        roi_json JSONB
    );
    """)

    # --- ตาราง 3: detections (รวม detections และ events) ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id BIGSERIAL PRIMARY KEY,
        ts TIMESTAMP NOT NULL,
        
        camera_id INTEGER NOT NULL,
        FOREIGN KEY(camera_id) REFERENCES cameras(camera_id),
        
        bin_15min_id INTEGER,
        FOREIGN KEY(bin_15min_id) REFERENCES bins_15min(bin_15min_id),
        
        track_id INTEGER NOT NULL,
        vehicle_class VARCHAR(50) NOT NULL, -- คลาสสุดท้าย (12 คลาส)
        conf NUMERIC(5, 4) NOT NULL,
        x1 INTEGER NOT NULL, y1 INTEGER NOT NULL, x2 INTEGER NOT NULL, y2 INTEGER NOT NULL,
        direction VARCHAR(50),
        lane VARCHAR(50),
        snapshot_path VARCHAR(512)
    );
    """)

    # --- สร้าง index สำหรับ detections ---
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_detections_ts_camera_id ON detections (ts, camera_id);
    """)

    # --- ตาราง 4: counts (5 กลุ่มคลาส) ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS counts (
        id BIGSERIAL PRIMARY KEY,
        
        bin_15min_id INTEGER NOT NULL,
        FOREIGN KEY(bin_15min_id) REFERENCES bins_15min(bin_15min_id),
        
        camera_id INTEGER NOT NULL,
        FOREIGN KEY(camera_id) REFERENCES cameras(camera_id),
        
        -- คอลัมน์การนับ 5 กลุ่มใหม่
        motorcycle_tuk_tuk INTEGER DEFAULT 0,
        sedan_pickup_suv INTEGER DEFAULT 0,
        van INTEGER DEFAULT 0,
        minibus_bus INTEGER DEFAULT 0,
        truck6_truck10_trailer INTEGER DEFAULT 0,
        
        UNIQUE (bin_15min_id, camera_id)
    );
    """)

    conn.commit()
    cursor.close()
    conn.close()
    logger.info("✅ Database and tables initialized successfully with final schema.")

if __name__ == "__main__":
    init_db()
    get_connection()