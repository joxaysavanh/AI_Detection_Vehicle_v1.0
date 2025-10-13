import uvicorn
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import os
from pathlib import Path
import logging
from database.db_config import get_connection, DB_CONFIG

from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Vehicle Detection API",
    description="API for managing vehicle detection system with ROI, counts, and snapshots",
    version="1.0.0"
)

origins = [
    "http://localhost:3000",   # React dev
    "https://127.0.0.1:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # คุณสามารถเปลี่ยนเป็นเฉพาะโดเมน เช่น ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ROILine(BaseModel):
    """ROI line configuration"""
    x1: int = Field(..., description="Start X coordinate")
    y1: int = Field(..., description="Start Y coordinate") 
    x2: int = Field(..., description="End X coordinate")
    y2: int = Field(..., description="End Y coordinate")
    direction: str = Field(default="any", description="Direction: up, down, left, right, any")

class ROIConfig(BaseModel):
    """ROI configuration for camera"""
    camera_id: int = Field(..., description="Camera ID")
    lines: List[ROILine] = Field(..., description="List of counting lines")
    roi_polygon: Optional[List[List[int]]] = Field(None, description="ROI polygon coordinates")

class CountsResponse(BaseModel):
    """Response model for counts data"""
    camera_id: int
    bin_15min_id: int
    start_ts: datetime
    end_ts: datetime
    motorcycle_tuk_tuk: int
    sedan_pickup_suv: int
    van: int
    minibus_bus: int
    truck6_truck10_trailer: int
    total_count: int

class CameraInfo(BaseModel):
    """Camera information"""
    camera_id: int
    name: str
    source_path: str
    roi_json: Optional[Dict[str, Any]] = None

# Database dependency
def get_db_connection():
    """Get database connection"""
    try:
        conn = get_connection()
        if conn is None:
            raise HTTPException(status_code=500, detail="Database connection failed")
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")


# ดึงข้อมูลทั้งหมดจาก database (detections, counts, cameras)
@app.get("/")
async def root(conn = Depends(get_db_connection)):
    """Root endpoint: Return all data from database (detections, counts, cameras)"""
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get all cameras
            cursor.execute("SELECT * FROM cameras ORDER BY camera_id")
            cameras = cursor.fetchall()

            # Get all counts
            cursor.execute("SELECT * FROM counts ORDER BY bin_15min_id DESC, camera_id")
            counts = cursor.fetchall()

            # Get all detections (limit 1000 for safety)
            cursor.execute("SELECT * FROM detections ORDER BY ts DESC LIMIT 1000")
            detections = cursor.fetchall()

        return {"message": "Vehicle Detection", "version": "1.0.0"},{
            "cameras": cameras,
            "counts": counts,
            "detections": detections
        }
    except Exception as e:
        logger.error(f"Error fetching all data: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch all data")

@app.get("/cameras", response_model=List[CameraInfo])
async def get_cameras(conn = Depends(get_db_connection)):
    """Get list of all cameras"""
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT camera_id, name, source_path, roi_json FROM cameras ORDER BY camera_id")
            cameras = cursor.fetchall()
            
            result = []
            for camera in cameras:
                result.append(CameraInfo(
                    camera_id=camera['camera_id'],
                    name=camera['name'],
                    source_path=camera['source_path'],
                    roi_json=camera['roi_json']
                ))
            
            return result
            
    except Exception as e:
        logger.error(f"Error fetching cameras: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch cameras")

@app.post("/roi")
async def update_roi(roi_config: ROIConfig, conn = Depends(get_db_connection)):
    """
    Update ROI (Region of Interest) configuration for a camera
    Updates counting lines and ROI polygon
    """
    try:
        # Prepare ROI JSON data
        roi_data = {
            "lines": [
                {
                    "x1": line.x1,
                    "y1": line.y1, 
                    "x2": line.x2,
                    "y2": line.y2,
                    "direction": line.direction
                }
                for line in roi_config.lines
            ],
            "roi_polygon": roi_config.roi_polygon
        }
        
        with conn.cursor() as cursor:
            # Check if camera exists
            cursor.execute("SELECT camera_id FROM cameras WHERE camera_id = %s", (roi_config.camera_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Camera not found")
            
            # Update ROI configuration
            cursor.execute(
                "UPDATE cameras SET roi_json = %s WHERE camera_id = %s",
                (json.dumps(roi_data), roi_config.camera_id)
            )
            conn.commit()
            
            logger.info(f"Updated ROI for camera {roi_config.camera_id}")
            return {
                "message": "ROI updated successfully",
                "camera_id": roi_config.camera_id,
                "lines_count": len(roi_config.lines)
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating ROI: {e}")
        conn.rollback()
        raise HTTPException(status_code=500, detail="Failed to update ROI")

@app.get("/counts", response_model=List[CountsResponse])
async def get_counts(
    from_time: Optional[datetime] = Query(None, description="Start time (ISO format)"),
    to_time: Optional[datetime] = Query(None, description="End time (ISO format)"),
    camera_id: Optional[int] = Query(None, description="Filter by camera ID"),
    conn = Depends(get_db_connection)
):
    """
    Get vehicle counts summary
    Returns aggregated counts for specified time range and camera
    """
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Build query with optional filters
            query = """
                SELECT 
                    c.camera_id,
                    c.bin_15min_id,
                    b.start_ts,
                    b.end_ts,
                    c.motorcycle_tuk_tuk,
                    c.sedan_pickup_suv,
                    c.van,
                    c.minibus_bus,
                    c.truck6_truck10_trailer,
                    (c.motorcycle_tuk_tuk + c.sedan_pickup_suv + c.van + c.minibus_bus + c.truck6_truck10_trailer) as total_count
                FROM counts c
                JOIN bins_15min b ON c.bin_15min_id = b.bin_15min_id
                WHERE 1=1
            """
            params = []
            
            if from_time:
                query += " AND b.start_ts >= %s"
                params.append(from_time)
            
            if to_time:
                query += " AND b.end_ts <= %s"
                params.append(to_time)
            
            if camera_id:
                query += " AND c.camera_id = %s"
                params.append(camera_id)
            
            query += " ORDER BY b.start_ts DESC"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            # Convert to response model
            counts_list = []
            for row in results:
                counts_list.append(CountsResponse(
                    camera_id=row['camera_id'],
                    bin_15min_id=row['bin_15min_id'],
                    start_ts=row['start_ts'],
                    end_ts=row['end_ts'],
                    motorcycle_tuk_tuk=row['motorcycle_tuk_tuk'],
                    sedan_pickup_suv=row['sedan_pickup_suv'],
                    van=row['van'],
                    minibus_bus=row['minibus_bus'],
                    truck6_truck10_trailer=row['truck6_truck10_trailer'],
                    total_count=row['total_count']
                ))
            
            return counts_list
            
    except Exception as e:
        logger.error(f"Error fetching counts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch counts")

@app.get("/snapshot/{event_id}")
async def get_snapshot(event_id: int, conn = Depends(get_db_connection)):
    """
    Get snapshot image for a specific detection event
    Returns the snapshot file if it exists
    """
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get detection with snapshot path
            cursor.execute(
                "SELECT snapshot_path FROM detections WHERE id = %s AND snapshot_path IS NOT NULL",
                (event_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Snapshot not found for this event")
            
            snapshot_path = result['snapshot_path']
            
            # Check if file exists
            if not os.path.exists(snapshot_path):
                raise HTTPException(status_code=404, detail="Snapshot file not found on disk")
            
            # Return file
            return FileResponse(
                path=snapshot_path,
                media_type="image/jpeg",
                filename=f"snapshot_{event_id}.jpg"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching snapshot: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch snapshot")

@app.get("/detections")
async def get_detections(
    from_time: Optional[datetime] = Query(None, description="Start time (ISO format)"),
    to_time: Optional[datetime] = Query(None, description="End time (ISO format)"),
    camera_id: Optional[int] = Query(None, description="Filter by camera ID"),
    vehicle_class: Optional[str] = Query(None, description="Filter by vehicle class"),
    limit: int = Query(100, description="Maximum number of records to return"),
    offset: int = Query(0, description="Number of records to skip"),
    conn = Depends(get_db_connection)
):
    """
    Get detailed detection records
    Returns individual detection events with full details
    """
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Build query with optional filters
            query = """
                SELECT 
                    d.id,
                    d.ts,
                    d.camera_id,
                    d.track_id,
                    d.vehicle_class,
                    d.conf,
                    d.x1, d.y1, d.x2, d.y2,
                    d.direction,
                    d.lane,
                    d.snapshot_path
                FROM detections d
                WHERE 1=1
            """
            params = []
            
            if from_time:
                query += " AND d.ts >= %s"
                params.append(from_time)
            
            if to_time:
                query += " AND d.ts <= %s"
                params.append(to_time)
            
            if camera_id:
                query += " AND d.camera_id = %s"
                params.append(camera_id)
            
            if vehicle_class:
                query += " AND d.vehicle_class = %s"
                params.append(vehicle_class)
            
            query += " ORDER BY d.ts DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            return [dict(row) for row in results]
            
    except Exception as e:
        logger.error(f"Error fetching detections: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch detections")

@app.get("/stats/summary")
async def get_stats_summary(
    from_time: Optional[datetime] = Query(None, description="Start time (ISO format)"),
    to_time: Optional[datetime] = Query(None, description="End time (ISO format)"),
    conn = Depends(get_db_connection)
):
    """
    Get summary statistics for the specified time range
    """
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Build time filter
            time_filter = ""
            params = []
            
            if from_time and to_time:
                time_filter = "WHERE b.start_ts >= %s AND b.end_ts <= %s"
                params = [from_time, to_time]
            elif from_time:
                time_filter = "WHERE b.start_ts >= %s"
                params = [from_time]
            elif to_time:
                time_filter = "WHERE b.end_ts <= %s"
                params = [to_time]
            
            # Get total counts by vehicle group
            query = f"""
                SELECT 
                    SUM(c.motorcycle_tuk_tuk) as total_motorcycle_tuk_tuk,
                    SUM(c.sedan_pickup_suv) as total_sedan_pickup_suv,
                    SUM(c.van) as total_van,
                    SUM(c.minibus_bus) as total_minibus_bus,
                    SUM(c.truck6_truck10_trailer) as total_truck6_truck10_trailer,
                    COUNT(DISTINCT c.camera_id) as active_cameras,
                    COUNT(DISTINCT c.bin_15min_id) as time_bins
                FROM counts c
                JOIN bins_15min b ON c.bin_15min_id = b.bin_15min_id
                {time_filter}
            """
            
            cursor.execute(query, params)
            summary = cursor.fetchone()
            
            # Get detection count
            det_query = f"""
                SELECT COUNT(*) as total_detections
                FROM detections d
                {time_filter.replace('b.start_ts', 'd.ts').replace('b.end_ts', 'd.ts')}
            """
            
            cursor.execute(det_query, params)
            det_count = cursor.fetchone()
            
            return {
                "summary": dict(summary),
                "total_detections": det_count['total_detections'] if det_count else 0,
                "time_range": {
                    "from": from_time.isoformat() if from_time else None,
                    "to": to_time.isoformat() if to_time else None
                }
            }
            
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8020)
