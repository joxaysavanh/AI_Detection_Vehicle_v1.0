# 🚗 AI Vehicle Detection System
ระบบตรวจจับและนับยานพาหนะอัตโนมัติด้วยเทคโนโลยี AI

## 📋 ภาพรวมของระบบ

ระบบตรวจจับยานพาหนะนี้ใช้โมเดล YOLO (You Only Look Once) ในการตรวจจับและติดตามยานพาหนะแบบเรียลไทม์ พร้อมทั้งนับจำนวนยานพาหนะที่ผ่านเส้นนับที่กำหนด สามารถรองรับการตรวจจับยานพาหนะได้ 12 ประเภท และจัดกลุ่มเป็น 5 หมวดหมู่หลัก

### 🎯 ฟีเจอร์หลัก
- **การตรวจจับยานพาหนะแบบเรียลไทม์** ด้วยโมเดล YOLO
- **การติดตามวัตถุ** ด้วย ByteTrack algorithm
- **การนับยานพาหนะ** ที่ผ่านเส้นนับที่กำหนด
- **การจัดกลุ่มยานพาหนะ** เป็น 5 หมวดหมู่
- **การบันทึกข้อมูล** ลงฐานข้อมูล PostgreSQL
- **API และ Dashboard** สำหรับดูข้อมูลและสถิติ
- **การบันทึกภาพ** ของยานพาหนะที่ผ่านเส้นนับ

## 🚀 การติดตั้งและใช้งาน

### 📋 ความต้องการของระบบ
- Python 3.8+
- PostgreSQL 12+
- CUDA (สำหรับ GPU acceleration - ไม่บังคับ)

### 🔧 การติดตั้ง

1. **Clone โปรเจค**
```bash
git clone <repository-url>
cd AI_Detection_Vehicle
```

2. **ติดตั้ง Dependencies**
```bash
python -m venv venv ให้สภาพแวดล้อมเหมาะสมในการติดตั้ง requirements จากนั้นให้ใช้คำสั่ง venv\scripts\activate.bat ใน Terminal
pip install -r requirements.txt
```

3. **ตั้งค่าฐานข้อมูล PostgreSQL**
   - ติดตั้ง PostgreSQL
   - สร้างฐานข้อมูล `vehicle_detection_db`
   - แก้ไขรหัสผ่านในไฟล์ `scripts/database/db_config.py`

4. **เริ่มต้นฐานข้อมูล**
```bash
cd scripts/database
python db_config.py
```

### ⚙️ การตั้งค่า

แก้ไขไฟล์ `configs/config.yaml` ตามความต้องการ:

```yaml
# การตั้งค่าพื้นฐาน
device: "cpu"  # หรือ "cuda:0" สำหรับ GPU
model:
  weights: "weights/best.pt"
  img_size: 640
  conf_thres: 0.25
  iou_thres: 0.45

# การตั้งค่าการนับ
counting:
  mode: "line"  # หรือ "roi"
  line: [0, 500, 1920, 500]  # x1,y1,x2,y2
  direction: "down"  # up, down, left, right

# การตั้งค่าฐานข้อมูล
database:
  enabled: true
  host: "localhost"
  user: "postgres"
  password: "YOUR_PASSWORD_HERE"
  database: "vehicle_detection_db"
  port: 5432
```

## 🎮 วิธีการใช้งาน

### 1. การตรวจจับยานพาหนะจากวิดีโอ

```bash
cd scripts
python pipeline_batch.py
```

**การควบคุมในขณะทำงาน:**
- **คลิกซ้าย 2 ครั้ง**: สร้างเส้นนับใหม่
- **คลิกซ้าย 1 ครั้ง**: เลือกเส้นนับ (จะเปลี่ยนเป็นสีเหลือง)
- **Backspace/Delete**: ลบเส้นนับที่เลือก
- **กด 'q'**: ออกจากโปรแกรม

### 2. การฝึกโมเดล

```bash
cd scripts
python train_detector.py
```

### 3. การแปลงวิดีโอเป็นภาพ

```bash
cd scripts
python videoToFrame.py
```

### 4. การใช้งาน API

```bash
cd scripts
python api.py
```

API จะทำงานที่ `http://localhost:8020`

**API Endpoints:**
- `GET /` - ข้อมูลทั้งหมด
- `GET /cameras` - รายการกล้อง
- `GET /counts` - ข้อมูลการนับ
- `GET /detections` - ข้อมูลการตรวจจับ
- `POST /roi` - อัปเดต ROI configuration
- `GET /snapshot/{event_id}` - ดาวน์โหลดภาพ snapshot

### 5. การใช้งาน Dashboard

```bash
cd scripts
streamlit run dashboard.py
```

Dashboard จะทำงานที่ `http://localhost:8050`

## 📊 ประเภทยานพาหนะที่รองรับ

### 🏍️ หมวดหมู่ที่ 1: Motorcycle/Tuk-Tuk
- Motorcycle
- Tuk-tuk

### 🚗 หมวดหมู่ที่ 2: Sedan/Pickup/SUV
- Sedan
- Single-pick-up
- Double-pick-up
- SUV

### 🚙 หมวดหมู่ที่ 3: Van
- Van

### 🚌 หมวดหมู่ที่ 4: Minibus/Bus
- Mini Bus
- Bus

### 🚛 หมวดหมู่ที่ 5: Truck/Trailer
- Truck 6 wheels
- Truck 10 wheels
- Trailer

## 🗄️ โครงสร้างฐานข้อมูล

### ตาราง `cameras`
- `camera_id`: ID ของกล้อง
- `name`: ชื่อกล้อง
- `source_path`: เส้นทางของวิดีโอหรือ webcam
- `roi_json`: การตั้งค่า ROI

### ตาราง `detections`
- `id`: ID การตรวจจับ
- `ts`: เวลาที่ตรวจจับ
- `camera_id`: ID กล้อง
- `track_id`: ID การติดตาม
- `vehicle_class`: ประเภทยานพาหนะ
- `conf`: ความมั่นใจในการตรวจจับ
- `x1, y1, x2, y2`: พิกัด bounding box
- `snapshot_path`: เส้นทางภาพ snapshot

### ตาราง `counts`
- `bin_15min_id`: ID ช่วงเวลา 15 นาที
- `camera_id`: ID กล้อง
- `motorcycle_tuk_tuk`: จำนวนรถมอเตอร์ไซค์/ตุ๊กตุ๊ก
- `sedan_pickup_suv`: จำนวนรถเก๋ง/ปิคอัพ/SUV
- `van`: จำนวนรถตู้
- `minibus_bus`: จำนวนรถมินิบัส/รถบัส
- `truck6_truck10_trailer`: จำนวนรถบรรทุก/รถพ่วง

## 📁 โครงสร้างไฟล์

```
AI_Detection_Vehicle/
├── configs/
│   └── config.yaml          # การตั้งค่าหลัก
├── scripts/
│   ├── api.py               # FastAPI server
│   ├── dashboard.py         # Streamlit dashboard
│   ├── pipeline_batch.py    # การประมวลผลวิดีโอหลัก
│   ├── train_detector.py    # การฝึกโมเดล
│   ├── videoToFrame.py      # แปลงวิดีโอเป็นภาพ
│   ├── database/
│   │   └── db_config.py     # การตั้งค่าฐานข้อมูล
│   └── datasets/           # ข้อมูลสำหรับฝึกโมเดล
├── weights/
│   ├── best.pt             # โมเดลที่ฝึกแล้ว
│   └── yolo12x.pt          # โมเดล YOLO12x
├── CSV_File/               # ไฟล์ CSV ที่ส่งออก
├── snapshots/              # ภาพ snapshot
└── requirements.txt        # Dependencies
```

## 🔧 การแก้ไขปัญหา

### ปัญหาการเชื่อมต่อฐานข้อมูล
1. ตรวจสอบว่า PostgreSQL ทำงานอยู่
2. ตรวจสอบรหัสผ่านใน `db_config.py`
3. ตรวจสอบว่า database `vehicle_detection_db` ถูกสร้างแล้ว

### ปัญหาการโหลดโมเดล
1. ตรวจสอบว่าไฟล์ `weights/best.pt` และ `weights/yolo12x.pt` มีอยู่
2. ตรวจสอบการตั้งค่า `device` ใน `config.yaml`

### ปัญหาการแสดงผลวิดีโอ
1. ตรวจสอบว่า OpenCV ติดตั้งถูกต้อง
2. ตรวจสอบเส้นทางไฟล์วิดีโอ
3. สำหรับ webcam ให้ใช้ `'0'` เป็น source_path

## 📈 การปรับปรุงประสิทธิภาพ

### สำหรับ GPU
```yaml
device: "cuda:0"  # ใน config.yaml
```

### สำหรับ CPU
```yaml
device: "cpu"  # ใน config.yaml
```

#   A I _ D e t e c t i o n _ V e h i c l e _ v 1 . 0  
 