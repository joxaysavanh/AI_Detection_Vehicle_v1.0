import cv2
import os

# ===== ตั้งค่า =====
video_path = "videos/night3.mp4"          # path ของวิดีโอ
output_folder = "frames_output"   # โฟลเดอร์สำหรับเก็บภาพ

# สร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(output_folder, exist_ok=True)

# เปิดไฟล์วิดีโอ
cap = cv2.VideoCapture(video_path)

# อ่านค่า FPS ของวิดีโอ (frames per second)
fps = int(cap.get(cv2.CAP_PROP_FPS))

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:  # ถ้าอ่านไม่สำเร็จ แสดงว่าจบวิดีโอ
        break

    # เลือกเก็บภาพทุก fps frame (1 วินาทีต่อภาพ)
    if frame_count % fps == 0:
        filename = os.path.join(output_folder, f"frame_Night2_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        saved_count += 1

    frame_count += 1

cap.release()
print("✅ Successfuly")
