from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from ultralytics import YOLO
import shutil
import uuid
import os
import json  # 👈 สำหรับแปลง dictionary เป็น string ที่อ่านง่าย

app = FastAPI()

# สำหรับเชื่อมกับแอป Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดล
model = YOLO("E:/acne_backend/best.pt")  # แก้ path ตามโมเดลคุณ

# สร้างโฟลเดอร์ชั่วคราว
os.makedirs("temp_images", exist_ok=True)

@app.post("/analyze/")
async def analyze_images(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        # 1. บันทึกไฟล์ภาพลง temp
        image_id = str(uuid.uuid4())
        image_path = f"temp_images/{image_id}_{file.filename}"
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. วิเคราะห์ภาพด้วยโมเดล
        prediction = model(image_path)[0]

        # 3. สร้าง dictionary สำหรับส่งกลับ
        image_data = {
            "filename": file.filename,
            "detections": [],
            "acne_count_by_type": {},
            "total_acne_count": 0,
        }

        for box in prediction.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = box
            class_name = model.names[int(cls_id)]

            detection = {
                "class": class_name,
                "confidence": round(conf, 2),
                "box": [round(x1), round(y1), round(x2), round(y2)],
            }

            image_data["detections"].append(detection)
            image_data["acne_count_by_type"][class_name] = image_data["acne_count_by_type"].get(class_name, 0) + 1
            image_data["total_acne_count"] += 1

        results.append(image_data)

        # ลบไฟล์ภาพหลังวิเคราะห์เสร็จ
        os.remove(image_path)

    # ✅ พิมพ์ JSON ที่จะส่งกลับ เพื่อ debug
    print("🔍 JSON response:")
    print(json.dumps({"results": results}, indent=2))

    return {"results": results}
