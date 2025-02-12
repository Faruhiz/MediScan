from db_manager import DatabaseManager

# ✅ สร้าง instance ของ DatabaseManager
db = DatabaseManager()

# ✅ สร้าง project ก่อนถ้ายังไม่มี
# project_id = db.insert_project("Test Project")

# print(f"✅ Created Project ID: {project_id}")

# ✅ ลองแทรกโมเดลลงในฐานข้อมูล
# model_id = db.insert_model(
#     project_id=1,
#     name="test_model",
#     version="v1.0",
#     model_type="detect",
#     file_path="runs/detect/train/best.pt"
# )

# print(f"✅ Inserted Model ID: {model_id}")

# # ✅ ดึงข้อมูลจากฐานข้อมูลเพื่อตรวจสอบ
# db.cursor.execute("SELECT * FROM models ")
# model = db.cursor.fetchone()
# print("🔍 Retrieved model:", model)

# # ✅ ปิด DB
# db.close()

# ✅ ตรวจสอบว่ามีข้อมูลในตารางหรือไม่
db.cursor.execute("SELECT COUNT(*) FROM models")
count = db.cursor.fetchone()[0]

if count == 0:
    print("⚠️ ไม่มีข้อมูลในตาราง models!")
else:
    db.cursor.execute("SELECT * FROM projects")
    models = db.cursor.fetchall()
    
    print("🔍 Retrieved models:")
    for model in models:
        print(model)


# db.cursor.execute("PRAGMA table_info(models)")
# columns = db.cursor.fetchall()
# if not columns:
#     print("❌ ตาราง 'models' ไม่มีอยู่จริง!")
# else:
#     print("✅ ตาราง 'models' มีอยู่จริง:", columns)


model_id = db.insert_model(
    project_id=1,
    name="test_model",
    version="v1.0",
    model_type="detect",
    file_path="runs/detect/train/best.pt"
)

db.close()