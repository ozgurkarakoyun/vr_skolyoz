"""
download_models.py
──────────────────────────────────────────────────
Railway deploy sırasında çalıştırılır.
YOLO26n-pose modelini önceden indirir,
soğuk başlatma gecikmesini ortadan kaldırır.
"""
import os
import sys

def download_pose_model():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    pose_path = os.environ.get('MODEL_PATH', os.path.join(model_dir, 'yolo26n-pose.pt'))
    scol_path = os.environ.get('SCOL_MODEL_PATH', os.path.join(model_dir, "model_point4.pt"))

    print("[MODEL] Pose model kontrol ediliyor...")
    if not os.path.exists(pose_path):
        try:
            from ultralytics import YOLO
            model_name = os.path.basename(pose_path)
            print(f"[MODEL] {model_name} indiriliyor...")
            model = YOLO(model_name)
            import shutil
            downloaded = model_name
            if os.path.exists(downloaded):
                shutil.move(downloaded, pose_path)
            print(f"[MODEL] ✅ Pose model hazır: {pose_path}")
        except Exception as e:
            print(f"[MODEL] ⚠ Pose model indirilemedi: {e}", file=sys.stderr)
    else:
        print(f"[MODEL] ✅ Pose model zaten mevcut: {pose_path}")

    if os.path.exists(scol_path):
        print(f"[MODEL] ✅ Skolyoz model mevcut: {scol_path}")
    else:
        print(f"[MODEL] ℹ Skolyoz model bulunamadı: {scol_path}")
        print("[MODEL] ℹ Pose keypoint tahmini kullanılacak")

if __name__ == "__main__":
    download_pose_model()
