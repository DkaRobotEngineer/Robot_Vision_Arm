"""
================================================================================
CÔNG CỤ THU THẬP DỮ LIỆU & HUẤN LUYỆN AI YOLOv8 TỰ ĐỘNG CHO ROBOT ARM
Đồ án Cơ cấu chấp hành - UET
--------------------------------------------------------------------------------
Hướng dẫn:
1. Đặt đồ vật (nắp bút, khối vuông, khối tròn, tam giác...) lên bàn.
2. Dùng chuột KÉO THẢ CHỮ NHẬT bao quanh đồ vật (hoặc chọn tự động).
3. Bấm [SPACE]: Lưu ảnh và nhãn tự động (chụp khoảng 15 - 30 ảnh ở các góc khác nhau).
4. Bấm [t]: BẮT ĐẦU HUẤN LUYỆN AI (mất khoảng 1 - 2 phút).
5. Xuất ra file 'best_robot_model.pt' dùng trực tiếp trong main_auto_pick_place.py!
================================================================================
"""

import cv2
import numpy as np
import os
import shutil
import time
from ultralytics import YOLO

DATASET_DIR = "ai_dataset"
IMG_TRAIN_DIR = os.path.join(DATASET_DIR, "images", "train")
LBL_TRAIN_DIR = os.path.join(DATASET_DIR, "labels", "train")

CLASSES = ["NAP_BUT", "KHOI_TRON", "KHOI_VUONG", "TAM_GIAC"]
current_class_id = 0

# Biến vẽ khung chữ nhật bằng chuột
drawing = False
ix, iy = -1, -1
current_rect = None
captured_count = 0

def init_directories():
    os.makedirs(IMG_TRAIN_DIR, exist_ok=True)
    os.makedirs(LBL_TRAIN_DIR, exist_ok=True)

def count_existing_samples():
    if os.path.exists(IMG_TRAIN_DIR):
        return len([f for f in os.listdir(IMG_TRAIN_DIR) if f.endswith(".jpg")])
    return 0

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, current_rect

    # Bỏ qua khu vực thanh công cụ trên cùng
    if y < 100:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_rect = (ix, iy, x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_rect = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1 = min(ix, x)
        y1 = min(iy, y)
        x2 = max(ix, x)
        y2 = max(iy, y)
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            current_rect = (x1, y1, x2, y2)
        else:
            current_rect = None

def save_sample(frame, rect, class_id):
    global captured_count
    if rect is None:
        return False

    x1, y1, x2, y2 = rect
    h_img, w_img = frame.shape[:2]

    # Chuẩn hóa tọa độ YOLO (x_center, y_center, width, height) trong dải 0..1
    cx = ((x1 + x2) / 2.0) / w_img
    cy = ((y1 + y2) / 2.0) / h_img
    bw = (x2 - x1) / float(w_img)
    bh = (y2 - y1) / float(h_img)

    timestamp = int(time.time() * 1000)
    img_name = f"sample_{timestamp}.jpg"
    lbl_name = f"sample_{timestamp}.txt"

    img_path = os.path.join(IMG_TRAIN_DIR, img_name)
    lbl_path = os.path.join(LBL_TRAIN_DIR, lbl_name)

    cv2.imwrite(img_path, frame)
    with open(lbl_path, "w") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    captured_count += 1
    print(f"[OK] Đã lưu mẫu {captured_count}: Lớp '{CLASSES[class_id]}' vào {img_name}")
    return True

def train_yolo_model():
    samples = count_existing_samples()
    if samples < 5:
        print(f"[!] Dữ liệu quá ít ({samples} ảnh). Vui lòng chụp tối thiểu 5 - 10 ảnh trước khi Train!")
        return False

    print("\n" + "=" * 65)
    print(f"[*] BẮT ĐẦU HUẤN LUYỆN AI YOLOV8 VỚI {samples} ẢNH...")
    print("    Quá trình này mất khoảng 1 - 2 phút trên CPU...")
    print("=" * 65)

    base_dir = os.path.abspath(DATASET_DIR).replace("\\", "/")
    yaml_path = os.path.abspath("ai_robot_data.yaml")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {base_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n\n")
        f.write("names:\n")
        for i, name in enumerate(CLASSES):
            f.write(f"  {i}: {name}\n")

    try:
        model = YOLO("yolov8n.pt")
        results = model.train(
            data=yaml_path,
            epochs=20,
            imgsz=416,
            batch=4,
            device="cpu",
            verbose=True
        )

        # Lấy weights tốt nhất
        best_pt = os.path.join("runs", "detect", "train", "weights", "best.pt")
        if not os.path.exists(best_pt):
            # Tìm trong các thư mục train gần nhất
            runs_dir = os.path.join("runs", "detect")
            if os.path.exists(runs_dir):
                all_trains = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if d.startswith("train")]
                all_trains.sort(key=os.path.getmtime, reverse=True)
                if len(all_trains) > 0:
                    cand = os.path.join(all_trains[0], "weights", "best.pt")
                    if os.path.exists(cand):
                        best_pt = cand

        target_model = "best_robot_model.pt"
        if os.path.exists(best_pt):
            shutil.copy(best_pt, target_model)
            print("\n" + "=" * 65)
            print(f">>> HUẤN LUYỆN THÀNH CÔNG! ĐÃ XUẤT: {target_model} <<<")
            print(">>> Bây giờ bạn có thể mở main_auto_pick_place.py để dùng AI! <<<")
            print("=" * 65 + "\n")
            return True
        else:
            print("[!] Không tìm thấy file best.pt sau khi train.")
            return False

    except Exception as e:
        print(f"[!] Lỗi khi huấn luyện: {e}")
        return False

def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, frame = cap.read()
        if ret and frame is not None:
            return cap
    return None

def main():
    global current_class_id, current_rect, captured_count

    init_directories()
    captured_count = count_existing_samples()

    print("=" * 65)
    print("CÔNG CỤ THU THẬP DỮ LIỆU & HUẤN LUYỆN AI YOLOV8 - ROBOT VISION")
    print("=" * 65)
    print(f"[*] Dữ liệu hiện có trong thư mục: {captured_count} ảnh.")

    current_idx = 0
    cap = open_camera(current_idx)
    if cap is None:
        current_idx = 2
        cap = open_camera(current_idx)

    if cap is None:
        print("[!] Không thể mở camera. Vui lòng kiểm tra cáp cắm!")
        return

    win_name = "AI Data Collector & Trainer"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)
    cv2.setMouseCallback(win_name, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        display = frame.copy()

        # 1. Vẽ khung chữ nhật người dùng đang kéo
        if current_rect is not None:
            x1, y1, x2, y2 = current_rect
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{CLASSES[current_class_id]}", (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 2. Thanh hướng dẫn trên cùng (Top Banner)
        header = display.copy()
        cv2.rectangle(header, (10, 10), (1270, 95), (20, 20, 20), -1)
        cv2.addWeighted(header, 0.85, display, 0.15, 0, display)

        cur_name = CLASSES[current_class_id]
        cv2.putText(display, f">> DANG CHON LOP: [{cur_name}] (Phim 1, 2, 3, 4 de doi lop)",
                    (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

        info_str = f"Da chup: {captured_count} anh | Phim [SPACE]: Luu mau | Phim [t]: TRAIN AI | Phim [c]: Xoa khung | Phim [q]: Thoat"
        cv2.putText(display, info_str, (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)

        guide_str = "Keo chuot trai de ve o vuong bao quanh do vat tren ban -> Bam [SPACE] de luu!"
        cv2.putText(display, guide_str, (25, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        cv2.imshow(win_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('1'):
            current_class_id = 0
            print(f"[*] Đổi lớp: {CLASSES[0]}")
        elif key == ord('2'):
            current_class_id = 1
            print(f"[*] Đổi lớp: {CLASSES[1]}")
        elif key == ord('3'):
            current_class_id = 2
            print(f"[*] Đổi lớp: {CLASSES[2]}")
        elif key == ord('4'):
            current_class_id = 3
            print(f"[*] Đổi lớp: {CLASSES[3]}")

        elif key == ord('c'):
            current_rect = None

        elif key == ord(' '):
            if current_rect is not None:
                save_sample(frame, current_rect, current_class_id)
            else:
                print("[!] Hãy dùng chuột kéo khung chữ nhật bao quanh đồ vật trước khi bấm Space!")

        elif key == ord('t'):
            cap.release()
            cv2.destroyAllWindows()
            train_yolo_model()
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
