"""
CHƯƠNG TRÌNH THỊ GIÁC & HIỆU CHUẨN TỌA ĐỘ ROBOT (BƯỚC 3)
Đồ án Cơ cấu chấp hành - UET
Cấu hình: Tờ giấy A3 (420x297 mm), khoảng cách đế tới mép giấy d = 55 mm
Tầm với tối đa: R_max = 200 mm
"""

import cv2
import numpy as np
import json
import os
import math

CONFIG_FILE = "calibration.json"

# Kích thước chuẩn tờ A3 và vị trí đặt Robot (tính theo mm)
PAPER_W = 420.0       # Chiều ngang tờ giấy (mm)
PAPER_H = 297.0       # Chiều dọc tờ giấy (mm)
ROBOT_OFFSET_X = 55.0 # Khoảng cách từ tâm J1 đến mép dưới giấy (mm)
R_MAX = 200.0         # Tầm với tối đa của cánh tay (mm)

# Tọa độ thực tế của 4 góc tờ giấy A3 trong hệ tọa độ Robot {R}:
# Gốc (0,0) là tâm trục J1, trục X hướng về phía trước, trục Y hướng sang phải
REAL_CORNERS = np.float32([
    [-PAPER_W / 2.0, ROBOT_OFFSET_X + PAPER_H], # Góc Trên - Trái: (-210, 352)
    [ PAPER_W / 2.0, ROBOT_OFFSET_X + PAPER_H], # Góc Trên - Phải: ( 210, 352)
    [ PAPER_W / 2.0, ROBOT_OFFSET_X],           # Góc Dưới - Phải: ( 210,  55)
    [-PAPER_W / 2.0, ROBOT_OFFSET_X]            # Góc Dưới - Trái: (-210,  55)
])

# Biến toàn cục
calib_points = []
H_matrix = None
is_calibrating = False
target_place = None

def load_calibration():
    """Tải ma trận biến đổi tọa độ từ file nếu đã lưu trước đó."""
    global H_matrix, calib_points
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                H_matrix = np.array(data["H_matrix"], dtype=np.float32)
                calib_points = data.get("calib_points", [])
                print("[OK] Đã tải dữ liệu hiệu chuẩn từ file calibration.json thành công!")
                return True
        except Exception as e:
            print(f"[!] Lỗi đọc file config: {e}")
    return False

def save_calibration():
    """Lưu ma trận và các điểm mốc vào file calibration.json."""
    if H_matrix is not None:
        data = {
            "H_matrix": H_matrix.tolist(),
            "calib_points": calib_points,
            "paper_w": PAPER_W,
            "paper_h": PAPER_H,
            "offset_x": ROBOT_OFFSET_X
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print("[OK] ĐÃ LƯU CẤU HÌNH VÀO calibration.json THÀNH CÔNG!")

def pixel_to_robot(u, v):
    """Chuyển đổi tọa độ pixel camera (u, v) sang tọa độ thực tế Robot (X_r, Y_r) mm."""
    if H_matrix is None:
        return None, None
    pt = np.array([u, v, 1.0], dtype=np.float32)
    res = np.dot(H_matrix, pt)
    if res[2] == 0:
        return None, None
    y_r = res[0] / res[2] # Trục Y robot (ngang: âm là trái, dương là phải)
    x_r = res[1] / res[2] # Trục X robot (dọc: khoảng cách vươn xa)
    return x_r, y_r

def mouse_callback(event, x, y, flags, param):
    """Bắt sự kiện click chuột để hiệu chuẩn hoặc chọn điểm thả."""
    global calib_points, H_matrix, is_calibrating, target_place

    if event == cv2.EVENT_LBUTTONDOWN:
        if is_calibrating:
            # Chế độ hiệu chuẩn 4 góc tờ giấy
            if len(calib_points) < 4:
                calib_points.append([x, y])
                print(f"[*] Đã chọn điểm {len(calib_points)}: Pixel ({x}, {y})")
                
                if len(calib_points) == 4:
                    pts_src = np.float32(calib_points)
                    pts_dst = REAL_CORNERS
                    H_matrix, _ = cv2.findHomography(pts_src, pts_dst)
                    save_calibration()
                    is_calibrating = False
                    print("\n" + "="*50)
                    print(">>> HIỆU CHUẨN HOÀN TẤT! BÂY GIỜ ĐÃ SẴN SÀNG! <<<")
                    print("="*50 + "\n")
        else:
            # Chế độ bình thường: Click chuột chọn vị trí đích cần thả vật
            x_r, y_r = pixel_to_robot(x, y)
            if x_r is not None:
                dist = math.sqrt(x_r**2 + y_r**2)
                target_place = {
                    "pixel": (x, y),
                    "robot_x": round(x_r, 1),
                    "robot_y": round(y_r, 1),
                    "dist": round(dist, 1),
                    "reachable": dist <= R_MAX and x_r > 0
                }
                status = "TRONG TẦM VỚI" if target_place["reachable"] else "NGOÀI TẦM VỚI (R > 200mm)"
                print(f"[ĐÍCH THẢ ĐÃ CHỌN] X: {target_place['robot_x']} mm, Y: {target_place['robot_y']} mm | {status}")

def detect_shape(approx, contour):
    """Nhận diện hình học: Tam giác, Hình vuông/Chữ nhật, Hình tròn."""
    num_vertices = len(approx)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "KHONG XAC DINH"

    # Tính độ tròn (circularity) = 4 * pi * Area / Perimeter^2
    circularity = 4 * math.pi * area / (perimeter * perimeter)

    if circularity > 0.75:
        return "HINH TRON"
    elif num_vertices == 3:
        return "TAM GIAC"
    elif num_vertices == 4:
        return "HINH VUONG"
    else:
        return "VAT THE"

def main():
    global is_calibrating, calib_points, target_place

    print("=" * 65)
    print("HỆ THỐNG VISION & ĐỊNH VỊ KHÔNG GIAN ROBOT 5-DOF - UET")
    print("=" * 65)

    has_calib = load_calibration()

    # Mở camera Logitech C270 bằng DirectShow
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Robot Vision System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Vision System", 1280, 720)
    cv2.setMouseCallback("Robot Vision System", mouse_callback)

    if not has_calib:
        print("\n[!] Chưa có dữ liệu hiệu chuẩn. Vui lòng bấm phím 'k' để căn chỉnh 4 góc giấy A3!")

    # Dải màu xanh dương mặc định
    lower_blue = np.array([90, 90, 50])
    upper_blue = np.array([135, 255, 255])

    calib_names = [
        "1. Goc TREN - BEN TRAI to A3",
        "2. Goc TREN - BEN PHAI to A3",
        "3. Goc DUOI - BEN PHAI to A3",
        "4. Goc DUOI - BEN TRAI to A3"
    ]

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 1. Vẽ các điểm mốc hiệu chuẩn (nếu có)
        if len(calib_points) > 0:
            for i, pt in enumerate(calib_points):
                cv2.circle(frame, tuple(pt), 7, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if len(calib_points) == 4:
                cv2.polylines(frame, [np.array(calib_points, np.int32)], True, (255, 100, 0), 2)

        # 2. Xử lý nhận diện vật thể
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 350:
                # Xấp xỉ đa giác để nhận diện hình dạng
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                shape_name = detect_shape(approx, cnt)

                # Bounding box & Góc xoay
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect
                box = np.int32(cv2.boxPoints(rect))
                if w < h:
                    angle += 90

                # Chuyển đổi tọa độ sang Robot Frame (mm)
                x_r, y_r = pixel_to_robot(cx, cy)

                # Kiểm tra tầm với robot (R <= 200 mm)
                is_reachable = False
                dist_r = 0
                if x_r is not None:
                    dist_r = math.sqrt(x_r**2 + y_r**2)
                    is_reachable = (dist_r <= R_MAX and x_r > 0)

                # Vẽ viền (Xanh lá nếu với tới, Đỏ nếu ngoài tầm với)
                draw_color = (0, 255, 0) if is_reachable else (0, 0, 255)
                cv2.drawContours(frame, [box], 0, draw_color, 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

                # Hiển thị thông tin
                if x_r is not None:
                    label = f"[{shape_name}] X:{x_r:.1f} Y:{y_r:.1f}mm | {int(angle)} deg"
                    reach_str = "VUA TAM VOI" if is_reachable else f"QUA XA (R={dist_r:.0f}mm)"
                else:
                    label = f"[{shape_name}] Chua Calib"
                    reach_str = ""

                cv2.putText(frame, label, (int(cx) - 100, int(cy) - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                if reach_str:
                    cv2.putText(frame, reach_str, (int(cx) - 60, int(cy) + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)

        # 3. Vẽ điểm đích cần thả (Place Target) nếu người dùng đã click
        if target_place is not None:
            tx, ty = target_place["pixel"]
            t_color = (0, 255, 255) if target_place["reachable"] else (0, 0, 255)
            cv2.circle(frame, (tx, ty), 15, t_color, 2)
            cv2.circle(frame, (tx, ty), 3, (0, 0, 255), -1)
            t_text = f"DICH THA: X={target_place['robot_x']} Y={target_place['robot_y']}mm"
            cv2.putText(frame, t_text, (tx - 100, ty - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, t_color, 2)

        # 4. Hiển thị bảng hướng dẫn ở mép trên
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (700, 85), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if is_calibrating:
            step = len(calib_points)
            if step < 4:
                guide = f">> CHE DO HIEU CHUAN: Click vao [{calib_names[step]}]"
            else:
                guide = ">> Da chon du 4 diem! Dang luu du lieu..."
            cv2.putText(frame, guide, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "PHIM TAT: [k] Can chinh 4 goc A3 | [r] Xoa diem dich | [q] Thoat", 
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, "CLICK CHUOT vao bat ky vi tri nao tren ban de chon VI TRI THA!", 
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow("Robot Vision System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('k'):
            # Bật chế độ hiệu chuẩn lại 4 góc
            is_calibrating = True
            calib_points = []
            print("\n[*] Đã kích hoạt chế độ Hiệu chuẩn! Click chuột lần lượt 4 góc tờ giấy A3...")
        elif key == ord('r'):
            target_place = None
            print("[*] Đã xóa điểm đích cần thả.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
