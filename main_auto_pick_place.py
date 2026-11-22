"""
================================================================================
HỆ THỐNG GẮP - ĐẶT TỰ ĐỘNG BẰNG THỊ GIÁC MÁY TÍNH (PHIÊN BẢN HOÀN THIỆN ĐỒ ÁN)
Đồ án Cơ cấu chấp hành - UET (Đại học Công nghệ - ĐHQGHN)
--------------------------------------------------------------------------------
Các tính năng nâng cấp:
1. Hiệu chuẩn trực tiếp bằng Thước kẻ + Gốc Robot (KHÔNG CẦN GIẤY A3, chống xô lệch 100%)
2. Tự động nhận diện hình dạng vật thể: HÌNH TRÒN, HÌNH VUÔNG, HÌNH TAM GIÁC
3. Chế độ Tự động phân loại (Auto-Sort) theo hình dạng vào 3 khay đích hoặc Click tự do
4. Khắc phục triệt để đảo chiều nâng/hạ, đóng/mở kẹp và lệch tọa độ
5. Bộ lọc màu HSV tinh chuẩn chống nhận nhầm thân máy robot màu tím
6. Hiển thị cung tròn bán kính làm việc (110mm - 200mm) trực quan trên camera
7. Phím kiểm tra nhanh: [t] Ngắm tâm lơ lửng, [g] Kiểm tra độ cao sát bàn, [o/p] Mở/Đóng kẹp
================================================================================
"""

import cv2
import numpy as np
import json
import os
import math
import time
import websocket
import threading

CONFIG_FILE = "calibration.json"
WS_URL = "ws://192.168.4.1:81/"

# Giới hạn bán kính làm việc thực tế của cánh tay robot (mm)
R_MIN = 110.0   # Bán kính gần nhất có thể với tới
R_MAX = 200.0   # Bán kính xa nhất vươn tới được

# Khóa luồng gửi lệnh robot
ws_lock = threading.Lock()
ws = None
ws_connected = False

# ==========================================
# CẤU HÌNH & HIỆU CHUẨN MẶC ĐỊNH
# ==========================================
config_data = {
    "calib_mode": "base_scale",
    "base_pixel": [736, 15],       # Tọa độ pixel tâm đế xoay J1
    "scale_mm_per_pixel": 0.495,   # Tỉ lệ mm / pixel
    "j1_invert": False,            # Đảo chiều quay đế J1 nếu cần
    "gripper_open": 35,            # Góc mở kẹp FinRay (nhả vật)
    "gripper_close": 110,          # Góc đóng kẹp FinRay (bóp chặt vật)
    "grasp_depth_offset": 0,       # Bù trừ độ sâu khi hạ kẹp (độ)
    "auto_sort_enabled": False,    # Tự động phân loại theo hình dạng
    "active_slot": "2",            # Khay đích mặc định (1: Trái, 2: Giữa, 3: Phải)
    "slots": {
        "1": {"name": "Khay 1 (Trai)", "x_r": 145.0, "y_r": -75.0, "shape": "TRON"},
        "2": {"name": "Khay 2 (Giua)", "x_r": 175.0, "y_r": 0.0,   "shape": "VUONG"},
        "3": {"name": "Khay 3 (Phai)", "x_r": 145.0, "y_r": 75.0,  "shape": "TAM GIAC"}
    }
}

# Biến trạng thái runtime
is_calibrating = False
calib_step = 0
calib_pts = []
target_place = None
detected_object = None
is_running_sequence = False
status_message = "SAN SANG: Click chuot chon dich hoac bam [1/2/3] -> Bam [SPACE] de Gap!"

# ==========================================
# QUẢN LÝ CẤU HÌNH & KẾT NỐI
# ==========================================
def load_config():
    global config_data
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                config_data.update(loaded)
                print("[OK] Đã nạp cấu hình thành công từ calibration.json!")
                return True
        except Exception as e:
            print(f"[!] Lỗi đọc file config: {e}")
    return False

def save_config():
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        print("[OK] Đã lưu cấu hình vào calibration.json!")
        return True
    except Exception as e:
        print(f"[!] Lỗi ghi file config: {e}")
        return False

def get_ws():
    """Khởi tạo hoặc trả về kết nối WebSocket tới ESP32."""
    global ws, ws_connected
    if ws is not None:
        try:
            # Kiểm tra nhanh kết nối còn sống không
            ws.sock.settimeout(0.5)
            return ws
        except Exception:
            ws = None
            ws_connected = False

    try:
        ws = websocket.create_connection(WS_URL, timeout=2.5)
        ws_connected = True
        print("[OK] Đã kết nối WebSocket thành công tới ESP32!")
        # Đặt tốc độ phản hồi êm ái
        ws.send("spd 1.0")
        return ws
    except Exception as e:
        ws = None
        ws_connected = False
        return None

def send_robot_cmd(cmd):
    """Gửi lệnh chuỗi tới ESP32 Robot Arm."""
    global ws, ws_connected
    with ws_lock:
        sock = get_ws()
        if sock is not None:
            try:
                sock.send(cmd)
                return True
            except Exception as e:
                print(f"[!] Lỗi gửi lệnh '{cmd}': {e}")
                ws = None
                ws_connected = False
        return False

# ==========================================
# BIẾN ĐỔI HỆ TỌA ĐỘ (CAMERA -> ROBOT)
# ==========================================
def pixel_to_robot(u, v):
    """
    Chuyển đổi từ tọa độ điểm ảnh (u, v) sang hệ tọa độ thực tế của Robot (X_r, Y_r) mm.
    Camera gắn thẳng từ trên nhìn xuống, đế robot nằm ở phía trên của khung hình.
    Trục X: khoảng cách vươn về phía trước (hướng xuống dưới của ảnh).
    Trục Y: độ lệch ngang (sang phải của ảnh là bên trái của robot).
    """
    u0, v0 = config_data["base_pixel"]
    scale = config_data["scale_mm_per_pixel"]
    du = u - u0
    dv = v - v0
    
    x_r = dv * scale
    y_r = du * scale
    r = math.sqrt(x_r**2 + y_r**2)
    theta = math.degrees(math.atan2(y_r, x_r))
    return x_r, y_r, r, theta

def robot_to_pixel(x_r, y_r):
    """Chuyển đổi ngược từ tọa độ thực tế Robot (X_r, Y_r) mm sang điểm ảnh (u, v)."""
    u0, v0 = config_data["base_pixel"]
    scale = config_data["scale_mm_per_pixel"]
    u = int(round(u0 + y_r / scale))
    v = int(round(v0 + x_r / scale))
    return u, v

# ==========================================
# ĐỘNG HỌC NGHỊCH (IK) CHO CÁNH TAY 5-DOF
# ==========================================
def calculate_ik(x_r, y_r, state="approach", obj_angle=0):
    """
    Tính toán góc điều khiển 5 khớp servo từ tọa độ vật (x_r, y_r) mm.
    state:
      - 'home': Tư thế gập an toàn
      - 'approach' / 'lift': Tư thế lơ lửng trên cao (~6-8 cm cách mặt bàn)
      - 'grasp': Tư thế hạ sát mặt bàn để đầu kẹp chạm chắc vào vật
    """
    j1_invert = config_data["j1_invert"]
    grasp_offset = config_data["grasp_depth_offset"]
    
    # 1. Khớp J1 (Đế xoay ngang)
    # alpha: góc lệch tâm so với đường thẳng chính diện
    alpha = math.degrees(math.atan2(y_r, x_r))
    if not j1_invert:
        j1 = int(round(90.0 + alpha))
    else:
        j1 = int(round(90.0 - alpha))
    j1 = int(np.clip(j1, 10, 170))

    # Khoảng cách bán kính từ tâm trục xoay
    r = math.sqrt(x_r**2 + y_r**2)
    ratio = float(np.clip((r - R_MIN) / (R_MAX - R_MIN), 0.0, 1.0))

    # 2. Các khớp J2 (Vai), J3 (Khuỷu), J4 (Cổ tay gập)
    if state == "home":
        return 90, 90, 90, 90, 90

    elif state in ("approach", "lift"):
        # Tư thế trên cao: Vai ngẩng nhẹ (~85-95 độ), khuỷu gập (~75-90 độ), cổ tay vuông
        j2 = int(round(85.0 + 10.0 * ratio))
        j3 = int(round(75.0 + 15.0 * ratio))
        j4 = 85

    elif state == "grasp":
        # Tư thế hạ sát mặt bàn:
        # Vai J2 nghiêng mạnh về phía trước (> 90 để hạ thấp)
        # Khuỷu J3 duỗi cẳng tay xuống sát mặt bàn
        # Cổ tay J4 bẻ gập hướng đầu kẹp vuông góc với bàn
        j2 = int(round(116.0 + 10.0 * ratio + grasp_offset))
        j3 = int(round(62.0 - 8.0 * ratio - grasp_offset))
        j4 = int(round(112.0 + 8.0 * ratio))
    else:
        j2, j3, j4 = 90, 90, 90

    # Giới hạn an toàn cơ khí
    j2 = int(np.clip(j2, 20, 150))
    j3 = int(np.clip(j3, 20, 150))
    j4 = int(np.clip(j4, 10, 170))

    # 3. Khớp J5 (Xoay cổ tay ôm theo góc nghiêng của vật thể)
    # obj_angle: [-90, 90]
    j5 = int(round(np.clip(90.0 + obj_angle, 15, 165)))

    return j1, j2, j3, j4, j5

# ==========================================
# NHẬN DIỆN HÌNH DẠNG (TRÒN / VUÔNG / TAM GIÁC)
# ==========================================
def classify_shape(cnt):
    """Phân loại hình dạng vật thể bằng xấp xỉ đa giác và hệ số tròn Circularity."""
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.038 * peri, True)
    num_v = len(approx)
    area = cv2.contourArea(cnt)
    
    if area < 10 or peri < 10:
        return "KHAC", num_v

    circularity = (4.0 * math.pi * area) / (peri * peri)
    
    if num_v == 3:
        return "TAM GIAC", num_v
    elif num_v == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ratio = float(w) / float(h)
        if 0.78 <= ratio <= 1.28:
            return "VUONG", num_v
        else:
            return "CHU NHAT", num_v
    elif circularity >= 0.72 or num_v >= 6:
        return "TRON", num_v
    else:
        return "DA GIAC", num_v

# ==========================================
# CÁC KỊCH BẢN ĐIỀU KHIỂN & CHU TRÌNH GẮP - ĐẶT
# ==========================================
def execute_pick_and_place(pick_data, place_data):
    """Chu trình gắp - nâng - chuyển - đặt tự động hoàn chỉnh."""
    global is_running_sequence, status_message
    is_running_sequence = True

    px, py = pick_data["x_r"], pick_data["y_r"]
    p_ang = pick_data["angle"]
    tx, ty = place_data["x_r"], place_data["y_r"]

    open_angle = config_data["gripper_open"]
    close_angle = config_data["gripper_close"]

    try:
        # Bước 1: Mở kẹp & Vươn tới lơ lửng trên đỉnh vật
        status_message = "1/6: Dang mo kep & vuon toi tren dinh vat..."
        send_robot_cmd(f"j 5 {open_angle}") # Mở kẹp nhả
        time.sleep(0.4)

        j1, j2, j3, j4, j5 = calculate_ik(px, py, state="approach", obj_angle=p_ang)
        send_robot_cmd(f"j 0 {j1}")
        send_robot_cmd(f"j 4 {j5}")
        send_robot_cmd(f"j 1 {j2}")
        send_robot_cmd(f"j 2 {j3}")
        send_robot_cmd(f"j 3 {j4}")
        time.sleep(1.8)

        # Bước 2: Hạ tay xuống sát vật để kẹp
        status_message = "2/6: Dang ha tay xuong sat mat ban..."
        j1_g, j2_g, j3_g, j4_g, j5_g = calculate_ik(px, py, state="grasp", obj_angle=p_ang)
        send_robot_cmd(f"j 1 {j2_g}")
        send_robot_cmd(f"j 2 {j3_g}")
        send_robot_cmd(f"j 3 {j4_g}")
        time.sleep(1.4)

        # Bước 3: Đóng kẹp FinRay bóp chặt vật
        status_message = "3/6: Dang kep chat vat..."
        send_robot_cmd(f"j 5 {close_angle}") # Kẹp chặt
        time.sleep(1.2)

        # Bước 4: Nhấc bổng vật lên khỏi mặt bàn
        status_message = "4/6: Dang nhac vat len cao..."
        send_robot_cmd(f"j 1 {j2}")
        send_robot_cmd(f"j 2 {j3}")
        send_robot_cmd(f"j 3 {j4}")
        time.sleep(1.2)

        # Bước 5: Xoay sang vị trí đích và hạ xuống
        status_message = "5/6: Dang chuyen sang vi tri dich..."
        j1_t, j2_t, j3_t, j4_t, j5_t = calculate_ik(tx, ty, state="approach", obj_angle=0)
        send_robot_cmd(f"j 0 {j1_t}")
        send_robot_cmd(f"j 4 {j5_t}")
        send_robot_cmd(f"j 1 {j2_t}")
        send_robot_cmd(f"j 2 {j3_t}")
        send_robot_cmd(f"j 3 {j4_t}")
        time.sleep(1.8)

        # Hạ tay xuống điểm đích
        j1_tg, j2_tg, j3_tg, j4_tg, j5_tg = calculate_ik(tx, ty, state="grasp", obj_angle=0)
        send_robot_cmd(f"j 1 {j2_tg}")
        send_robot_cmd(f"j 2 {j3_tg}")
        send_robot_cmd(f"j 3 {j4_tg}")
        time.sleep(1.4)

        # Bước 6: Nhả kẹp & Rút tay về vị trí Home an toàn
        status_message = "6/6: Tha vat & Rut tay ve Home..."
        send_robot_cmd(f"j 5 {open_angle}") # Mở kẹp nhả vật
        time.sleep(0.8)
        send_robot_cmd("home")
        time.sleep(1.6)

        status_message = "THANH CONG: Da hoan thanh chu trinh! Chon vat tiep theo."
        print(f"\n[OK] Chu trình Gắp - Đặt thành công từ ({px:.1f}, {py:.1f}) sang ({tx:.1f}, {ty:.1f})!")

    except Exception as e:
        status_message = f"LOI THUC THI: {e}"
        print(f"[!] Lỗi chu trình gắp: {e}")
    finally:
        is_running_sequence = False

def test_hover_target(pick_data):
    """Thử nghiệm ngắm tâm: Vươn đến lơ lửng trên vật rồi về Home (không kẹp)."""
    global is_running_sequence, status_message
    is_running_sequence = True
    px, py = pick_data["x_r"], pick_data["y_r"]
    p_ang = pick_data["angle"]

    try:
        status_message = "TEST HOVER: Dang vuon den lo lung tren vat de ngam..."
        open_angle = config_data["gripper_open"]
        send_robot_cmd(f"j 5 {open_angle}")
        j1, j2, j3, j4, j5 = calculate_ik(px, py, state="approach", obj_angle=p_ang)
        send_robot_cmd(f"j 0 {j1}")
        send_robot_cmd(f"j 4 {j5}")
        send_robot_cmd(f"j 1 {j2}")
        send_robot_cmd(f"j 2 {j3}")
        send_robot_cmd(f"j 3 {j4}")
        time.sleep(3.0)
        send_robot_cmd("home")
        time.sleep(1.5)
        status_message = "HOAN TAT TEST HOVER: Tam robot va goc xoay da can chuan!"
    except Exception as e:
        status_message = f"LOI TEST HOVER: {e}"
    finally:
        is_running_sequence = False

def test_grasp_target(pick_data):
    """Thử nghiệm độ cao gắp Z: Hạ xuống sát vật rồi nhấc lên về Home (không kẹp chặt)."""
    global is_running_sequence, status_message
    is_running_sequence = True
    px, py = pick_data["x_r"], pick_data["y_r"]
    p_ang = pick_data["angle"]

    try:
        status_message = "TEST Z: Dang ha tay sat mat ban de ngam do cao..."
        open_angle = config_data["gripper_open"]
        send_robot_cmd(f"j 5 {open_angle}")
        j1, j2, j3, j4, j5 = calculate_ik(px, py, state="grasp", obj_angle=p_ang)
        send_robot_cmd(f"j 0 {j1}")
        send_robot_cmd(f"j 4 {j5}")
        send_robot_cmd(f"j 1 {j2}")
        send_robot_cmd(f"j 2 {j3}")
        send_robot_cmd(f"j 3 {j4}")
        time.sleep(3.0)
        send_robot_cmd("home")
        time.sleep(1.5)
        status_message = "HOAN TAT TEST Z: Bam [ [ ] de ha sau them hoac [ ] ] de nang len!"
    except Exception as e:
        status_message = f"LOI TEST Z: {e}"
    finally:
        is_running_sequence = False

# ==========================================
# BẮT SỰ KIỆN CHUỘT (HIỆU CHUẨN & CHỌN ĐÍCH)
# ==========================================
def mouse_callback(event, x, y, flags, param):
    global is_calibrating, calib_step, calib_pts, target_place, status_message

    if event == cv2.EVENT_LBUTTONDOWN:
        if is_calibrating:
            if calib_step == 1:
                # Bước 1: Chọn tâm đế xoay Robot J1
                config_data["base_pixel"] = [x, y]
                calib_step = 2
                status_message = "CALIB (2/3): Dat thuoc len ban -> Click vao vach 0 cm!"
                print(f"[*] Đã chọn Tâm Đế Robot: Pixel ({x}, {y})")

            elif calib_step == 2:
                # Bước 2: Chọn vạch 0 cm của thước kẻ
                calib_pts = [[x, y]]
                calib_step = 3
                status_message = "CALIB (3/3): Click vao vach 10 cm (100mm) tren thuoc ke!"
                print(f"[*] Đã chọn vạch 0 cm: Pixel ({x}, {y})")

            elif calib_step == 3:
                # Bước 3: Chọn vạch 10 cm của thước kẻ
                calib_pts.append([x, y])
                x1, y1 = calib_pts[0]
                x2, y2 = calib_pts[1]
                dist_px = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if dist_px > 20:
                    scale = 100.0 / dist_px  # 10 cm = 100 mm
                    config_data["scale_mm_per_pixel"] = round(scale, 4)
                    save_config()
                    is_calibrating = False
                    calib_step = 0
                    status_message = f"HIEU CHUAN XONG! Ti le: {scale:.4f} mm/px. Goc: {config_data['base_pixel']}"
                    print(f"[OK] HIỆU CHUẨN THÀNH CÔNG! 100mm = {dist_px:.1f}px -> Tỉ lệ: {scale:.4f} mm/px")
                else:
                    status_message = "Khoang cach qua ngan, vui long click lai vach 10 cm!"

        else:
            # Chọn điểm đích cần thả tự do bằng chuột
            xr, yr, dist, theta = pixel_to_robot(x, y)
            is_reach = (R_MIN <= dist <= R_MAX and xr > 0)
            target_place = {
                "pixel": (x, y),
                "x_r": round(xr, 1),
                "y_r": round(yr, 1),
                "dist": round(dist, 1),
                "reachable": is_reach,
                "label": "DICH TU CHON"
            }
            print(f"[ĐÍCH THẢ] X: {target_place['x_r']}mm, Y: {target_place['y_r']}mm | Tầm với: {is_reach}")

# ==========================================
# MỞ CAMERA (DirectShow Backend)
# ==========================================
def open_camera(index):
    print(f"[*] Đang mở Camera index [{index}]...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(3):
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[OK] Đã mở thành công Camera index [{index}]!")
                return cap
    if cap is not None:
        cap.release()
    return None

# ==========================================
# CHƯƠNG TRÌNH CHÍNH (MAIN LOOP)
# ==========================================
def main():
    global is_calibrating, calib_step, calib_pts, target_place, detected_object
    global is_running_sequence, status_message, ws_connected

    print("=" * 70)
    print("HỆ THỐNG GẮP - ĐẶT TỰ ĐỘNG BẰNG THỊ GIÁC MÁY TÍNH - UET ROBOTICS")
    print("=" * 70)

    load_config()
    get_ws()

    # Mặc định mở camera Logitech C270
    current_index = 0
    cap = open_camera(current_index)
    if cap is None:
        current_index = 2
        cap = open_camera(current_index)

    if cap is None:
        print("[!] Không thể mở camera. Vui lòng cắm lại cáp USB!")
        return

    cv2.namedWindow("Robot Vision System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Vision System", 1280, 720)
    cv2.setMouseCallback("Robot Vision System", mouse_callback)

    # Dải màu xanh dương bão hòa cao (loại trừ triệt để màu tím thân robot)
    lower_blue = np.array([104, 150, 65])
    upper_blue = np.array([122, 255, 255])

    # Thiết lập điểm đích mặc định từ Slot 2 (Khay Giữa)
    active_slot = config_data.get("active_slot", "2")
    slot_info = config_data["slots"].get(active_slot, config_data["slots"]["2"])
    u_slot, v_slot = robot_to_pixel(slot_info["x_r"], slot_info["y_r"])
    target_place = {
        "pixel": (u_slot, v_slot),
        "x_r": slot_info["x_r"],
        "y_r": slot_info["y_r"],
        "dist": math.sqrt(slot_info["x_r"]**2 + slot_info["y_r"]**2),
        "reachable": True,
        "label": slot_info["name"]
    }

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        u0, v0 = config_data["base_pixel"]
        scale = config_data["scale_mm_per_pixel"]

        # -------------------------------------------------------------
        # 1. VẼ VÙNG LÀM VIỆC & BÁN KÍNH TẦM VỚI CỦA ROBOT (Reach Overlay)
        # -------------------------------------------------------------
        # Vẽ tâm chân đế Robot (0, 0)
        cv2.circle(frame, (u0, v0), 9, (0, 0, 255), -1)
        cv2.circle(frame, (u0, v0), 16, (0, 255, 255), 2)
        cv2.putText(frame, "GOC ROBOT (0,0)", (u0 - 65, max(25, v0 - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Trục chính diện thẳng về trước (Centerline)
        cv2.line(frame, (u0, v0), (u0, v0 + int(R_MAX / scale)), (100, 100, 100), 1, cv2.LINE_AA)

        # Cung tròn bán kính tối thiểu R_MIN (110 mm)
        r_min_px = int(R_MIN / scale)
        cv2.ellipse(frame, (u0, v0), (r_min_px, r_min_px), 0, 30, 150, (0, 100, 255), 1, cv2.LINE_AA)

        # Cung tròn bán kính lý tưởng R_MID (150 mm)
        r_mid_px = int(150.0 / scale)
        cv2.ellipse(frame, (u0, v0), (r_mid_px, r_mid_px), 0, 30, 150, (0, 255, 120), 1, cv2.LINE_AA)

        # Cung tròn bán kính tối đa R_MAX (200 mm)
        r_max_px = int(R_MAX / scale)
        cv2.ellipse(frame, (u0, v0), (r_max_px, r_max_px), 0, 30, 150, (0, 100, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 2. VẼ 3 KHAY ĐÍCH ĐẾN MẶC ĐỊNH (SLOT 1, 2, 3)
        # -------------------------------------------------------------
        for s_id, s_data in config_data["slots"].items():
            su, sv = robot_to_pixel(s_data["x_r"], s_data["y_r"])
            is_active = (target_place is not None and abs(target_place["x_r"] - s_data["x_r"]) < 2 and abs(target_place["y_r"] - s_data["y_r"]) < 2)
            s_col = (0, 255, 255) if is_active else (180, 180, 180)
            cv2.circle(frame, (su, sv), 18, s_col, 2 if is_active else 1)
            cv2.circle(frame, (su, sv), 4, s_col, -1)
            cv2.putText(frame, f"[{s_id}] {s_data['shape']}", (su - 35, sv + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, s_col, 1)

        # -------------------------------------------------------------
        # 3. QUÉT NHẬN DIỆN VẬT THỂ & PHÂN LOẠI HÌNH DẠNG
        # -------------------------------------------------------------
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_objects = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Chỉ nhận diện vật thể có diện tích từ 350 đến 2500 px (loại trừ chi tiết nhỏ và thân robot)
            if 350 < area < 2500:
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect

                # Tỉ lệ cạnh: vật gắp (khối/nắp bút) có tỉ lệ < 2.6 (thân robot rất dài ngoẵng > 3.5)
                aspect = max(w, h) / (min(w, h) + 1e-5)
                if aspect > 2.6:
                    continue

                # Chuẩn hóa góc quay cổ tay ôm vật
                if w < h:
                    angle += 90.0
                if angle > 90.0:
                    angle -= 180.0

                xr, yr, dist, theta = pixel_to_robot(cx, cy)
                
                # Loại trừ vùng sát chân đế robot (nơi gắn các link và dây điện)
                if dist < 95.0:
                    continue

                is_reach = (R_MIN <= dist <= R_MAX and xr > 0)
                box = np.int32(cv2.boxPoints(rect))
                shape_label, num_v = classify_shape(cnt)

                color = (0, 255, 0) if is_reach else (0, 0, 255)
                cv2.drawContours(frame, [box], 0, color, 2)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

                tag = f"[{shape_label}] X:{xr:.1f} Y:{yr:.1f} R:{dist:.1f}mm | {int(angle)} deg"
                cv2.putText(frame, tag, (int(cx) - 100, int(cy) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 2)

                if is_reach:
                    valid_objects.append({
                        "x_r": xr, "y_r": yr, "angle": angle, "dist": dist,
                        "shape": shape_label, "area": area, "pixel": (int(cx), int(cy))
                    })

        # Cập nhật vật thể mục tiêu (chỉ cập nhật khi không đang chạy chu trình)
        if not is_running_sequence:
            if len(valid_objects) > 0:
                # Ưu tiên vật thể có hình khối chuẩn nhất
                valid_objects.sort(key=lambda o: abs(o["area"] - 800))
                detected_object = valid_objects[0]
            else:
                detected_object = None

        # -------------------------------------------------------------
        # 4. CHẾ ĐỘ AUTO-SORT (TỰ ĐỘNG CHỌN KHAY THEO HÌNH DẠNG)
        # -------------------------------------------------------------
        if config_data.get("auto_sort_enabled", False) and detected_object is not None:
            s_type = detected_object["shape"]
            target_slot_id = "2" # Mặc định
            if s_type == "TRON":
                target_slot_id = "1"
            elif s_type in ("VUONG", "CHU NHAT"):
                target_slot_id = "2"
            elif s_type == "TAM GIAC":
                target_slot_id = "3"

            s_info = config_data["slots"][target_slot_id]
            su, sv = robot_to_pixel(s_info["x_r"], s_info["y_r"])
            target_place = {
                "pixel": (su, sv),
                "x_r": s_info["x_r"],
                "y_r": s_info["y_r"],
                "dist": math.sqrt(s_info["x_r"]**2 + s_info["y_r"]**2),
                "reachable": True,
                "label": f"AUTO-SORT -> {s_info['name']}"
            }

        # -------------------------------------------------------------
        # 5. VẼ ĐIỂM ĐÍCH THẢ ĐÃ CHỌN
        # -------------------------------------------------------------
        if target_place is not None:
            tx, ty = target_place["pixel"]
            t_col = (0, 255, 255) if target_place["reachable"] else (0, 0, 255)
            cv2.circle(frame, (tx, ty), 22, t_col, 2)
            cv2.circle(frame, (tx, ty), 12, t_col, 1)
            cv2.circle(frame, (tx, ty), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"DICH: X={target_place['x_r']} Y={target_place['y_r']}mm ({target_place.get('label', '')})",
                        (tx - 110, ty - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, t_col, 2)

        # -------------------------------------------------------------
        # 6. BẢNG ĐIỀU KHIỂN & THANH TRẠNG THÁI (HUD OVERLAY)
        # -------------------------------------------------------------
        hud = frame.copy()
        cv2.rectangle(hud, (10, 10), (1270, 105), (20, 20, 20), -1)
        cv2.addWeighted(hud, 0.75, frame, 0.25, 0, frame)

        if is_calibrating:
            if calib_step == 1:
                guide_1 = ">> BUOC 1/3: CLICK VAO TAM CHAN DE ROBOT (TRUC XOAY J1) O PHIA TREN MAN HINH!"
                guide_2 = "Click vao tam truc xoay J1 cua robot de thiet lap goc toa do (0, 0)."
            elif calib_step == 2:
                guide_1 = ">> BUOC 2/3: DAT THUOC KE LEN BAN -> CLICK VAO VACH 0 CM!"
                guide_2 = "Dat thuoc ke ngang hoac doc tren mat ban trong tam nhin cua camera."
            elif calib_step == 3:
                guide_1 = ">> BUOC 3/3: CLICK VAO VACH 10 CM (100mm) TREN THUOC KE!"
                guide_2 = "He thong se tu dong tinh chinh xac ti le mm/pixel va luu vao calibration.json."
                if len(calib_pts) == 1:
                    cv2.circle(frame, tuple(calib_pts[0]), 6, (0, 255, 0), -1)
            else:
                guide_1 = "Dang hoan tat hieu chuan..."
                guide_2 = ""

            cv2.putText(frame, guide_1, (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, guide_2, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
            cv2.putText(frame, "[ESC] Huy hieu chuan", (1050, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)

        else:
            # Dòng 1: Thông báo trạng thái hoạt động chính
            cv2.putText(frame, f">> {status_message}", (25, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)

            # Dòng 2: Trạng thái phần cứng & thông số
            ws_str = "ESP32: DA KET NOI" if ws_connected else "ESP32: MAT KET NOI [w]"
            ws_col = (0, 255, 0) if ws_connected else (0, 0, 255)
            depth_str = f"Z-Offset: {config_data['grasp_depth_offset']:+d} do [ [ / ] ]"
            grip_str = f"Kep: {config_data['gripper_close']} do"
            j1_rev_str = f"J1-Rev: {'ON' if config_data['j1_invert'] else 'OFF'} [i]"
            auto_str = f"Auto-Sort: {'ON' if config_data['auto_sort_enabled'] else 'OFF'} [a]"

            status_line = f"{ws_str} | Cam: [{current_index}] | {depth_str} | {grip_str} | {j1_rev_str} | {auto_str}"
            cv2.putText(frame, status_line, (25, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.46, ws_col, 1)

            # Dòng 3: Hướng dẫn phím bấm
            keys_line = "PHIM: [SPACE] Gap | [t] Test Ngam | [g] Test Z | [o/p] Mo/Dong kep | [1/2/3] Khay | [k] Calib Thuoc | [h] Home | [q] Thoat"
            cv2.putText(frame, keys_line, (25, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)

        cv2.imshow("Robot Vision System", frame)

        # -------------------------------------------------------------
        # 7. XỬ LÝ PHÍM BẤM
        # -------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == 27: # Phím ESC
            if is_calibrating:
                is_calibrating = False
                calib_step = 0
                status_message = "Da huy che do Hieu chuan!"

        elif key == ord('k'):
            # Bật chế độ Hiệu chuẩn Thước kẻ + Tâm Robot (3 clicks)
            is_calibrating = True
            calib_step = 1
            calib_pts = []
            status_message = "CALIB (1/3): Click vao TAM CHAN DE ROBOT (Truc J1) o phia tren man hinh!"
            print("\n[*] Đã kích hoạt Chế độ Hiệu chuẩn không cần giấy A3!")

        elif key == ord('o'):
            # Mở kẹp Gripper (góc nhỏ ~35 độ)
            open_ang = config_data["gripper_open"]
            send_robot_cmd(f"j 5 {open_ang}")
            status_message = f"Da mo kep ({open_ang} do)!"
            print(f"[*] Mở kẹp: j 5 {open_ang}")

        elif key == ord('p'):
            # Đóng kẹp Gripper chặt (góc lớn ~110 độ)
            close_ang = config_data["gripper_close"]
            send_robot_cmd(f"j 5 {close_ang}")
            status_message = f"Da dong kep chat ({close_ang} do)!"
            print(f"[*] Đóng kẹp: j 5 {close_ang}")

        elif key == ord('['):
            # Hạ sâu thêm độ cao kẹp (J2 tăng thêm 2 độ)
            config_data["grasp_depth_offset"] += 2
            save_config()
            status_message = f"Da ha sau them: {config_data['grasp_depth_offset']:+d} do. Bam [g] de thu!"
            print(f"[*] Độ sâu hạ tay hiện tại: {config_data['grasp_depth_offset']:+d} độ")

        elif key == ord(']'):
            # Nâng cao lên nếu kẹp cọ sát mặt bàn
            config_data["grasp_depth_offset"] -= 2
            save_config()
            status_message = f"Da nang cao len: {config_data['grasp_depth_offset']:+d} do. Bam [g] de thu!"
            print(f"[*] Độ sâu hạ tay hiện tại: {config_data['grasp_depth_offset']:+d} độ")

        elif key == ord('{'):
            # Tăng góc kẹp chặt thêm 3 độ
            config_data["gripper_close"] = min(140, config_data["gripper_close"] + 3)
            save_config()
            status_message = f"Goc kep chat moi: {config_data['gripper_close']} do"

        elif key == ord('}'):
            # Giảm góc kẹp đi 3 độ (nhả bớt lực)
            config_data["gripper_close"] = max(80, config_data["gripper_close"] - 3)
            save_config()
            status_message = f"Goc kep chat moi: {config_data['gripper_close']} do"

        elif key == ord('i'):
            # Đảo chiều quay trái/phải của khớp đế J1
            config_data["j1_invert"] = not config_data["j1_invert"]
            save_config()
            status_message = f"Da dao chieu de J1: {'ON' if config_data['j1_invert'] else 'OFF'}"
            print(f"[*] Đảo chiều J1: {config_data['j1_invert']}")

        elif key == ord('a'):
            # Bật/Tắt chế độ Auto-Sort theo hình dạng
            config_data["auto_sort_enabled"] = not config_data["auto_sort_enabled"]
            save_config()
            status_message = f"Che do Auto-Sort: {'BAT' if config_data['auto_sort_enabled'] else 'TAT'}"
            print(f"[*] Auto-Sort: {config_data['auto_sort_enabled']}")

        elif key in (ord('1'), ord('2'), ord('3')):
            # Chọn đích thả là Khay 1, Khay 2 hoặc Khay 3
            slot_id = chr(key)
            config_data["active_slot"] = slot_id
            s_data = config_data["slots"][slot_id]
            su, sv = robot_to_pixel(s_data["x_r"], s_data["y_r"])
            target_place = {
                "pixel": (su, sv),
                "x_r": s_data["x_r"],
                "y_r": s_data["y_r"],
                "dist": math.sqrt(s_data["x_r"]**2 + s_data["y_r"]**2),
                "reachable": True,
                "label": s_data["name"]
            }
            save_config()
            status_message = f"Da chon {s_data['name']} (X={s_data['x_r']}, Y={s_data['y_r']}mm)"
            print(f"[*] Đã chọn đích: {s_data['name']}")

        elif key == ord('t'):
            # Thử nghiệm Hover ngắm tâm trên đỉnh vật
            if is_running_sequence:
                print("[!] Robot đang thực hiện chu trình!")
            elif detected_object is None:
                status_message = "Chua co vat the nao trong tam voi de Test!"
            else:
                th = threading.Thread(target=test_hover_target, args=(detected_object,))
                th.daemon = True
                th.start()

        elif key == ord('g'):
            # Thử nghiệm kiểm tra độ cao sát bàn Z
            if is_running_sequence:
                print("[!] Robot đang thực hiện chu trình!")
            elif detected_object is None:
                status_message = "Chua co vat the nao trong tam voi de Test!"
            else:
                th = threading.Thread(target=test_grasp_target, args=(detected_object,))
                th.daemon = True
                th.start()

        elif key == ord('h'):
            # Về tư thế Home an toàn
            send_robot_cmd("home")
            status_message = "Da gui lenh robot ve vi tri Home an toan!"

        elif key == ord('w'):
            # Kết nối lại WebSocket ESP32
            print("[*] Đang thử kết nối lại WebSocket ESP32...")
            if get_ws() is not None:
                status_message = "Ket noi WebSocket ESP32 thanh cong!"
            else:
                status_message = "Ket noi ESP32 that bai! Hay kiem tra WiFi 'RobotArm' & DONG tab web 192.168.4.1!"

        elif key == ord('c'):
            # Đổi sang camera tiếp theo: 0 -> 2 -> 1 -> 0
            order = [0, 2, 1]
            cur_pos = order.index(current_index) if current_index in order else 0
            next_idx = order[(cur_pos + 1) % len(order)]
            print(f"[*] Đang chuyển sang Camera index [{next_idx}]...")
            cap.release()
            new_cap = open_camera(next_idx)
            if new_cap is not None:
                cap = new_cap
                current_index = next_idx
            else:
                print(f"[!] Không mở được index [{next_idx}], giữ lại index [{current_index}]")
                cap = open_camera(current_index)

        elif key == ord('s'):
            # Lưu cấu hình hiện tại
            save_config()
            status_message = "Da luu toan bo cau hinh vao calibration.json!"

        elif key == ord(' '): # Phím SPACE: Bắt đầu chu trình Gắp - Đặt tự động hoàn chỉnh
            if is_running_sequence:
                print("[!] Robot đang trong chu trình di chuyển, vui lòng đợi!")
            elif detected_object is None:
                status_message = "CANH BAO: Chua phat hien vat the mau xanh trong tam voi (110-200mm)!"
                print("[!] Chưa nhận diện được vật thể hợp lệ trong vùng làm việc!")
            elif target_place is None or not target_place["reachable"]:
                status_message = "CANH BAO: Vui long chon dich tha hop le (Bam 1, 2, 3 hoac click chuot)!"
                print("[!] Điểm thả chưa hợp lệ!")
            else:
                print("\n" + "=" * 60)
                print(f"[*] BẮT ĐẦU CHU TRÌNH GẮP - ĐẶT TỰ ĐỘNG...")
                print(f"    - Vật thể: {detected_object['shape']} tại X={detected_object['x_r']}mm, Y={detected_object['y_r']}mm")
                print(f"    - Điểm thả: {target_place.get('label', '')} tại X={target_place['x_r']}mm, Y={target_place['y_r']}mm")
                print("=" * 60 + "\n")
                t = threading.Thread(target=execute_pick_and_place, args=(detected_object, target_place))
                t.daemon = True
                t.start()

    cap.release()
    cv2.destroyAllWindows()
    if ws is not None:
        try:
            ws.close()
        except Exception:
            pass
    print("\n[OK] Đã thoát chương trình an toàn!")

if __name__ == "__main__":
    main()
