"""
HỆ THỐNG GẮP - ĐẶT TỰ ĐỘNG BẰNG THỊ GIÁC MÁY TÍNH (BƯỚC 5 - HOÀN THIỆN)
Đồ án Cơ cấu chấp hành - UET
Quy trình: Camera nhận diện vật -> Tính tọa độ & góc xoay -> Gửi WebSocket tới ESP32 -> Gắp & Đặt tự động
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

# Kích thước bàn & Tầm với
PAPER_W = 420.0
PAPER_H = 297.0
ROBOT_OFFSET_X = 55.0
R_MAX = 200.0

REAL_CORNERS = np.float32([
    [-PAPER_W / 2.0, ROBOT_OFFSET_X + PAPER_H], # Góc Trên - Trái
    [ PAPER_W / 2.0, ROBOT_OFFSET_X + PAPER_H], # Góc Trên - Phải
    [ PAPER_W / 2.0, ROBOT_OFFSET_X],           # Góc Dưới - Phải
    [-PAPER_W / 2.0, ROBOT_OFFSET_X]            # Góc Dưới - Trái
])

# Biến toàn cục
H_matrix = None
calib_points = []
is_calibrating = False
target_place = None
detected_object = None
is_running_sequence = False
status_message = "SAN SANG: Click chuot chon dich tha -> Bam [SPACE] de Gap!"

# WebSocket Client
ws = None

def init_websocket():
    global ws
    try:
        ws = websocket.create_connection(WS_URL, timeout=3)
        print("[OK] Đã kết nối WebSocket thành công tới ESP32!")
        # Đặt tốc độ ổn định
        ws.send("spd 1.0")
        return True
    except Exception as e:
        print(f"[!] Chưa kết nối được WebSocket ESP32: {e}")
        ws = None
        return False

def send_robot_cmd(cmd):
    global ws
    if ws is None:
        init_websocket()
    if ws is not None:
        try:
            ws.send(cmd)
            return True
        except Exception as e:
            print(f"[!] Lỗi gửi lệnh '{cmd}': {e}")
            ws = None
    return False

def load_calibration():
    global H_matrix, calib_points
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                H_matrix = np.array(data["H_matrix"], dtype=np.float32)
                calib_points = data.get("calib_points", [])
                return True
        except Exception as e:
            print(f"[!] Lỗi đọc file config: {e}")
    return False

def pixel_to_robot(u, v):
    if H_matrix is None:
        return None, None
    pt = np.array([u, v, 1.0], dtype=np.float32)
    res = np.dot(H_matrix, pt)
    if res[2] == 0:
        return None, None
    y_r = res[0] / res[2] # Trục Y ngang (âm: trái, dương: phải)
    x_r = res[1] / res[2] # Trục X dọc (vươn xa)
    return x_r, y_r

def calculate_ik(x_r, y_r, z_height_state="approach", obj_angle=0):
    """
    Tính toán Động học nghịch (IK) chuyển đổi tọa độ thực tế sang góc 6 servo.
    z_height_state: 'home', 'approach' (trên cao), 'grasp' (hạ sát mặt bàn), 'lift' (nhấc lên)
    """
    # 1. Khớp J1 (Đế xoay): 90 độ là chính diện, <90 sang phải, >90 sang trái
    alpha_deg = math.degrees(math.atan2(y_r, x_r))
    j1 = int(np.clip(90 - alpha_deg, 0, 180))

    # Khoảng cách mặt phẳng ngang từ tâm trục J1 tới vật
    dist_r = math.sqrt(x_r**2 + y_r**2)
    
    # Chuẩn hóa khoảng cách trong dải 130mm - 195mm
    ratio = np.clip((dist_r - 130.0) / 65.0, 0.0, 1.0)

    # 2. Các khớp J2 (Vai), J3 (Khuỷu), J4 (Cổ tay gập) theo từng trạng thái độ cao:
    if z_height_state == "home":
        j2 = 90
        j3 = 90
        j4 = 90
    elif z_height_state == "approach" or z_height_state == "lift":
        # Tư thế trên cao (cách mặt bàn ~8-10 cm)
        j2 = int(95 - 15 * ratio)
        j3 = int(85 + 25 * ratio)
        j4 = 90
    elif z_height_state == "grasp":
        # Tư thế hạ sát mặt bàn để kẹp vật
        j2 = int(72 - 20 * ratio)
        j3 = int(105 + 30 * ratio)
        j4 = int(105 + 10 * ratio)
    else:
        j2, j3, j4 = 90, 90, 90

    # Giới hạn an toàn cơ khí theo firmware Galaxy Arm
    j2 = int(np.clip(j2, 20, 160))
    j3 = int(np.clip(j3, 20, 160))
    j4 = int(np.clip(j4, 0, 180))

    # 3. Khớp J5 (Xoay cổ tay ôm theo hướng vật)
    j5 = int(np.clip(90 + (obj_angle % 90 - 45), 0, 180))

    return j1, j2, j3, j4, j5

def execute_pick_and_place(pick_data, place_data):
    """Luồng chạy kịch bản gắp - nâng - chuyển - đặt tự động."""
    global is_running_sequence, status_message
    is_running_sequence = True

    px, py = pick_data["x_r"], pick_data["y_r"]
    p_ang = pick_data["angle"]
    tx, ty = place_data["x_r"], place_data["y_r"]

    try:
        # Bước 1: Mở kẹp & đưa tay lên cao trên đỉnh vật
        status_message = "1/6: Dang vuon toi vi tri tren cao cua vat..."
        send_robot_cmd("j 5 100") # Mở kẹp
        time.sleep(0.4)
        j1, j2, j3, j4, j5 = calculate_ik(px, py, z_height_state="approach", obj_angle=p_ang)
        send_robot_cmd(f"j 0 {j1}")
        send_robot_cmd(f"j 4 {j5}")
        send_robot_cmd(f"j 1 {j2}")
        send_robot_cmd(f"j 2 {j3}")
        send_robot_cmd(f"j 3 {j4}")
        time.sleep(1.8)

        # Bước 2: Hạ tay xuống sát vật để kẹp
        status_message = "2/6: Dang ha tay xuong sat vat..."
        j1, j2, j3, j4, j5 = calculate_ik(px, py, z_height_state="grasp", obj_angle=p_ang)
        send_robot_cmd(f"j 1 {j2}")
        send_robot_cmd(f"j 2 {j3}")
        send_robot_cmd(f"j 3 {j4}")
        time.sleep(1.5)

        # Bước 3: Đóng kẹp FinRay kẹp chặt vật
        status_message = "3/6: Dang kep chat vat..."
        send_robot_cmd("j 5 40") # Đóng kẹp FinRay
        time.sleep(1.0)

        # Bước 4: Nhấc bổng vật lên khỏi mặt bàn
        status_message = "4/6: Dang nhac vat len khoi mat ban..."
        j1, j2, j3, j4, j5 = calculate_ik(px, py, z_height_state="lift", obj_angle=p_ang)
        send_robot_cmd(f"j 1 {j2}")
        send_robot_cmd(f"j 2 {j3}")
        send_robot_cmd(f"j 3 {j4}")
        time.sleep(1.2)

        # Bước 5: Xoay sang vị trí đích và hạ xuống
        status_message = "5/6: Dang chuyen sang vi tri dich..."
        j1_t, j2_t, j3_t, j4_t, j5_t = calculate_ik(tx, ty, z_height_state="approach", obj_angle=0)
        send_robot_cmd(f"j 0 {j1_t}")
        send_robot_cmd(f"j 4 {j5_t}")
        send_robot_cmd(f"j 1 {j2_t}")
        send_robot_cmd(f"j 2 {j3_t}")
        send_robot_cmd(f"j 3 {j4_t}")
        time.sleep(2.0)

        # Hạ tay xuống điểm đích
        j1_g, j2_g, j3_g, j4_g, j5_g = calculate_ik(tx, ty, z_height_state="grasp", obj_angle=0)
        send_robot_cmd(f"j 1 {j2_g}")
        send_robot_cmd(f"j 2 {j3_g}")
        send_robot_cmd(f"j 3 {j4_g}")
        time.sleep(1.5)

        # Bước 6: Nhả kẹp thả vật & rút tay về Home
        status_message = "6/6: Tha vat & Rut tay ve vi tri Home!"
        send_robot_cmd("j 5 100") # Mở kẹp nhả vật
        time.sleep(0.8)
        send_robot_cmd("home") # Về vị trí Home
        time.sleep(1.5)

        status_message = "HOAN THANH GAP - DAT THANH CONG! Click chon dich tiep theo."

    except Exception as e:
        status_message = f"LOI THUC THI: {e}"
        print(f"[!] Lỗi chu trình gắp: {e}")
    finally:
        is_running_sequence = False

def mouse_callback(event, x, y, flags, param):
    global target_place, is_calibrating, calib_points, H_matrix

    if event == cv2.EVENT_LBUTTONDOWN:
        if is_calibrating:
            if len(calib_points) < 4:
                calib_points.append([x, y])
                if len(calib_points) == 4:
                    pts_src = np.float32(calib_points)
                    pts_dst = REAL_CORNERS
                    H_matrix, _ = cv2.findHomography(pts_src, pts_dst)
                    data = {
                        "H_matrix": H_matrix.tolist(),
                        "calib_points": calib_points,
                        "paper_w": PAPER_W, "paper_h": PAPER_H, "offset_x": ROBOT_OFFSET_X
                    }
                    with open(CONFIG_FILE, "w") as f:
                        json.dump(data, f, indent=4)
                    is_calibrating = False
        else:
            # Chọn điểm đích cần thả
            xr, yr = pixel_to_robot(x, y)
            if xr is not None:
                d = math.sqrt(xr**2 + yr**2)
                target_place = {
                    "pixel": (x, y),
                    "x_r": round(xr, 1),
                    "y_r": round(yr, 1),
                    "reachable": (d <= R_MAX and xr > 0)
                }
                print(f"[ĐÍCH ĐẾN] X: {target_place['x_r']}mm, Y: {target_place['y_r']}mm | Tầm với: {target_place['reachable']}")

def main():
    global is_calibrating, calib_points, target_place, detected_object, is_running_sequence

    print("=" * 65)
    print("HỆ THỐNG GẮP ĐẶT TỰ ĐỘNG - ROBOT VISION 5-DOF - UET")
    print("=" * 65)

    load_calibration()
    init_websocket()

    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Robot Vision System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Vision System", 1280, 720)
    cv2.setMouseCallback("Robot Vision System", mouse_callback)

    lower_blue = np.array([90, 90, 50])
    upper_blue = np.array([135, 255, 255])

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 1. Vẽ khung tờ giấy A3 đã hiệu chuẩn
        if len(calib_points) == 4:
            cv2.polylines(frame, [np.array(calib_points, np.int32)], True, (255, 100, 0), 2)
            for i, pt in enumerate(calib_points):
                cv2.circle(frame, tuple(pt), 6, (0, 0, 255), -1)

        # 2. Quét tìm vật thể
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        temp_obj = None
        for cnt in contours:
            if cv2.contourArea(cnt) > 350:
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect
                box = np.int32(cv2.boxPoints(rect))
                if w < h:
                    angle += 90

                xr, yr = pixel_to_robot(cx, cy)
                if xr is not None:
                    dist = math.sqrt(xr**2 + yr**2)
                    is_reach = (dist <= R_MAX and xr > 0)
                    color = (0, 255, 0) if is_reach else (0, 0, 255)
                    cv2.drawContours(frame, [box], 0, color, 2)
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

                    txt = f"VAT GAP: X:{xr:.1f} Y:{yr:.1f}mm | {int(angle)} deg"
                    cv2.putText(frame, txt, (int(cx) - 100, int(cy) - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

                    if is_reach:
                        temp_obj = {"x_r": xr, "y_r": yr, "angle": angle}

        detected_object = temp_obj

        # 3. Vẽ điểm đích cần thả
        if target_place is not None:
            tx, ty = target_place["pixel"]
            t_col = (0, 255, 255) if target_place["reachable"] else (0, 0, 255)
            cv2.circle(frame, (tx, ty), 16, t_col, 2)
            cv2.circle(frame, (tx, ty), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"DICH THA: X={target_place['x_r']} Y={target_place['y_r']}mm", 
                        (tx - 90, ty - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, t_col, 2)

        # 4. Thanh thông báo trạng thái
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (900, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, f">> {status_message}", (25, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, "PHIM: [SPACE] Bat dau Gap-Dat | [r] Xoa dich | [h] Ve Home | [q] Thoat", 
                    (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Robot Vision System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            target_place = None
        elif key == ord('h'):
            send_robot_cmd("home")
        elif key == ord(' '): # Phím SPACE: Bắt đầu chu trình gắp - đặt tự động
            if is_running_sequence:
                print("[!] Robot đang chạy chu trình, vui lòng đợi!")
            elif detected_object is None:
                print("[!] Chưa nhận diện được vật thể trong tầm với!")
            elif target_place is None or not target_place["reachable"]:
                print("[!] Vui lòng click chuột chọn ĐIỂM THẢ hợp lệ trên bàn!")
            else:
                print("[*] BẮT ĐẦU CHU TRÌNH GẮP - ĐẶT TỰ ĐỘNG...")
                t = threading.Thread(target=execute_pick_and_place, args=(detected_object, target_place))
                t.daemon = True
                t.start()

    cap.release()
    cv2.destroyAllWindows()
    if ws is not None:
        ws.close()

if __name__ == "__main__":
    main()
