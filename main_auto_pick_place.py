"""
================================================================================
HỆ THỐNG GẮP - ĐẶT TỰ ĐỘNG BẰNG THỊ GIÁC MÁY TÍNH & AI YOLOV8 (PHIÊN BẢN V3.1)
Đồ án Cơ cấu chấp hành - UET (Đại học Công nghệ - ĐHQGHN)
--------------------------------------------------------------------------------
Cập nhật V3.1:
1. Đảo chiều quay đế J1 chuẩn xác (j1_invert = True).
2. Tự động kết nối lại WebSocket ngầm mỗi 2 giây (Auto-Reconnect khi đổi Wi-Fi).
3. Cảnh báo ĐỎ rõ ràng trên màn hình khi chưa kết nối vào Wi-Fi 'RobotArm'.
4. Chặn lệnh di chuyển giả khi chưa có kết nối phần cứng.
================================================================================
"""

import os
import sys

os.environ["YOLO_OFFLINE"] = "1"
os.environ["ULTRALYTICS_OFFLINE"] = "1"

import cv2
import numpy as np
import json
import math
import time
import websocket
import threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "calibration.json")
LOCAL_YOLO_PATH = os.path.join(SCRIPT_DIR, "yolov8n.pt")
CUSTOM_YOLO_PATH = os.path.join(SCRIPT_DIR, "best_robot_model.pt")
WS_URL = "ws://192.168.4.1:81/"

ws_lock = threading.Lock()
ws = None
ws_connected = False

# ==========================================
# CẤU HÌNH & HIỆU CHUẨN
# ==========================================
config_data = {
    "calib_mode": "base_scale",
    "base_pixel": [716, 117],
    "scale_mm_per_pixel": 0.6438,
    "r_min": 80.0,
    "r_max": 340.0,
    "reach_offset": -35.0,         # Offset tam vuon xa/gan (mm), giup tinh chinh dung tam vat
    "j1_invert": True,             # Huong quay khop de J1
    "gripper_open": 35,
    "gripper_close": 110,
    "grasp_depth_offset": 0,
    "auto_sort_enabled": False,
    "use_ai_yolo": True,
    "color_mode": "BLUE",
    "active_slot": "1",
    "slots": {
        "1": {"name": "Khay 1 (Trai)", "x_r": 180.0, "y_r": -90.0, "shape": "TRON"},
        "2": {"name": "Khay 2 (Giua)", "x_r": 230.0, "y_r": 0.0,   "shape": "VUONG"},
        "3": {"name": "Khay 3 (Phai)", "x_r": 180.0, "y_r": 90.0,  "shape": "TAM GIAC"}
    }
}

COLOR_MODES = ["BLUE", "ALL", "RED", "YELLOW", "GREEN"]

is_calibrating = False
calib_step = 0
calib_pts = []
target_place = None
manual_pick = None
detected_object = None
is_running_sequence = False
status_message = "SAN SANG: Click TRAI chon vat, Click PHAI chon dich -> Bam [SPACE] de Gap!"

yolo_model = None
model_name = "Dang khoi tao..."

def init_yolo():
    global yolo_model, model_name
    try:
        from ultralytics import YOLO
        if os.path.exists(CUSTOM_YOLO_PATH):
            yolo_model = YOLO(CUSTOM_YOLO_PATH)
            model_name = "Custom Model"
            return True
        elif os.path.exists(LOCAL_YOLO_PATH):
            yolo_model = YOLO(LOCAL_YOLO_PATH)
            model_name = "YOLOv8 Nano (Local)"
            return True
        else:
            yolo_model = None
            model_name = "Khong co model"
            config_data["use_ai_yolo"] = False
            return False
    except Exception as e:
        yolo_model = None
        model_name = "Loi nap model"
        config_data["use_ai_yolo"] = False
        return False

def load_config():
    global config_data
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                config_data.update(loaded)
                return True
        except Exception:
            pass
    return False

def save_config():
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        return True
    except Exception:
        return False

def get_ws():
    global ws, ws_connected
    if ws is not None:
        try:
            ws.sock.settimeout(0.3)
            return ws
        except Exception:
            ws = None
            ws_connected = False

    try:
        ws = websocket.create_connection(WS_URL, timeout=1.2)
        ws_connected = True
        print("[OK] Đã kết nối WebSocket thành công tới ESP32!")
        ws.send("spd 1.0")
        return ws
    except Exception:
        ws = None
        ws_connected = False
        return None

def auto_reconnect_loop():
    """Luồng chạy ngầm tự động thử kết nối lại ESP32 mỗi 2.5s khi mất kết nối."""
    global ws_connected
    while True:
        if not ws_connected:
            get_ws()
        time.sleep(2.5)

def send_robot_cmd(cmd):
    global ws, ws_connected, status_message
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
        else:
            status_message = "LOI: CHUA KET NOI ROBOT! Vui long ket noi WiFi 'RobotArm'!"
        return False

def pixel_to_robot(u, v):
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
    u0, v0 = config_data["base_pixel"]
    scale = config_data["scale_mm_per_pixel"]
    u = int(round(u0 + y_r / scale))
    v = int(round(v0 + x_r / scale))
    return u, v

def calculate_ik(x_r, y_r, state="approach", obj_angle=0):
    j1_invert = config_data.get("j1_invert", True)
    grasp_offset = config_data.get("grasp_depth_offset", 0)
    reach_offset = config_data.get("reach_offset", -35.0)
    r_min = config_data.get("r_min", 80.0)
    r_max = config_data.get("r_max", 340.0)
    
    # 1. Khớp J1 (Đế xoay)
    alpha = math.degrees(math.atan2(y_r, x_r))
    if not j1_invert:
        j1 = int(round(90.0 + alpha))
    else:
        j1 = int(round(90.0 - alpha))
    j1 = int(np.clip(j1, 10, 170))

    # Khoảng cách bán kính R (co/duỗi theo reach_offset)
    r = math.sqrt(x_r**2 + y_r**2)
    r_eff = max(r_min, r + reach_offset)
    ratio = float(np.clip((r_eff - r_min) / (r_max - r_min + 1e-5), 0.0, 1.0))

    if state == "home":
        return 90, 90, 90, 90, 90

    elif state in ("approach", "lift"):
        j2 = int(round(80.0 + 26.0 * ratio))
        j3 = int(round(72.0 + 24.0 * ratio))
        j4 = 85

    elif state == "grasp":
        j2 = int(round(105.0 + 32.0 * ratio + grasp_offset))
        j3 = int(round(48.0 + 34.0 * ratio - grasp_offset))
        j4 = int(round(118.0 - 10.0 * ratio))
    else:
        j2, j3, j4 = 90, 90, 90

    j2 = int(np.clip(j2, 20, 160))
    j3 = int(np.clip(j3, 20, 160))
    j4 = int(np.clip(j4, 10, 170))

    j5 = int(round(np.clip(90.0 + obj_angle, 15, 165)))
    return j1, j2, j3, j4, j5

def get_color_mask(hsv, mode="BLUE"):
    masks = []
    if mode in ("ALL", "BLUE"):
        m_blue = cv2.inRange(hsv, np.array([92, 50, 40]), np.array([130, 255, 255]))
        masks.append(m_blue)
    if mode in ("ALL", "RED"):
        m_red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        m_red2 = cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
        masks.append(cv2.bitwise_or(m_red1, m_red2))
    if mode in ("ALL", "YELLOW"):
        m_yel = cv2.inRange(hsv, np.array([18, 70, 70]), np.array([38, 255, 255]))
        masks.append(m_yel)
    if mode in ("ALL", "GREEN"):
        m_grn = cv2.inRange(hsv, np.array([38, 50, 50]), np.array([88, 255, 255]))
        masks.append(m_grn)

    if len(masks) == 0:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)

    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)

    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    return combined

def execute_pick_and_place(pick_data, place_data):
    global is_running_sequence, status_message, ws_connected
    if not ws_connected:
        status_message = "LOI: CHUA KET NOI ROBOT! Vui long ket noi WiFi 'RobotArm'!"
        return

    is_running_sequence = True
    px, py = pick_data["x_r"], pick_data["y_r"]
    p_ang = pick_data.get("angle", 0.0)
    tx, ty = place_data["x_r"], place_data["y_r"]

    open_angle = config_data["gripper_open"]
    close_angle = config_data["gripper_close"]

    try:
        status_message = "1/6: Dang mo kep & vuon toi tren dinh vat..."
        send_robot_cmd(f"j 5 {open_angle}")
        time.sleep(0.4)

        j1, j2, j3, j4, j5 = calculate_ik(px, py, state="approach", obj_angle=p_ang)
        send_robot_cmd(f"j 0 {j1}")
        send_robot_cmd(f"j 4 {j5}")
        send_robot_cmd(f"j 1 {j2}")
        send_robot_cmd(f"j 2 {j3}")
        send_robot_cmd(f"j 3 {j4}")
        time.sleep(1.8)

        status_message = "2/6: Dang ha tay xuong sat mat ban..."
        j1_g, j2_g, j3_g, j4_g, j5_g = calculate_ik(px, py, state="grasp", obj_angle=p_ang)
        send_robot_cmd(f"j 1 {j2_g}")
        send_robot_cmd(f"j 2 {j3_g}")
        send_robot_cmd(f"j 3 {j4_g}")
        time.sleep(1.4)

        status_message = "3/6: Dang kep chat vat..."
        send_robot_cmd(f"j 5 {close_angle}")
        time.sleep(1.2)

        status_message = "4/6: Dang nhac vat len cao..."
        send_robot_cmd(f"j 1 {j2}")
        send_robot_cmd(f"j 2 {j3}")
        send_robot_cmd(f"j 3 {j4}")
        time.sleep(1.2)

        status_message = "5/6: Dang chuyen sang vi tri dich..."
        j1_t, j2_t, j3_t, j4_t, j5_t = calculate_ik(tx, ty, state="approach", obj_angle=0)
        send_robot_cmd(f"j 0 {j1_t}")
        send_robot_cmd(f"j 4 {j5_t}")
        send_robot_cmd(f"j 1 {j2_t}")
        send_robot_cmd(f"j 2 {j3_t}")
        send_robot_cmd(f"j 3 {j4_t}")
        time.sleep(1.8)

        j1_tg, j2_tg, j3_tg, j4_tg, j5_tg = calculate_ik(tx, ty, state="grasp", obj_angle=0)
        send_robot_cmd(f"j 1 {j2_tg}")
        send_robot_cmd(f"j 2 {j3_tg}")
        send_robot_cmd(f"j 3 {j4_tg}")
        time.sleep(1.4)

        status_message = "6/6: Tha vat & Rut tay ve Home..."
        send_robot_cmd(f"j 5 {open_angle}")
        time.sleep(0.8)
        send_robot_cmd("home")
        time.sleep(1.6)

        status_message = f"THANH CONG! Da gap vat tu ({px:.0f}, {py:.0f}) sang ({tx:.0f}, {ty:.0f})!"
    except Exception as e:
        status_message = f"LOI THUC THI: {e}"
    finally:
        is_running_sequence = False

def trigger_hover_test():
    global is_running_sequence, status_message, ws_connected
    if not ws_connected:
        status_message = "LOI: CHUA KET NOI ROBOT! Vui long ket noi WiFi 'RobotArm'!"
        print("[!] Chưa kết nối ESP32. Hãy kết nối Wi-Fi 'RobotArm'!")
        return

    if is_running_sequence:
        return
    pick_obj = manual_pick if manual_pick is not None else detected_object
    if pick_obj is None:
        status_message = "Chua co vat nao de Test! Click TRAI chon vat tren ban."
        return

    def _worker(target):
        global is_running_sequence, status_message
        is_running_sequence = True
        try:
            r_eff = max(80.0, target['dist'] + config_data.get('reach_offset', -35.0))
            status_message = f"TEST HOVER: Dang vuon den lo lung tren vat (R={target['dist']}mm -> R_thuc={r_eff:.0f}mm)..."
            send_robot_cmd(f"j 5 {config_data['gripper_open']}")
            j1, j2, j3, j4, j5 = calculate_ik(target["x_r"], target["y_r"], state="approach", obj_angle=target.get("angle", 0))
            send_robot_cmd(f"j 0 {j1}")
            send_robot_cmd(f"j 4 {j5}")
            send_robot_cmd(f"j 1 {j2}")
            send_robot_cmd(f"j 2 {j3}")
            send_robot_cmd(f"j 3 {j4}")
            time.sleep(3.0)
            send_robot_cmd("home")
            time.sleep(1.5)
            status_message = "HOAN TAT TEST HOVER: Tam robot da can chuan!"
        except Exception as e:
            status_message = f"LOI: {e}"
        finally:
            is_running_sequence = False

    th = threading.Thread(target=_worker, args=(pick_obj,))
    th.daemon = True
    th.start()

def trigger_grasp_test():
    global is_running_sequence, status_message, ws_connected
    if not ws_connected:
        status_message = "LOI: CHUA KET NOI ROBOT! Vui long ket noi WiFi 'RobotArm'!"
        print("[!] Chưa kết nối ESP32. Hãy kết nối Wi-Fi 'RobotArm'!")
        return

    if is_running_sequence:
        return
    pick_obj = manual_pick if manual_pick is not None else detected_object
    if pick_obj is None:
        status_message = "Chua co vat nao de Test! Click TRAI chon vat tren ban."
        return

    def _worker(target):
        global is_running_sequence, status_message
        is_running_sequence = True
        try:
            r_eff = max(80.0, target['dist'] + config_data.get('reach_offset', -35.0))
            status_message = f"TEST Z: Dang ha tay xuong sat ban (R={target['dist']}mm -> R_thuc={r_eff:.0f}mm)..."
            send_robot_cmd(f"j 5 {config_data['gripper_open']}")
            j1, j2, j3, j4, j5 = calculate_ik(target["x_r"], target["y_r"], state="grasp", obj_angle=target.get("angle", 0))
            send_robot_cmd(f"j 0 {j1}")
            send_robot_cmd(f"j 4 {j5}")
            send_robot_cmd(f"j 1 {j2}")
            send_robot_cmd(f"j 2 {j3}")
            send_robot_cmd(f"j 3 {j4}")
            time.sleep(3.0)
            send_robot_cmd("home")
            time.sleep(1.5)
            status_message = "HOAN TAT TEST Z! Bam [ [ ] de ha sau them hoac [ ] ] de nang len."
        except Exception as e:
            status_message = f"LOI: {e}"
        finally:
            is_running_sequence = False

    th = threading.Thread(target=_worker, args=(pick_obj,))
    th.daemon = True
    th.start()

def trigger_pick_and_place():
    global is_running_sequence, status_message, ws_connected
    if not ws_connected:
        status_message = "LOI: CHUA KET NOI ROBOT! Vui long ket noi WiFi 'RobotArm'!"
        print("[!] Chưa kết nối ESP32. Hãy kết nối Wi-Fi 'RobotArm'!")
        return

    if is_running_sequence:
        return

    pick_obj = manual_pick if manual_pick is not None else detected_object
    if pick_obj is None:
        status_message = "CANH BAO: Vui long Click CHUOT TRAI chon vat can gap tren ban!"
        return

    r_min = config_data.get("r_min", 80.0)
    r_max = config_data.get("r_max", 340.0)
    if not (r_min <= pick_obj["dist"] <= r_max):
        status_message = f"VAT NGOAI TAM VOI (R={pick_obj['dist']}mm). Hay keo vat vao gan hon!"
        return

    place_obj = target_place
    if place_obj is None:
        slot_2 = config_data["slots"]["2"]
        su, sv = robot_to_pixel(slot_2["x_r"], slot_2["y_r"])
        place_obj = {
            "pixel": (su, sv), "x_r": slot_2["x_r"], "y_r": slot_2["y_r"],
            "dist": math.sqrt(slot_2["x_r"]**2 + slot_2["y_r"]**2),
            "reachable": True, "label": slot_2["name"]
        }

    t = threading.Thread(target=execute_pick_and_place, args=(pick_obj, place_obj))
    t.daemon = True
    t.start()

def toggle_ai_yolo():
    global status_message
    config_data["use_ai_yolo"] = not config_data.get("use_ai_yolo", True)
    save_config()
    status_message = f"Che do AI YOLOv8: {'BAT' if config_data['use_ai_yolo'] else 'TAT'}"

def mouse_callback(event, x, y, flags, param):
    global is_calibrating, calib_step, calib_pts, target_place, manual_pick
    global status_message, config_data

    # Click các nút ảo ở đáy
    if event == cv2.EVENT_LBUTTONDOWN and y >= 660:
        if 15 <= x <= 165:
            is_calibrating = True
            calib_step = 1
            calib_pts = []
            status_message = "CALIB (1/3): Click vao TAM CHAN DE ROBOT (Truc J1) o mep tren video!"
            return
        elif 175 <= x <= 345:
            toggle_ai_yolo()
            return
        elif 355 <= x <= 505:
            trigger_hover_test()
            return
        elif 515 <= x <= 655:
            trigger_grasp_test()
            return
        elif 665 <= x <= 825:
            cur_grip = config_data["gripper_close"]
            send_robot_cmd(f"j 5 {cur_grip}")
            status_message = f"Da dong kep ({cur_grip} do)!"
            return
        elif 835 <= x <= 975:
            send_robot_cmd("home")
            status_message = "Da gui lenh ve Home an toan!"
            return
        elif 985 <= x <= 1265:
            trigger_pick_and_place()
            return

    # Chế độ hiệu chuẩn
    if is_calibrating and event == cv2.EVENT_LBUTTONDOWN:
        if calib_step == 1:
            config_data["base_pixel"] = [x, y]
            calib_step = 2
            status_message = "CALIB (2/3): Dat cay thuoc ke len ban -> Click vao vach 0 CM!"
            return
        elif calib_step == 2:
            calib_pts = [[x, y]]
            calib_step = 3
            status_message = "CALIB (3/3): Click vao vach 10 CM (100mm) tren cay thuoc ke!"
            return
        elif calib_step == 3:
            calib_pts.append([x, y])
            x1, y1 = calib_pts[0]
            x2, y2 = calib_pts[1]
            dist_px = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if dist_px > 15:
                scale = 100.0 / dist_px
                config_data["scale_mm_per_pixel"] = round(scale, 4)
                save_config()
                is_calibrating = False
                calib_step = 0
                status_message = f"HIEU CHUAN XONG! Ti le: {scale:.4f} mm/px. Goc Robot: {config_data['base_pixel']}"
            else:
                status_message = "Khoang cach qua ngan, vui long click lai vach 10 cm!"
            return

    # Chế độ bình thường
    if not is_calibrating:
        if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY):
            config_data["base_pixel"] = [x, y]
            save_config()
            status_message = f"Da doi GOC ROBOT J1 ve Pixel ({x}, {y})!"
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            xr, yr, dist, theta = pixel_to_robot(x, y)
            r_min = config_data.get("r_min", 80.0)
            r_max = config_data.get("r_max", 340.0)
            is_reach = (r_min <= dist <= r_max and xr > 0)
            manual_pick = {
                "pixel": (x, y),
                "x_r": round(xr, 1),
                "y_r": round(yr, 1),
                "dist": round(dist, 1),
                "angle": 0.0,
                "shape": "TU CHON",
                "reachable": is_reach
            }
            reach_str = "TRONG TAM VOI" if is_reach else "NGOAI TAM VOI"
            status_message = f"DA CHON VAT GAP: X={manual_pick['x_r']} Y={manual_pick['y_r']}mm (R={manual_pick['dist']}mm - {reach_str}). Click PHAI chon DICH THA!"
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            xr, yr, dist, theta = pixel_to_robot(x, y)
            r_min = config_data.get("r_min", 80.0)
            r_max = config_data.get("r_max", 340.0)
            is_reach = (r_min <= dist <= r_max and xr > 0)
            target_place = {
                "pixel": (x, y),
                "x_r": round(xr, 1),
                "y_r": round(yr, 1),
                "dist": round(dist, 1),
                "reachable": is_reach,
                "label": "DICH TU CHON"
            }
            reach_str = "TRONG TAM VOI" if is_reach else "NGOAI TAM VOI"
            status_message = f"DA CHON DICH THA: X={target_place['x_r']} Y={target_place['y_r']}mm. Bam [SPACE] de Gap!"
            return

def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(3):
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
    if cap is not None:
        cap.release()
    return None

def main():
    global is_calibrating, calib_step, calib_pts, target_place, manual_pick, detected_object
    global is_running_sequence, status_message, ws_connected

    print("=" * 75)
    print("HỆ THỐNG GẮP - ĐẶT TỰ ĐỘNG BẰNG THỊ GIÁC MÁY TÍNH & AI YOLOV8 (V3.1)")
    print("=" * 75)

    load_config()

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

    init_yolo()
    
    # Khởi chạy luồng tự động kết nối lại WebSocket ngầm
    t_reconnect = threading.Thread(target=auto_reconnect_loop, daemon=True)
    t_reconnect.start()

    active_slot = config_data.get("active_slot", "1")
    slot_info = config_data["slots"].get(active_slot, config_data["slots"]["1"])
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

        u0, v0 = config_data["base_pixel"]
        scale = config_data["scale_mm_per_pixel"]
        r_min = config_data.get("r_min", 80.0)
        r_max = config_data.get("r_max", 340.0)
        use_ai = config_data.get("use_ai_yolo", True) and (yolo_model is not None)

        # 1. Vẽ cung tròn tầm với
        cv2.circle(frame, (u0, v0), 9, (0, 0, 255), -1)
        cv2.circle(frame, (u0, v0), 16, (0, 255, 255), 2)
        cv2.putText(frame, "GOC ROBOT (0,0)", (u0 - 65, max(25, v0 - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.line(frame, (u0, v0), (u0, v0 + int(r_max / scale)), (100, 100, 100), 1, cv2.LINE_AA)

        r_min_px = int(r_min / scale)
        cv2.ellipse(frame, (u0, v0), (r_min_px, r_min_px), 0, 30, 150, (0, 100, 255), 1, cv2.LINE_AA)

        r_mid1_px = int(180.0 / scale)
        cv2.ellipse(frame, (u0, v0), (r_mid1_px, r_mid1_px), 0, 30, 150, (0, 255, 120), 1, cv2.LINE_AA)

        r_mid2_px = int(260.0 / scale)
        cv2.ellipse(frame, (u0, v0), (r_mid2_px, r_mid2_px), 0, 30, 150, (0, 255, 180), 1, cv2.LINE_AA)

        r_max_px = int(r_max / scale)
        cv2.ellipse(frame, (u0, v0), (r_max_px, r_max_px), 0, 30, 150, (0, 100, 255), 2, cv2.LINE_AA)

        # 2. Vẽ 3 khay đích mặc định
        for s_id, s_data in config_data["slots"].items():
            su, sv = robot_to_pixel(s_data["x_r"], s_data["y_r"])
            is_active = (target_place is not None and abs(target_place["x_r"] - s_data["x_r"]) < 2 and abs(target_place["y_r"] - s_data["y_r"]) < 2)
            s_col = (0, 255, 255) if is_active else (180, 180, 180)
            cv2.circle(frame, (su, sv), 18, s_col, 2 if is_active else 1)
            cv2.circle(frame, (su, sv), 4, s_col, -1)
            cv2.putText(frame, f"[{s_id}] {s_data['shape']}", (su - 35, sv + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, s_col, 1)

        # 3. Nhận diện vật thể
        valid_objects = []

        if use_ai:
            try:
                results = yolo_model(frame, verbose=False, conf=0.25)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        cls_id = int(box.cls[0])
                        cls_name = yolo_model.names.get(cls_id, str(cls_id))
                        conf = float(box.conf[0])

                        if cls_name == "person":
                            continue

                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        xr, yr, dist, theta = pixel_to_robot(cx, cy)

                        if dist < 65.0:
                            continue

                        is_reach = (r_min <= dist <= r_max and xr > 0)
                        box_col = (0, 255, 0) if is_reach else (0, 0, 255)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 2)
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                        tag = f"[AI: {cls_name} {conf:.0%}] X:{xr:.0f} Y:{yr:.0f} R:{dist:.0f}mm"
                        cv2.putText(frame, tag, (x1, max(25, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)

                        if is_reach:
                            valid_objects.append({
                                "x_r": xr, "y_r": yr, "angle": 0.0, "dist": dist,
                                "shape": cls_name, "area": (x2 - x1) * (y2 - y1), "pixel": (cx, cy)
                            })
            except Exception:
                pass

        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cur_color = config_data.get("color_mode", "BLUE")
            mask = get_color_mask(hsv, cur_color)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if 100 < area < 3500:
                    rect = cv2.minAreaRect(cnt)
                    (cx, cy), (w, h), angle = rect

                    aspect = max(w, h) / (min(w, h) + 1e-5)
                    if aspect > 3.2:
                        continue

                    if w < h:
                        angle += 90.0
                    if angle > 90.0:
                        angle -= 180.0

                    xr, yr, dist, theta = pixel_to_robot(cx, cy)
                    if dist < 65.0:
                        continue

                    is_reach = (r_min <= dist <= r_max and xr > 0)
                    box = np.int32(cv2.boxPoints(rect))

                    color = (0, 255, 0) if is_reach else (0, 0, 255)
                    cv2.drawContours(frame, [box], 0, color, 2)
                    cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

                    tag = f"[COLOR] X:{xr:.0f} Y:{yr:.0f} R:{dist:.0f}mm"
                    cv2.putText(frame, tag, (int(cx) - 80, int(cy) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 2)

                    if is_reach:
                        valid_objects.append({
                            "x_r": xr, "y_r": yr, "angle": angle, "dist": dist,
                            "shape": "VAT_MAU", "area": area, "pixel": (int(cx), int(cy))
                        })

        if not is_running_sequence:
            if len(valid_objects) > 0:
                valid_objects.sort(key=lambda o: abs(o["area"] - 500))
                detected_object = valid_objects[0]
            else:
                detected_object = None

        # 4. Vẽ vật gắp
        active_pick = manual_pick if manual_pick is not None else detected_object
        if active_pick is not None:
            px, py = active_pick["pixel"]
            p_col = (0, 255, 0) if active_pick.get("reachable", True) else (0, 0, 255)
            cv2.circle(frame, (px, py), 24, p_col, 2)
            cv2.circle(frame, (px, py), 14, p_col, 1)
            cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
            p_label = "VAT GAP (CHON TAY)" if manual_pick is not None else f"VAT GAP ({active_pick.get('shape', '')})"
            cv2.putText(frame, f"[{p_label}] X={active_pick['x_r']} Y={active_pick['y_r']}mm",
                        (px - 110, py - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, p_col, 2)

        # 5. Vẽ điểm thả
        if target_place is not None:
            tx, ty = target_place["pixel"]
            t_col = (0, 255, 255) if target_place.get("reachable", True) else (0, 0, 255)
            cv2.circle(frame, (tx, ty), 24, t_col, 2)
            cv2.circle(frame, (tx, ty), 14, t_col, 1)
            cv2.circle(frame, (tx, ty), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"[DICH THA] X={target_place['x_r']} Y={target_place['y_r']}mm ({target_place.get('label', '')})",
                        (tx - 110, ty - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, t_col, 2)

        # 6. Header HUD
        top_hud = frame.copy()
        cv2.rectangle(top_hud, (10, 10), (1270, 95), (20, 20, 20), -1)
        cv2.addWeighted(top_hud, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, f">> {status_message}", (25, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Trạng thái ESP32 nổi bật
        if ws_connected:
            ws_str = "ESP32: DA KET NOI"
            ws_col = (0, 255, 0)
        else:
            ws_str = "ESP32: CHUA BAT WIFI RobotArm! [w]"
            ws_col = (0, 0, 255)
            # Cảnh báo nhấp nháy trên màn hình
            cv2.rectangle(frame, (850, 15), (1260, 50), (0, 0, 200), -1)
            cv2.putText(frame, "CHUA KET NOI WIFI RobotArm!", (860, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        ai_str = f"AI: {'ON' if use_ai else 'OFF'} [y]"
        j1_str = f"J1-Rev: {'ON' if config_data.get('j1_invert', True) else 'OFF'} [i]"
        reach_offset_val = config_data.get('reach_offset', -35.0)
        reach_str = f"Tam vuon: {reach_offset_val:+.0f}mm [+/-]"
        z_str = f"Z: {config_data.get('grasp_depth_offset', 0):+d} [[]/]]"

        status_line = f"{ws_str} | {ai_str} | {j1_str} | {reach_str} | {z_str}"
        cv2.putText(frame, status_line, (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.46, ws_col, 1)

        guide_line = "PHIM: [t] Thu Ngam | [g] Thu Gap | [+/-] Vuon Xa/Gan | [ [ / ] ] Ha/Nang Z | [o/p] Kep | [SPACE] Gap-Tha"
        cv2.putText(frame, guide_line, (25, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # 7. Bảng Calib khổ lớn
        if is_calibrating:
            box = frame.copy()
            cv2.rectangle(box, (160, 150), (1120, 470), (15, 15, 15), -1)
            cv2.addWeighted(box, 0.88, frame, 0.12, 0, frame)
            cv2.rectangle(frame, (160, 150), (1120, 470), (0, 215, 255), 3)

            cv2.putText(frame, "=== CHE DO HIEU CHUAN THUOC KE 3 BUOC (KHONG CAN GIAY A3) ===",
                        (190, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2)

            if calib_step == 1:
                step_title = "BUOC 1/3: CLICK CHUOT VAO TAM CHAN DE ROBOT (TRUC XOAY J1)"
                step_desc1 = "Nhin vao mep tren cua video, click chuot trai dung vao giua de xoay robot."
                step_desc2 = "Goc toa do (0, 0) se duoc thiet lap tai diem nay."
            elif calib_step == 2:
                step_title = "BUOC 2/3: DAT THUOC KE LEN BAN -> CLICK VAO VACH 0 CM"
                step_desc1 = "Dat cay thuoc ke hoc sinh ngang hoac doc tren mat ban trong tam nhin camera."
                step_desc2 = "Click chuot trai dung vao vach so 0 tren thuoc ke."
            elif calib_step == 3:
                step_title = "BUOC 3/3: CLICK CHUOT VAO VACH 10 CM (100mm) TREN CAY THUOC"
                step_desc1 = "Click chuot trai vao vach 10 cm. He thong se tu dong tinh ti le mm/pixel."
                step_desc2 = "Du lieu se duoc luu vinh vien vao calibration.json!"
                if len(calib_pts) == 1:
                    cv2.circle(frame, tuple(calib_pts[0]), 8, (0, 255, 0), -1)
            else:
                step_title = "HIEU CHUAN HOAN TAT!"
                step_desc1 = ""
                step_desc2 = ""

            cv2.putText(frame, step_title, (190, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 165, 255), 2)
            cv2.putText(frame, step_desc1, (190, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            cv2.putText(frame, step_desc2, (190, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

            cv2.putText(frame, ">> Bam [ESC] tren ban phim de Huy hieu chuan bat ky luc nao.",
                        (190, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 100, 255), 1)

        # 8. Nút bấm ảo ở đáy
        btn_bar = frame.copy()
        cv2.rectangle(btn_bar, (10, 660), (1270, 715), (25, 25, 25), -1)
        cv2.addWeighted(btn_bar, 0.85, frame, 0.15, 0, frame)

        ai_btn_text = f"[2] AI: {'ON' if use_ai else 'OFF'} (y)"
        ai_btn_col = (0, 255, 255) if use_ai else (150, 150, 150)

        buttons = [
            ((15, 665), (165, 710), "[1] CALIB (k)", (0, 165, 255)),
            ((175, 665), (345, 710), ai_btn_text, ai_btn_col),
            ((355, 665), (505, 710), "[3] TEST NGAM (t)", (255, 200, 0)),
            ((515, 665), (655, 710), "[4] TEST Z (g)", (0, 255, 200)),
            ((665, 665), (825, 710), "[5] KEP (o/p)", (200, 150, 100)),
            ((835, 665), (975, 710), "[6] HOME (h)", (180, 180, 180)),
            ((985, 665), (1265, 710), ">> GAP TU DONG (SPACE) <<", (0, 255, 0))
        ]

        for pt1, pt2, label, col in buttons:
            cv2.rectangle(frame, pt1, pt2, col, 1)
            cv2.putText(frame, label, (pt1[0] + 8, pt1[1] + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44 if len(label) > 16 else 0.48, col, 1 if len(label) > 20 else 2)

        cv2.imshow("Robot Vision System", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == 27:
            if is_calibrating:
                is_calibrating = False
                calib_step = 0
                status_message = "Da huy che do Hieu chuan!"

        elif key == ord('y'):
            toggle_ai_yolo()

        elif key == ord('k'):
            is_calibrating = True
            calib_step = 1
            calib_pts = []
            status_message = "CALIB (1/3): Click vao TAM CHAN DE ROBOT (Truc J1) o mep tren video!"

        elif key in (ord('+'), ord('='), ord('.')):
            config_data["reach_offset"] = config_data.get("reach_offset", -35.0) + 5.0
            save_config()
            status_message = f"Da tang tam vuon (+5mm xa hon): Offset={config_data['reach_offset']:+.0f}mm. Bam [t] de thu!"

        elif key in (ord('-'), ord('_'), ord(',')):
            config_data["reach_offset"] = config_data.get("reach_offset", -35.0) - 5.0
            save_config()
            status_message = f"Da thu tam vuon (-5mm gan lai): Offset={config_data['reach_offset']:+.0f}mm. Bam [t] de thu!"

        elif key == ord('r'):
            manual_pick = None
            status_message = "Da xoa vat chon tay! Quay lai che do tu dong."

        elif key == ord('t'):
            trigger_hover_test()

        elif key == ord('g'):
            trigger_grasp_test()

        elif key == ord('o'):
            open_ang = config_data["gripper_open"]
            send_robot_cmd(f"j 5 {open_ang}")
            status_message = f"Da mo kep ({open_ang} do)!"

        elif key == ord('p'):
            close_ang = config_data["gripper_close"]
            send_robot_cmd(f"j 5 {close_ang}")
            status_message = f"Da dong kep chat ({close_ang} do)!"

        elif key == ord('['):
            config_data["grasp_depth_offset"] += 2
            save_config()
            status_message = f"Da ha sau them: {config_data['grasp_depth_offset']:+d} do. Bam [g] de thu!"

        elif key == ord(']'):
            config_data["grasp_depth_offset"] -= 2
            save_config()
            status_message = f"Da nang cao len: {config_data['grasp_depth_offset']:+d} do. Bam [g] de thu!"

        elif key == ord('i'):
            config_data["j1_invert"] = not config_data.get("j1_invert", True)
            save_config()
            status_message = f"Da dao chieu de J1: {'ON' if config_data['j1_invert'] else 'OFF'}"

        elif key in (ord('1'), ord('2'), ord('3')):
            slot_id = chr(key)
            config_data["active_slot"] = slot_id
            s_data = config_data["slots"][slot_id]
            su, sv = robot_to_pixel(s_data["x_r"], s_data["y_r"])
            target_place = {
                "pixel": (su, sv), "x_r": s_data["x_r"], "y_r": s_data["y_r"],
                "dist": math.sqrt(s_data["x_r"]**2 + s_data["y_r"]**2),
                "reachable": True, "label": s_data["name"]
            }
            save_config()
            status_message = f"Da chon dich: {s_data['name']} (X={s_data['x_r']}, Y={s_data['y_r']}mm)"

        elif key == ord('h'):
            send_robot_cmd("home")
            status_message = "Da gui lenh ve Home an toan!"

        elif key == ord('w'):
            print("[*] Đang thử kết nối lại WebSocket ESP32...")
            if get_ws() is not None:
                status_message = "Ket noi WebSocket ESP32 thanh cong!"
            else:
                status_message = "Ket noi ESP32 that bai! Hay kiem tra WiFi 'RobotArm'!"

        elif key == ord('c'):
            order = [0, 2, 1]
            cur_pos = order.index(current_index) if current_index in order else 0
            next_idx = order[(cur_pos + 1) % len(order)]
            cap.release()
            new_cap = open_camera(next_idx)
            if new_cap is not None:
                cap = new_cap
                current_index = next_idx
            else:
                cap = open_camera(current_index)

        elif key == ord(' '):
            trigger_pick_and_place()

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
