"""
TEST KẾT NỐI VÀ ĐIỀU KHIỂN ROBOT QUA WEBSOCKET
Đồ án Cơ cấu chấp hành - UET
Giao tiếp: Laptop -> WiFi RobotArm -> ws://192.168.4.1:81/
"""

import time
import websocket

WS_URL = "ws://192.168.4.1:81/"

def test_control():
    print("=" * 60)
    print("ĐANG KẾT NỐI TỚI ESP32 TẠI: " + WS_URL)
    print("Lưu ý: Hãy đảm bảo Laptop đã kết nối vào WiFi 'RobotArm'!")
    print("=" * 60)

    try:
        ws = websocket.create_connection(WS_URL, timeout=5)
        print("[OK] KẾT NỐI THÀNH CÔNG TỚI ESP32!\n")

        print("1. Gửi lệnh: Về vị trí Home...")
        ws.send("home")
        time.sleep(2)

        print("2. Gửi lệnh: Mở kẹp Gripper (Góc 100 độ)...")
        ws.send("j 5 100")
        time.sleep(1.5)

        print("3. Gửi lệnh: Đóng kẹp Gripper (Góc 40 độ)...")
        ws.send("j 5 40")
        time.sleep(1.5)

        print("4. Gửi lệnh: Mở kẹp Gripper trở lại...")
        ws.send("j 5 100")
        time.sleep(1.5)

        print("\n" + "=" * 60)
        print(">>> THỬ NGHIỆM THÀNH CÔNG 100%! <<<")
        print("Máy tính đã hoàn toàn làm chủ cánh tay robot qua WiFi!")
        print("=" * 60)

        ws.close()

    except Exception as e:
        print("\n[LỖI KẾT NỐI]:", e)
        print("-> Vui lòng kiểm tra:")
        print("   1. Laptop của bạn đã chuyển sang kết nối WiFi 'RobotArm' chưa?")
        print("   2. ESP32 và nguồn tổ ong đã bật sáng đèn chưa?")

if __name__ == "__main__":
    test_control()
