"""
Chương trình Bước 2: Kiểm tra Camera Logitech C270 & Hiệu chuẩn màu nhận diện vật thể
Môn: Cơ Cấu Chấp Hành - UET
"""

import cv2
import numpy as np

def nothing(x):
    pass

def open_camera(index):
    """Mở camera bằng DirectShow (CAP_DSHOW) - Không bao giờ bị treo trên Windows."""
    print(f"[*] Đang kết nối tới Camera index [{index}] qua DirectShow...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    
    if cap.isOpened():
        # Đọc thử 1 frame
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"[OK] Mở thành công Camera index [{index}]!")
            return cap
    
    if cap is not None:
        cap.release()
    return None

def main():
    print("=" * 60)
    print("HỆ THỐNG KIỂM TRA VISION ROBOT - UET")
    print("=" * 60)

    cap = None
    active_index = -1

    # Thử lần lượt các index: 0, 1, 2 bằng DirectShow
    # Thường C270 sẽ là 0, 1 hoặc 2 tùy thứ tự cắm
    for idx in [2, 1, 0]:
        cap = open_camera(idx)
        if cap is not None:
            active_index = idx
            break

    if cap is None:
        print("\n[LỖI] Không thể đọc được hình ảnh từ bất kỳ Camera nào!")
        print("-> Bạn kiểm tra lại: Cáp USB của Logitech C270 đã cắm chặt chưa?")
        input("\nNhấn Enter để thoát...")
        return

    # Cài đặt độ phân giải nét
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Tạo cửa sổ thanh trượt Trackbar để chỉnh màu
    cv2.namedWindow("Dieu Chinh Mau HSV", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dieu Chinh Mau HSV", 420, 280)

    # Khởi tạo giá trị mặc định cho màu Xanh Dương (Blue)
    cv2.createTrackbar("H Min", "Dieu Chinh Mau HSV", 90, 179, nothing)
    cv2.createTrackbar("H Max", "Dieu Chinh Mau HSV", 135, 179, nothing)
    cv2.createTrackbar("S Min", "Dieu Chinh Mau HSV", 90, 255, nothing)
    cv2.createTrackbar("S Max", "Dieu Chinh Mau HSV", 255, 255, nothing)
    cv2.createTrackbar("V Min", "Dieu Chinh Mau HSV", 50, 255, nothing)
    cv2.createTrackbar("V Max", "Dieu Chinh Mau HSV", 255, 255, nothing)

    print("\n" + "=" * 60)
    print(f">>> ĐANG HIỂN THỊ HÌNH ẢNH CAMERA INDEX [{active_index}] <<<")
    print("- Nhấn phím 'c': Chuyển đổi camera khác (nếu đang nhầm sang camera khác)")
    print("- Nhấn phím 'q': Thoát chương trình")
    print("=" * 60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Chuyển đổi không gian màu BGR sang HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Lấy giá trị thanh trượt hiện tại
        h_min = cv2.getTrackbarPos("H Min", "Dieu Chinh Mau HSV")
        h_max = cv2.getTrackbarPos("H Max", "Dieu Chinh Mau HSV")
        s_min = cv2.getTrackbarPos("S Min", "Dieu Chinh Mau HSV")
        s_max = cv2.getTrackbarPos("S Max", "Dieu Chinh Mau HSV")
        v_min = cv2.getTrackbarPos("V Min", "Dieu Chinh Mau HSV")
        v_max = cv2.getTrackbarPos("V Max", "Dieu Chinh Mau HSV")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # Tạo mặt nạ lọc màu (Mask)
        mask = cv2.inRange(hsv, lower, upper)

        # Lọc nhiễu hạt nhỏ
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Tìm các đường bao (Contours) của vật thể
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Chỉ xét các vật có diện tích đủ lớn để tránh nhiễu
            if area > 300:
                # Tìm khung hình chữ nhật xoay bám sát vật thể
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect

                # Tính 4 đỉnh của khung chữ nhật bao quanh
                box = cv2.boxPoints(rect)
                box = np.int32(box)

                # Chuẩn hóa góc xoay
                if w < h:
                    angle = angle + 90

                # Vẽ khung bao quanh vật (màu xanh lá)
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

                # Vẽ tâm vật thể (chấm tròn đỏ)
                cv2.circle(frame, (int(cx), int(cy)), 6, (0, 0, 255), -1)

                # Hiển thị tọa độ tâm và góc xoay
                info_text = f"X:{int(cx)} Y:{int(cy)} | Goc:{int(angle)} do"
                cv2.putText(frame, info_text, (int(cx) - 80, int(cy) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Hiển thị hướng dẫn phím tắt lên ảnh
        cv2.putText(frame, f"Camera [{active_index}] - Nhan 'c' de doi cam, 'q' de thoat", 
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # Hiển thị 2 cửa sổ
        cv2.imshow("Robot Vision - Logitech C270", frame)
        cv2.imshow("Mat Na Loc Mau (Mask)", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Đổi sang camera tiếp theo
            next_idx = (active_index + 1) % 3
            print(f"[*] Đang chuyển sang Camera index [{next_idx}]...")
            cap.release()
            new_cap = open_camera(next_idx)
            if new_cap is not None:
                cap = new_cap
                active_index = next_idx
            else:
                print(f"[!] Không mở được index [{next_idx}], quay lại index [{active_index}]")
                cap = open_camera(active_index)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
