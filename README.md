# Vision-Guided 5-DOF Robotic Arm for Pick-and-Place Automation
**Đồ án Cơ cấu chấp hành - Trường Đại học Công nghệ, ĐHQGHN (UET)**  
*Tác giả / Nhóm trưởng: [DkaRobotEngineer](https://github.com/DkaRobotEngineer)*

---

## 📌 1. Giới thiệu tổng quan (Overview)
Dự án cánh tay robot 5 bậc tự do (5-DOF) tích hợp hệ thống thị giác máy tính (Eye-to-Hand Vision System) phục vụ bài toán gắp - đặt (Pick-and-Place) và phân loại vật thể tự động theo tọa độ chỉ định.

Hệ thống kết hợp giữa xử lý ảnh thời gian thực (Python + OpenCV), thuật toán Động học nghịch (Inverse Kinematics) và mạch điều khiển nhúng vi điều khiển ESP32 kết hợp driver PCA9685.

---

## ⚙️ 2. Cấu trúc phần cứng (Hardware Specifications)
* **Khung cơ khí:** Cánh tay robot 5-DOF in 3D bằng nhựa PLA chịu lực.
* **Cơ cấu kẹp:** Kẹp mềm sinh học **FinRay (Soft Gripper)** in 3D bằng nhựa dẻo TPU, có khả năng tự thích ứng ôm khít biên dạng vật thể.
* **Hệ thống động cơ (7 Servos):**
  * `J1 (Base)`: 1x Servo kim loại MG996R (Xoay đế $180^\circ$).
  * `J2 (Shoulder)`: 2x Servo kim loại MG996R (Cụm vai chịu tải chính, hoạt động đồng bộ).
  * `J3 (Elbow)`: 1x Servo kim loại MG996R (Khớp khuỷu tay).
  * `J4 (Wrist Pitch)`: 1x Servo mini MG90S (Khớp gập cổ tay).
  * `J5 (Wrist Roll)`: 1x Servo mini MG90S (Khớp xoay cổ tay theo hướng vật).
  * `Gripper`: 1x Servo mini MG90S (Đóng / mở cơ cấu kẹp FinRay).
* **Vi điều khiển & Điều khiển công suất:**
  * Kit **ESP32** (Dual-Core 240MHz, tích hợp WiFi/BLE).
  * Mạch mở rộng xung PWM 16 kênh **PCA9685** giao tiếp qua chuẩn $I^2C$.
  * Nguồn tổ ong công suất rời 5V - 10A cấp riêng cho mạch servo.
* **Thị giác máy tính (Vision):**
  * Webcam **Logitech C270 HD (720p)** bố trí theo cấu hình **Eye-to-Hand** (nhìn vuông góc $90^\circ$ từ trên xuống mặt bàn làm việc).

---

## 💻 3. Kiến trúc phần mềm & Thuật toán (Software & Algorithms)
* **Xử lý ảnh (Computer Vision):**
  * Sử dụng không gian màu **HSV (Hue - Saturation - Value)** và bộ lọc ngưỡng động để trích xuất vật thể không phụ thuộc ánh sáng môi trường.
  * Phép toán hình thái học (Morphology Open & Close) lọc sạch nhiễu nền.
  * Thuật toán trích xuất đường bao (Contour Extraction) & Khung chữ nhật bao quanh nhỏ nhất (`cv2.minAreaRect`) để tính toán chính xác:
    * **Tâm vật $(X_{pixel}, Y_{pixel})$**
    * **Góc xoay của vật ($\theta$)** để điều khiển servo xoay kẹp (J5) đón đúng hướng vật.
* **Hiệu chuẩn tọa độ (Calibration & Homography):**
  * Ánh xạ tọa độ từ hệ ảnh (Pixel Frame) sang hệ tọa độ thực của bàn Robot (Robot Base Frame theo milimét) bằng ma trận biến đổi phối cảnh (Perspective Transform).
* **Động học nghịch (Analytical Inverse Kinematics - IK):**
  * Giải bài toán động học vị trí để chuyển đổi từ tọa độ mục tiêu $(X, Y, Z)$ thành 6 góc quay tương ứng của các khớp servo.

---

## 🚀 4. Cài đặt & Hướng dẫn sử dụng (Quickstart)

### 4.1. Yêu cầu môi trường
* Python 3.10 / 3.11 / 3.12
* Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

### 4.2. Chạy kiểm tra nhận diện vật thể
Cắm webcam Logitech C270 vào cổng USB máy tính và chạy:
```bash
python test_camera.py
```
* **Phím tắt điều khiển:**
  * `c`: Chuyển đổi giữa các camera (nếu cắm nhiều camera).
  * `q`: Thoát chương trình.
  * Kéo các thanh trượt trong cửa sổ `Dieu Chinh Mau HSV` để bắt màu vật thể mong muốn.

---

## 📈 5. Tiến độ thực hiện (Project Status)
- [x] Thiết kế 3D, gia công in và lắp ráp cơ khí 5-DOF + Gripper FinRay TPU.
- [x] Lắp đặt mạch công suất ESP32 + PCA9685 và cấp nguồn servo an toàn.
- [x] Hoàn thiện module Vision thời gian thực: lọc màu HSV, bắt tâm và góc xoay vật thể.
- [ ] Tích hợp công cụ Calibration 4 điểm chuyển đổi Pixel $\rightarrow$ Milimét.
- [ ] Ghép nối thuật toán IK và chu trình gắp - đặt tự động theo điểm click chỉ định.
