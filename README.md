# Vision-Guided 5-DOF Robotic Arm for Pick-and-Place Automation
**Đồ án Cơ cấu chấp hành - Trường Đại học Công nghệ, ĐHQGHN (UET)**  
*Tác giả / Nhóm trưởng: [DkaRobotEngineer](https://github.com/DkaRobotEngineer)*

---

## 📌 1. Giới thiệu tổng quan (Overview)
Dự án cánh tay robot 5 bậc tự do (5-DOF) tích hợp hệ thống thị giác máy tính (Eye-to-Hand Vision System) phục vụ bài toán gắp - đặt (Pick-and-Place) và phân loại vật thể tự động theo hình dạng (Hình tròn, Hình vuông, Hình tam giác) vào các khay chỉ định.

Hệ thống kết hợp giữa xử lý ảnh thời gian thực (Python + OpenCV), thuật toán Động học nghịch (Inverse Kinematics) và mạch điều khiển nhúng vi điều khiển ESP32 kết hợp driver PCA9685 qua giao thức WebSocket WiFi tốc độ cao.

---

## ⚙️ 2. Cấu trúc phần cứng (Hardware Specifications)
* **Khung cơ khí:** Cánh tay robot 5-DOF in 3D bằng nhựa PLA chịu lực.
* **Cơ cấu kẹp:** Kẹp mềm sinh học **FinRay (Soft Gripper)** in 3D bằng nhựa dẻo TPU, có khả năng tự thích ứng ôm khít biên dạng vật thể.
* **Hệ thống động cơ (7 Servos):**
  * `J1 (Base)`: 1x Servo kim loại MG996R (Xoay đế $180^\circ$).
  * `J2 (Shoulder)`: 2x Servo kim loại MG996R (Cụm vai chịu tải chính, hoạt động đồng bộ).
  * `J3 (Elbow)`: 1x Servo kim loại MG996R (Khớp khuỷu tay).
  * `J4 (Wrist Pitch)`: 1x Servo mini MG90S (Khớp gập cổ tay).
  * `J5 (Wrist Roll)`: 1x Servo mini MG90S (Khớp xoay cổ tay đón góc vật).
  * `Gripper`: 1x Servo mini MG90S (Đóng / mở cơ cấu kẹp FinRay).
* **Vi điều khiển & Nguồn công suất:**
  * Kit **ESP32** (Dual-Core 240MHz, tích hợp WiFi WebSocket Server `ws://192.168.4.1:81/`).
  * Mạch mở rộng xung PWM 16 kênh **PCA9685** giao tiếp qua chuẩn $I^2C$.
  * Nguồn tổ ong 5V - 10A cấp riêng cho dàn servo.
* **Thị giác máy tính (Vision):**
  * Webcam **Logitech C270 HD (720p)** bố trí theo cấu hình **Eye-to-Hand** (nhìn thẳng $90^\circ$ từ trên xuống mặt bàn).

---

## 💻 3. Kiến trúc phần mềm & Thuật toán
1. **Xử lý ảnh & Phân loại hình dạng:**
   - Trích xuất dải màu HSV xanh dương bão hòa cao ($104 \le H \le 122, S \ge 150$) loại trừ triệt để nhiễu phản chiếu từ thân robot màu tím.
   - Nhận dạng tự động hình dạng: **HÌNH TRÒN, HÌNH VUÔNG, HÌNH TAM GIÁC** bằng thuật toán đa giác xấp xỉ `cv2.approxPolyDP` và hệ số tròn Circularity.
2. **Hiệu chuẩn Không cần Giấy A3 (Ruler + Base Calibration):**
   - Không phụ thuộc vào giấy A3 mỏng dễ xô lệch.
   - Quy trình 3 click chuột: Click tâm đế robot J1 $\rightarrow$ Click vạch 0cm thước $\rightarrow$ Click vạch 10cm thước $\rightarrow$ Tự động tính tỉ lệ `mm_per_pixel` và lưu vĩnh viễn vào `calibration.json`.
3. **Hiển thị trực quan Vùng làm việc (Workspace Reach Overlay):**
   - Cung tròn bán kính $R_{min} = 110\text{ mm}$, $R_{mid} = 150\text{ mm}$, $R_{max} = 200\text{ mm}$ hiển thị trực tiếp trên video.
4. **Động học nghịch & Điều khiển 6 bước tự động:**
   - Approach (Lơ lửng trên cao) $\rightarrow$ Descend (Hạ sát bàn) $\rightarrow$ Grasp (Kẹp chặt) $\rightarrow$ Lift (Nhấc cao) $\rightarrow$ Transfer (Chuyển sang đích) $\rightarrow$ Release & Home (Nhả vật & Về vị trí an toàn).

---

## 🚀 4. Hướng dẫn sử dụng hệ thống

### 4.1. Khởi động nhanh
1. Bật công tắc nguồn tổ ong 5V cấp cho Robot và ESP32.
2. Trên Laptop, kết nối vào mạng Wi-Fi: **`RobotArm`** (Mật khẩu: `12345678`).
3. **LƯU Ý:** Đóng tất cả tab web `192.168.4.1` trên Chrome/điện thoại để không bị nghẽn WebSocket!
4. Nhấp đúp vào file **`chay_tu_dong.bat`**.

### 4.2. Bảng phím tắt điều khiển trên giao diện camera
| Phím tắt | Chức năng |
| :---: | :--- |
| **`SPACE`** | **Bắt đầu chu trình Gắp & Đặt tự động hoàn toàn** |
| **`t`** | **Hover Test**: Vươn tới lơ lửng trên vật ngắm tâm X, Y (chờ 3s rồi về Home, không kẹp) |
| **`g`** | **Grasp Test**: Hạ xuống sát bàn ngắm độ cao Z (chờ 3s rồi về Home) |
| **`o`** | Mở kẹp Gripper (góc $35^\circ$) |
| **`p`** | Đóng kẹp Gripper chặt (góc $110^\circ$) |
| **`[`** / **`]`** | Hạ sâu thêm $+2^\circ$ / Nâng cao lên $-2^\circ$ độ cao gắp sát bàn |
| **`{`** / **`}`** | Tăng $+3^\circ$ / Giảm $-3^\circ$ lực kẹp chặt của Gripper |
| **`1`, `2`, `3`** | Chọn nhanh điểm thả: Khay 1 (Trái), Khay 2 (Giữa), Khay 3 (Phải) |
| **`a`** | **Bật/Tắt chế độ Auto-Sort**: Tự động thả Tròn $\rightarrow$ Khay 1, Vuông $\rightarrow$ Khay 2, Tam giác $\rightarrow$ Khay 3 |
| **`k`** | **Bật chế độ Hiệu chuẩn** (Click tâm đế J1 $\rightarrow$ 0cm thước $\rightarrow$ 10cm thước) |
| **`i`** | Đảo chiều quay trái/phải của khớp đế J1 (J1 Invert) |
| **`c`** | Đổi Camera (Index 0 $\rightarrow$ 2 $\rightarrow$ 1 $\rightarrow$ 0) |
| **`w`** | Thử kết nối lại WebSocket tới ESP32 |
| **`h`** | Đưa robot về vị trí an toàn (Home) |
| **`s`** | Lưu toàn bộ thông số hiệu chuẩn vào file `calibration.json` |
| **`q`** | Thoát chương trình an toàn |

---

## 📈 5. Tiến độ thực hiện (Project Status)
- [x] Thiết kế 3D, gia công in và lắp ráp cơ khí 5-DOF + Gripper FinRay TPU.
- [x] Lắp đặt mạch công suất ESP32 + PCA9685 và cấp nguồn servo an toàn.
- [x] Hoàn thiện module Vision thời gian thực: lọc màu HSV, bắt tâm và góc xoay vật thể.
- [x] Nâng cấp hiệu chuẩn không cần giấy A3 (Ruler + Base 3-click calibration).
- [x] Tích hợp thuật toán nhận diện hình dạng (Tròn / Vuông / Tam giác) và Auto-Sorting.
- [x] Khắc phục triệt để các lỗi cơ học đảo chiều servo và lệch tọa độ.
- [x] Hoàn thiện giao diện HUD công nghiệp phục vụ báo cáo và bảo vệ đồ án UET.
