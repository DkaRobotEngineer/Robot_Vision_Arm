@echo off
title He Thong Robot Gap Dat Tu Dong Vision 5-DOF - UET
color 0b
cd /d "%~dp0"

echo =======================================================================
echo          HE THONG ROBOT GAP DAT TU DONG 5-DOF - UET ROBOTICS
echo =======================================================================
echo [!] LUU Y QUAN TRONG TRUOC KHI CHAY:
echo     1. Laptop da ket noi vao Wi-Fi "RobotArm" (Mat khau: 12345678)
echo     2. DONG TAT CA tab web "192.168.4.1" tren Chrome de tranh nghen kenh!
echo     3. Nguon to ong 5V va ESP32 da bat sang den.
echo =======================================================================
echo.

"D:\Python3-12\python.exe" -u main_auto_pick_place.py

echo.
echo =======================================================================
echo Chuong trinh da ket thuc. Nhan phim bat ky de dong cua so...
pause >nul
