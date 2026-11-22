@echo off
title Thu Thap Du Lieu va Huan Luyen AI YOLOv8 - UET Robotics
color 0e
cd /d "%~dp0"

echo =======================================================================
echo          CONG CU THU THAP DU LIEU VA HUAN LUYEN AI YOLOV8
echo =======================================================================
echo [!] Huong dan:
echo     1. Dung chuot keo o vuong bao quanh vat the tren ban (nap but, khoi...).
echo     2. Bam [SPACE] de luu anh (chup khoang 15 - 30 anh o cac goc).
echo     3. Bam phim [t] de may tu dong Huan luyen AI trong 1 - 2 phut!
echo =======================================================================
echo.

"D:\Python3-12\python.exe" ai_dataset_and_train.py

echo.
echo =======================================================================
echo Hoan tat! Nhan phim bat ky de thoat...
pause >nul
