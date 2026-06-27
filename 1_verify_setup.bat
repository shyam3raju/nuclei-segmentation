@echo off
echo ========================================
echo Verifying Project Setup
echo ========================================
echo.

cd src
python test_setup.py
cd ..

echo.
echo ========================================
echo Press any key to exit...
pause >nul
