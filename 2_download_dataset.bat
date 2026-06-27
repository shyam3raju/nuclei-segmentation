@echo off
echo ========================================
echo Downloading Dataset
echo ========================================
echo.
echo Make sure you have:
echo 1. Kaggle account created
echo 2. kaggle.json placed in C:\Users\%USERNAME%\.kaggle\
echo.
echo Press any key to continue...
pause >nul

cd data
python download_data.py
cd ..

echo.
echo ========================================
echo Dataset download complete!
echo Press any key to exit...
pause >nul
