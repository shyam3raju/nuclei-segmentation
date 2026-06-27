@echo off
echo ========================================
echo Quick Test Training (5 minutes)
echo ========================================
echo.
echo This will train on 50 images for 5 epochs
echo to verify everything works.
echo.
echo Press any key to start...
pause >nul

cd src
python train.py --max_samples 50 --num_epochs 5
cd ..

echo.
echo ========================================
echo Test complete! Check outputs folder.
echo Press any key to exit...
pause >nul
