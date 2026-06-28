@echo off
echo ========================================
echo Regenerating Training Curves
echo ========================================
echo.
echo This will create new training_curves.png
echo based on your actual training results.
echo.
echo Press any key to continue...
pause >nul

cd src
python regenerate_training_curves.py
cd ..

echo.
echo ========================================
echo Done! Check outputs/training_curves.png
echo Press any key to exit...
pause >nul
