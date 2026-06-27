@echo off
echo ========================================
echo Creating Prediction Visualizations
echo ========================================
echo.

cd src
python inference.py --data_dir ..\data\stage1_train --num_samples 10
cd ..

echo.
echo ========================================
echo Visualizations created!
echo Check outputs/predictions/ folder
echo Press any key to exit...
pause >nul
