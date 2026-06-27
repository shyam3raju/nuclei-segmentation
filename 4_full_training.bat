@echo off
echo ========================================
echo Full Training (30-60 min GPU / 2-3 hours CPU)
echo ========================================
echo.
echo This will train on the full dataset for 30 epochs.
echo Make sure you have time for this!
echo.
echo Press any key to start training...
pause >nul

cd src
python train.py --num_epochs 30 --batch_size 8
cd ..

echo.
echo ========================================
echo Training complete!
echo Check outputs/best_model.pth
echo Press any key to exit...
pause >nul
