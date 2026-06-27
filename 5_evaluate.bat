@echo off
echo ========================================
echo Evaluating Trained Model
echo ========================================
echo.

cd src
python evaluate.py --model_path ..\outputs\best_model.pth --save_results
cd ..

echo.
echo ========================================
echo Evaluation complete!
echo Check outputs/evaluation_results.txt
echo Press any key to exit...
pause >nul
