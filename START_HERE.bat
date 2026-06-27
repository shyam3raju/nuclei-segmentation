@echo off
cls
echo ========================================
echo NUCLEI SEGMENTATION PROJECT
echo ========================================
echo.
echo Project Location: %CD%
echo.
echo ========================================
echo QUICK START MENU
echo ========================================
echo.
echo 0. Install Dependencies (run this first!)
echo 1. Verify Setup (check if everything is installed)
echo 2. Download Dataset (required before training)
echo 3. Quick Test (5 min test run)
echo 4. Full Training (30-60 min GPU / 2-3 hours CPU)
echo 5. Evaluate Model (get Dice and IoU scores)
echo 6. Visualize Predictions (create sample images)
echo.
echo 7. Open Demo Notebook (Jupyter)
echo 8. View HOW_TO_RUN.txt guide
echo.
echo 9. Exit
echo.
echo ========================================
echo.

set /p choice="Enter your choice (0-9): "

if "%choice%"=="0" (
    call 0_INSTALL.bat
    goto end
)
if "%choice%"=="1" (
    call 1_verify_setup.bat
    goto end
)
if "%choice%"=="2" (
    call 2_download_dataset.bat
    goto end
)
if "%choice%"=="3" (
    call 3_quick_test.bat
    goto end
)
if "%choice%"=="4" (
    call 4_full_training.bat
    goto end
)
if "%choice%"=="5" (
    call 5_evaluate.bat
    goto end
)
if "%choice%"=="6" (
    call 6_visualize.bat
    goto end
)
if "%choice%"=="7" (
    cd notebooks
    jupyter notebook demo.ipynb
    cd ..
    goto end
)
if "%choice%"=="8" (
    notepad HOW_TO_RUN.txt
    goto end
)
if "%choice%"=="9" (
    exit
)

echo Invalid choice. Please try again.
pause
goto end

:end
echo.
echo Press any key to return to menu...
pause >nul
cls
goto START_HERE.bat
