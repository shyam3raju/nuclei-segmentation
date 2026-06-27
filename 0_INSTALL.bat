@echo off
echo ========================================
echo Installing Dependencies
echo ========================================
echo.
echo This will install all required Python packages.
echo This may take 5-10 minutes.
echo.
echo Press any key to start installation...
pause >nul

pip install -r requirements.txt

echo.
echo ========================================
echo Installation complete!
echo.
echo Next step: Run "1_verify_setup.bat"
echo Press any key to exit...
pause >nul
