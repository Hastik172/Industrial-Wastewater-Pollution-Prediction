@echo off
title AquaPredict - Industrial Wastewater Pollution Prediction
cd /d "%~dp0"

echo =======================================================
echo   Starting AquaPredict (Streamlit Local Server)...
echo =======================================================
echo.
echo Opening browser at http://localhost:8501 ...
start "" "http://localhost:8501"

echo.
python -m streamlit run app.py --browser.gatherUsageStats false
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Could not start Streamlit with "python -m streamlit".
    echo Trying direct "py" launcher...
    start "" "http://localhost:8501"
    py -m streamlit run app.py --browser.gatherUsageStats false
)

pause
