@echo off
cd /d "%~dp0"
echo Starting Streamlit Dashboard at http://localhost:8501 ...
streamlit run frontend/app.py
pause
