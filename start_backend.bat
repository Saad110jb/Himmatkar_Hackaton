@echo off
cd /d "%~dp0"
echo Starting FastAPI Backend at http://localhost:8000 ...
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
pause
