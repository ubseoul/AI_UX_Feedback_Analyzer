@echo off
setlocal
cd /d "%~dp0"

if not exist .venv (
  py -m venv .venv
)
call .venv\Scripts\activate

".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

".venv\Scripts\python.exe" -m streamlit run app.py --server.address 0.0.0.0 --server.headless true

pause
