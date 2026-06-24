@echo off
cd /d "%~dp0"
echo Starting Brain MRI Classifier...
streamlit run app/streamlit_app.py
pause
