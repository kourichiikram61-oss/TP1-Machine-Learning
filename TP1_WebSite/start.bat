@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
python train_models.py
python -m streamlit run app.py
pause
