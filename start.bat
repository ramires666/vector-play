@echo off
cd /d "%~dp0"
if not defined VECTOR_CALC_ENABLED_MODELS set VECTOR_CALC_ENABLED_MODELS=ru-rusvectores
if not defined VECTOR_CALC_DEFAULT_MODEL set VECTOR_CALC_DEFAULT_MODEL=ru-rusvectores
"%~dp0.venv\Scripts\python.exe" server.py
pause
