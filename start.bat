@echo off
cd /d "%~dp0"
if not defined VECTOR_CALC_ENABLED_MODELS set VECTOR_CALC_ENABLED_MODELS=ru-rusvectores
if not defined VECTOR_CALC_DEFAULT_MODEL set VECTOR_CALC_DEFAULT_MODEL=ru-rusvectores
if not defined VECTOR_CALC_MODERATION set VECTOR_CALC_MODERATION=queue
if not defined VECTOR_CALC_MODERATION_PROFILE set VECTOR_CALC_MODERATION_PROFILE=teen
if not defined VECTOR_CALC_LLM_URL set VECTOR_CALC_LLM_URL=http://192.168.1.102:8080/v1
if not defined VECTOR_CALC_LLM_MODEL set VECTOR_CALC_LLM_MODEL=local
if not defined VECTOR_CALC_LLM_NO_THINK set VECTOR_CALC_LLM_NO_THINK=1
if not defined VECTOR_CALC_ADMIN_TOKEN set VECTOR_CALC_ADMIN_TOKEN=local-admin
"%~dp0.venv\Scripts\python.exe" server.py
pause
