@echo off
cd /d "%~dp0"
rem Локальные настройки из .env (файл в .gitignore; шаблон — .env.example)
if exist .env for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do if not "%%A"=="" set "%%A=%%B"
if not defined VECTOR_CALC_ENABLED_MODELS set VECTOR_CALC_ENABLED_MODELS=en-small,ru-rusvectores
if not defined VECTOR_CALC_DEFAULT_MODEL set VECTOR_CALC_DEFAULT_MODEL=en-small
if not defined VECTOR_CALC_MODERATION set VECTOR_CALC_MODERATION=queue
if not defined VECTOR_CALC_MODERATION_PROFILE set VECTOR_CALC_MODERATION_PROFILE=teen
if not defined VECTOR_CALC_DEFAULT_MODE set VECTOR_CALC_DEFAULT_MODE=kids
if not defined VECTOR_CALC_LLM_URL set VECTOR_CALC_LLM_URL=http://192.168.1.102:8080/v1
if not defined VECTOR_CALC_LLM_MODEL set VECTOR_CALC_LLM_MODEL=local
if not defined VECTOR_CALC_LLM_NO_THINK set VECTOR_CALC_LLM_NO_THINK=1
if not defined VECTOR_CALC_ADMIN_TOKEN set VECTOR_CALC_ADMIN_TOKEN=xxx111xxx333
"%~dp0.venv\Scripts\python.exe" server.py
pause
