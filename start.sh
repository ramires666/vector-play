#!/usr/bin/env bash
# VectorPlay launcher for Linux and other POSIX systems.
#
# First run: creates .venv, installs requirements.txt and downloads the
# Word2Vec archive. Later runs: starts server.py directly.
#
# Usage: ./start.sh
# Host/port can be changed with VECTOR_CALC_HOST and VECTOR_CALC_PORT.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"
VENV_PY="$VENV/bin/python"
ARCHIVE="models/ruwikiruscorpora_upos_skipgram_300_2_2019.zip"

# Модели: по умолчанию только две самые лёгкие — en-small (GloVe Wiki+Gigaword
# 100D, загрузка 128 МБ) и русская (RusVectōrēs, 0.6 ГБ). Тяжёлые
# en-google/en-glove остаются в коде (MODELS в word2vec_engine.py), не грузятся
# и не показываются в меню; включить: VECTOR_CALC_ENABLED_MODELS=en-google,en-glove,ru-rusvectores
export VECTOR_CALC_ENABLED_MODELS="${VECTOR_CALC_ENABLED_MODELS:-en-small,ru-rusvectores}"
export VECTOR_CALC_DEFAULT_MODEL="${VECTOR_CALC_DEFAULT_MODEL:-en-small}"

# Модерация коллекции: словари + векторные эвристики + LLM (OpenAI-совместимый адрес).
export VECTOR_CALC_MODERATION="${VECTOR_CALC_MODERATION:-queue}"
export VECTOR_CALC_MODERATION_PROFILE="${VECTOR_CALC_MODERATION_PROFILE:-teen}"   # kids — строже (7+)
export VECTOR_CALC_LLM_URL="${VECTOR_CALC_LLM_URL:-http://192.168.1.102:8080/v1}"
export VECTOR_CALC_LLM_MODEL="${VECTOR_CALC_LLM_MODEL:-local}"
export VECTOR_CALC_LLM_NO_THINK="${VECTOR_CALC_LLM_NO_THINK:-1}"                  # ~1 c вместо 5-10 c
export VECTOR_CALC_ADMIN_TOKEN="${VECTOR_CALC_ADMIN_TOKEN:-xxx111xxx333}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: $PYTHON not found; install Python 3.10+ or set PYTHON=/path/to/python3" >&2
    exit 1
fi

if [ -e "$VENV" ] && [ ! -x "$VENV_PY" ]; then
    echo "error: $VENV exists but has no Linux interpreter at $VENV_PY." >&2
    echo "       It was probably created on Windows; remove it or set VENV" >&2
    echo "       to use a different directory." >&2
    exit 1
fi

if [ ! -x "$VENV_PY" ]; then
    echo "Creating virtual environment in $VENV ..."
    "$PYTHON" -m venv "$VENV"
fi

if ! "$VENV_PY" -c "import numpy, pymorphy3, fastapi, uvicorn" >/dev/null 2>&1; then
    echo "Installing dependencies from requirements.txt ..."
    "$VENV_PY" -m pip install --disable-pip-version-check -r requirements.txt
fi

if [ ! -f "$ARCHIVE" ]; then
    echo "Word2Vec model not found; downloading it now (resumable, may take a while) ..."
    "$VENV_PY" download_model.py
fi

exec "$VENV_PY" server.py
