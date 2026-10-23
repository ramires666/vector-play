#!/bin/sh
# VectorPlay entrypoint. The project folder is mounted at /app, so the SQLite
# database and the model files live on the host. Fail early with a readable
# message when the container user cannot write there.
set -eu

APP_DIR=/app

if ! mkdir -p "$APP_DIR/models" 2>/dev/null; then
    echo "error: cannot create $APP_DIR/models inside the container." >&2
    echo "       The project folder must be writable by user $(id -u):$(id -g)." >&2
    exit 1
fi

for path in "$APP_DIR" "$APP_DIR/models" "$APP_DIR/memories.sqlite3"; do
    if [ -e "$path" ] && [ ! -w "$path" ]; then
        echo "error: $path is not writable by the container user $(id -u):$(id -g)." >&2
        echo "       Fix the ownership on the host, or rebuild with matching ids:" >&2
        echo "         VECTORPLAY_UID=\$(id -u) VECTORPLAY_GID=\$(id -g) docker compose up -d --build" >&2
        echo "       Last resort: uncomment 'user: \"0:0\"' in docker-compose.yml." >&2
        exit 1
    fi
done

# Модель по умолчанию (VECTOR_CALC_DEFAULT_MODEL) качаем В ФОНЕ, чтобы сервер
# поднимался сразу: в интерфейсе будет «Модель скачивается…», а модель
# подхватится, как только файл появится. Остальные модели:
#   docker compose exec vectorplay python download_model.py --all
DEFAULT_ARCHIVE="$(python -c 'import word2vec_engine as e; print(e.MODELS_BY_ID[e.MANAGER.default_id].path)' 2>/dev/null || true)"
if [ -n "$DEFAULT_ARCHIVE" ] && [ ! -f "$DEFAULT_ARCHIVE" ]; then
    echo "Модель по умолчанию не найдена, качаю в фоне: $(basename "$DEFAULT_ARCHIVE")"
    echo "Прогресс и результат видны в docker compose logs -f; сервер уже поднимается."
    python download_model.py &
fi

exec python server.py
