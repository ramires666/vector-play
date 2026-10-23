# VectorPlay — shared LAN Word2Vec calculator and memories gallery.
FROM python:3.12-slim

# The container runs as an unprivileged user. On a Linux host, build with
# matching ids when the project folder belongs to another UID:
#   VECTORPLAY_UID=$(id -u) VECTORPLAY_GID=$(id -g) docker compose up -d --build
ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV VECTOR_CALC_HOST=0.0.0.0
ENV VECTOR_CALC_PORT=58765

RUN groupadd --gid "${APP_GID}" vectorplay \
 && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin vectorplay

WORKDIR /app

COPY requirements.txt ./
RUN pip install --disable-pip-version-check -r requirements.txt

COPY --chown=vectorplay:vectorplay server.py memories_db.py word2vec_engine.py download_model.py ./
COPY --chown=vectorplay:vectorplay index.html memories.html app.js ./
COPY --chown=vectorplay:vectorplay docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh \
 && mkdir -p /app/models \
 && chown -R vectorplay:vectorplay /app

USER vectorplay

EXPOSE 58765

# docker-compose.yml mounts the project folder at /app, so memories.sqlite3
# and models/ live on the host and survive restarts and rebuilds.
ENTRYPOINT ["/bin/sh", "/app/docker-entrypoint.sh"]
