# VectorPlay

**Word-algebra calculator on Word2Vec / GloVe embeddings with a shared, kid-safe "Memories" gallery — a self-hosted FastAPI service with a mobile-first web UI.**

Type an equation of words (`king − man + woman`) and the app finds the closest concept in the model's **entire vocabulary** (`queen`). Save funny results to a shared collection, browse what other visitors have saved, and keep the gallery safe for children with a three-layer moderation pipeline (dictionaries + vector heuristics + an optional local LLM).

Repository: <https://github.com/ramires666/vector-play>

---

## Table of contents

- [What it is](#what-it-is)
- [Highlights](#highlights)
- [Quick start](#quick-start)
- [Models](#models)
  - [Enabling and disabling models](#enabling-and-disabling-models)
  - [Downloading models](#downloading-models)
- [Docker deployment](#docker-deployment)
- [Configuration reference](#configuration-reference)
- [HTTP API](#http-api)
- [Moderation and access groups](#moderation-and-access-groups)
- [The memories database](#the-memories-database)
- [UI guide](#ui-guide)
- [Architecture and internals](#architecture-and-internals)
- [Tests](#tests)
- [Measured performance](#measured-performance)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)

---

## What it is

VectorPlay is a tiny self-hosted web service for playing with **word embeddings**:

- you write an expression of words with `+` / `−` signs, e.g. `king − man + woman`;
- the server projects the expression vector onto the whole vocabulary and returns the nearest words;
- the result can be saved to a shared **Memories** gallery (SQLite) with automatic deduplication and moderation;
- the gallery can be browsed in three display modes (`kids` / `teen` / `off`), so a child sees only examples approved for children.

Verified classic results:

| Model | Equation | Nearest word | Cosine |
|---|---|---|---|
| `en-small` | king − man + woman | **queen** | 0.690 |
| `en-small` | paris − france + italy | **rome** | — |
| `en-small` | moscow − russia + japan | **tokyo** | — |
| `ru-rusvectores` | король − мужчина + женщина | **королева** | 0.746 |

Verified on real models with the shipped code, not on a hand-picked candidate list: similarity is computed as the cosine against every token vector in the model.

**Tech stack:** Python 3.10+ (3.12 in Docker), FastAPI + uvicorn, NumPy, pymorphy3 (Russian morphology), SQLite, vanilla JavaScript (no build step, no npm), Docker Compose.

---

## Highlights

- **Two compact full-size models enabled by default** — English GloVe 300D (400k words) and Russian RusVectōrēs 300D (249k words); the heavy 1.6 GB models stay in the code but are disabled and never downloaded.
- **Full-vocabulary search** — up to 12 terms per equation, each term can be a multi-word phrase (lemmatized for Russian, case-variant matched for English); source words can be excluded from the results.
- **Shared "Memories" gallery** — SQLite-backed, per-model lists, content-based deduplication, live update about once a second, click a card to load its equation back into the calculator.
- **Kid-safe by design** — display modes per browser (`kids` / `teen` / `off`) and save-time moderation profiles (`kids` / `teen`) with three layers: hashed dictionaries, vector-neighbor heuristics and an optional OpenAI-compatible LLM. Moderation runs with thinking disabled for speed when the LLM supports it.
- **Admin workflow** — queue of questionable examples, approve/reject, re-assess one or all records (via vectors or via LLM), manual access-group changes, all protected by `X-Admin-Token`.
- **Sweet main page** — the calculator sits right under the heading; examples and memories are merged into one block: built-in examples are shown while the collection is empty or until you calculate for the first time, then they disappear forever and only fresh community examples remain.
- **Model manager with a memory budget** — models are loaded in the background (the server never blocks for a minute), kept together in an LRU cache while they fit into the auto-detected memory budget, and evicted when idle.
- **Hardened server** — rate limits per IP, body-size cap, concurrency cap for calculations, security headers and CSP, no `/docs` by default, non-root read-only Docker container.
- **Bilingual UI** — English and Russian interface strings; the language follows the selected model, and the switch hides itself when only one language is available.

---

## Quick start

### Windows

```powershell
git clone https://github.com/ramires666/vector-play.git
cd vector-play
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_model.py     # default model: en-small, 376 MB
start.bat                                        # or: .\.venv\Scripts\python.exe server.py
```

### Linux / macOS

```bash
git clone https://github.com/ramires666/vector-play.git
cd vector-play
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_model.py
./start.sh                                       # or: .venv/bin/python server.py
```

Open <http://localhost:58765/>. On the first request the default model is loaded in the background (about 20 seconds); the header status shows the progress and the vocabulary size when ready.

`start.sh` / `start.bat` pre-set the recommended configuration: the two compact models enabled, `en-small` as default, moderation in `queue` mode with the `teen` profile, the LLM endpoint of the author's LAN (override it), and the admin token. Bare `python server.py` without those variables exposes the whole model registry and moderation in `warn` mode.

Optional: copy `.env.example` to `.env` and adjust the deployment-specific values there (LLM endpoint and key, admin token, moderation profile, default gallery mode). Docker Compose, `start.sh` / `start.bat` and `server.py` itself all read this file automatically (explicit environment variables always win over it); `.env` is git-ignored, so `git pull` never touches it.

---

## Models

The registry lives in `word2vec_engine.py` (`MODELS`). Each model is described by an archive file, download mirrors, size and RAM estimates.

| id | In the menu | Model | Vocabulary | Dimensions | Download | RAM (vectors) |
|---|---|---|---|---|---|---|
| `en-small` | ✅ default | 🇬🇧 English Small — GloVe Wiki+Gigaword 300D | 400,000 | 300 | **376 MB** | ~505 MB |
| `ru-rusvectores` | ✅ | 🇷🇺 Русский — RusVectōrēs Word2Vec 300D | 248,978 | 300 | 608 MB | ~314 MB |
| `en-google` | — (code only) | 🇬🇧 English Classic — Google News Word2Vec 300D | ~3M | 300 | 1663 MB | ~3.8 GB (planned) |
| `en-glove` | — (code only) | 🇬🇧 English Modern — Stanford GloVe 2024 300D | ~1.29M | 300 | 1626 MB | ~1.6 GB (planned) |

Notes:

- `en-small` is the classic Stanford GloVe trained on Wikipedia 2014 + Gigaword 5, distributed through the `gensim-data` release mirror. The file is a gzip-compressed **word2vec text** file with a `400000 300` header; the loader detects this format automatically (it also supports `.gz` word2vec binary and `.zip` archives with `model.bin`).
- `ru-rusvectores` is the RusVectōrēs `ruwikiruscorpora_upos_skipgram_300_2_2019` model; tokens carry POS tags (`королева_NOUN`) and requests are lemmatized with pymorphy3.
- The two heavy models are **kept in the code but disabled**: they are not downloaded by the entrypoint, not loaded into memory, not listed in the UI menu and do not appear in the shared gallery feed. Enable them explicitly if the server has 4+ GB of free RAM.

### Enabling and disabling models

The full registry is always in the code; a specific run exposes only the models listed in `VECTOR_CALC_ENABLED_MODELS` (comma-separated ids):

```yaml
VECTOR_CALC_ENABLED_MODELS: "en-small,ru-rusvectores"                      # default in Docker/start scripts
# "en-small,en-google,en-glove,ru-rusvectores"                            # everything (download the heavy files first)
# "ru-rusvectores"                                                        # Russian only
```

What this gives you: disabled models are not loaded, not preloaded at startup, not downloaded by the entrypoint, do not appear in the model/language switcher (if only one language remains, the EN/RU buttons hide themselves), and their records are filtered out of the "all models" gallery feed. A typo in an id fails fast at startup with a readable message instead of silently exposing something unwanted. You can still download a disabled model explicitly.

### Downloading models

```bash
python download_model.py                  # the default model (VECTOR_CALC_DEFAULT_MODEL, currently en-small)
python download_model.py ru-rusvectores   # specific models by id
python download_model.py --all            # every enabled model (en-small + Russian ≈ 1 GB)
python download_model.py --force en-small # re-download even if the file is already there
python download_model.py en-google        # a disabled model can be fetched explicitly
```

Downloads are **resumable** (HTTP Range), validated after completion (gzip CRC / zip integrity), and marked with a `<archive>.ok` file so a second run skips them. All model files live in `models/` (git-ignored) and survive Docker rebuilds because the project folder is mounted into the container.

---

## Docker deployment

```bash
docker compose up -d
docker compose logs -f        # watch the server start and the model download progress
```

What happens on the first run:

1. `docker-entrypoint.sh` checks that the mounted project folder is writable by the container user;
2. if the default model file is missing, it starts `python download_model.py` **in the background**, so the server starts immediately;
3. the web UI shows "Model is downloading…" until the file is ready, then the model loads automatically (the page retries on its own).

Useful commands:

```bash
docker compose exec vectorplay python download_model.py --all       # fetch the other enabled models
docker compose exec vectorplay python download_model.py en-google   # fetch a disabled model
docker compose down                                                  # stop (models and DB stay on the host)
docker compose up -d --build                                         # rebuild the image
```

Deployment notes:

- The whole project folder is mounted at `/app` (`- ./:/app`), so `memories.sqlite3` and `models/` live on the host and survive rebuilds and updates.
- The port is published directly (`58765:58765`). To put the app behind nginx, bind `127.0.0.1:58765:58765` and set `VECTOR_CALC_TRUST_PROXY=1` so the rate limiter sees real client IPs from `X-Forwarded-For`. Without a proxy in front, keep it `0` — otherwise clients can spoof their IP.
- On Linux, build with matching ids when the project folder belongs to another user: `VECTORPLAY_UID=$(id -u) VECTORPLAY_GID=$(id -g) docker compose up -d --build`. The last resort is uncommenting `user: "0:0"` in `docker-compose.yml`.
- If Docker cannot allocate a subnet ("all predefined address pools have been fully subnetted"), run `docker network prune -f` or extend `default-address-pools` in `daemon.json`. As an alternative, `network_mode: host` with `VECTOR_CALC_HOST: 127.0.0.1` works without a Docker network at all.
- Any compose value can be overridden by a `.env` file next to `docker-compose.yml` (it is git-ignored, so local settings never conflict with `git pull`). Copy the template: `cp .env.example .env`, then set the LLM endpoint and key, the admin token, the moderation profile and the default gallery mode there. `start.sh` / `start.bat` read the same `.env` for local runs.
- Container hardening: unprivileged user, `read_only: true`, `tmpfs: /tmp`, `cap_drop: [ALL]`, `no-new-privileges`, `pids_limit: 256`, log rotation `10m × 3`.

---

## Configuration reference

Everything is configured through environment variables. Values in **bold** are the built-in code defaults; Docker Compose and the `start` scripts override some of them (noted below).

### Server and network

| Variable | Default | Meaning |
|---|---|---|
| `VECTOR_CALC_HOST` | `0.0.0.0` | Bind address |
| `VECTOR_CALC_PORT` | `58765` | Port |
| `VECTOR_CALC_LOG_LEVEL` | `info` | uvicorn log level |
| `VECTOR_CALC_DOCS` | `0` | `1` enables the interactive OpenAPI docs at `/docs` |
| `VECTOR_CALC_TRUST_PROXY` | `0` | Trust `X-Forwarded-For` (only behind your own proxy!) |
| `VECTOR_CALC_ALLOWED_HOSTS` | *(empty = any)* | Comma-separated `Host` allowlist |
| `VECTOR_CALC_MAX_BODY` | `16384` | Maximum request body size in bytes |
| `VECTOR_CALC_MAX_CONNECTIONS` | `128` | uvicorn connection limit |

### Models and memory

| Variable | Default | Meaning |
|---|---|---|
| `VECTOR_CALC_ENABLED_MODELS` | *(empty = whole registry)* | Comma-separated model ids exposed by this run; Compose uses `en-small,ru-rusvectores` |
| `VECTOR_CALC_DEFAULT_MODEL` | `en-small` | Model loaded at startup and preselected in the UI |
| `VECTOR_CALC_MODEL_CACHE_MB` | *(auto)* | Memory budget for resident models; `0` disables the limit. Auto = available RAM − 1 GB reserve (cgroup limit in Docker, `/proc/meminfo`, or the Windows API; falls back to 3072 MB, minimum 768 MB) |
| `VECTOR_CALC_MODEL_IDLE_EVICTION_S` | `180` | Evict a model that has been idle for this many seconds when the budget is exceeded |
| `VECTOR_CALC_MAX_MEMORIES` | `50000` | Keep only the newest N records in the gallery; `0` disables pruning |

### Limits

| Variable | Default | Meaning |
|---|---|---|
| `VECTOR_CALC_RATE_API` | `600` | API/page requests per minute per IP |
| `VECTOR_CALC_RATE_WRITE` | `30` | Heavy POST requests (calculate/save/moderation) per minute per IP |
| `VECTOR_CALC_MAX_CONCURRENT` | `4` | Simultaneous CPU-heavy calculations |
| `VECTOR_CALC_SLOT_TIMEOUT` | `10` | Seconds to wait for a free calculation slot (then `503` + `Retry-After: 5`) |

### Moderation and admin

| Variable | Default | Meaning |
|---|---|---|
| `VECTOR_CALC_MODERATION` | `warn` *(Compose: `queue`)* | `off` — no checks, everything approved; `warn` — check and log, still approve; `queue` — anything not `ok` goes to the admin queue; `strict` — everything goes to the queue |
| `VECTOR_CALC_MODERATION_PROFILE` | `teen` | Audience profile: `kids` (stricter, 7+) or `teen` (softer, under 18) |
| `VECTOR_CALC_DEFAULT_MODE` | `kids` | Default gallery display mode served to the UI (`kids` / `teen` / `off`); a manual choice in the gear is stored in the browser and overrides it |
| `VECTOR_CALC_MODERATION_NEIGHBORS` | `10` | How many nearest words of the answer to check against the dictionaries |
| `VECTOR_CALC_MODERATION_MIN_HITS` | profile default (`2` kids / `4` teen) | Dictionary hits needed to trigger a vector-based flag |
| `VECTOR_CALC_LLM_URL` | *(empty = LLM disabled)* | OpenAI-compatible base URL, e.g. `http://192.168.1.102:8080/v1` |
| `VECTOR_CALC_LLM_MODEL` | `local` | Model name sent in the chat completion request |
| `VECTOR_CALC_LLM_API_KEY` | *(empty)* | Bearer token for the LLM endpoint |
| `VECTOR_CALC_LLM_TIMEOUT` | `60` | LLM request timeout, seconds |
| `VECTOR_CALC_LLM_MAX_TOKENS` | `800` | Response token cap |
| `VECTOR_CALC_LLM_NO_THINK` | `1` | Ask the LLM to skip its reasoning phase (`chat_template_kwargs.enable_thinking=false`) — ~1 s per verdict instead of 5–10 s |
| `VECTOR_CALC_ADMIN_TOKEN` | *(empty = admin disabled)* | Token for all `/api/moderation/*` endpoints and manual group changes. Compose ships a placeholder value — **override it in `.env`** |

---

## HTTP API

Base URL: `http://<host>:58765`. All responses are JSON; errors have the shape `{"error": "..."}`. The API is a single process: run one uvicorn worker (the model cache and rate limiter are in-process state).

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/` | — | Calculator page |
| `GET` | `/memories` | — | Gallery page |
| `GET` | `/api/models` | — | Enabled models with state, sizes, the cache budget and `default_mode` |
| `GET` | `/api/status` | — | Same, plus which model is currently loaded |
| `POST` | `/api/models/{id}/load` | — | Start a background load of a model (`{"started": true/false, "state": {...}}`) |
| `GET` | `/api/health` | — | Liveness probe (default model, resident models, resident MB) |
| `POST` | `/api/calculate` | — | Run an equation |
| `GET` | `/api/memories` | — | List gallery records (`after`, `limit` ≤ 100, `model`, `status`, `mode`) |
| `POST` | `/api/memories` | — | Save an equation (deduplicated) |
| `POST` | `/api/moderate` | admin* | Run all moderation layers on an equation without saving |
| `GET` | `/api/moderation/queue` | admin | Pending records |
| `POST` | `/api/moderation/{id}?action=approve\|reject` | admin | Approve or reject a record |
| `POST` | `/api/moderation/{id}/recheck?method=vectors\|llm` | admin | Re-assess one record |
| `POST` | `/api/moderation/{id}/group?group=kids\|teen\|adult` | admin | Manually set the access group |

\* `/api/moderate` requires the token only when `VECTOR_CALC_ADMIN_TOKEN` is configured. The admin token is passed as the `X-Admin-Token` header or a `?token=` query parameter.

### `POST /api/calculate`

```json
{
  "model": "en-small",
  "terms": [
    {"text": "king", "weight": 1},
    {"text": "man", "weight": -1},
    {"text": "woman", "weight": 1}
  ],
  "top_k": 5,
  "exclude_terms": true
}
```

Real response (trimmed):

```json
{
  "results": [
    {"text": "queen", "token": "queen", "similarity": 0.689616, "source": "word2vec"},
    {"text": "monarch", "token": "monarch", "similarity": 0.557549, "source": "word2vec"},
    {"text": "throne", "token": "throne", "similarity": 0.556537, "source": "word2vec"}
  ],
  "candidate_count": 399997,
  "dimensions": 300,
  "mode": "word2vec",
  "model": "en-small",
  "model_name": "GloVe Wiki+Gigaword 300D",
  "language": "en",
  "backend": "Word2Vec / NumPy",
  "resolved": [
    {"text": "king",  "weight": 1.0,  "matched": ["king"]},
    {"text": "man",   "weight": -1.0, "matched": ["man"]},
    {"text": "woman", "weight": 1.0,  "matched": ["woman"]}
  ],
  "vector_preview": [ -0.027174, -0.044044, 0.001833, "..."]
}
```

Validation rules: 1–12 terms; `text` 1–600 characters; `weight` from −100 to 100; `top_k` 1–50; unknown fields are rejected (`extra="forbid"`). Empty word fields are silently ignored by the UI, not by the API.

### `POST /api/memories`

```json
{
  "model": "ru-rusvectores",
  "terms": [
    {"text": "король", "weight": 1},
    {"text": "мужчина", "weight": -1},
    {"text": "женщина", "weight": 1}
  ]
}
```

- Terms are limited to 1–12 items, `text` ≤ 120 characters, `weight` is exactly `+1` or `−1`.
- The equation is deduplicated by a SHA-256 signature over the model id and the canonically normalized terms.
- Response `201` with `{"item": {...}, "created": true, "verdict": "ok", "reasons": [...]}` for a new record, `200` with `created: false` for a duplicate. If moderation returns `pending` (queue/strict mode), the record waits in the queue and is hidden from the public list until approved.

### `GET /api/memories`

Query parameters: `after` (id cursor for incremental polling), `limit` (1–100), `model`, `status` (`approved` by default; any other value requires the admin token), `mode` (`kids` / `teen` / `off` — filters records by access group). Response: `{"total": N, "items": [...]}`; with `mode` also `"shown"` and `"mode"`. A record looks like:

```json
{
  "id": 50,
  "model": "ru-rusvectores",
  "status": "approved",
  "group": "kids",
  "moderation": {"verdict": "ok", "reasons": [], "source": "vectors", "group": "kids", "mode": "queue", "profile": "teen"},
  "created_at": "2026-10-23T06:42:25+00:00",
  "terms": [{"text": "король", "weight": 1}, {"text": "мужчина", "weight": -1}, {"text": "женщина", "weight": 1}],
  "answer": "королева",
  "similarity": 0.746273
}
```

### Status codes

| Code | When |
|---|---|
| `400` | Validation error, unknown model id, unparsable JSON, unknown word in the equation |
| `403` | Missing or wrong admin token, or admin access not configured |
| `404` | Unknown model id in `/api/models/{id}/load`, unknown memory id, unknown paths |
| `413` | Request body larger than `VECTOR_CALC_MAX_BODY` |
| `429` | Per-IP rate limit exceeded (`Retry-After: 60`) |
| `503` | Model is loading or unavailable (`Retry-After: 10`), no free calculation slot (`Retry-After: 5`), or a model file is missing |

---

## Moderation and access groups

Every saved equation can be assessed by three layers, from cheapest to smartest:

1. **Hashed dictionaries** — `moderation/*.txt` contain SHA-256 hashes of words and stems (so the repository holds no explicit obscene words), split into categories (`obscene`, `sensitive`) plus an allow list. Russian words are checked both as written and as pymorphy3 lemmas/stems; English words are case-folded. Category severity depends on the profile.
2. **Vector heuristics** — the nearest neighbors of the answer (and of the terms) are checked against the same dictionaries. If enough neighbors hit a category (`VECTOR_CALC_MODERATION_MIN_HITS`), the record is flagged even when the exact word is not in the list.
3. **LLM (optional)** — if `VECTOR_CALC_LLM_URL` is configured, a local OpenAI-compatible model reviews the whole equation (terms + answer) and returns strict JSON `{"verdict": "ok|review|block", "reasons": [...]}`. With `VECTOR_CALC_LLM_NO_THINK=1` (default) the request asks the engine to disable its reasoning phase; the client also falls back to `reasoning_content` if `content` is empty.

**Verdicts → access groups:** `ok → kids`, `review → teen`, `block → adult`.

**Profiles:**

| Profile | Obscene words | Sensitive words | Neighbor hits | Audience |
|---|---|---|---|---|
| `kids` | `block` | `review` | 2 | children from 7 |
| `teen` | `review` | ignored | 4 | teenagers under 18 |

**Save-time modes** (`status_for`): `off` → everything `approved`; `warn` → everything `approved` (assessment is stored for the record); `queue` → `pending` unless the verdict is `ok`; `strict` → always `pending`.

**Display modes** (client side, stored per browser): `kids` shows only the `kids` group, `teen` shows `kids` + `teen`, `off` shows everything. Records without an assessment are treated as `kids` (visible everywhere). The default mode is `kids`.

**Admin workflow:** the gear button on `/memories` opens the settings block — assessment method (`vectors` / `llm`), "Re-assess all records" (runs sequentially with a progress line and stores who assessed each record and why), and the admin token field. Pending records can be approved/rejected; each card shows the assessment line (`Assessment: safe · by: vectors · Access group: Kids`) and a group dropdown; a manual group change is recorded as `source: "human"`.

**CLI helper:**

```bash
python moderation.py --mode                       # current profile, mode and LLM settings
python moderation.py --check "текст"              # dictionary check
python moderation.py --hash "слово"               # sha256 value for the dictionary files
python moderation.py --llm "король - мужчина + женщина = королева"
```

---

## The memories database

A single SQLite file `memories.sqlite3` next to the code (git-ignored, mounted from the host in Docker).

```sql
CREATE TABLE memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  signature TEXT NOT NULL UNIQUE,   -- sha256(model | canonical terms, casefolded)
  model TEXT NOT NULL DEFAULT 'ru-rusvectores',
  status TEXT NOT NULL DEFAULT 'approved',   -- approved | pending | rejected
  moderation TEXT,                  -- JSON verdict report
  group_name TEXT,                  -- kids | teen | adult (NULL = not assessed yet)
  created_at TEXT NOT NULL,
  terms_json TEXT NOT NULL,
  answer TEXT NOT NULL,
  similarity REAL NOT NULL
);
```

- **Indexes** on `(id DESC)`, `(model, id DESC)`, `(status, id DESC)`, `(group_name, id DESC)`.
- **Migrations** run on startup: legacy databases get the `model` / `status` / `moderation` / `group_name` columns, and content signatures are recomputed for the multi-model era (duplicates found by the new signature are merged). All migrations are idempotent.
- **Pruning** keeps only the newest `VECTOR_CALC_MAX_MEMORIES` rows (50 000 by default) so the shared gallery cannot fill the disk.
- To reset the gallery, stop the server and delete `memories.sqlite3` — it is recreated automatically.

---

## UI guide

### Main page (`/`)

1. **Header** — brand, EN/RU switch (hidden when one language is available), model select with RAM hints, and a status pill (checking / loading / downloading / vocabulary size).
2. **Calculator** — right under the heading. 1–12 terms, each with a `+`/`−` toggle (click the sign), an input and a delete button; "Add word"; "Hide source words"; answers count 5/10/20; **Calculate**. Empty rows are ignored silently. Results appear in the card below: the winner with its cosine similarity, the rest of the top list, the recognized tokens and the candidate count.
3. **Save** — `⭐ Save to memories` (enabled after a calculation) and a link to the full gallery. The feedback line reports "added", "already in the collection" or "sent for review" depending on moderation.
4. **💡 Examples & memories** — one merged block:
   - built-in examples (language-dependent: `King − man + woman`, `Paris − France + Italy`, `Sushi − Japan + Italy` / `Король − мужчина + женщина`, `Россия − Путин`, `Россия − Москва + Франция`) are shown **while the collection is empty or until you run the calculator for the first time**;
   - after the first successful calculation the built-in examples disappear **forever** for this browser (`localStorage: vp.usedCalculator`) and only fresh community examples remain; they only come back as a fallback if the collection is completely empty;
   - community chips poll the gallery once a second (newest 12 for the current model and display mode); tapping one loads its equation into the calculator.

### Gallery (`/memories`)

- Counter (`Total: N · shown: M`), model filter including "All models", and a `↻ Refresh` implied by the live refresh.
- **Gear button** with the current display mode (🧑 Kids / 🧑‍🎓 Teens / 🌐 All): choosing a mode filters the gallery and is stored for this browser forever. The settings block right under the heading contains the assessment method, "Re-assess all records", the admin token and short explanations.
- **Cards** show the equation and answer, the model tag, cosine similarity and date, the assessment line, the assessment reason (including a quoted LLM reason when present) and the access-group dropdown. Clicking a card (or pressing Enter/Space) stores `vp.pendingExample` and navigates to the calculator, which loads the equation automatically.

### Local storage keys

| Key | Content |
|---|---|
| `vp.lang` | `en` / `ru` |
| `vp.model` | Current model id |
| `vp.lastModel.<lang>` | Last model used for each language |
| `vp.memoriesMode` | Display mode: `kids` (default) / `teen` / `off` |
| `vp.usedCalculator` | `1` after the first successful calculation (hides built-in examples) |
| `vp.recheckMethod` | `vectors` / `llm` for "Re-assess all records" |
| `vp.memoriesFilter` | Gallery model filter (`all` or a model id) |
| `vp.adminToken` | Admin token for the settings block |
| `vp.pendingExample` | Transient equation handed from the gallery to the calculator |

If `vp.model` points to a model that was since disabled, the page self-heals: it clears the stored value and reloads once (the `vp.staleModel` session flag prevents reload loops). After changing `VECTOR_CALC_ENABLED_MODELS`, a browser refresh (F5) is enough.

---

## Architecture and internals

### File map

| File | Role |
|---|---|
| `server.py` | FastAPI app: middleware (body cap, rate limits, security headers), endpoints, request validation, admin guard, lifecycle, `uvicorn` bootstrap |
| `word2vec_engine.py` | Model registry (`MODELS`), readers (word2vec binary, gzip binary, gzip text, zip with `model.bin`), tokenization/lemmatization, the calculation core and the `ModelManager` LRU cache |
| `moderation.py` | Dictionaries, profiles, vector heuristics, the LLM client, verdict/group/status mapping, CLI |
| `memories_db.py` | SQLite schema, migrations, signatures, dedup, pruning, listing, assessments |
| `app.js` | Shared front-end: i18n tables, presets, language/model switcher, display modes, RAM hints |
| `index.html` | Calculator page (inline UI logic) |
| `memories.html` | Gallery page (inline UI logic) |
| `download_model.py` | Resumable downloader with mirrors, validation and `.ok` markers |
| `docker-entrypoint.sh` | Writability checks + background download of the default model |
| `Dockerfile`, `docker-compose.yml` | Non-root image (Python 3.12-slim) and the hardened Compose service |
| `test_word2vec.py`, `test_moderation.py` | Unit tests |
| `test_ui.py`, `test_memories_ui.py` | Playwright smoke tests (need a running server) |

### Model manager and the memory budget

- Only one background load runs at a time; `POST /api/models/{id}/load` returns `{"started": false}` if another load is in progress — the web UI retries every few seconds, so no request ever blocks on a multi-hundred-megabyte load.
- Models stay resident together while they fit into the budget (`available RAM − 1 GB`, overridable). When the budget is exceeded, the model that has been idle longest is evicted (idle threshold 180 s); a just-used model is never evicted, so a request gets a clear `503` with `Retry-After` instead of endless reload thrashing.
- `GET /api/models` and `GET /api/status` expose the budget, the resident list and the actual size of each loaded model; startup logs a line like `Модель ... загружена: 400000 слов, ~505 МБ`.
- The registry supports reading only the first N tokens of a binary model (`max_words`, frequency order) — a pruning knob kept for future "small" variants.

### Calculation

1. Each term is resolved to tokens: English tries the exact case, lowercase and capitalized forms; Russian lemmatizes with pymorphy3 and picks the best POS form (`король → король_NOUN`).
2. Term vectors are added with their weights: `v = Σ wᵢ · normalize(vec(tokenᵢ))`.
3. Similarity is the cosine between `v` and every normalized token vector: one matrix–vector product for the whole vocabulary (NumPy), `candidate_count` in the response shows how many tokens were compared.
4. Source words are optionally excluded from the results (`exclude_terms`, on by default).

---

## Tests

### Unit tests (no server needed)

```bash
python -m unittest test_word2vec test_moderation -v
```

35 tests cover: all three readers plus gzip text detection and the `N D` header skip, tokenizers and case variants, the similarity math against a normalized-dot reference, registry invariants (four models, the compact English model is a standalone <500 MB download, default is English), the LRU cache policy, moderation profiles (`kids` stricter than `teen`), modes, verdict→status mapping and LLM JSON parsing.

### Playwright smoke tests (server on `127.0.0.1:58765`, Chrome)

```bash
python server.py            # in one terminal
python test_ui.py           # 4 mobile/desktop viewports: layout, add/remove terms, preset, calculation
python test_memories_ui.py  # merged examples behavior, save, dedup, 1s sync, gallery, second visitor
```

`test_memories_ui.py` verifies the merged block explicitly: built-in examples are visible for a fresh browser, disappear after the first calculation, stay hidden after a reload, and fresh community examples propagate to a second browser and to the gallery within ~1 second.

---

## Measured performance

Measured on a desktop CPU (Windows, Python 3.12) with the shipped models:

| Model | Load time | Tokens | Dimensions | Vector data |
|---|---|---|---|---|
| `en-small` (GloVe Wiki+Gigaword 300D) | ~22 s (text parse, first load) | 400,000 | 300 | 505 MB |
| `ru-rusvectores` (RusVectōrēs Word2Vec 300D) | ~2 s (zip, binary) | 248,978 | 300 | 314 MB |

- Both default models together need roughly **1–1.2 GB of process memory** including Python and vocabularies; a server with 2 GB of RAM runs them comfortably.
- A calculation is a single 400k × 300 matrix–vector product: a few dozen milliseconds per request once the model is loaded.
- Cold start is not blocked by the model: the server is up immediately, the model loads in the background, and the UI shows the state.

---

## Troubleshooting

**The page shows "Server unavailable" right after `docker compose up -d`.**
Give it a couple of seconds — the entrypoint starts uvicorn last. Then check `docker compose logs -f`.

**"Model not found" / "Model is downloading…" for a long time.**
The model file is missing or still downloading. Watch `docker compose logs -f`, or download manually: `docker compose exec vectorplay python download_model.py --all`.

**The UI shows "Not enough memory for two models".**
The cache budget (available RAM − 1 GB) is smaller than the two models together. Raise `VECTOR_CALC_MODEL_CACHE_MB`, keep only one model in `VECTOR_CALC_ENABLED_MODELS`, or add RAM.

**A model id disappeared after a config change and the page shows occasional 400s.**
Your browser stored the old model in `localStorage`. The page clears it and reloads once by itself; a manual F5 also helps.

**Docker: "all predefined address pools have been fully subnetted".**
`docker network prune -f`, extend `default-address-pools` in `daemon.json`, or switch the service to `network_mode: host`.

**Docker: "cannot create /app/models" or "not writable".**
The mounted folder belongs to another user. Rebuild with `VECTORPLAY_UID=$(id -u) VECTORPLAY_GID=$(id -g) docker compose up -d --build`, fix the ownership, or (last resort) run the container as root.

**Calculations return 429/503 under automation.**
Raise `VECTOR_CALC_RATE_WRITE` / `VECTOR_CALC_RATE_API` or `VECTOR_CALC_MAX_CONCURRENT`. `503` with `Retry-After: 10` simply means the model is still loading; `503` with `Retry-After: 5` means all calculation slots are busy.

**The LLM moderation is slow.**
Keep `VECTOR_CALC_LLM_NO_THINK=1` (default), lower `VECTOR_CALC_LLM_MAX_TOKENS`, or leave the LLM disabled (dictionaries and vector heuristics still work).

**I want to reduce disk usage.**
Disabled models are never downloaded; the two default archives take about 1 GB. `models/` is safe to inspect — the downloader only checks the file it needs. Deleting an archive and re-running `download_model.py` re-fetches it (resumable).

---

## Repository layout

```
vector-play/
├── server.py               # FastAPI app and endpoints
├── word2vec_engine.py      # models, readers, calculation, cache
├── moderation.py           # moderation pipeline + CLI
├── memories_db.py          # SQLite schema and queries
├── download_model.py       # resumable model downloader
├── app.js                  # shared front-end (i18n, switchers)
├── index.html              # calculator page
├── memories.html           # gallery page
├── moderation/             # hashed dictionaries (obscene/sensitive/allow)
├── models/                 # model archives (git-ignored, created on demand)
├── memories.sqlite3        # shared gallery database (git-ignored)
├── docker-compose.yml
├── Dockerfile
├── docker-entrypoint.sh
├── start.sh / start.bat
├── test_word2vec.py        # engine + cache unit tests
├── test_moderation.py      # moderation unit tests
├── test_ui.py              # Playwright smoke test (calculator)
└── test_memories_ui.py     # Playwright smoke test (memories)
```

## Notes

- The service is built for a home LAN / small group: single process, in-memory rate limiting and cache, no user accounts.
- Server-side messages (errors, logs, moderation reasons) are in Russian; the web interface itself is bilingual (EN/RU) and follows the selected model's language.
- Model files are redistributed from their public mirrors (gensim-data release assets, Stanford NLP) and keep their original licenses; the application code is a personal project of [@ramires666](https://github.com/ramires666).
