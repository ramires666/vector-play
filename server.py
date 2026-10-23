"""VectorPlay — shared LAN Word2Vec calculator and memories gallery.

FastAPI + uvicorn front end, hardened for internet-facing deployments:
  * strict pydantic validation and a hard request-body cap (16 KB by default);
  * per-IP rate limits plus a concurrency cap for CPU-heavy calculations;
  * CSP and other security headers on every response, /docs disabled,
    the uvicorn Server header suppressed;
  * the model loads in a background thread at startup; requests that arrive
    during the load get a clean 503 with Retry-After instead of blocking.

The public API is unchanged: /, /memories, /api/status, /api/memories,
/api/calculate, plus /api/health for monitoring.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

import memories_db as memories
import moderation
import word2vec_engine as engine

ROOT = Path(__file__).resolve().parent
HOST = os.getenv("VECTOR_CALC_HOST", "0.0.0.0")
PORT = int(os.getenv("VECTOR_CALC_PORT", "58765"))
LOG_LEVEL = os.getenv("VECTOR_CALC_LOG_LEVEL", "info")

# Trust X-Forwarded-For only when a reverse proxy rewrites it (see README).
TRUST_PROXY = os.getenv("VECTOR_CALC_TRUST_PROXY", "0").strip().lower() in {"1", "true", "yes", "on"}
MAX_BODY = int(os.getenv("VECTOR_CALC_MAX_BODY", "16384"))
MAX_CONCURRENT = max(1, int(os.getenv("VECTOR_CALC_MAX_CONCURRENT", "4")))
SLOT_TIMEOUT = float(os.getenv("VECTOR_CALC_SLOT_TIMEOUT", "10"))
RATE_API = int(os.getenv("VECTOR_CALC_RATE_API", "600"))        # requests per minute per IP
RATE_WRITE = int(os.getenv("VECTOR_CALC_RATE_WRITE", "30"))     # heavy POSTs per minute per IP
HEAVY_POST_PATHS = {"/api/calculate", "/api/memories", "/api/moderate"}   # они идут через WRITE_LIMITER
ADMIN_TOKEN = os.getenv("VECTOR_CALC_ADMIN_TOKEN", "").strip()
# Режимы показа коллекции (шестерёнка на /memories) → какие группы допуска видеть.
# Запись без оценки считается безопасной (kids) и видна во всех режимах.
MODE_GROUPS: dict[str, tuple] = {
    "kids": ("kids",),
    "teen": ("kids", "teen"),
    "off": ("kids", "teen", "adult"),
}


def _item_group(item: dict) -> str:
    if item.get("group"):
        return item["group"]
    verdict = (item.get("moderation") or {}).get("verdict")
    return {"ok": "kids", "review": "teen", "block": "adult"}.get(verdict, "kids")
MAX_CONNECTIONS = int(os.getenv("VECTOR_CALC_MAX_CONNECTIONS", "128"))
DOCS_ENABLED = os.getenv("VECTOR_CALC_DOCS", "0").strip().lower() in {"1", "true", "yes", "on"}
ALLOWED_HOSTS = [item.strip() for item in os.getenv("VECTOR_CALC_ALLOWED_HOSTS", "").split(",") if item.strip()]

HTML_CSP = ("default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'unsafe-inline'; "
            "connect-src 'self'; img-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
API_CSP = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"

logger = logging.getLogger("vectorplay")


class RateLimiter:
    """In-memory sliding-window limiter, one bucket per client key."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = max(0, per_minute)
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._cleanup = 0.0

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                hits = self._hits[key] = deque()
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            if now - self._cleanup >= 300.0:
                self._cleanup = now
                for bucket_key in list(self._hits):
                    bucket = self._hits[bucket_key]
                    while bucket and bucket[0] <= cutoff:
                        bucket.popleft()
                    if not bucket:
                        del self._hits[bucket_key]
            return True


API_LIMITER = RateLimiter(RATE_API)
WRITE_LIMITER = RateLimiter(RATE_WRITE)
CALC_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT)


class BodyTooLarge(HTTPException):
    """Raised while reading a request body that exceeds the configured cap.

    Inherits from HTTPException because FastAPI converts any other exception
    raised while parsing the body into a generic 400.
    """

    def __init__(self) -> None:
        super().__init__(status_code=413, detail="Тело запроса слишком большое")


class BodySizeLimitMiddleware:
    """Reject oversized bodies even when Content-Length is absent (chunked)."""

    def __init__(self, app, limit: int) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        seen = 0

        async def limited_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    raise BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except BodyTooLarge as exc:
            await _json_error(exc.status_code, exc.detail)(scope, receive, send)


def _client_ip(request: Request) -> str:
    if TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for", "")
        hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
        if hops:
            return hops[-1][:64]
    return request.client.host if request.client else "unknown"


def _security_headers(response: Response) -> None:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=(), payment=(), usb=()"
    response.headers["Cache-Control"] = "no-store"
    content_type = response.headers.get("content-type", "")
    response.headers["Content-Security-Policy"] = HTML_CSP if content_type.startswith("text/html") else API_CSP


def _json_error(status_code: int, message: str, retry_after: str | None = None) -> JSONResponse:
    headers = {"Retry-After": retry_after} if retry_after else None
    response = JSONResponse(status_code=status_code, content={"error": message}, headers=headers)
    _security_headers(response)
    return response


@contextmanager
def _calc_slot():
    """Cap the number of simultaneous CPU-heavy calculations."""
    if not CALC_SLOTS.acquire(timeout=SLOT_TIMEOUT):
        raise HTTPException(status_code=503, detail="Сервер занят вычислениями, попробуйте ещё раз",
                            headers={"Retry-After": "5"})
    try:
        yield
    finally:
        CALC_SLOTS.release()


def _require_admin(request: Request) -> None:
    """Доступ к очереди модерации — только по токену из VECTOR_CALC_ADMIN_TOKEN."""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Админ-доступ не настроен: задайте VECTOR_CALC_ADMIN_TOKEN")
    token = request.headers.get("x-admin-token") or request.query_params.get("token", "")
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Неверный админ-токен")


class CalcTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    weight: float = 1

    @field_validator("text")
    @classmethod
    def check_text(cls, value: str) -> str:
        value = value.strip()
        if not 1 <= len(value) <= 600:
            raise ValueError("Каждое выражение — от 1 до 600 символов")
        return value

    @field_validator("weight")
    @classmethod
    def check_weight(cls, value: float) -> float:
        if not -100 <= value <= 100 or value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Коэффициент должен быть числом от -100 до 100")
        return value


class MemoryTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    weight: int

    @field_validator("text")
    @classmethod
    def check_text(cls, value: str) -> str:
        value = value.strip()
        if not 1 <= len(value) <= 120:
            raise ValueError("Для коллекции каждое выражение — от 1 до 120 символов")
        return value

    @field_validator("weight")
    @classmethod
    def check_weight(cls, value: int) -> int:
        if value not in (-1, 1):
            raise ValueError("В коллекции поддерживаются только сложение и вычитание")
        return value


class CalculateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = engine.MANAGER.default_id
    terms: list[CalcTerm]
    top_k: int = 10
    exclude_terms: bool = True

    @field_validator("model")
    @classmethod
    def check_model(cls, value: str) -> str:
        if value not in engine.ENABLED_MODEL_IDS:
            raise ValueError(f"Неизвестная модель: {value}")
        return value

    @field_validator("terms")
    @classmethod
    def check_terms(cls, value: list[CalcTerm]) -> list[CalcTerm]:
        if not 1 <= len(value) <= 12:
            raise ValueError("Нужно от 1 до 12 слов")
        return value

    @field_validator("top_k")
    @classmethod
    def check_top_k(cls, value: int) -> int:
        if not 1 <= value <= 50:
            raise ValueError("Ответов должно быть от 1 до 50")
        return value


class MemoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = engine.MANAGER.default_id
    terms: list[MemoryTerm]

    @field_validator("model")
    @classmethod
    def check_model(cls, value: str) -> str:
        if value not in engine.ENABLED_MODEL_IDS:
            raise ValueError(f"Неизвестная модель: {value}")
        return value

    @field_validator("terms")
    @classmethod
    def check_terms(cls, value: list[MemoryTerm]) -> list[MemoryTerm]:
        if not 1 <= len(value) <= 12:
            raise ValueError("Нужно от 1 до 12 слов")
        return value


@asynccontextmanager
async def lifespan(_: FastAPI):
    budget = engine.MANAGER.budget_mb()
    default_id = engine.MANAGER.default_id
    default_mb = engine.MANAGER.spec(default_id).planned_ram_mb
    logger.info("VectorPlay on %s:%s (модель по умолчанию %s, включены: %s, бюджет кэша %d МБ: %s)",
                HOST, PORT, default_id, ", ".join(spec.id for spec in engine.ENABLED_MODELS),
                budget, engine.MANAGER.budget_source)
    others = [spec.planned_ram_mb for spec in engine.ENABLED_MODELS if spec.id != default_id]
    if others and default_mb + min(others) > budget:
        logger.warning("Бюджета кэша не хватает, чтобы держать «%s» (~%d МБ) вместе со второй моделью: "
                       "переключение между моделями будет упираться в память. "
                       "Увеличьте VECTOR_CALC_MODEL_CACHE_MB или выберите модель полегче.", default_id, default_mb)
    started = engine.MANAGER.ensure_async(default_id)
    logger.info("Предзагрузка модели по умолчанию: %s", "запущена" if started else "не нужна")
    yield


app = FastAPI(
    title="VectorPlay",
    docs_url="/docs" if DOCS_ENABLED else None,
    redoc_url=None,
    openapi_url="/openapi.json" if DOCS_ENABLED else None,
    lifespan=lifespan,
)
if ALLOWED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
app.add_middleware(BodySizeLimitMiddleware, limit=MAX_BODY)


@app.middleware("http")
async def guard(request: Request, call_next):
    heavy = request.method == "POST" and request.url.path in HEAVY_POST_PATHS
    limiter = WRITE_LIMITER if heavy else API_LIMITER
    if not limiter.allow(_client_ip(request)):
        return _json_error(429, "Слишком много запросов, подождите минуту", retry_after="60")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY:
        return _json_error(413, "Тело запроса слишком большое")
    response = await call_next(request)
    _security_headers(response)
    return response


@app.get("/", include_in_schema=False)
def index_page() -> FileResponse:
    return FileResponse(ROOT / "index.html", media_type="text/html; charset=utf-8")


@app.get("/memories", include_in_schema=False)
def memories_page() -> FileResponse:
    return FileResponse(ROOT / "memories.html", media_type="text/html; charset=utf-8")


@app.get("/app.js", include_in_schema=False)
def app_script() -> FileResponse:
    return FileResponse(ROOT / "app.js", media_type="text/javascript; charset=utf-8")


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {"default": engine.MANAGER.default_id, "cache_budget_mb": engine.MANAGER.budget_mb(),
            "resident": engine.MANAGER.resident_ids(), "models": engine.MANAGER.states()}


@app.post("/api/models/{model_id}/load")
def load_model(model_id: str) -> dict[str, Any]:
    if model_id not in engine.ENABLED_MODEL_IDS:
        raise HTTPException(status_code=404, detail=f"Неизвестная модель: {model_id}")
    started = engine.MANAGER.ensure_async(model_id)
    return {"started": started, "state": engine.MANAGER.state(model_id)}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "default_model": engine.MANAGER.default_id,
            "enabled_models": sorted(engine.ENABLED_MODEL_IDS),
            "resident_models": engine.MANAGER.resident_ids(),
            "resident_mb": round(engine.MANAGER.resident_mb())}


@app.get("/api/status")
def status() -> dict[str, Any]:
    return {"default": engine.MANAGER.default_id, "loaded": engine.MANAGER.loaded_id(),
            "cache_budget_mb": engine.MANAGER.budget_mb(),
            "resident": engine.MANAGER.resident_ids(), "models": engine.MANAGER.states()}


@app.get("/api/memories")
def list_memories(request: Request,
                  after: Annotated[int, Query(ge=0)] = 0,
                  limit: Annotated[int, Query(ge=1, le=100)] = 50,
                  model: Annotated[str | None, Query(max_length=64)] = None,
                  status: Annotated[str | None, Query(max_length=16)] = None,
                  mode: Annotated[str | None, Query(max_length=16)] = None) -> dict[str, Any]:
    if model is not None and model not in engine.ENABLED_MODEL_IDS:
        raise HTTPException(status_code=400, detail=f"Неизвестная модель: {model}")
    # Публично показываем только одобренное; очередь модерации — по админ-токену.
    if status is None:
        status = "approved"
    elif status != "approved":
        _require_admin(request)
    # Без явного фильтра показываем коллекцию только включённых моделей:
    # записи выключенных (например, английских) в интерфейсе не мелькают.
    enabled = None if model is not None else sorted(engine.ENABLED_MODEL_IDS)
    result = memories.list_memories(after=after, limit=limit, model=model, models=enabled, status=status)
    # Режим показа (шестерёнка на странице коллекции): фильтруем по группе допуска —
    # kids видит только детские, teen — детские и подростковые, off — всё.
    groups = MODE_GROUPS.get(mode or "")
    if groups is not None:
        result["items"] = [item for item in result["items"] if _item_group(item) in groups]
        result["shown"] = len(result["items"])
        result["mode"] = mode
    return result


@app.post("/api/calculate")
def calculate(payload: CalculateRequest) -> dict[str, Any]:
    terms = [term.model_dump() for term in payload.terms]
    try:
        with _calc_slot():
            return engine.calculate(terms, top_k=payload.top_k, exclude_terms=payload.exclude_terms,
                                    model_id=payload.model)
    except engine.ModelNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "10"}) from exc
    except engine.ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except engine.UnknownModel as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/memories")
def add_memory(payload: MemoryRequest, response: Response) -> dict[str, Any]:
    terms = [term.model_dump() for term in payload.terms]
    try:
        existing = memories.find_existing(terms, model_id=payload.model)
        if existing is not None:
            response.status_code = 200
            return {"item": existing, "created": False}
        report = None
        status = "approved"
        if moderation.MODE != "off":
            with _calc_slot():
                calculation = engine.calculate(terms, top_k=10, exclude_terms=True, model_id=payload.model)
            # Модерация (включая LLM) идёт вне слота расчёта: LLM может думать секунды,
            # и держать из-за неё очередь вычислений нельзя.
            report = moderation.review(terms, calculation["results"], model_id=payload.model)
            status = moderation.status_for(report["verdict"])
        item, created = memories.add_memory(terms, model_id=payload.model, status=status, moderation=report)
    except engine.ModelNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "10"}) from exc
    except engine.ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except engine.UnknownModel as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response.status_code = 201 if created else 200
    return {"item": item, "created": created, "verdict": (report or {}).get("verdict"),
            "reasons": (report or {}).get("reasons", [])}


@app.post("/api/moderate")
def moderate(payload: CalculateRequest, request: Request) -> dict[str, Any]:
    """Проверка уравнения всеми слоями модерации (тесты и админка)."""
    if ADMIN_TOKEN:
        _require_admin(request)
    terms = [term.model_dump() for term in payload.terms]
    try:
        with _calc_slot():
            calculation = engine.calculate(terms, top_k=10, exclude_terms=True, model_id=payload.model)
    except engine.ModelNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "10"}) from exc
    except engine.ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    report = moderation.review(terms, calculation["results"], model_id=payload.model)
    report["proposed_status"] = moderation.status_for(report["verdict"])
    report["sample"] = [item["token"] for item in calculation["results"][:5]]
    return report


@app.get("/api/moderation/queue")
def moderation_queue(request: Request,
                     limit: Annotated[int, Query(ge=1, le=100)] = 50) -> dict[str, Any]:
    _require_admin(request)
    return memories.list_memories(limit=limit, status="pending")


@app.post("/api/moderation/{memory_id}")
def moderation_decide(memory_id: int, request: Request,
                      action: Annotated[str, Query(pattern="^(approve|reject)$")]) -> dict[str, Any]:
    _require_admin(request)
    item = memories.set_status(memory_id, "approved" if action == "approve" else "rejected")
    if item is None:
        raise HTTPException(status_code=404, detail="Запись не найдена")
    return {"item": item, "action": action}


def _memory_row(memory_id: int):
    from contextlib import closing
    with closing(memories.connect()) as conn:
        row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
    return memories.as_item(row) if row is not None else None


@app.post("/api/moderation/{memory_id}/recheck")
def moderation_recheck(memory_id: int, request: Request,
                       method: Annotated[str, Query(pattern="^(llm|vectors)$")] = "vectors") -> dict[str, Any]:
    """Переоценка одной записи: словари+вектора или ещё и LLM (по выбору)."""
    _require_admin(request)
    item = _memory_row(memory_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Запись не найдена")
    try:
        with _calc_slot():
            calculation = engine.calculate(item["terms"], top_k=10, exclude_terms=True, model_id=item["model"])
    except engine.ModelNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "10"}) from exc
    except engine.ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (engine.UnknownModel, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    report = moderation.review(item["terms"], calculation["results"], model_id=item["model"],
                               use_llm=(method == "llm"))
    report["source"] = "llm" if (method == "llm" and report.get("llm")) else "vectors"
    report["group"] = moderation.group_for(report["verdict"])
    updated = memories.set_assessment(memory_id, moderation=report, group_name=report["group"])
    return {"item": updated, "method": method, "group": report["group"], "verdict": report["verdict"]}


@app.post("/api/moderation/{memory_id}/group")
def moderation_set_group(memory_id: int, request: Request,
                         group: Annotated[str, Query(pattern="^(kids|teen|adult)$")]) -> dict[str, Any]:
    """Ручная смена группы допуска: оценка помечается как сделанная человеком."""
    _require_admin(request)
    item = _memory_row(memory_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Запись не найдена")
    report = dict(item.get("moderation") or {})
    report.update({
        "verdict": {"kids": "ok", "teen": "review", "adult": "block"}[group],
        "group": group,
        "source": "human",
        "reasons": ["Группу допуска изменили вручную"],
        "mode": moderation.MODE,
        "profile": moderation.PROFILE,
    })
    updated = memories.set_assessment(memory_id, moderation=report, group_name=group)
    return {"item": updated, "group": group}


def _first_error_message(exc: RequestValidationError) -> str:
    for error in exc.errors():
        if error.get("type") == "json_invalid":
            return "Некорректный JSON в теле запроса"
        message = str(error.get("msg", "")).removeprefix("Value error, ")
        if message:
            return message
    return "Некорректный запрос"


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return _json_error(400, _first_error_message(exc))


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) and exc.detail else "Ошибка запроса"
    response = _json_error(exc.status_code, detail)
    if exc.headers:
        response.headers.update(exc.headers)
    return response


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _json_error(500, "Внутренняя ошибка сервера")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):   # консоль Windows может быть в cp1251
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL,
        server_header=False,
        timeout_keep_alive=5,
        limit_concurrency=MAX_CONNECTIONS,
        proxy_headers=False,
        access_log=True,
    )
