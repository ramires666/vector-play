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
MAX_CONNECTIONS = int(os.getenv("VECTOR_CALC_MAX_CONNECTIONS", "128"))
DOCS_ENABLED = os.getenv("VECTOR_CALC_DOCS", "0").strip().lower() in {"1", "true", "yes", "on"}
ALLOWED_HOSTS = [item.strip() for item in os.getenv("VECTOR_CALC_ALLOWED_HOSTS", "").split(",") if item.strip()]

HTML_CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
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

    terms: list[CalcTerm]
    top_k: int = 10
    exclude_terms: bool = True

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

    terms: list[MemoryTerm]

    @field_validator("terms")
    @classmethod
    def check_terms(cls, value: list[MemoryTerm]) -> list[MemoryTerm]:
        if not 1 <= len(value) <= 12:
            raise ValueError("Нужно от 1 до 12 слов")
        return value


@asynccontextmanager
async def lifespan(_: FastAPI):
    started = engine.ensure_loaded_async()
    logger.info("VectorPlay on %s:%s (model available=%s, preload started=%s)",
                HOST, PORT, engine.available(), started)
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
    limiter = WRITE_LIMITER if request.method == "POST" and request.url.path.startswith("/api/") else API_LIMITER
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


@app.get("/api/health")
def health() -> dict[str, Any]:
    state = engine.state()
    return {"status": "ok", "model_available": state["available"], "model_loaded": state["loaded"]}


@app.get("/api/status")
def status() -> dict[str, Any]:
    return engine.state()


@app.get("/api/memories")
def list_memories(after: Annotated[int, Query(ge=0)] = 0,
                  limit: Annotated[int, Query(ge=1, le=100)] = 50) -> dict[str, Any]:
    return memories.list_memories(after=after, limit=limit)


@app.post("/api/calculate")
def calculate(payload: CalculateRequest) -> dict[str, Any]:
    terms = [term.model_dump() for term in payload.terms]
    try:
        with _calc_slot():
            return engine.calculate(terms, top_k=payload.top_k, exclude_terms=payload.exclude_terms)
    except engine.ModelNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "10"}) from exc
    except engine.ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/memories")
def add_memory(payload: MemoryRequest, response: Response) -> dict[str, Any]:
    terms = [term.model_dump() for term in payload.terms]
    try:
        with _calc_slot():
            item, created = memories.add_memory(terms)
    except engine.ModelNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "10"}) from exc
    except engine.ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response.status_code = 201 if created else 200
    return {"item": item, "created": created}


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
