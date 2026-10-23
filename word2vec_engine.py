"""Multi-model word-vector engine: Word2Vec binary and GloVe text.

Models are registered in MODELS; each one is loaded on demand, and only one
model is kept in RAM at a time (the large English models need several GB).
Loading happens in a background thread: requests that arrive while a model is
being loaded get ModelNotReady and can answer 503 with Retry-After.

Memory note: similarity is computed as (vectors @ target) / norms, which is
mathematically identical to dotting with a normalized copy of the matrix but
keeps only a single float32 matrix in RAM.
"""
from __future__ import annotations

import array
import gzip
import logging
import math
import os
import re
import threading
import time
import warnings
import zipfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymorphy3

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DEFAULT_MODEL = os.getenv("VECTOR_CALC_DEFAULT_MODEL", "en-google").strip() or "en-google"

logger = logging.getLogger("vectorplay.engine")

CACHE_RESERVE_MB = 1024        # оставляем системе поверх бюджета моделей
UNKNOWN_BUDGET_MB = 3072       # бюджет, когда объём памяти определить не удалось
MIN_BUDGET_MB = 768
IDLE_EVICTION_S = float(os.getenv("VECTOR_CALC_MODEL_IDLE_EVICTION_S", "180"))


@dataclass(frozen=True)
class ModelSpec:
    id: str
    flag: str
    label: str
    language: str
    name: str
    detail: str
    archive: str
    urls: tuple[str, ...]
    size_mb: int
    ram_mb: int = 0          # ожидаемый резидентный размер; 0 = оценка по размеру архива
    max_words: int = 0       # >0 — брать только первые N слов (частотный порядок, для прунинга)

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.archive

    @property
    def planned_ram_mb(self) -> int:
        return self.ram_mb or int(self.size_mb * 2)

    def download_in_progress(self) -> bool:
        """Файл-спутник part-url остаётся на время незавершённой закачки."""
        return self.path.with_suffix(self.path.suffix + ".part-url").exists()

    def available(self) -> bool:
        if self.download_in_progress():
            return False
        try:
            if not self.path.is_file():
                return False
            with self.path.open("rb") as stream:
                magic = stream.read(4)
        except OSError:
            return False
        if self.path.suffix.lower() == ".zip":
            return magic[:2] == b"PK"
        if self.path.suffix.lower() == ".gz":
            return magic[:2] == b"\x1f\x8b"
        return bool(magic)


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="en-small", flag="🇬🇧", label="English Small", language="en",
        name="Google News Word2Vec 300D · top-250k",
        detail="250k tokens · 300D · Google News, обрезанная до частотного топа",
        archive="GoogleNews-vectors-negative300.bin.gz",
        urls=(
            "https://github.com/RaRe-Technologies/gensim-data/releases/download/word2vec-google-news-300/word2vec-google-news-300.gz",
            "https://huggingface.co/LoganKilpatrick/GoogleNews-vectors-negative300/resolve/main/GoogleNews-vectors-negative300.bin.gz",
        ),
        size_mb=1650, ram_mb=420, max_words=250_000,
    ),
    ModelSpec(
        id="en-google", flag="🇬🇧", label="English Classic", language="en",
        name="Google News Word2Vec 300D",
        detail="3M tokens · 300D · classic word2vec",
        archive="GoogleNews-vectors-negative300.bin.gz",
        urls=(
            "https://github.com/RaRe-Technologies/gensim-data/releases/download/word2vec-google-news-300/word2vec-google-news-300.gz",
            "https://huggingface.co/LoganKilpatrick/GoogleNews-vectors-negative300/resolve/main/GoogleNews-vectors-negative300.bin.gz",
        ),
        size_mb=1650, ram_mb=3800,
    ),
    ModelSpec(
        id="en-glove", flag="🇬🇧", label="English Modern", language="en",
        name="Stanford GloVe 2024 300D",
        detail="1.2M tokens · 300D · Wikipedia + Gigaword 2024",
        archive="glove.2024.wikigiga.300d.zip",
        urls=("https://nlp.stanford.edu/data/glove.2024.wikigiga.300d.zip",),
        size_mb=1600, ram_mb=1650,
    ),
    ModelSpec(
        id="ru-rusvectores", flag="🇷🇺", label="Русский", language="ru",
        name="RusVectōrēs Word2Vec 300D",
        detail="249k tokens · 300D · Russian Wikipedia 2019",
        archive="ruwikiruscorpora_upos_skipgram_300_2_2019.zip",
        urls=("https://vectors.nlpl.eu/repository/20/182.zip",),
        size_mb=600, ram_mb=320,
    ),
)

MODELS_BY_ID = {spec.id: spec for spec in MODELS}


def enabled_specs(raw: str | None = None) -> tuple[ModelSpec, ...]:
    """Модели, доступные этому запуску: VECTOR_CALC_ENABLED_MODELS=id,id,...

    Пустое значение — включены все модели реестра: код всегда содержит полный
    набор, а выключается ненужное переменной окружения. Неизвестный id — ошибка
    на старте, чтобы опечатка не открыла лишние модели молча.
    """
    value = (raw if raw is not None else os.getenv("VECTOR_CALC_ENABLED_MODELS", "")).strip()
    if not value:
        return MODELS
    wanted = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in wanted if item not in MODELS_BY_ID]
    selected = tuple(spec for spec in MODELS if spec.id in wanted)
    if unknown or not selected:
        raise RuntimeError(
            "VECTOR_CALC_ENABLED_MODELS: неизвестные модели " + ", ".join(unknown or wanted)
            + "; доступные: " + ", ".join(MODELS_BY_ID)
        )
    return selected


ENABLED_MODELS = enabled_specs()
ENABLED_MODEL_IDS = {spec.id for spec in ENABLED_MODELS}


class ModelUnavailable(RuntimeError):
    """The model archive is missing or could not be read."""


class ModelNotReady(RuntimeError):
    """The model is being loaded right now; retry shortly."""


class UnknownModel(RuntimeError):
    """The requested model id is not in the registry."""


_TAGS = {"NOUN": "NOUN", "ADJF": "ADJ", "ADJS": "ADJ", "COMP": "ADJ",
         "VERB": "VERB", "INFN": "VERB", "PRTF": "VERB", "PRTS": "VERB",
         "GRND": "VERB", "ADVB": "ADV", "NPRO": "PRON", "NUMR": "NUM",
         "PREP": "ADP", "CONJ": "CCONJ", "PRCL": "PART", "INTJ": "INTJ"}


class WordSpace:
    """Vocabulary plus a single float32 matrix; similarity avoids a normed copy."""

    def __init__(self, words, matrix, spec: ModelSpec):
        self.index_to_key = words
        self.key_to_index = {word: i for i, word in enumerate(words)}
        self.vectors = matrix
        self.vector_size = matrix.shape[1]
        self.spec = spec
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms < 1e-12] = 1.0
        self.norms = norms.astype(np.float32, copy=False)

    def __contains__(self, word):
        return word in self.key_to_index

    def __getitem__(self, word):
        return self.vectors[self.key_to_index[word]]

    def __len__(self):
        return len(self.index_to_key)

    def similarities(self, target):
        return (self.vectors @ target) / self.norms

    def memory_mb(self) -> float:
        """Приблизительный резидентный размер: матрица, нормы и питоновская обвязка словаря."""
        return (self.vectors.nbytes + self.norms.nbytes + 120 * len(self.index_to_key)) / 1024 ** 2


class _ChunkReader:
    """Buffered reader for word2vec binary: words end with a space, rows with a newline."""

    def __init__(self, stream, chunk: int = 1 << 20):
        self._stream = stream
        self._chunk = chunk
        self._buf = b""
        self._pos = 0
        self._eof = False

    def _fill(self) -> bool:
        if self._pos < len(self._buf):
            return True
        if self._eof:
            return False
        self._buf = self._stream.read(self._chunk)
        self._pos = 0
        if not self._buf:
            self._eof = True
            return False
        return True

    def read_until(self, sep: bytes) -> bytes:
        parts = []
        while True:
            if not self._fill():
                raise EOFError("Unexpected EOF in model file")
            index = self._buf.find(sep, self._pos)
            if index != -1:
                parts.append(self._buf[self._pos:index])
                self._pos = index + len(sep)
                return b"".join(parts)
            parts.append(self._buf[self._pos:])
            self._pos = len(self._buf)

    def read_token(self) -> bytes:
        while True:
            if not self._fill():
                raise EOFError("Unexpected EOF in model file")
            if self._buf[self._pos] in (10, 13):
                self._pos += 1
                continue
            return self.read_until(b" ")

    def read_exact(self, count: int) -> bytes:
        parts = []
        need = count
        while need:
            if not self._fill():
                raise EOFError("Unexpected EOF in model file")
            take = min(need, len(self._buf) - self._pos)
            parts.append(self._buf[self._pos:self._pos + take])
            self._pos += take
            need -= take
        return parts[0] if len(parts) == 1 else b"".join(parts)


def _read_word2vec_binary(stream, spec: ModelSpec, max_words: int = 0) -> WordSpace:
    """Читает word2vec binary. max_words>0 — брать только первые N слов (частотный порядок)."""
    reader = _ChunkReader(stream)
    header = reader.read_until(b"\n").strip().split()
    if len(header) != 2:
        raise ValueError("Invalid Word2Vec binary header")
    count, dims = map(int, header)
    if not 1 <= dims <= 4096 or not 1 <= count <= 5_000_000:
        raise ValueError(f"Unexpected Word2Vec shape {count} x {dims}")
    take = count if max_words <= 0 else min(count, max_words)
    matrix = np.empty((take, dims), dtype=np.float32)
    words = []
    append = words.append
    for i in range(take):
        token = reader.read_token()
        if len(token) > 1024:
            raise ValueError("Word2Vec token too long")
        append(token.decode("utf-8", errors="replace"))
        raw = reader.read_exact(dims * 4)
        matrix[i] = np.frombuffer(raw, dtype="<f4")
    if take < count:
        logger.info("Модель %s обрезана: %d из %d слов (частотный порядок)",
                    spec.id, take, count)
    return WordSpace(words, matrix, spec)


def _parse_glove_line(text: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            return np.fromstring(text, sep=" ", dtype=np.float32)
        except ValueError:
            return None


def _read_glove_text(stream, spec: ModelSpec) -> WordSpace:
    buffer = array.array("f")
    words = []
    dims = None
    skipped = 0
    for raw_line in stream:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        separator = line.find(" ")
        if separator <= 0:
            skipped += 1
            continue
        word = line[:separator]
        values = _parse_glove_line(line[separator + 1:])
        if values is None or values.shape[0] == 0:
            skipped += 1
            continue
        if dims is None:
            dims = values.shape[0]
        elif values.shape[0] != dims:
            skipped += 1
            continue
        buffer.frombytes(values.tobytes())
        words.append(word)
    if not words or dims is None:
        raise ValueError("В файле GloVe не найдено ни одного вектора")
    if skipped:
        print(f"GloVe: строк без корректного вектора пропущено: {skipped}", flush=True)
    matrix = np.frombuffer(buffer, dtype=np.float32).reshape(len(words), dims)
    return WordSpace(words, matrix, spec)


def _load_model(spec: ModelSpec) -> WordSpace:
    path = spec.path
    if not spec.available():
        raise ModelUnavailable(f"Файл модели не найден: {path.name}")
    suffix = path.suffix.lower()
    with ExitStack() as stack:
        if suffix == ".zip":
            archive = stack.enter_context(zipfile.ZipFile(path))
            names = archive.namelist()
            binary = next((n for n in names if n.endswith("model.bin")), None)
            text = next((n for n in names if n.lower().endswith((".txt", ".vec"))), None)
            if binary:
                stream = stack.enter_context(archive.open(binary))
                return _read_word2vec_binary(stream, spec)
            if text:
                stream = stack.enter_context(archive.open(text))
                return _read_glove_text(stream, spec)
            raise ValueError(f"В архиве {path.name} нет model.bin или текстовых векторов")
        if suffix == ".gz":
            stream = stack.enter_context(gzip.open(path, "rb"))
            return _read_word2vec_binary(stream, spec, spec.max_words)
        stream = stack.enter_context(path.open("rb"))
        return _read_word2vec_binary(stream, spec, spec.max_words)


def available_memory_mb() -> int | None:
    """Оценка доступной памяти: cgroup v2 (Docker), /proc/meminfo, затем Windows."""
    try:
        limit_raw = Path("/sys/fs/cgroup/memory.max").read_text(encoding="utf-8").strip()
        if limit_raw.isdigit():
            return int(limit_raw) // 1024 ** 2
    except OSError:
        pass
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        meminfo = ""
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            try:
                return int(line.split()[1]) // 1024
            except (IndexError, ValueError):
                return None
    try:  # Windows / Docker Desktop
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys) // 1024 ** 2
    except Exception:
        pass
    return None


def resolve_cache_budget_mb() -> tuple[int, str]:
    """Бюджет кэша моделей в мегабайтах и пояснение, откуда он взялся."""
    raw = os.getenv("VECTOR_CALC_MODEL_CACHE_MB", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = -1
        if value > 0:
            return value, "VECTOR_CALC_MODEL_CACHE_MB"
        if value == 0:
            return 10 ** 6, "VECTOR_CALC_MODEL_CACHE_MB=0 (без предела)"
    detected = available_memory_mb()
    if detected is None:
        return UNKNOWN_BUDGET_MB, "оценка по умолчанию (память не определилась)"
    return max(MIN_BUDGET_MB, detected - CACHE_RESERVE_MB), f"авто ({detected} МБ доступно)"


@dataclass
class _Entry:
    model: WordSpace
    morph: object | None
    used_at: float

    def resident_mb(self) -> float:
        """Занятая память: фактическая, но не меньше плановой оценки модели."""
        return max(self.model.memory_mb(), self.model.spec.planned_ram_mb)


class ModelManager:
    """LRU-кэш моделей с бюджетом памяти.

    Держит в памяти столько моделей, сколько влезает в бюджет, поэтому два
    посетителя на разных моделях не перезагружают вектора туда-обратно. Если
    модели не помещаются вместе, вытесняется самая давно неиспользованная, но
    только если она простаивает дольше IDLE_EVICTION_S — иначе загрузка
    отклоняется с понятным сообщением, чтобы не было бесконечной перезагрузки.
    """

    def __init__(self, specs: tuple[ModelSpec, ...] = MODELS, default_id: str = DEFAULT_MODEL,
                 budget_mb: int | None = None):
        self._specs = {spec.id: spec for spec in specs}
        self.default_id = default_id if default_id in self._specs else specs[0].id
        if budget_mb:
            self._budget_mb, self.budget_source = int(budget_mb), "параметр"
        else:
            self._budget_mb, self.budget_source = resolve_cache_budget_mb()
        self._lock = threading.RLock()
        self._entries: dict[str, _Entry] = {}
        self._loading_id: str | None = None
        self._errors: dict[str, str | None] = {}
        self._notes: dict[str, str | None] = {}

    def spec(self, model_id: str) -> ModelSpec:
        try:
            return self._specs[model_id]
        except KeyError:
            raise UnknownModel(f"Неизвестная модель: {model_id}") from None

    def budget_mb(self) -> int:
        return self._budget_mb

    def set_budget_mb(self, value: int, source: str = "параметр") -> None:
        with self._lock:
            self._budget_mb = int(value)
            self.budget_source = source

    def resident_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._entries, key=lambda model_id: self._entries[model_id].used_at, reverse=True)

    def resident_mb(self) -> float:
        with self._lock:
            return sum(entry.resident_mb() for entry in self._entries.values())

    def loaded_id(self) -> str | None:
        residents = self.resident_ids()
        return residents[0] if residents else None

    def morph_for(self, model_id: str):
        with self._lock:
            entry = self._entries.get(model_id)
            return entry.morph if entry is not None else None

    def install(self, model_id: str, model: WordSpace, morph=None) -> None:
        """Register an already-built model (used by tests)."""
        self.spec(model_id)
        with self._lock:
            self._entries[model_id] = _Entry(model, morph, time.monotonic())
            self._loading_id = None
            self._errors[model_id] = None
            self._notes[model_id] = None

    def _make_room(self, needed_mb: float, keep: str):
        """Вытеснить простаивающие модели, чтобы `needed_mb` влезло.

        Возвращает (вытесненные, причина_отказа). Отказ означает, что место
        занято недавно использованной моделью и трогать её нельзя.
        """
        evicted = []
        while True:
            candidates = {model_id: entry for model_id, entry in self._entries.items() if model_id != keep}
            if not candidates:
                return evicted, None
            if self.resident_mb() + needed_mb <= self._budget_mb:
                return evicted, None
            victim = min(candidates, key=lambda model_id: candidates[model_id].used_at)
            idle_for = time.monotonic() - candidates[victim].used_at
            if idle_for < IDLE_EVICTION_S:
                return evicted, (victim, idle_for)
            del self._entries[victim]
            evicted.append(victim)

    def ensure_async(self, model_id: str) -> bool:
        """Start a background load when needed. Returns True when a load was started."""
        spec = self.spec(model_id)
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is not None:
                # «горячесть» модели обновляет только реальное использование
                # (расчёт или сохранение), а не опрос статуса страницей.
                self._notes[model_id] = None
                return False
            if self._loading_id is not None:
                return False
            if not spec.available():
                return False
            evicted, blocked = self._make_room(spec.planned_ram_mb, keep=model_id)
            if blocked is not None:
                victim, _ = blocked
                self._notes[model_id] = (
                    f"Не хватает памяти, чтобы держать «{self._specs[victim].label}» и «{spec.label}» сразу "
                    f"(бюджет кэша {self._budget_mb} МБ). Увеличьте VECTOR_CALC_MODEL_CACHE_MB "
                    f"или пользуйтесь одной моделью."
                )
                return False
            self._loading_id = model_id
            self._errors[model_id] = None
            self._notes[model_id] = None
        for model_id_victim in evicted:
            logger.info("Кэш моделей: выгружена «%s», чтобы загрузить «%s» (бюджет %d МБ)",
                        model_id_victim, model_id, self._budget_mb)
        threading.Thread(target=self._worker, args=(model_id, spec), name="model-loader", daemon=True).start()
        return True

    def _worker(self, model_id: str, spec: ModelSpec) -> None:
        try:
            model = _load_model(spec)
            morph = pymorphy3.MorphAnalyzer() if spec.language == "ru" else None
        except Exception as exc:
            with self._lock:
                self._errors[model_id] = f"{type(exc).__name__}: {exc}"
                self._loading_id = None
            logger.warning("Модель %s не загрузилась: %s", model_id, exc)
            return
        size_mb = model.memory_mb()
        with self._lock:
            self._entries[model_id] = _Entry(model, morph, time.monotonic())
            self._loading_id = None
            self._notes[model_id] = None
        logger.info("Модель %s загружена: %d слов, ~%.0f МБ; в памяти моделей: %d",
                    model_id, len(model), size_mb, len(self.resident_ids()))

    def get(self, model_id: str) -> WordSpace:
        """Return the loaded model.

        A missing model starts a background load and answers ModelNotReady, so a
        web request never blocks for minutes on a multi-GB load.
        """
        spec = self.spec(model_id)
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is not None:
                entry.used_at = time.monotonic()
                return entry.model
            if self._loading_id == model_id:
                raise ModelNotReady(f"Модель «{spec.label}» ещё загружается, попробуйте через несколько секунд")
            if not spec.available():
                raise ModelUnavailable(f"Модель «{spec.name}» не найдена на сервере. Запустите download_model.py")
            error = self._errors.get(model_id)
        if error:
            raise ModelUnavailable(f"Модель «{spec.name}» не загрузилась: {error}")
        started = self.ensure_async(model_id)
        notice = self._notes.get(model_id)
        if not started and notice:
            raise ModelNotReady(notice)
        raise ModelNotReady(f"Модель «{spec.label}» загружается, попробуйте через несколько секунд")

    def state(self, model_id: str) -> dict:
        spec = self.spec(model_id)
        with self._lock:
            entry = self._entries.get(model_id)
            return {
                "id": spec.id, "flag": spec.flag, "label": spec.label, "language": spec.language,
                "name": spec.name, "detail": spec.detail, "size_mb": spec.size_mb,
                "ram_mb": round(entry.model.memory_mb()) if entry is not None else spec.planned_ram_mb,
                "available": spec.available(), "loaded": entry is not None,
                "downloading": spec.download_in_progress(),
                "loading": self._loading_id == model_id, "error": self._errors.get(model_id),
                "note": self._notes.get(model_id),
                "vocab_size": len(entry.model) if entry is not None else 0,
                "dimensions": entry.model.vector_size if entry is not None else 300,
            }

    def states(self) -> list[dict]:
        return [self.state(spec.id) for spec in self._specs.values()]


MANAGER = ModelManager(ENABLED_MODELS)
logger.info("Кэш моделей: бюджет %d МБ (%s), резидентно сейчас %s; включены модели: %s",
            MANAGER.budget_mb(), MANAGER.budget_source, MANAGER.resident_ids() or "ничего",
            ", ".join(spec.id for spec in ENABLED_MODELS))


def tokens(text: str, model: WordSpace, morph):
    if model.spec.language == "ru":
        return _tokens_ru(text, model, morph)
    return _tokens_en(text, model)


def _tokens_ru(text: str, model: WordSpace, morph):
    result = []
    for word in re.findall(r"[\w-]+", text.lower(), re.UNICODE):
        if word in model:
            result.append(word)
            continue
        possibilities = []
        for parse in morph.parse(word)[:5]:
            tag = _TAGS.get(parse.tag.POS)
            if tag:
                possibilities.append(f"{parse.normal_form}_{tag}")
            possibilities.append(f"{parse.normal_form}_PROPN")
            possibilities.append(parse.normal_form)
        possibilities.extend((word + "_PROPN", word + "_NOUN", word + "_ADJ", word + "_VERB"))
        match = next((value for value in possibilities if value in model), None)
        if match is not None:
            result.append(match)
    return result


def _tokens_en(text: str, model: WordSpace):
    result = []
    for word in re.findall(r"[A-Za-z][A-Za-z'’\-]*", text):
        for candidate in (word, word.lower(), word.capitalize(), word.upper()):
            if candidate in model:
                result.append(candidate)
                break
    return result


def embed(text: str, model: WordSpace, morph):
    matched = tokens(text, model, morph)
    if not matched:
        raise ValueError(f"Ни одно слово из {text!r} не найдено в модели")
    vector = np.mean([model[word] for word in matched], axis=0)
    return np.asarray(vector, dtype=np.float32), matched


def calculate(terms, top_k: int = 10, exclude_terms: bool = True, model_id: str | None = None):
    model_id = model_id or MANAGER.default_id
    model = MANAGER.get(model_id)
    morph = MANAGER.morph_for(model_id)
    if not isinstance(terms, list) or not 1 <= len(terms) <= 12:
        raise ValueError("Необходимо от 1 до 12 переменных")
    target = np.zeros(model.vector_size, dtype=np.float32)
    resolved = []
    excluded = set()
    for item in terms:
        if not isinstance(item, dict):
            raise ValueError("Некорректная переменная")
        phrase = str(item.get("text", "")).strip()
        weight = item.get("weight", 1)
        if not phrase or len(phrase) > 600 or isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or abs(weight) > 100:
            raise ValueError("Некорректный текст или коэффициент")
        vector, keys = embed(phrase, model, morph)
        target += float(weight) * vector
        resolved.append({"text": phrase, "weight": weight, "matched": keys})
        if exclude_terms:
            excluded.update(keys)
    norm = float(np.linalg.norm(target))
    if norm < 1e-12:
        raise ValueError("Нулевой результирующий вектор")
    target /= norm
    scores = model.similarities(target)
    k = max(1, min(int(top_k), 50))
    sample = min(len(scores), max(k * 40, k + len(excluded) + 20))
    nearest = np.argpartition(scores, -sample)[-sample:]
    nearest = nearest[np.argsort(scores[nearest])[::-1]]
    results = []
    seen_words = set()
    for position in nearest:
        token = model.index_to_key[int(position)]
        surface = token.rsplit("_", 1)[0].casefold()
        if token in excluded or surface in seen_words:
            continue
        seen_words.add(surface)
        display = token.replace("_", " · ", 1) if model.spec.language == "ru" else token.replace("_", " ")
        results.append({"text": display, "token": token,
                        "similarity": round(float(scores[position]), 6), "source": "word2vec"})
        if len(results) >= k:
            break
    return {"results": results, "candidate_count": len(model) - len(excluded),
            "dimensions": model.vector_size, "mode": "word2vec",
            "model": model.spec.id, "model_name": model.spec.name, "language": model.spec.language,
            "backend": "Word2Vec / NumPy", "resolved": resolved,
            "vector_preview": [round(float(value), 6) for value in target[:20]]}
