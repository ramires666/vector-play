"""Оценка приличности комбинации слов: словари + векторные эвристики + LLM.

Слои (defense in depth):
  1. Словари — хеши слов и стемов (мат в репозитории не лежит открытым текстом),
     лемматизация через pymorphy3 для русских слов. Мгновенно и объяснимо.
  2. Векторные эвристики — ответ и его соседи по уже посчитанному расчёту
     проверяются теми же словарями: если «плохие» слова лезут в топ, это review.
     Ничего дополнительно считать не нужно, вектора уже в памяти.
  3. LLM через OpenAI-совместимый API (/chat/completions) — смысловая оценка:
     сексизм, расизм, спорные темы. Необязательный слой, включается env-переменной.

Итог: {"verdict": "ok|review|block", "categories": [...], "reasons": [...],
       "scores": {...}, "source": "..."}.

CLI:
    python moderation.py --hash "слово"     # sha256 для списка модерации
    python moderation.py --check "текст"    # проверка словарями
    python moderation.py --llm "уравнение"  # проверка через LLM
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
import urllib.request
from pathlib import Path

import word2vec_engine as engine

log = logging.getLogger("vectorplay.moderation")

BASE_DIR = Path(__file__).resolve().parent
LISTS_DIR = BASE_DIR / "moderation"

MODE = os.getenv("VECTOR_CALC_MODERATION", "warn").strip().lower()
LLM_URL = os.getenv("VECTOR_CALC_LLM_URL", "").strip().rstrip("/")
LLM_MODEL = os.getenv("VECTOR_CALC_LLM_MODEL", "").strip() or "local"
LLM_KEY = os.getenv("VECTOR_CALC_LLM_API_KEY", "").strip()
LLM_TIMEOUT = float(os.getenv("VECTOR_CALC_LLM_TIMEOUT", "120"))
LLM_MAX_TOKENS = int(os.getenv("VECTOR_CALC_LLM_MAX_TOKENS", "800"))
LLM_NO_THINK = os.getenv("VECTOR_CALC_LLM_NO_THINK", "0").strip().lower() in {"1", "true", "yes", "on"}
NEIGHBORS = int(os.getenv("VECTOR_CALC_MODERATION_NEIGHBORS", "10"))
MIN_NEIGHBOR_HITS = int(os.getenv("VECTOR_CALC_MODERATION_MIN_HITS", "2"))

VERDICTS = ("ok", "review", "block")
_LEVEL = {"ok": 0, "review": 1, "block": 2}
STEM_MIN = 4

SYSTEM_PROMPT = (
    "Ты — строгий, но доброжелательный модератор коллекции словесных примеров, "
    "которую смотрят дети от 7 лет. Тебе дают уравнение из сложения и вычитания слов "
    "и ответ модели эмбеддингов. Оцени, можно ли показать этот пример детям.\n"
    "Недопустимо: мат и грубые оскорбления, секс и порнография, насилие и жестокость, "
    "расизм, сексизм, гомофобия, травля, наркотики, политические выпады и агитация.\n"
    "Допустимо: нейтральные и смешные примеры про животных, еду, города, профессии, "
    "сказки, технику, науку.\n"
    "Важно: оценивай КОМБИНАЦИЮ, а не отдельные слова. Например, результат может быть "
    "обидным для группы людей, даже если каждое слово по отдельности нейтрально.\n"
    'Примеры: "король − мужчина + женщина = королева" → ok; '
    '"россия − москва + франция = париж" → ok; '
    '"петух − курица + гусь = петух" → ok (птицы); '
    '"женщина − человек + мужчина = козёл" → block (оскорбление женщин); '
    '"армянин − человек + грузин = ..." → block (национальная неприязнь).\n'
    "Сомневаешься — выбирай review, а не ok.\n"
    'Ответь СТРОГО одним JSON без пояснений: {"verdict":"ok|review|block",'
    '"categories":["..."],"reason":"кратко по-русски"}'
)

_CACHE: dict[str, frozenset[str]] = {}


def _hash(word: str) -> str:
    return hashlib.sha256(word.strip().lower().encode("utf-8")).hexdigest()


def _load(name: str) -> frozenset[str]:
    cached = _CACHE.get(name)
    if cached is not None:
        return cached
    try:
        lines = (LISTS_DIR / name).read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    values = set()
    for line in lines:
        line = line.split("#", 1)[0].strip().lower()
        if not line:
            continue
        if line.startswith("sha256:"):
            values.add(line[7:])
        elif re.fullmatch(r"[0-9a-f]{64}", line):
            values.add(line)                      # в файле уже готовый хеш
        else:
            values.add(_hash(line))               # в файле обычное слово
    result = frozenset(values)
    _CACHE[name] = result
    return result


def _candidate_hashes(text: str) -> set[str]:
    """Хеши самого слова и его стемов (префиксов от STEM_MIN символов)."""
    text = text.strip().lower()
    hashes = {_hash(text)} if text else set()
    for index in range(STEM_MIN, len(text)):
        hashes.add(_hash(text[:index]))
    return hashes


def _allowed(word: str, lang: str) -> bool:
    return _hash(word) in _load(f"allow_{lang}.txt")


def word_level(word: str, lang: str) -> str | None:
    """Уровень одного слова: block (мат/оскорбление), review (спорное) или None."""
    word = word.strip().lower()
    if not word or len(word) < 2 or _allowed(word, lang):
        return None
    if not lang in ("ru", "en"):
        lang = "ru"
    for name, level in ((f"obscene_{lang}.txt", "block"), (f"sensitive_{lang}.txt", "review")):
        table = _load(name)
        if table and (_candidate_hashes(word) & table):
            return level
    return None


def word_levels(word: str, lang: str, morph=None) -> list[str]:
    """Уровни слова с учётом лемм (pymorphy3) — ловит падежи и формы."""
    candidates = {word.strip().lower()}
    if lang == "ru" and morph is not None:
        try:
            for parse in morph.parse(word)[:5]:
                candidates.add(parse.normal_form.lower())
        except Exception:
            pass
    levels = []
    for candidate in candidates:
        level = word_level(candidate, lang)
        if level is not None:
            levels.append(level)
    return levels


def check_text(text: str, lang: str = "ru", morph=None) -> dict[str, str]:
    """Словарная проверка текста: {слово: 'block'|'review'}."""
    pattern = r"[\w'’\-]+" if lang == "ru" else r"[A-Za-z][A-Za-z'’\-]*"
    hits: dict[str, str] = {}
    for word in re.findall(pattern, str(text).lower(), re.UNICODE):
        for level in word_levels(word, lang, morph):
            if level == "block" or word not in hits:
                hits[word] = "block" if level == "block" else hits.get(word, "review")
    return hits


def format_equation(terms, answer: str | None) -> str:
    parts = []
    for index, term in enumerate(terms):
        text = str(term.get("text", "")).strip()
        if not text:
            continue
        sign = "− " if float(term.get("weight", 1)) < 0 else ("+ " if index else "")
        parts.append(f"{sign}{text}".strip())
    equation = " ".join(parts) if parts else "—"
    return f"{equation} = {answer}" if answer else equation


def llm_review(equation: str) -> dict | None:
    """Смысловая проверка через OpenAI-совместимый API. None — если недоступно."""
    if not LLM_URL:
        return None
    payload = {
        "model": LLM_MODEL,
        "temperature": 0,
        "max_tokens": LLM_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Уравнение: {equation}"},
        ],
    }
    if LLM_NO_THINK:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if LLM_KEY:
        headers["Authorization"] = f"Bearer {LLM_KEY}"
    request = urllib.request.Request(LLM_URL + "/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"), headers=headers)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=LLM_TIMEOUT) as response:
            body = json.loads(response.read())
    except Exception as exc:
        log.warning("LLM-проверка не удалась (%s): %s", type(exc).__name__, exc)
        return None
    elapsed = time.monotonic() - started
    message = (body.get("choices") or [{}])[0].get("message", {}) or {}
    # Thinking-модели кладут ответ в content, а размышления — в reasoning_content:
    # если content пуст (модель не успела закончить мысль), пробуем reasoning.
    for text in (message.get("content"), message.get("reasoning_content")):
        parsed = _first_json(text or "")
        if parsed is not None:
            parsed["elapsed"] = round(elapsed, 2)
            parsed["usage"] = (body.get("usage") or {}).get("completion_tokens")
            return parsed
    log.warning("LLM вернула ответ без JSON за %.1f c", elapsed)
    return None


def _first_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        return None
    categories = data.get("categories")
    return {
        "verdict": verdict,
        "categories": [str(item)[:64] for item in categories][:8] if isinstance(categories, list) else [],
        "reason": str(data.get("reason", ""))[:300],
    }


def review(terms, results, model_id: str | None = None, use_llm: bool = True) -> dict:
    """Полная проверка уравнения: словари + векторные эвристики + LLM."""
    started = time.monotonic()
    lang, morph = "ru", None
    try:
        spec = engine.MANAGER.spec(model_id or engine.MANAGER.default_id)
        lang = spec.language
        morph = engine.MANAGER.morph_for(spec.id)
    except Exception:
        pass

    categories: list[str] = []
    reasons: list[str] = []
    term_hits: dict[str, str] = {}
    for term in terms:
        term_hits.update(check_text(term.get("text", ""), lang, morph))
    result_hits: dict[str, str] = {}
    for item in results[:max(1, NEIGHBORS)]:
        result_hits.update(check_text(item.get("token", item.get("text", "")), lang, morph))
    answer = str(results[0].get("token", results[0].get("text", ""))) if results else None
    answer_hits = check_text(answer, lang, morph) if answer else {}

    verdict = "ok"
    if any(level == "block" for level in list(term_hits.values()) + list(answer_hits.values())):
        verdict = "block"
        categories.append("obscene")
        reasons.append("Мат или грубое оскорбление в словах/ответе (словарь)")
    else:
        suspicious = [word for word, level in result_hits.items() if level in ("block", "review")]
        if any(level == "review" for level in answer_hits.values()):
            verdict = "review"
            categories.append("sensitive-answer")
            reasons.append("Ответ задевает спорную тему (словарь)")
        elif len(suspicious) >= MIN_NEIGHBOR_HITS:
            verdict = "review"
            categories.append("sensitive-neighbors")
            reasons.append(f"Среди ближайших слов {len(suspicious)} спорных (вектора)")

    llm = None
    if use_llm and LLM_URL and MODE != "off":
        llm = llm_review(format_equation(terms, answer))
        if llm is not None and _LEVEL[llm["verdict"]] > _LEVEL[verdict]:
            verdict = llm["verdict"]
            categories.extend(item for item in llm["categories"] if item not in categories)
            if llm["reason"]:
                reasons.append("LLM: " + llm["reason"])

    return {
        "verdict": verdict,
        "categories": categories,
        "reasons": reasons,
        "terms": term_hits,
        "answer": answer_hits,
        "neighbors": {word: level for word, level in result_hits.items() if level != "ok"},
        "llm": llm,
        "mode": MODE,
        "elapsed": round(time.monotonic() - started, 2),
    }


def status_for(verdict: str) -> str:
    """Каким становится сохранение при текущем режиме модерации."""
    if MODE == "off":
        return "approved"
    if MODE == "strict":
        return "pending"
    if MODE == "queue" and verdict != "ok":
        return "pending"
    return "approved"          # warn и queue с verdict=ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Модерация VectorPlay")
    parser.add_argument("--hash", metavar="СЛОВО", help="sha256 для строки списка модерации")
    parser.add_argument("--check", metavar="ТЕКСТ", help="проверить текст словарями")
    parser.add_argument("--llm", metavar="УРАВНЕНИЕ", help="проверить уравнение через LLM")
    parser.add_argument("--mode", action="store_true", help="показать режим и настройки LLM")
    args = parser.parse_args(argv)

    if args.mode:
        print(json.dumps({"mode": MODE, "llm_url": LLM_URL or None, "llm_model": LLM_MODEL,
                          "no_think": LLM_NO_THINK, "neighbors": NEIGHBORS,
                          "min_neighbor_hits": MIN_NEIGHBOR_HITS}, ensure_ascii=False, indent=2))
    if args.hash:
        print(_hash(args.hash))
    if args.check:
        print(json.dumps(check_text(args.check), ensure_ascii=False))
    if args.llm:
        print(json.dumps(llm_review(args.llm), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
