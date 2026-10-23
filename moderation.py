"""Оценка приличности комбинации слов: словари + векторные эвристики + LLM.

Слои (defense in depth):
  1. Словари — хеши слов и стемов (мат в репозитории не лежит открытым текстом),
     лемматизация через pymorphy3 для русских слов. Мгновенно и объяснимо.
  2. Векторные эвристики — ответ и его соседи по уже посчитанному расчёту
     проверяются теми же словарями: если «плохие» слова лезут в топ, это review.
     Ничего дополнительно считать не нужно, вектора уже в памяти.
  3. LLM через OpenAI-совместимый API (/chat/completions) — смысловая оценка:
     сексизм, расизм, оскорбительные комбинации. Необязательный слой.

Профиль аудитории (VECTOR_CALC_MODERATION_PROFILE) задаёт, что считать недопустимым:
  kids — строгий (7+): мат блокируется, «взрослые» слова уходят в очередь;
  teen — мягче (до 18): мат уходит на проверку (не блокируется), а слова вроде
         «оружие»/«секс» сами по себе пример не тормозят, порог срабатывания выше.

Итог: {"verdict": "ok|review|block", "categories": [...], "reasons": [...],
       "scores": {...}, "profile": "...", "source": "..."}.

CLI:
    python moderation.py --hash "слово"     # sha256 для списка модерации
    python moderation.py --check "текст"    # проверка словарями
    python moderation.py --llm "уравнение"  # проверка через LLM
    python moderation.py --mode             # профиль, режим и настройки LLM
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
LLM_TIMEOUT = float(os.getenv("VECTOR_CALC_LLM_TIMEOUT", "60"))
LLM_MAX_TOKENS = int(os.getenv("VECTOR_CALC_LLM_MAX_TOKENS", "800"))
# По умолчанию просим модель отвечать без «размышлений»: с ними проверка занимает
# 5-10 секунд вместо ~1 секунды, а вердикт нужен быстрый и короткий.
LLM_NO_THINK = os.getenv("VECTOR_CALC_LLM_NO_THINK", "1").strip().lower() not in {"0", "false", "no", "off"}
NEIGHBORS = int(os.getenv("VECTOR_CALC_MODERATION_NEIGHBORS", "10"))

VERDICTS = ("ok", "review", "block")
_LEVEL = {"ok": 0, "review": 1, "block": 2}
STEM_MIN = 4

# Профиль аудитории. kids — строгий (7+), teen — мягче (до 18).
PROFILES: dict[str, dict] = {
    "kids": {
        "age": "дети от 7 лет",
        "obscene": "block",
        "sensitive": "review",
        "min_hits": 2,
        "forbid": ("мат и грубые оскорбления, секс и порнография, насилие и жестокость, "
                   "расизм, сексизм, гомофобия, травля, наркотики, политические выпады и агитация"),
        "allow": ("нейтральные и смешные примеры про животных, еду, города, профессии, "
                  "сказки, технику, науку"),
        "examples": ('"король − мужчина + женщина = королева" → ok; '
                     '"петух − курица + гусь = петух" → ok (птицы); '
                     '"женщина − человек + мужчина = козёл" → block (оскорбление женщин)'),
        "age_en": "children from 7 years old",
        "forbid_en": ("profanity and coarse insults, sex and pornography, violence and cruelty, "
                      "racism, sexism, homophobia, bullying, drugs, political attacks and campaigning"),
        "allow_en": ("neutral and funny examples about animals, food, cities, professions, "
                     "fairy tales, technology and science"),
        "examples_en": ('"king − man + woman = queen" → ok; '
                        '"rooster − hen + goose = rooster" → ok (birds); '
                        '"woman − human + man = goat" → block (an insult to women)'),
    },
    "teen": {
        "age": "подростки до 18 лет",
        "obscene": "review",
        "sensitive": None,
        "min_hits": 4,
        "forbid": ("травля и унижение конкретных людей или групп, расизм, сексизм, гомофобия, "
                   "порнография, сцены насилия, пропаганда наркотиков, политическая агитация"),
        "allow": ("грубоватый юмор, сленг, интернет-мемы, взрослые темы без пропаганды, "
                  "а также животных, еду, города, профессии, технику, науку"),
        "examples": ('"король − мужчина + женщина = королева" → ok; '
                     '"петух − курица + гусь = петух" → ok (птицы); '
                     '"женщина − человек + мужчина = козёл" → review (грубо, но не блокировать); '
                     '"армянин − человек + грузин = ..." → block (национальная неприязнь)'),
        "age_en": "teenagers under 18",
        "forbid_en": ("bullying and humiliation of particular people or groups, racism, sexism, "
                      "homophobia, pornography, scenes of violence, drug propaganda, political campaigning"),
        "allow_en": ("rough humour, slang, internet memes, adult topics without propaganda, "
                     "as well as animals, food, cities, professions, technology and science"),
        "examples_en": ('"king − man + woman = queen" → ok; '
                        '"rooster − hen + goose = rooster" → ok (birds); '
                        '"woman − human + man = goat" → review (rude, but not a block); '
                        '"armenian − human + georgian = ..." → block (ethnic hostility)'),
    },
}

PROFILE = os.getenv("VECTOR_CALC_MODERATION_PROFILE", "teen").strip().lower()
if PROFILE not in PROFILES:
    PROFILE = "teen"
MIN_NEIGHBOR_HITS = int(os.getenv("VECTOR_CALC_MODERATION_MIN_HITS", str(PROFILES[PROFILE]["min_hits"])))

_CACHE: dict[str, frozenset[str]] = {}


def level_for(category: str | None) -> str | None:
    """Уровень категории словаря при текущем профиле (None — не считать замечанием)."""
    if category is None:
        return None
    return PROFILES[PROFILE].get(category)


def language_for(model_id: str | None = None) -> str:
    """Язык модели: от него зависят язык промпта LLM и язык обоснований оценки."""
    try:
        return engine.MANAGER.spec(model_id or engine.MANAGER.default_id).language
    except Exception:
        return "ru"


# Обоснования оценки пишутся на языке модели (русская — по-русски, английская —
# по-английски), чтобы в интерфейсе не смешивались языки.
REASONS: dict[str, dict[str, str]] = {
    "ru": {
        "block": "Мат или грубое оскорбление в словах/ответе (словарь)",
        "sensitive_answer": "Ответ задевает спорную тему (словарь)",
        "neighbors": "Среди ближайших слов {n} спорных (вектора)",
        "clean": "Замечаний не найдено: словари и ближайшие слова чистые",
        "group_manual": "Группу допуска изменили вручную",
    },
    "en": {
        "block": "Profanity or a coarse insult in the terms/answer (dictionary)",
        "sensitive_answer": "The answer touches a sensitive topic (dictionary)",
        "neighbors": "{n} suspicious words among the nearest neighbours (vectors)",
        "clean": "No issues found: dictionaries and nearest neighbours are clean",
        "group_manual": "The access group was changed manually",
    },
}


def reason_text(key: str, model_id: str | None = None, **params) -> str:
    """Обоснование на языке модели; неизвестный ключ возвращается как есть."""
    table = REASONS.get(language_for(model_id), REASONS["ru"])
    text = table.get(key) or REASONS["ru"].get(key, key)
    for name, value in params.items():
        text = text.replace("{" + name + "}", str(value))
    return text


def system_prompt(model_id: str | None = None) -> str:
    """Промпт для LLM под текущий профиль аудитории и язык модели."""
    profile = PROFILES[PROFILE]
    if language_for(model_id) == "en":
        return (
            f"You are a strict but friendly moderator of a collection of word equations "
            f"viewed by {profile['age_en']}. You get an equation built by adding and subtracting "
            "words, plus the embedding model's answer. Decide whether this example may be shown.\n"
            f"Forbidden: {profile['forbid_en']}.\n"
            f"Allowed: {profile['allow_en']}.\n"
            "Important: judge the COMBINATION, not individual words. The result may be offensive "
            "to a group of people even when every single word is neutral.\n"
            f"Examples: {profile['examples_en']}.\n"
            "When in doubt, choose review instead of ok.\n"
            'Answer with STRICTLY one JSON object and nothing else: {"verdict":"ok|review|block",'
            '"categories":["..."],"reason":"brief, in English"}'
        )
    return (
        f"Ты — строгий, но доброжелательный модератор коллекции словесных примеров, "
        f"которую смотрят {profile['age']}. Тебе дают уравнение из сложения и вычитания слов "
        "и ответ модели эмбеддингов. Оцени, можно ли показать этот пример.\n"
        f"Недопустимо: {profile['forbid']}.\n"
        f"Допустимо: {profile['allow']}.\n"
        "Важно: оценивай КОМБИНАЦИЮ, а не отдельные слова. Например, результат может быть "
        "обидным для группы людей, даже если каждое слово по отдельности нейтрально.\n"
        f"Примеры: {profile['examples']}.\n"
        "Сомневаешься — выбирай review, а не ok.\n"
        'Ответь СТРОГО одним JSON без пояснений: {"verdict":"ok|review|block",'
        '"categories":["..."],"reason":"кратко по-русски"}'
    )


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


def word_category(word: str, lang: str) -> str | None:
    """Категория одного слова: obscene, sensitive или None."""
    word = word.strip().lower()
    if not word or len(word) < 2 or _allowed(word, lang):
        return None
    if lang not in ("ru", "en"):
        lang = "ru"
    for name, category in ((f"obscene_{lang}.txt", "obscene"), (f"sensitive_{lang}.txt", "sensitive")):
        table = _load(name)
        if table and (_candidate_hashes(word) & table):
            return category
    return None


def word_categories(word: str, lang: str, morph=None) -> list[str]:
    """Категории слова с учётом лемм (pymorphy3) — ловит падежи и формы."""
    candidates = {word.strip().lower()}
    if lang == "ru" and morph is not None:
        try:
            for parse in morph.parse(word)[:5]:
                candidates.add(parse.normal_form.lower())
        except Exception:
            pass
    found = []
    for candidate in candidates:
        category = word_category(candidate, lang)
        if category is not None and category not in found:
            found.append(category)
    return found


def check_text(text: str, lang: str = "ru", morph=None) -> dict[str, str]:
    """Словарная проверка текста: {слово: 'obscene'|'sensitive'} (без учёта профиля)."""
    pattern = r"[\w'’\-]+" if lang == "ru" else r"[A-Za-z][A-Za-z'’\-]*"
    hits: dict[str, str] = {}
    for word in re.findall(pattern, str(text).lower(), re.UNICODE):
        for category in word_categories(word, lang, morph):
            if category == "obscene" or word not in hits:
                hits[word] = "obscene" if category == "obscene" else hits.get(word, "sensitive")
    return hits


def level_of_text(text: str, lang: str = "ru", morph=None) -> str:
    """Уровень текста при текущем профиле: ok / review / block."""
    levels = [level_for(category) for category in check_text(text, lang, morph).values()]
    levels = [level for level in levels if level is not None]
    if not levels:
        return "ok"
    return max(levels, key=lambda level: _LEVEL[level])


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


def llm_review(equation: str, model_id: str | None = None) -> dict | None:
    """Смысловая проверка через OpenAI-совместимый API. None — если недоступно.

    Язык промпта (и, значит, обоснования) — язык модели: английская модель
    получает английский промпт, русская — русский.
    """
    if not LLM_URL:
        return None
    english = language_for(model_id) == "en"
    payload = {
        "model": LLM_MODEL,
        "temperature": 0,
        "max_tokens": LLM_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system_prompt(model_id)},
            {"role": "user", "content": ("Equation: " if english else "Уравнение: ") + equation},
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
    terms_level = max([level_for(c) for c in term_hits.values() if level_for(c)] or ["ok"], key=lambda l: _LEVEL[l])
    answer_level = max([level_for(c) for c in answer_hits.values() if level_for(c)] or ["ok"], key=lambda l: _LEVEL[l])
    if terms_level == "block" or answer_level == "block":
        verdict = "block"
        categories.append("obscene")
        reasons.append(reason_text("block", model_id))
    else:
        suspicious = [word for word, category in result_hits.items() if level_for(category)]
        if answer_level == "review":
            verdict = "review"
            categories.append("sensitive-answer")
            reasons.append(reason_text("sensitive_answer", model_id))
        elif len(suspicious) >= MIN_NEIGHBOR_HITS:
            verdict = "review"
            categories.append("sensitive-neighbors")
            reasons.append(reason_text("neighbors", model_id, n=len(suspicious)))

    llm = None
    if use_llm and LLM_URL and MODE != "off":
        llm = llm_review(format_equation(terms, answer), model_id)
        if llm is not None:
            if _LEVEL[llm["verdict"]] > _LEVEL[verdict]:
                verdict = llm["verdict"]
                categories.extend(item for item in llm["categories"] if item not in categories)
            if llm["reason"]:
                reasons.append("LLM: " + llm["reason"])

    if verdict == "ok" and not reasons:
        reasons.append(reason_text("clean", model_id))

    return {
        "verdict": verdict,
        "profile": PROFILE,
        "categories": categories,
        "reasons": reasons,
        "terms": term_hits,
        "answer": answer_hits,
        "neighbors": {word: category for word, category in result_hits.items()},
        "llm": llm,
        "mode": MODE,
        "elapsed": round(time.monotonic() - started, 2),
    }


GROUPS = ("kids", "teen", "adult")


def group_for(verdict: str) -> str:
    """Группа допуска по вердикту: ok -> kids, review -> teen, block -> adult."""
    return {"ok": "kids", "review": "teen", "block": "adult"}.get(verdict, "teen")


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
    parser.add_argument("--model", metavar="ID", help="модель для --llm/--check: язык промпта и обоснований")
    parser.add_argument("--mode", action="store_true", help="показать профиль, режим и настройки LLM")
    args = parser.parse_args(argv)

    if args.mode:
        print(json.dumps({"profile": PROFILE, "mode": MODE, "llm_url": LLM_URL or None,
                          "llm_model": LLM_MODEL, "no_think": LLM_NO_THINK, "neighbors": NEIGHBORS,
                          "min_neighbor_hits": MIN_NEIGHBOR_HITS,
                          "profiles": {name: {"obscene": p["obscene"], "sensitive": p["sensitive"],
                                              "min_hits": p["min_hits"]} for name, p in PROFILES.items()}},
                         ensure_ascii=False, indent=2))
    if args.hash:
        print(_hash(args.hash))
    if args.check:
        lang = language_for(args.model)
        hits = check_text(args.check, lang)
        print(json.dumps({"hits": hits, "level": level_of_text(args.check, lang),
                          "profile": PROFILE, "language": lang}, ensure_ascii=False))
    if args.llm:
        print(json.dumps(llm_review(args.llm, args.model), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
