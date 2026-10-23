"""Словарный слой модерации и профили аудитории (без сети и LLM).

Слова для проверок собраны из кодпоинтов, чтобы мат не лежал в репозитории открытым текстом.
"""
import unittest
from unittest.mock import patch

import moderation

MAT = "\u0445\u0443\u0439"                    # 3 буквы, точное слово
MAT_FORM = "\u0445\u0443\u0451\u0432\u044b\u0439"   # производное, ловится стемом
MAT_VERB = "\u043d\u0430\u0445\u0443\u0439"
MAT_NOUN = "\u0431\u043b\u044f\u0434\u044c"
SLUR = "\u043f\u0438\u0434\u043e\u0440"
SENSITIVE = "\u0441\u0435\u043a\u0441"
SENSITIVE_2 = "\u043e\u0440\u0443\u0436\u0438\u0435"
INNOCENT_ROOSTER = "\u043f\u0435\u0442\u0443\u0445"        # птица, не оскорбление
INNOCENT_GOAT = "\u043a\u043e\u0437\u0451\u043b"
INNOCENT_CLOTH = "\u0441\u0443\u043a\u043d\u043e"          # не мат
INNOCENT_WISE = "\u043c\u0443\u0434\u0440\u0430\u044f"     # стем "муд" не должен ловить


class DictionaryTests(unittest.TestCase):
    def level(self, text, profile="kids"):
        with patch.object(moderation, "PROFILE", profile):
            return moderation.level_of_text(text)

    def test_obscene_words_detected(self):
        for word in (MAT, MAT_FORM, MAT_VERB, MAT_NOUN, SLUR):
            self.assertEqual(moderation.check_text(word).get(word), "obscene", word)

    def test_sensitive_words_detected(self):
        for word in (SENSITIVE, SENSITIVE_2):
            self.assertEqual(moderation.check_text(word).get(word), "sensitive", word)

    def test_obscene_inside_sentence(self):
        self.assertIn("obscene", moderation.check_text(f"ты {MAT_VERB} пошёл").values())

    def test_allowlist_keeps_innocent_words(self):
        for word in (INNOCENT_ROOSTER, INNOCENT_GOAT, INNOCENT_CLOTH, INNOCENT_WISE):
            self.assertEqual(moderation.check_text(word), {}, word)

    def test_clean_equations_are_ok(self):
        self.assertEqual(self.level("король − мужчина + женщина"), "ok")
        self.assertEqual(self.level("россия − москва + франция"), "ok")


class ProfileTests(unittest.TestCase):
    def test_kids_profile_is_strict(self):
        with patch.object(moderation, "PROFILE", "kids"):
            self.assertEqual(moderation.level_of_text(MAT), "block")
            self.assertEqual(moderation.level_of_text(SENSITIVE), "review")

    def test_teen_profile_is_softer(self):
        with patch.object(moderation, "PROFILE", "teen"):
            self.assertEqual(moderation.level_of_text(MAT), "review")      # не блокируем
            self.assertEqual(moderation.level_of_text(SENSITIVE), "ok")    # слово само по себе не мешает
            self.assertEqual(moderation.level_of_text(INNOCENT_GOAT), "ok")

    def test_teen_has_higher_neighbor_threshold(self):
        self.assertGreater(moderation.PROFILES["teen"]["min_hits"], moderation.PROFILES["kids"]["min_hits"])

    def test_default_profile_is_teen(self):
        self.assertIn(moderation.PROFILE, moderation.PROFILES)

    def test_prompt_mentions_age_and_json(self):
        for profile in ("kids", "teen"):
            with patch.object(moderation, "PROFILE", profile):
                prompt = moderation.system_prompt("ru-rusvectores")
                self.assertIn(moderation.PROFILES[profile]["age"], prompt)
                self.assertIn('"verdict"', prompt)
                self.assertIn("по-русски", prompt)

    def test_english_model_gets_english_prompt(self):
        for profile in ("kids", "teen"):
            with patch.object(moderation, "PROFILE", profile):
                prompt = moderation.system_prompt("en-small")
                self.assertIn(moderation.PROFILES[profile]["age_en"], prompt)
                self.assertIn("in English", prompt)
                self.assertNotIn(moderation.PROFILES[profile]["age"], prompt)


class ReasonLanguageTests(unittest.TestCase):
    """Обоснования оценки пишутся на языке модели: en → английский, ru → русский."""

    def test_reason_text_follows_model_language(self):
        self.assertEqual(moderation.reason_text("group_manual", "en-small"),
                         "The access group was changed manually")
        self.assertEqual(moderation.reason_text("group_manual", "ru-rusvectores"),
                         "Группу допуска изменили вручную")
        self.assertIn("3", moderation.reason_text("neighbors", "en-small", n=3))
        self.assertEqual(moderation.reason_text("unknown-key", "en-small"), "unknown-key")

    def test_review_reasons_are_english_for_english_model(self):
        terms = [{"text": "king", "weight": 1}, {"text": "man", "weight": -1}, {"text": "woman", "weight": 1}]
        results = [{"token": "queen", "text": "queen"}]
        report = moderation.review(terms, results, model_id="en-small", use_llm=False)
        self.assertEqual(report["verdict"], "ok")
        self.assertTrue(any("No issues found" in reason for reason in report["reasons"]), report["reasons"])
        report = moderation.review(terms, results, model_id="ru-rusvectores", use_llm=False)
        self.assertTrue(any("Замечаний не найдено" in reason for reason in report["reasons"]), report["reasons"])


class ModeTests(unittest.TestCase):
    def test_status_for_modes(self):
        with patch.object(moderation, "MODE", "off"):
            self.assertEqual(moderation.status_for("block"), "approved")
        with patch.object(moderation, "MODE", "warn"):
            self.assertEqual(moderation.status_for("block"), "approved")
        with patch.object(moderation, "MODE", "queue"):
            self.assertEqual(moderation.status_for("ok"), "approved")
            self.assertEqual(moderation.status_for("review"), "pending")
            self.assertEqual(moderation.status_for("block"), "pending")
        with patch.object(moderation, "MODE", "strict"):
            self.assertEqual(moderation.status_for("ok"), "pending")


class LlmParsingTests(unittest.TestCase):
    """Разбор ответа LLM без обращения к сети."""

    def test_plain_json(self):
        parsed = moderation._first_json('{"verdict":"block","categories":["obscene"],"reason":"мат"}')
        self.assertEqual(parsed["verdict"], "block")
        self.assertEqual(parsed["categories"], ["obscene"])

    def test_json_inside_prose_and_code_fence(self):
        text = 'Хм, подумаю.\n```json\n{"verdict":"review","reason":"спорно"}\n```\nВот так.'
        self.assertEqual(moderation._first_json(text)["verdict"], "review")

    def test_garbage_and_unknown_verdict(self):
        self.assertIsNone(moderation._first_json("без json"))
        self.assertIsNone(moderation._first_json('{"verdict":"maybe"}'))

    def test_no_think_is_default(self):
        self.assertTrue(moderation.LLM_NO_THINK)


if __name__ == "__main__":
    unittest.main(verbosity=2)
