"""Словарный слой модерации (без сети и LLM).

Слова для проверок собраны из кодпоинтов, чтобы мат не лежал в репозитории открытым текстом.
"""
import unittest

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
    def level(self, text):
        hits = moderation.check_text(text)
        if "block" in hits.values():
            return "block"
        return "review" if hits else "ok"

    def test_obscene_words_blocked(self):
        for word in (MAT, MAT_FORM, MAT_VERB, MAT_NOUN, SLUR):
            self.assertEqual(self.level(word), "block", word)

    def test_obscene_inside_sentence(self):
        self.assertEqual(self.level(f"ты {MAT_VERB} пошёл"), "block")

    def test_sensitive_words_go_to_review(self):
        for word in (SENSITIVE, SENSITIVE_2):
            self.assertEqual(self.level(word), "review", word)

    def test_allowlist_keeps_innocent_words(self):
        for word in (INNOCENT_ROOSTER, INNOCENT_GOAT, INNOCENT_CLOTH, INNOCENT_WISE):
            self.assertEqual(self.level(word), "ok", word)

    def test_clean_equations_are_ok(self):
        self.assertEqual(self.level("король − мужчина + женщина"), "ok")
        self.assertEqual(self.level("россия − москва + франция"), "ok")

    def test_format_equation(self):
        text = moderation.format_equation(
            [{"text": "король", "weight": 1}, {"text": "мужчина", "weight": -1},
             {"text": "женщина", "weight": 1}], "королева")
        self.assertEqual(text, "король − мужчина + женщина = королева")


class ModeTests(unittest.TestCase):
    def test_status_for_modes(self):
        original = moderation.MODE
        try:
            moderation.MODE = "off"
            self.assertEqual(moderation.status_for("block"), "approved")
            moderation.MODE = "warn"
            self.assertEqual(moderation.status_for("block"), "approved")
            moderation.MODE = "queue"
            self.assertEqual(moderation.status_for("ok"), "approved")
            self.assertEqual(moderation.status_for("review"), "pending")
            self.assertEqual(moderation.status_for("block"), "pending")
            moderation.MODE = "strict"
            self.assertEqual(moderation.status_for("ok"), "pending")
        finally:
            moderation.MODE = original


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
