"""Engine tests: binary/gzip/GloVe readers, tokenizers and the similarity math."""
import gzip
import io
import struct
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import word2vec_engine as engine

BINARY_PAYLOAD = b"4 3\n"
WORDS = (
    ("король_NOUN", [1, 1, 0]),
    ("мужчина_NOUN", [1, 0, 0]),
    ("женщина_NOUN", [0, 0, 1]),
    ("королева_NOUN", [0, 1, 1]),
)
for _word, _vec in WORDS:
    BINARY_PAYLOAD += _word.encode("utf-8") + b" " + struct.pack("<3f", *_vec) + b"\n"

RU_SPEC = engine.ModelSpec(id="fake-ru", flag="🇷🇺", label="Fake", language="ru",
                           name="Fake 300D", detail="", archive="fake.zip", urls=(), size_mb=0)
EN_SPEC = engine.ModelSpec(id="fake-en", flag="🇬🇧", label="Fake", language="en",
                           name="Fake 300D", detail="", archive="fake.zip", urls=(), size_mb=0)


class FakeMorph:
    def parse(self, word):
        class Tag:
            POS = "NOUN"

        class Parse:
            tag = Tag()
            normal_form = word

        return [Parse()]


class FakeManager:
    def __init__(self, model, morph=None):
        self._model = model
        self._morph = morph
        self.default_id = model.spec.id

    def get(self, model_id):
        return self._model

    def morph_for(self, model_id):
        return self._morph


class ReaderTests(unittest.TestCase):
    def test_word2vec_binary_reader(self):
        model = engine._read_word2vec_binary(io.BytesIO(BINARY_PAYLOAD), RU_SPEC)
        self.assertEqual(len(model), 4)
        self.assertEqual(model.vector_size, 3)
        self.assertIn("королева_NOUN", model)
        np.testing.assert_allclose(model["король_NOUN"], [1, 1, 0])

    def test_gzip_binary_reader(self):
        stream = io.BytesIO(gzip.compress(BINARY_PAYLOAD))
        model = engine._read_word2vec_binary(gzip.GzipFile(fileobj=stream), RU_SPEC)
        self.assertEqual(len(model), 4)

    def test_glove_text_reader(self):
        text = b"king 1 1 0\nqueen 0 1 1\nbroken line\nman 1 0 0\n"
        model = engine._read_glove_text(io.BytesIO(text), EN_SPEC)
        self.assertEqual(len(model), 3)
        self.assertEqual(model.vector_size, 3)
        self.assertNotIn("broken", model)

    def test_similarities_match_normalized_dot(self):
        # (vectors @ target) / norms must equal normed_matrix @ target
        model = engine._read_word2vec_binary(io.BytesIO(BINARY_PAYLOAD), EN_SPEC)
        target = np.array([0.3, -0.2, 0.9], dtype=np.float32)
        expected = model.vectors / np.linalg.norm(model.vectors, axis=1, keepdims=True) @ target
        np.testing.assert_allclose(model.similarities(target), expected, rtol=1e-5, atol=1e-6)


class CalculateTests(unittest.TestCase):
    def setUp(self):
        self.model = engine._read_word2vec_binary(io.BytesIO(BINARY_PAYLOAD), RU_SPEC)

    def test_analogy_and_result_shape(self):
        with patch.object(engine, "MANAGER", FakeManager(self.model, FakeMorph())):
            result = engine.calculate([
                {"text": "король", "weight": 1},
                {"text": "мужчина", "weight": -1},
                {"text": "женщина", "weight": 1},
            ])
        self.assertEqual(result["results"][0]["token"], "королева_NOUN")
        self.assertEqual(result["dimensions"], 3)
        self.assertEqual(result["resolved"][0]["matched"], ["король_NOUN"])
        self.assertEqual(result["model"], "fake-ru")
        self.assertEqual(result["language"], "ru")

    def test_english_tokenizer_tries_case_variants(self):
        text = b"King 1 1 0\nking 0 1 1\n"
        glove = engine._read_glove_text(io.BytesIO(text), EN_SPEC)
        self.assertEqual(engine.tokens("King", glove, None), ["King"])
        self.assertEqual(engine.tokens("king", glove, None), ["king"])
        self.assertEqual(engine.tokens("KING", glove, None), ["king"])
        self.assertEqual(engine.tokens("Queen", glove, None), [])
        self.assertEqual(engine.tokens("Король", glove, None), [])

    def test_unknown_words_raise(self):
        with patch.object(engine, "MANAGER", FakeManager(self.model, FakeMorph())):
            with self.assertRaises(ValueError):
                engine.calculate([{"text": "абракадабра", "weight": 1}], model_id="fake-ru")


class RegistryTests(unittest.TestCase):
    def test_registry_has_three_models(self):
        ids = {spec.id for spec in engine.MODELS}
        self.assertEqual(ids, {"en-google", "en-glove", "ru-rusvectores"})

    def test_enabled_specs_default_is_full_registry(self):
        self.assertEqual(engine.enabled_specs(""), engine.MODELS)
        self.assertEqual(engine.enabled_specs("   "), engine.MODELS)

    def test_enabled_specs_filters_by_id(self):
        selected = engine.enabled_specs("ru-rusvectores")
        self.assertEqual([spec.id for spec in selected], ["ru-rusvectores"])
        selected = engine.enabled_specs(" en-glove , ru-rusvectores ")
        self.assertEqual([spec.id for spec in selected], ["en-glove", "ru-rusvectores"])

    def test_enabled_specs_rejects_unknown_ids(self):
        with self.assertRaises(RuntimeError):
            engine.enabled_specs("ru-rusvectores,en-gooooogle")
        with self.assertRaises(RuntimeError):
            engine.enabled_specs("совсем-не-модель")

    def test_default_model_is_english(self):
        self.assertEqual(engine.DEFAULT_MODEL, "en-google")

    def test_manager_reports_state_for_missing_model(self):
        state = engine.MANAGER.state("en-google")
        self.assertFalse(state["loaded"])
        self.assertEqual(state["language"], "en")
        with self.assertRaises(engine.UnknownModel):
            engine.MANAGER.state("nope")


class CachePolicyTests(unittest.TestCase):
    """Бюджет кэша: модели живут вместе, а при нехватке памяти нет перезагрузки по кругу."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="vectorplay-cache-")
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        original_dir = engine.MODELS_DIR
        self.addCleanup(setattr, engine, "MODELS_DIR", original_dir)
        engine.MODELS_DIR = self.dir

        self.spec_a = engine.ModelSpec(id="fake-a", flag="🇬🇧", label="A", language="en",
                                       name="Model A", detail="", archive="a.zip", urls=(),
                                       size_mb=1, ram_mb=1000)
        self.spec_b = engine.ModelSpec(id="fake-b", flag="🇷🇺", label="B", language="en",
                                       name="Model B", detail="", archive="b.zip", urls=(),
                                       size_mb=1, ram_mb=1000)
        for spec in (self.spec_a, self.spec_b):
            with zipfile.ZipFile(spec.path, "w") as archive:
                archive.writestr("model.bin", BINARY_PAYLOAD)

    def _manager(self, budget_mb):
        return engine.ModelManager((self.spec_a, self.spec_b), self.spec_a.id, budget_mb=budget_mb)

    @staticmethod
    def _wait(manager, model_id, timeout=15.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = manager.state(model_id)
            if state["loaded"] or state["error"]:
                return state
            time.sleep(0.05)
        raise AssertionError(f"model {model_id} did not finish loading")

    def test_both_models_stay_resident_when_budget_allows(self):
        manager = self._manager(10_000)
        self.assertTrue(manager.ensure_async("fake-a"))
        self._wait(manager, "fake-a")
        self.assertTrue(manager.ensure_async("fake-b"))
        self._wait(manager, "fake-b")
        self.assertEqual(set(manager.resident_ids()), {"fake-a", "fake-b"})
        self.assertFalse(manager.state("fake-b")["note"])

    def test_tight_budget_evicts_idle_model(self):
        with patch.object(engine, "IDLE_EVICTION_S", 0.0):
            manager = self._manager(1500)
            self.assertTrue(manager.ensure_async("fake-a"))
            self._wait(manager, "fake-a")
            self.assertTrue(manager.ensure_async("fake-b"))
            self._wait(manager, "fake-b")
            self.assertEqual(manager.resident_ids(), ["fake-b"])

    def test_tight_budget_refuses_to_thrash(self):
        with patch.object(engine, "IDLE_EVICTION_S", 180.0):
            manager = self._manager(1500)
            self.assertTrue(manager.ensure_async("fake-a"))
            self._wait(manager, "fake-a")
            self.assertFalse(manager.ensure_async("fake-b"))
            state = manager.state("fake-b")
            self.assertFalse(state["loaded"])
            self.assertIn("Не хватает памяти", state["note"] or "")
            self.assertEqual(manager.resident_ids(), ["fake-a"])
            self.assertIn("fake-a", manager.resident_ids())

    def test_model_larger_than_budget_still_loads(self):
        manager = self._manager(100)
        self.assertTrue(manager.ensure_async("fake-a"))
        self.assertTrue(self._wait(manager, "fake-a")["loaded"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
