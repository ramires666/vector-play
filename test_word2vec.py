import io
import struct
import unittest

import numpy as np
import word2vec_engine as engine
from unittest.mock import patch


class FakeMorph:
    def parse(self, word):
        class Tag:
            POS = "NOUN"
        class Parse:
            tag = Tag()
            normal_form = word
        return [Parse()]


class Word2VecTests(unittest.TestCase):
    def test_bin_parser_and_math(self):
        payload = b"4 3\n"
        for word, vec in (
            ("король_NOUN", [1, 1, 0]),
            ("мужчина_NOUN", [1, 0, 0]),
            ("женщина_NOUN", [0, 0, 1]),
            ("королева_NOUN", [0, 1, 1]),
        ):
            payload += word.encode("utf-8") + b" " + struct.pack("<3f", *vec) + b"\n"
        model = engine._read_binary(io.BytesIO(payload))
        self.assertEqual(len(model), 4)
        with patch.object(engine, "_MODEL", model), patch.object(engine, "_MORPH", FakeMorph()):
            r=engine.calculate([
                {"text":"король","weight":1},
                {"text":"мужчина","weight":-1},
                {"text":"женщина","weight":1},
            ])
            self.assertEqual(r["results"][0]["token"], "королева_NOUN")
            self.assertEqual(r["dimensions"], 3)
            self.assertEqual(r["resolved"][0]["matched"], ["король_NOUN"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
