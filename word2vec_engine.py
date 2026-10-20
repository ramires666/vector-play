"""Russian Word2Vec calculator using official RusVectores vectors in word2vec binary format.

No Gensim dependency: loads model.bin from the original NLPL zip archive.
"""
from __future__ import annotations
import math
import threading
import zipfile
from pathlib import Path

import numpy as np
import pymorphy3

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "models" / "ruwikiruscorpora_upos_skipgram_300_2_2019.zip"
LOCK = threading.RLock()
_MODEL = None
_MORPH = None
_ERROR = None
_LOADING = False
_TAGS = {"NOUN": "NOUN", "ADJF": "ADJ", "ADJS": "ADJ", "COMP": "ADJ",
         "VERB": "VERB", "INFN": "VERB", "PRTF": "VERB", "PRTS": "VERB",
         "GRND": "VERB", "ADVB": "ADV", "NPRO": "PRON", "NUMR": "NUM",
         "PREP": "ADP", "CONJ": "CCONJ", "PRCL": "PART", "INTJ": "INTJ"}


class WordSpace:
    def __init__(self, words, array):
        self.index_to_key = words
        self.key_to_index = {word: i for i, word in enumerate(words)}
        self.vectors = array
        self.vector_size = array.shape[1]
        lengths = np.linalg.norm(array, axis=1)
        lengths[lengths < 1e-12] = 1.
        self.normed = np.ascontiguousarray(array / lengths[:, None], dtype=np.float32)

    def __contains__(self, word):
        return word in self.key_to_index

    def __getitem__(self, word):
        return self.vectors[self.key_to_index[word]]

    def __len__(self):
        return len(self.index_to_key)

    def get_normed_vectors(self):
        return self.normed


def _read_binary(stream):
    header = stream.readline().strip().split()
    if len(header) != 2:
        raise ValueError("Invalid Word2Vec binary header")
    count, dims = map(int, header)
    if not 1 <= dims <= 4096 or not 1 <= count <= 5_000_000:
        raise ValueError(f"Unexpected Word2Vec shape {count} x {dims}")
    matrix = np.empty((count, dims), dtype=np.float32)
    words = []
    for i in range(count):
        chars = bytearray()
        while True:
            char = stream.read(1)
            if char == b" ":
                break
            if not char:
                raise EOFError(f"Unexpected EOF on word {i}")
            if char not in (b"\n", b"\r") or chars:
                chars.extend(char)
            if len(chars) > 1024:
                raise ValueError("Word2Vec token too long")
        words.append(chars.decode("utf-8", errors="replace"))
        binary = stream.read(dims*4)
        if len(binary) != dims*4:
            raise EOFError(f"Unexpected vector EOF on word {i}")
        matrix[i] = np.frombuffer(binary, dtype="<f4")
    return WordSpace(words, matrix)


def available():
    return ARCHIVE.is_file() and zipfile.is_zipfile(ARCHIVE)


def load():
    global _MODEL, _MORPH, _ERROR, _LOADING
    with LOCK:
        if _MODEL is not None:
            return _MODEL
        if not available():
            raise ValueError("Модель Word2Vec не загружена. Запустите download_model.py")
        _LOADING = True
        _ERROR = None
        try:
            with zipfile.ZipFile(ARCHIVE) as archive:
                name = next((n for n in archive.namelist() if n.endswith("model.bin")), None)
                if not name:
                    raise ValueError("В ZIP отсутствует model.bin")
                with archive.open(name) as stream:
                    model = _read_binary(stream)
            morph = pymorphy3.MorphAnalyzer()
            _MODEL, _MORPH = model, morph
            return model
        except Exception as exc:
            _ERROR = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _LOADING = False


def state():
    return {"available": available(), "loaded": _MODEL is not None,
            "loading": _LOADING, "error": _ERROR,
            "vocab_size": len(_MODEL) if _MODEL is not None else 0,
            "dimensions": _MODEL.vector_size if _MODEL is not None else 300}


def tokens(text):
    model = load()
    import re
    result = []
    for word in re.findall(r"[\w-]+", text.lower(), re.UNICODE):
        if word in model:
            result.append(word)
            continue
        possibilities = []
        for parse in _MORPH.parse(word)[:5]:
            tag = _TAGS.get(parse.tag.POS)
            if tag:
                possibilities.append(f"{parse.normal_form}_{tag}")
            possibilities.append(f"{parse.normal_form}_PROPN")
            possibilities.append(parse.normal_form)
        possibilities.extend((word + "_PROPN", word + "_NOUN", word + "_ADJ", word + "_VERB"))
        match = next((v for v in possibilities if v in model), None)
        if match is not None:
            result.append(match)
    return result


def embed(text):
    matched = tokens(text)
    if not matched:
        raise ValueError(f"Ни одно слово из {text!r} не найдено в Word2Vec")
    model = load()
    vec = np.mean([model[w] for w in matched], axis=0)
    return np.asarray(vec, dtype=np.float32), matched


def calculate(terms, top_k=12, exclude_terms=True):
    model = load()
    if not isinstance(terms, list) or not 1 <= len(terms) <= 12:
        raise ValueError("Необходимо от 1 до 12 переменных")
    target=np.zeros(model.vector_size,dtype=np.float32)
    resolved=[]
    excluded=set()
    for item in terms:
        if not isinstance(item, dict):
            raise ValueError("Некорректная переменная")
        phrase=str(item.get("text","")).strip()
        weight=item.get("weight",1)
        if not phrase or len(phrase)>600 or isinstance(weight,bool) or not isinstance(weight,(int,float)) or not math.isfinite(weight) or abs(weight)>100:
            raise ValueError("Некорректный текст или коэффициент")
        vec, keys=embed(phrase)
        target += float(weight)*vec
        resolved.append({"text":phrase,"weight":weight,"matched":keys})
        if exclude_terms:
            excluded.update(keys)
    norm=float(np.linalg.norm(target))
    if norm<1e-12:
        raise ValueError("Нулевой результирующий вектор")
    target/=norm
    scores=model.get_normed_vectors() @ target
    k=max(1,min(int(top_k),50))
    sample=min(len(scores),max(k*40,k+len(excluded)+20))
    nearest=np.argpartition(scores,-sample)[-sample:]
    nearest=nearest[np.argsort(scores[nearest])[::-1]]
    results=[]
    seen_words=set()
    for pos in nearest:
        token=model.index_to_key[int(pos)]
        surface=token.rsplit("_", 1)[0].casefold()
        if token in excluded or surface in seen_words:
            continue
        seen_words.add(surface)
        results.append({"text":token.replace("_"," · ",1),"token":token,
                        "similarity":round(float(scores[pos]),6),"source":"word2vec"})
        if len(results)>=k:
            break
    return {"results":results,"candidate_count":len(model)-len(excluded),
            "dimensions":model.vector_size,"mode":"word2vec",
            "model":"ruwikiruscorpora_upos_skipgram_300_2_2019",
            "backend":"Word2Vec / NumPy","resolved":resolved,
            "vector_preview":[round(float(v),6) for v in target[:20]]}


