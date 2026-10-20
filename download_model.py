"""Download the official RusVectores 2019 Word2Vec archive, resumably."""
from __future__ import annotations
import time
import urllib.request
import zipfile
from pathlib import Path
URL = "https://vectors.nlpl.eu/repository/20/182.zip"
ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "models" / "ruwikiruscorpora_upos_skipgram_300_2_2019.zip"


def download():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    offset = TARGET.stat().st_size if TARGET.exists() else 0
    request = urllib.request.Request(URL, headers={"User-Agent": "VectorLab/1.1"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=45) as response:
        status = response.status
        if offset and status != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        total = int(response.headers.get("Content-Length", "0")) + offset
        print(f"Download status={status} size={total:,} resume={offset:,}", flush=True)
        with TARGET.open(mode) as stream:
            previous = time.monotonic()
            copied = offset
            while data := response.read(1024*1024):
                stream.write(data)
                copied += len(data)
                now = time.monotonic()
                if now-previous > 4:
                    print(f"{copied/1024**2:.1f} / {total/1024**2:.1f} MiB ({copied*100/total:.1f}%)", flush=True)
                    previous = now
    print(f"Downloaded {TARGET.stat().st_size:,} bytes", flush=True)
    with zipfile.ZipFile(TARGET) as archive:
        print("Archive files:", flush=True)
        for f in archive.infolist():
            print(f.filename, f.file_size, flush=True)
        corrupted=archive.testzip()
        if corrupted is not None:
            raise RuntimeError(f"CRC error in {corrupted}")
    print("CRC OK",flush=True)


if __name__ == "__main__":
    download()
