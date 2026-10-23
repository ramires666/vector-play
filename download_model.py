"""Download VectorPlay model archives, resumably, with mirrors and validation.

Usage:
    python download_model.py                    # модель по умолчанию
    python download_model.py ru-rusvectores     # конкретные модели (любые из реестра)
    python download_model.py --all              # все включённые модели (VECTOR_CALC_ENABLED_MODELS)
    python download_model.py --force en-google  # перекачать заново

Модель по умолчанию задаётся VECTOR_CALC_DEFAULT_MODEL (сейчас en-small — 376 МБ).
Каждая полностью скачанная модель помечается файлом <архив>.ok, поэтому
повторный запуск ничего не перекачивает, а прерванная закачка продолжается
с места обрыва (HTTP Range).
"""
from __future__ import annotations

import argparse
import gzip
import sys
import time
import urllib.error
import urllib.request
import zipfile

import word2vec_engine as engine


def marker_path(spec: engine.ModelSpec):
    return spec.path.with_suffix(spec.path.suffix + ".ok")


def source_path(spec: engine.ModelSpec):
    """Какой URL использовался для текущей частичной закачки."""
    return spec.path.with_suffix(spec.path.suffix + ".part-url")


def is_complete(spec: engine.ModelSpec) -> bool:
    if not spec.path.is_file():
        return False
    try:
        return marker_path(spec).read_text(encoding="utf-8").strip() == str(spec.path.stat().st_size)
    except OSError:
        return False


def _download_from(url: str, spec: engine.ModelSpec) -> int:
    path = spec.path
    offset = path.stat().st_size if path.exists() else 0
    if offset:
        # Докачивать можно только тот же файл: у разных зеркал своя gzip-обёртка,
        # и склейка «половина с одного, половина с другого» даёт битый архив.
        previous = None
        try:
            previous = source_path(spec).read_text(encoding="utf-8").strip()
        except OSError:
            pass
        if previous != url:
            print(f"  частичная закачка была с другого зеркала ({previous or 'неизвестно'}), начинаю заново", flush=True)
            path.unlink(missing_ok=True)
            offset = 0
    request = urllib.request.Request(url, headers={"User-Agent": "VectorPlay/1.3"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        status = response.status
        if offset and status != 206:
            offset = 0
        total = int(response.headers.get("Content-Length", "0")) + offset
        print(f"{path.name}: status={status} size={total / 1024 ** 3:.2f} GiB resume={offset / 1024 ** 3:.2f} GiB", flush=True)
        source_path(spec).write_text(url, encoding="utf-8")
        with path.open("ab" if offset else "wb") as stream:
            previous_time = time.monotonic()
            copied = offset
            while data := response.read(1024 * 1024):
                stream.write(data)
                copied += len(data)
                now = time.monotonic()
                if now - previous_time > 5:
                    if total:
                        print(f"  {copied / 1024 ** 3:.2f} / {total / 1024 ** 3:.2f} GiB ({copied * 100 / total:.1f}%)", flush=True)
                    else:
                        print(f"  {copied / 1024 ** 3:.2f} GiB", flush=True)
                    previous_time = now
    size = path.stat().st_size
    if total and size != total:
        raise IOError(f"файл недокачан: {size} из {total} байт")
    print(f"  скачано {size / 1024 ** 3:.2f} GiB", flush=True)
    return size


def _check_gzip(path) -> None:
    """Полная проверка gzip: CRC и длина потока (минута на 1.6 ГБ)."""
    with gzip.open(path, "rb") as stream:
        head = stream.read(4096)
        header = head.split(b"\n", 1)[0].split()
        if len(header) != 2 or not all(part.isdigit() for part in header):
            raise RuntimeError("файл не похож на word2vec binary (плохой заголовок)")
        total = 0
        while chunk := stream.read(8 * 1024 * 1024):
            total += len(chunk)
            if total % (1024 ** 3) < 8 * 1024 * 1024:
                print(f"  проверено {total / 1024 ** 3:.1f} GiB", flush=True)
    print(f"  распаковано {total / 1024 ** 3:.2f} GiB, CRC в порядке", flush=True)


def validate(spec: engine.ModelSpec) -> None:
    path = spec.path
    suffix = path.suffix.lower()
    print(f"  проверяю {path.name} ...", flush=True)
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            print("  файлы в архиве: " + ", ".join(info.filename for info in archive.infolist()), flush=True)
            corrupted = archive.testzip()
            if corrupted is not None:
                raise RuntimeError(f"CRC ошибка в {corrupted}")
    elif suffix == ".gz":
        _check_gzip(path)
    else:
        with path.open("rb") as stream:
            head = stream.read(4096)
        if not head.strip():
            raise RuntimeError("пустой файл")
    print("  OK", flush=True)


def download(spec: engine.ModelSpec, force: bool = False) -> None:
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    if not force and is_complete(spec):
        print(f"{spec.id}: уже скачано ({spec.path.name}, {spec.path.stat().st_size / 1024 ** 3:.2f} GiB)", flush=True)
        return
    failures = []
    for url in spec.urls:
        print(f"{spec.id}: {spec.name} ({spec.size_mb} MiB) <- {url}", flush=True)
        try:
            _download_from(url, spec)
            validate(spec)
            marker_path(spec).write_text(str(spec.path.stat().st_size), encoding="utf-8")
            source_path(spec).unlink(missing_ok=True)
            print(f"{spec.id}: готово", flush=True)
            return
        except Exception as exc:
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
            print(f"  зеркало не сработало ({type(exc).__name__}: {exc}), пробую следующее", flush=True)
    raise SystemExit(f"Не удалось скачать {spec.id}:\n  " + "\n  ".join(failures))


def _force_utf8_console() -> None:
    """Windows-консоль по умолчанию в cp1251 и падает на «ō» в названии модели."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv=None) -> int:
    _force_utf8_console()
    parser = argparse.ArgumentParser(description="Скачать модели VectorPlay")
    parser.add_argument("models", nargs="*", help="id моделей; по умолчанию — модель по умолчанию")
    parser.add_argument("--all", action="store_true", help="скачать все модели из реестра")
    parser.add_argument("--force", action="store_true", help="перекачать заново, даже если файл уже есть")
    args = parser.parse_args(argv)

    if args.all:
        model_ids = [spec.id for spec in engine.ENABLED_MODELS]
    else:
        model_ids = args.models or [engine.MANAGER.default_id]

    unknown = [model_id for model_id in model_ids if model_id not in engine.MODELS_BY_ID]
    if unknown:
        parser.error(f"неизвестные модели: {', '.join(unknown)}; доступные: {', '.join(engine.MODELS_BY_ID)}")

    failures: list[str] = []
    for model_id in model_ids:
        try:
            download(engine.MODELS_BY_ID[model_id], force=args.force)
        except SystemExit as exc:
            failures.append(f"{model_id}: {exc}")
            print(f"Продолжаю со следующими моделями, к {model_id} вернёмся позже.", flush=True)
    if failures:
        raise SystemExit("Не удалось скачать:\n  " + "\n  ".join(failures))
    print("Все запрошенные модели на месте.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
