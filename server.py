"""VectorPlay — shared LAN Word2Vec calculator and memories gallery."""
from __future__ import annotations
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import word2vec_engine as engine
import memories_db as memories

ROOT = Path(__file__).resolve().parent
HOST = os.getenv("VECTOR_CALC_HOST", "0.0.0.0")
PORT = int(os.getenv("VECTOR_CALC_PORT", "58765"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args), flush=True)

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, filename):
        body = (ROOT / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path)
        if path.path == "/":
            self.send_html("index.html")
        elif path.path == "/memories":
            self.send_html("memories.html")
        elif path.path == "/api/status":
            self.send_json(200, engine.state())
        elif path.path == "/api/memories":
            try:
                query = parse_qs(path.query)
                after = int(query.get("after", ["0"])[0])
                limit = int(query.get("limit", ["50"])[0])
                self.send_json(200, memories.list_memories(after=after,limit=limit))
            except (ValueError, TypeError) as exc:
                self.send_json(400, {"error": str(exc)})
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path not in ("/api/calculate", "/api/memories"):
            self.send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 15_000:
                raise ValueError("Недопустимый размер запроса")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Ожидается JSON-объект")
            if self.path == "/api/memories":
                item, created = memories.add_memory(data.get("terms"))
                self.send_json(201 if created else 200, {"item": item, "created": created})
                return
            self.send_json(200, engine.calculate(
                data.get("terms"), data.get("top_k", 10),
                data.get("exclude_terms", True)))
        except (ValueError, TypeError, KeyError, OverflowError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            self.send_json(500, {"error": "Ошибка сервера: " + type(exc).__name__})


if __name__ == "__main__":
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"VectorPlay http://{HOST}:{PORT}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()
