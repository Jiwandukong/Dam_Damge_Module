"""Serve the local GPR viewer and its model/Overlay assets."""

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
GPR = HERE.parent
MODEL = GPR.parent / "Dam_model/Daecheongdam"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def send_head(self):
        route = unquote(urlsplit(self.path).path)
        static = {"/": "index.html", "/index.html": "index.html", "/style.css": "style.css",
                  "/viewer.js": "viewer.js", "/anomalies.json": "anomalies.json",
                  "/model_payload.json": "model_payload.json"}
        target = HERE / static[route] if route in static else None
        if route == "/ANM_result.csv":
            target = GPR / "Output/Result/ANM_result.csv"
        if target is None:
            for prefix, directory, suffixes in [
                ("/model/", MODEL, {".bin", ".gltf"}),
                ("/textures/", HERE / "textures", {".jpg", ".png"}),
                ("/overlays/", GPR / "Output/Overlay/ANM", {".png"}),
            ]:
                if route.startswith(prefix):
                    relative = Path(route[len(prefix):])
                    candidate = (directory / relative).resolve()
                    if not relative.is_absolute() and candidate.is_relative_to(directory.resolve()) and candidate.suffix.lower() in suffixes:
                        target = candidate
                    break
        if target is None or not target.is_file():
            self.send_error(404)
            return None
        stream = target.open("rb")
        self.send_response(200)
        kind = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if target.suffix in {".html", ".css", ".js", ".json", ".csv", ".gltf"}:
            kind += "; charset=utf-8"
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(target.stat().st_size))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        return stream


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    with ThreadingHTTPServer((args.bind, args.port), Handler) as server:
        print(f"GPR viewer: http://{args.bind}:{args.port}/", flush=True)
        server.serve_forever()
