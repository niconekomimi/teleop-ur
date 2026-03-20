#!/usr/bin/env python3
"""Local HTTP preview API for data collector frames."""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

import cv2
import numpy as np


FrameProvider = Callable[[str], Optional[np.ndarray]]


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class PreviewApiServer:
    """Serve the latest cached preview frames over lightweight HTTP endpoints."""

    def __init__(
        self,
        host: str,
        port: int,
        frame_provider: FrameProvider,
        jpeg_quality: int = 80,
        logger: Optional[object] = None,
    ) -> None:
        self._host = str(host).strip() or "127.0.0.1"
        self._port = int(port)
        self._frame_provider = frame_provider
        self._jpeg_quality = max(30, min(95, int(jpeg_quality)))
        self._logger = logger
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def _log(self, level: str, msg: str) -> None:
        if self._logger is None:
            return
        log_fn = getattr(self._logger, level, None)
        if callable(log_fn):
            log_fn(msg)

    def start(self) -> None:
        if self._server is not None:
            return

        server = _ReusableThreadingHTTPServer((self._host, self._port), self._build_handler())
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True, name="preview-api-server")
        thread.start()

        self._server = server
        self._thread = thread
        self._log("info", f"Preview API listening on http://{self._host}:{self._port}")

    def stop(self) -> None:
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None

        if server is not None:
            try:
                server.shutdown()
            except Exception as exc:  # noqa: BLE001
                self._log("warn", f"停止 Preview API 失败: {exc!r}")
            try:
                server.server_close()
            except Exception as exc:  # noqa: BLE001
                self._log("warn", f"关闭 Preview API socket 失败: {exc!r}")

        if thread is not None:
            thread.join(timeout=1.5)

    def _build_handler(self):
        outer = self

        class PreviewApiHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/healthz":
                    payload = json.dumps({"status": "ok"}).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                camera_name = None
                if self.path == "/preview/global.jpg":
                    camera_name = "global"
                elif self.path == "/preview/wrist.jpg":
                    camera_name = "wrist"

                if camera_name is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return

                frame = outer._frame_provider(camera_name)
                if frame is None:
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    return

                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), outer._jpeg_quality],
                )
                if not ok:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
                    return

                payload = encoded.tobytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, _format: str, *_args) -> None:
                return

        return PreviewApiHandler
