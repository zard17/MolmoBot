"""WebSocket server that streams JPEG-encoded simulation frames to browser clients.

Usage:
    streamer = FrameStreamer(port=9090)
    streamer.start()

    # In the rollout loop (from any thread):
    streamer.send_frame(numpy_rgb_array, metadata={"step": 42})

    # Browser clients connect to ws://host:9090 and receive JPEG frames.
    # Opening http://host:9090/ in a browser serves the built-in viewer.

    streamer.stop()
"""

from __future__ import annotations

import asyncio
import http
import io
import json
import logging
import threading
from pathlib import Path

import numpy as np
from PIL import Image

import websockets.asyncio.server as ws_server

logger = logging.getLogger(__name__)


class FrameStreamer:
    """Streams rendered frames over WebSocket to browser clients.

    Runs an asyncio WebSocket server in a daemon thread.  The rollout thread
    calls :meth:`send_frame` which JPEG-encodes the frame and makes it
    available to all connected clients.  Only the *latest* frame is kept — slow
    clients simply skip frames rather than building up a backlog.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9090,
        jpeg_quality: int = 70,
    ) -> None:
        self._host = host
        self._port = port
        self._quality = jpeg_quality

        # Latest encoded frame + metadata, guarded by a lock so the rollout
        # thread and the asyncio thread never race.
        self._lock = threading.Lock()
        self._frame_bytes: bytes | None = None
        self._metadata_bytes: bytes | None = None

        # Asyncio plumbing
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._new_frame: asyncio.Event | None = None  # created inside the loop thread
        self._clients: set[ws_server.ServerConnection] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the background server thread."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Frame streamer starting on ws://%s:%s", self._host, self._port)

    def stop(self) -> None:
        """Signal the server to shut down and wait for the thread."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Frame streamer stopped")

    # ------------------------------------------------------------------
    # Public API — called from the rollout thread
    # ------------------------------------------------------------------

    def send_frame(self, frame: np.ndarray, metadata: dict | None = None) -> None:
        """JPEG-encode *frame* and queue it for connected clients.

        If no clients are connected this is essentially free (just a set-length
        check).  Encoding errors are logged and swallowed so the evaluation is
        never interrupted.
        """
        if not self._clients:
            return

        try:
            # JPEG encode
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self._quality)
            jpeg = buf.getvalue()

            meta = json.dumps(metadata or {}).encode()

            with self._lock:
                self._frame_bytes = jpeg
                self._metadata_bytes = meta

            # Wake up the asyncio handlers
            if self._loop is not None and self._new_frame is not None:
                self._loop.call_soon_threadsafe(self._new_frame.set)
        except Exception:
            logger.warning("Frame encoding failed", exc_info=True)

    # ------------------------------------------------------------------
    # Internals — asyncio event loop in a background thread
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._new_frame = asyncio.Event()
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        async with ws_server.serve(
            self._ws_handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._http_handler,
        ):
            logger.info("Frame streamer listening on ws://%s:%s", self._host, self._port)
            await asyncio.get_event_loop().create_future()  # run forever

    async def _ws_handler(self, websocket: ws_server.ServerConnection) -> None:
        """Per-client handler: pushes the latest frame whenever a new one is available."""
        self._clients.add(websocket)
        logger.info("Viewer connected from %s (%d clients)", websocket.remote_address, len(self._clients))
        try:
            while True:
                await self._new_frame.wait()
                self._new_frame.clear()

                with self._lock:
                    meta = self._metadata_bytes
                    jpeg = self._frame_bytes

                if jpeg is not None:
                    if meta is not None:
                        await websocket.send(meta)  # text-ish JSON metadata
                    await websocket.send(jpeg)       # binary JPEG frame
        except websockets.ConnectionClosed:
            pass
        except Exception:
            logger.warning("Error in viewer handler", exc_info=True)
        finally:
            self._clients.discard(websocket)
            logger.info("Viewer disconnected (%d clients)", len(self._clients))

    def _http_handler(
        self,
        connection: ws_server.ServerConnection,
        request: ws_server.Request,
    ) -> ws_server.Response | None:
        """Serve the built-in viewer HTML at ``/`` and a health-check at ``/healthz``."""
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")

        if request.path in ("/", "/viewer"):
            viewer_path = Path(__file__).with_name("viewer.html")
            try:
                html = viewer_path.read_text()
            except FileNotFoundError:
                return connection.respond(
                    http.HTTPStatus.NOT_FOUND,
                    "viewer.html not found\n",
                )
            return connection.respond(
                http.HTTPStatus.OK,
                html,
                headers=websockets.Headers({"Content-Type": "text/html; charset=utf-8"}),
            )

        # Fall through to normal WebSocket upgrade for all other paths.
        return None
