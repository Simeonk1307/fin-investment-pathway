import pathway as pw
import asyncio
import aiohttp
from aiohttp.client_ws import ClientWebSocketResponse
from typing import Callable, Optional, List
import logging
import traceback


class AIOHttpWebsocketSubject(pw.io.python.ConnectorSubject):

    def __init__(
        self,
        url: str,
        reconnect_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: float = 30.0,
        heartbeat: float = 20.0,
        logger: Optional[logging.Logger] = None,
        on_error: Optional[Callable[[str, Exception, Optional[str]], None]] = None,
    ):
        super().__init__()
        self._url = url
        self._reconnect_delay = reconnect_delay
        self._max_delay = max_delay
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._heartbeat = heartbeat

        self._logger = logger or logging.getLogger("WebSocketSubject")
        self._on_error = on_error or (lambda typ, err, ctx: None)

        self._running = True

        self._logger.info(
            f"[WS INIT] url={url} reconnect_delay={reconnect_delay}s max_delay={max_delay}s timeout={timeout}s"
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def run(self):
        """Start connector event loop (blocking)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self._logger.info("[WS RUN] WebSocket loop starting...")

        try:
            loop.run_until_complete(self._reconnect_loop())
        except KeyboardInterrupt:
            self._logger.warning("[WS STOP] KeyboardInterrupt received — shutting down...")
        finally:
            self._running = False
            loop.stop()
            loop.close()
            self._logger.info("[WS STOP] Event loop closed.")

    def stop(self):
        self._logger.info("[WS STOP] Received stop() request.")
        self._running = False

    # -------------------------------------------------------------------------
    # Reconnection handling
    # -------------------------------------------------------------------------
    async def _reconnect_loop(self):
        delay = self._reconnect_delay

        while self._running:
            self._logger.info(f"[WS CONNECT] Attempting WebSocket connection to {self._url}")
            ok = await self._attempt_connection()

            if ok:
                delay = self._reconnect_delay  # reset on successful connection
                continue

            self._logger.warning(f"[WS RECONNECT] Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, self._max_delay)

    async def _attempt_connection(self) -> bool:
        """Try establishing one WebSocket connection."""
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.ws_connect(self._url, heartbeat=self._heartbeat) as ws:
                    self._logger.info("[WS CONNECTED] Connection established.")

                    setup_ok = await self._setup(ws)
                    if not setup_ok:
                        self._logger.error("[WS SETUP] on_connect() failed.")
                        return False

                    await self._stream(ws)
                    return True

        except (aiohttp.ClientError, asyncio.TimeoutError, asyncio.CancelledError) as e:
            self._report_error("connection", e)
            self._logger.error(f"[WS ERROR] Connection failure: {type(e).__name__}: {e}")
            return False

        except Exception as e:
            self._report_error("connection", e)
            self._logger.exception(f"[WS ERROR] Unexpected during connection: {e}")
            return False

    # -------------------------------------------------------------------------
    # Setup hook
    # -------------------------------------------------------------------------
    async def _setup(self, ws: ClientWebSocketResponse) -> bool:
        try:
            await self.on_connect(ws)
            self._logger.info("[WS SETUP] on_connect() completed successfully.")
            return True
        except Exception as e:
            self._report_error("setup", e)
            self._logger.exception("[WS SETUP ERROR] on_connect() raised an error.")
            return False

    # -------------------------------------------------------------------------
    # Streaming logic
    # -------------------------------------------------------------------------
    async def _stream(self, ws: ClientWebSocketResponse):
        """Main message processing loop."""
        self._logger.info("[WS STREAM] Listening for messages...")

        async for msg in ws:
            if not self._running:
                self._logger.info("[WS STREAM] stop() called — closing WebSocket.")
                await ws.close()
                break

            try:
                rows = await self.on_ws_message(msg, ws)

                if not isinstance(rows, list):
                    raise ValueError(f"on_ws_message must return a list[dict], got {type(rows)}")

                for row in rows:
                    if not isinstance(row, dict):
                        raise ValueError("Each row must be dict-like.")
                    self.next_json(row)

            except Exception as e:
                raw = msg.data if hasattr(msg, "data") else None
                self._report_error("message", e, raw)
                self._logger.error(
                    f"[WS MSG ERROR] Error processing message: {e}\n"
                    f"Raw: {raw}\n{traceback.format_exc()}"
                )

        self._logger.warning("[WS STREAM] WebSocket closed or lost connection.")

    # -------------------------------------------------------------------------
    # Error handling helper
    # -------------------------------------------------------------------------
    def _report_error(self, typ: str, err: Exception, ctx: Optional[str] = None):
        try:
            self._on_error(typ, err, ctx)
        except Exception:
            pass  # never let user error handler crash system

        self._logger.error(
            f"[WS ERROR] type={typ} error={err} ctx={ctx if ctx else 'N/A'}"
        )

    # -------------------------------------------------------------------------
    # Subclass hooks
    # -------------------------------------------------------------------------
    async def on_connect(self, ws: ClientWebSocketResponse):
        raise NotImplementedError("Implement on_connect()")

    async def on_ws_message(self, msg: aiohttp.WSMessage, ws: ClientWebSocketResponse) -> List[dict]:
        raise NotImplementedError("Implement on_ws_message()")
