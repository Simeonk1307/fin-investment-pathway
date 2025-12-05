import pathway as pw, asyncio, aiohttp, logging, traceback
from aiohttp.client_ws import ClientWebSocketResponse
from typing import Callable, Optional, List

class AIOHttpWebsocketSubject(pw.io.python.ConnectorSubject):
    def __init__(self, url: str, reconnect_delay: float = 1.0, max_delay: float = 60.0, timeout: float = 30.0, heartbeat: float = 20.0, logger: Optional[logging.Logger] = None, on_error: Optional[Callable] = None):
        super().__init__()
        self._url = url
        self._reconnect_delay = reconnect_delay
        self._max_delay = max_delay
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._heartbeat = heartbeat
        self._logger = logger or logging.getLogger("ws")
        self._on_error = on_error or (lambda t,e,c: None)
        self._running = True
        self._logger.info(f"WebSocket initialized (url={url})")

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._logger.info("Starting WebSocket loop...")
        try:
            loop.run_until_complete(self._reconnect_loop())
        except KeyboardInterrupt:
            self._logger.warning("Interrupted, stopping WebSocket...")
        finally:
            self._running = False
            loop.stop()
            loop.close()
            self._logger.info("WebSocket loop closed.")

    def stop(self):
        self._running = False
        self._logger.info("Stop requested.")

    async def _reconnect_loop(self):
        delay = self._reconnect_delay
        while self._running:
            self._logger.info(f"Connecting to {self._url}...")
            ok = await self._attempt_connection()
            if ok:
                delay = self._reconnect_delay
                continue
            self._logger.warning(f"Connection failed. Reconnecting in {delay:.1f}s...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, self._max_delay)

    async def _attempt_connection(self) -> bool:
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.ws_connect(self._url, heartbeat=self._heartbeat) as ws:
                    self._logger.info("Connected successfully.")
                    if not await self._setup(ws):
                        self._logger.error("Setup failed.")
                        return False
                    await self._stream(ws)
                    return True
        except Exception as e:
            self._report_error("connection", e)
            self._logger.error(f"Connection error: {e}")
            return False

    async def _setup(self, ws: ClientWebSocketResponse) -> bool:
        try:
            await self.on_connect(ws)
            self._logger.info("Connection setup complete.")
            return True
        except Exception as e:
            self._report_error("setup", e)
            self._logger.error(f"Setup error: {e}")
            return False

    async def _stream(self, ws: ClientWebSocketResponse):
        self._logger.info("Listening for messages...")
        async for msg in ws:
            if not self._running:
                self._logger.info("Stopping — closing connection.")
                await ws.close()
                break
            try:
                rows = await self.on_ws_message(msg, ws)
                if not isinstance(rows, list):
                    raise ValueError("on_ws_message must return a list of dicts")
                for r in rows:
                    if not isinstance(r, dict):
                        raise ValueError("Each returned row must be a dict")
                    self.next_json(r)
            except Exception as e:
                raw = getattr(msg, "data", None)
                self._report_error("message", e, raw)
                self._logger.error(
                    f"Message processing error: {e}\n"
                    f"Raw message: {raw}\n{traceback.format_exc()}"
                )
        self._logger.warning("Connection closed.")

    def _report_error(self, typ: str, err: Exception, ctx: Optional[str] = None):
        try:
            self._on_error(typ, err, ctx)
        except Exception:
            pass
        self._logger.error(f"{typ.capitalize()} error: {err}")

    async def on_connect(self, ws: ClientWebSocketResponse):
        raise NotImplementedError

    async def on_ws_message(self, msg: aiohttp.WSMessage, ws: ClientWebSocketResponse) -> List[dict]:
        raise NotImplementedError
