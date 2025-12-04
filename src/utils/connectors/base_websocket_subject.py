"""
Production WebSocket connector for Pathway with auto-reconnect.
Subclass and implement on_connect() and on_ws_message().
"""

import pathway as pw
import asyncio
import aiohttp
from aiohttp.client_ws import ClientWebSocketResponse
from typing import Callable, Optional


class AIOHttpWebsocketSubject(pw.io.python.ConnectorSubject):

    def __init__(
        self,
        url: str,
        reconnect_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: float = 30.0,
        on_error: Optional[Callable[[str, Exception, Optional[str]], None]] = None,
    ):
        super().__init__()
        self._url = url
        self._reconnect_delay = reconnect_delay
        self._max_delay = max_delay
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._on_error = on_error or (lambda *_: None)
        self._running = True

    def run(self):
        """Start connector and run until stopped."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._reconnect_loop())
        except KeyboardInterrupt:
            self._running = False
        finally:
            loop.close()

    def stop(self):
        self._running = False

    async def _reconnect_loop(self):
        """Reconnect with exponential backoff."""
        delay = self._reconnect_delay
        while self._running:
            connected = await self._attempt_connection()
            if connected:
                delay = self._reconnect_delay
            else:
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_delay)

    async def _attempt_connection(self) -> bool:
        """Single connection attempt with error handling."""
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.ws_connect(self._url, heartbeat=30) as ws:
                    try_setup = await self._setup(ws)
                    if not try_setup:
                        return False
                    await self._stream(ws)
                    return True
        except (aiohttp.ClientError, asyncio.TimeoutError, asyncio.CancelledError) as e:
            self._report_error(type(e).__name__, e)
            return False
        except Exception as e:
            self._report_error("unknown", e)
            return False

    async def _setup(self, ws: ClientWebSocketResponse) -> bool:
        """Call on_connect for subscription/auth setup."""
        try:
            await self.on_connect(ws)
            return True
        except Exception as e:
            self._report_error("setup", e)
            return False

    async def _stream(self, ws: ClientWebSocketResponse):
        """Process messages from WebSocket."""
        async for msg in ws:
            if not self._running:
                await ws.close()
                break
            try:
                rows = await self.on_ws_message(msg, ws)
                for row in rows:
                    self.next_json(row)
            except Exception as e:
                self._report_error("message", e, str(msg.data) if msg.data else None)

    def _report_error(self, typ: str, err: Exception, ctx: Optional[str] = None):
        try:
            self._on_error(typ, err, ctx)
        except Exception:
            pass

    # Subclass interface
    async def on_connect(self, ws: ClientWebSocketResponse):
        """Override to send subscription/auth messages on connect."""
        raise NotImplementedError("Implement on_connect()")

    async def on_ws_message(self, msg: aiohttp.WSMessage, ws: ClientWebSocketResponse) -> list[dict]:
        """Override to process messages and return dicts for Pathway."""
        raise NotImplementedError("Implement on_ws_message()")