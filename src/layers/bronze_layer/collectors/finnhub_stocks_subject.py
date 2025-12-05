import json
import aiohttp
from aiohttp.client_ws import ClientWebSocketResponse

from src.layers.bronze_layer.base.base_websocket_subject import AIOHttpWebsocketSubject
from src.layers.bronze_layer.event_envelope import create_event_envelope


class FinnhubSubject(AIOHttpWebsocketSubject):
    def __init__(self, api_key, symbols, logger=None, debug=False, debug_writer=None, **kwargs):
        super().__init__(
            url=f"wss://ws.finnhub.io?token={api_key}",
            logger=logger,
            **kwargs,
        )

        self._symbols = symbols
        self.debug = debug
        self.debug_writer = debug_writer or (lambda *_: None)

        self._logger.info(
            f"[FINNHUB:INIT] Symbols={len(symbols)} Debug={self.debug}"
        )

        if self.debug:
            self.debug_writer("stocks", "startup", {
                "symbols": symbols,
                "ws_url": f"wss://ws.finnhub.io?token={api_key}",
            })

    async def on_connect(self, ws: ClientWebSocketResponse):
        self._logger.info("[FINNHUB:CONNECT] WebSocket connected")

        try:
            for sym in self._symbols:
                await ws.send_json({"type": "subscribe", "symbol": sym})
                self._logger.info(f"[FINNHUB:SUB] {sym}")
        except Exception as e:
            self._logger.error(f"[FINNHUB:SUB ERROR] {e}")
            raise

        self._logger.info("[FINNHUB:CONNECT] Subscriptions sent")

    async def on_ws_message(self, msg: aiohttp.WSMessage, ws: ClientWebSocketResponse) -> list:
        if msg.type != aiohttp.WSMsgType.TEXT:
            self._logger.debug(f"[FINNHUB:SKIP] Non-text msg type={msg.type}")
            return []

        try:
            data = json.loads(msg.data)
        except Exception:
            self._logger.error(f"[FINNHUB:JSON] Invalid JSON msg={msg.data}")
            return []

        msg_type = data.get("type")

        if msg_type == "trade":
            trades = data.get("data", [])

            if not trades:
                self._logger.debug("[FINNHUB:TRADE] Empty batch")
                return []

            if not isinstance(trades, list):
                self._logger.error(f"[FINNHUB:TRADE] Invalid format: {trades}")
                return []

            out = []
            for tr in trades:
                if not isinstance(tr, dict):
                    self._logger.debug(f"[FINNHUB:TRADE:SKIP] Non-dict: {tr}")
                    continue

                try:
                    ev = create_event_envelope(
                        payload=tr,
                        source="finnhub",
                        source_type="websocket",
                    )
                    out.append(ev)
                except Exception as e:
                    self._logger.error(f"[FINNHUB:ENVELOPE] {e}")
                    continue

            self._logger.debug(f"[FINNHUB:TRADE] {len(out)} trades processed")

            if self.debug:
                self.debug_writer(
                    "stocks",
                    "trade_batch",
                    {"count": len(out)}
                )

            return out

        if msg_type == "error":
            err = data.get("msg", "Unknown error")
            self._logger.error(f"[FINNHUB:ERROR] {err}")
            raise RuntimeError(f"Finnhub error: {err}")

        self._logger.debug(f"[FINNHUB:OTHER] {data}")
        return []
