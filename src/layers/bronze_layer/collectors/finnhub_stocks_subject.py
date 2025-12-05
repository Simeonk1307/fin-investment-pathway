import json
import aiohttp
from aiohttp.client_ws import ClientWebSocketResponse

from src.layers.bronze_layer.base.base_websocket_subject import AIOHttpWebsocketSubject
from src.layers.bronze_layer.event_envelope import create_event_envelope


class FinnhubSubject(AIOHttpWebsocketSubject):

    def __init__(self, api_key: str, symbols: list[str], logger=None, **kwargs):
        super().__init__(
            url=f"wss://ws.finnhub.io?token={api_key}",
            logger=logger,
            **kwargs
        )
        self._symbols = symbols
        self._logger.info(f"[FINNHUB INIT] Subscribing to symbols: {symbols}")

    # -------------------------------------------------------------------------
    # Startup subscription logic
    # -------------------------------------------------------------------------
    async def on_connect(self, ws: ClientWebSocketResponse):
        """Subscribe to trade channels on connection."""
        self._logger.info("[FINNHUB CONNECT] Connected — sending subscriptions...")

        try:
            for symbol in self._symbols:
                await ws.send_json({"type": "subscribe", "symbol": symbol})
                self._logger.info(f"[FINNHUB SUBSCRIBE] {symbol}")
        except Exception as e:
            self._logger.exception(f"[FINNHUB SUBSCRIBE ERROR] {e}")
            raise

        self._logger.info("[FINNHUB CONNECT] Subscription complete.")

    # -------------------------------------------------------------------------
    # Main Finnhub message handler
    # -------------------------------------------------------------------------
    async def on_ws_message(self, msg: aiohttp.WSMessage, ws: ClientWebSocketResponse) -> list[dict]:
        """Decode, validate, and format Finnhub trade messages."""
        # Unsupported message type
        if msg.type != aiohttp.WSMsgType.TEXT:
            self._logger.debug(f"[FINNHUB IGNORE] Non-text message type={msg.type}")
            return []

        try:
            data = json.loads(msg.data)
        except Exception as e:
            self._logger.error(f"[FINNHUB JSON ERROR] Could not decode: {msg.data}")
            raise

        msg_type = data.get("type")

        # ---------------------------------------------------------------------
        # Trade messages
        # ---------------------------------------------------------------------
        if msg_type == "trade":
            trades = data.get("data", [])

            if not trades:
                self._logger.debug("[FINNHUB TRADE] Empty trade batch.")
                return []

            if not isinstance(trades, list):
                self._logger.error(f"[FINNHUB TRADE ERROR] Invalid format: {trades}")
                return []

            out = []
            for trade in trades:
                if not isinstance(trade, dict):
                    self._logger.warning(f"[FINNHUB TRADE WARN] Skipping non-dict trade: {trade}")
                    continue

                try:
                    ev = create_event_envelope(
                        payload=trade,
                        source="finnhub",
                        source_type="websocket",
                    )
                    out.append(ev)
                except Exception as e:
                    self._logger.exception(f"[FINNHUB ENVELOPE ERROR] {e}")
                    continue

            self._logger.debug(f"[FINNHUB TRADE] Processed {len(out)} trades.")
            return out

        # ---------------------------------------------------------------------
        # Finnhub error packets
        # ---------------------------------------------------------------------
        elif msg_type == "error":
            err_msg = data.get("msg", "Unknown error")
            self._logger.error(f"[FINNHUB ERROR] {err_msg}")
            raise RuntimeError(f"Finnhub error: {err_msg}")

        # ---------------------------------------------------------------------
        # Heartbeats, pings, unknowns
        # ---------------------------------------------------------------------
        else:
            self._logger.debug(f"[FINNHUB OTHER] type={msg_type} raw={data}")
            return []  # no output to Pathway
