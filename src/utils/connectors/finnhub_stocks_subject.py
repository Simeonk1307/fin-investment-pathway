"""Finnhub WebSocket connector for real-time stock trades."""
import json
import aiohttp
from aiohttp.client_ws import ClientWebSocketResponse
from src.utils.connectors.base_websocket_subject import AIOHttpWebsocketSubject
from src.utils.event_envelope import create_event_envelope

class FinnhubSubject(AIOHttpWebsocketSubject):

    def __init__(self, api_key: str, symbols: list[str], **kwargs):
        super().__init__(f"wss://ws.finnhub.io?token={api_key}", **kwargs)
        self._symbols = symbols

    async def on_connect(self, ws: ClientWebSocketResponse):
        """Subscribe to symbols on connect."""
        for symbol in self._symbols:
            await ws.send_json({"type": "subscribe", "symbol": symbol})

    async def on_ws_message(self, msg: aiohttp.WSMessage, ws: ClientWebSocketResponse) -> list[dict]:
        """Process Finnhub messages and return trade data."""
        if msg.type != aiohttp.WSMsgType.TEXT:
            return []
        
        data = json.loads(msg.data)
        msg_type = data.get("type")
        
        if msg_type == "trade":
            k = data.get("data", [])
            if k == []:
                return []
            
            return [create_event_envelope(payload= i, source="finnhub", source_type="websocket") for i in k]
        elif msg_type == "error":
            raise RuntimeError(f"Finnhub error: {data.get('msg', 'Unknown')}")
        
        return []  # ping or unknown types