"""Polygon WebSocket connector for real-time stock data."""
import json
import aiohttp
import datetime
from aiohttp.client_ws import ClientWebSocketResponse
from src.layers.bronze_layer.base.base_websocket_subject import AIOHttpWebsocketSubject
from src.layers.bronze_layer.event_envelope import create_event_envelope


class PolygonSubject(AIOHttpWebsocketSubject):

    def __init__(self, url: str, api_key: str, symbols: list[str], **kwargs):
        super().__init__(url, **kwargs)
        self._api_key = api_key
        self._symbols = symbols

    async def on_connect(self, ws: ClientWebSocketResponse):
        """Auth happens via message flow, not on connect."""
        pass

    async def on_ws_message(self, msg: aiohttp.WSMessage, ws: ClientWebSocketResponse) -> list[dict]:
        """
        Handle Polygon auth flow and data messages.
        Flow: connected → auth → auth_success → subscribe → data
        """
        if msg.type != aiohttp.WSMsgType.TEXT:
            return []
        
        result = []
        for obj in json.loads(msg.data):
            ev = obj.get("ev")
            status = obj.get("status")
            
            # Auth flow
            if ev == "status":
                if status == "connected":
                    await ws.send_json({"action": "auth", "params": self._api_key})
                elif status == "auth_success":
                    await ws.send_json({"action": "subscribe", "params": ",".join(self._symbols)})
                elif status == "error":
                    raise RuntimeError(f"Polygon error: {obj.get('message', 'Unknown')}")
            
            # Data events (A=aggregates, T=trades, Q=quotes)
            elif ev in ("A", "Q"):
                pass
            elif ev in ("T"):
                envelope = create_event_envelope(
                    payload=obj,
                    source="polygon",
                    source_type="websocket",
                )
                result.append(envelope)
        
        return result