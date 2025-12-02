import asyncio, datetime, os
from typing import List, Dict, Any
import yfinance as yf
from ..schemas.silver.stocks_schema import YFinanceEquitySchema
from ..config.logger_config import get_module_logger

import pathway as pw

class YFinanceStockConnector(pw.io.python.ConnectorSubject):
    # For more details refer to: https://ranaroussi.github.io/yfinance/reference/yfinance.websocket.html 
    """
    Asynchronous yFinance WebSocket connector for real-time stock prices
        yfinance AsyncWebSocket offers close(), subscribe(symbols), unsubscribe(symbols) and listen(message_handler) methods.
    """

    #---- MAIN FUNCTIONS ----#
    def __init__(self, tickers: List[str], logger_name):
        super().__init__() # for parent class initialization
        self.tickers = tickers
        self.logger = get_module_logger(logger_name)
        self.logger.info(f"Initialized Class with tickers: {tickers}")
    
    def run(self):
        asyncio.run(self._async_run())
    
    #----HELPER FUNCTIONS ----#
    async def _async_run(self):
        max_retry = 5
        while max_retry > 0:
            try:
                async with yf.AsyncWebSocket() as ws:
                    self.logger.info(f"WebSocket connection established")
                    await ws.subscribe(self.tickers)
                    self.logger.info(f"WebSocket subscribed to tickers: {self.tickers}")

                    await ws.listen(self._message_handler)
                    self.logger.info(f"Listening on WebSocket")
                    
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)
                self.logger.info(f"Will attempt to reconnect {max_retry} times")
                max_retry -= 1
        
    def _message_handler(self, msg: Dict[str, Any]):
        self.logger.info(f"Received message: {msg}")
        parsed = self._parse_stock_data(msg)
        if parsed:
            self.next(**parsed)
        else:
            self.logger.warning(f"Parsed data is None")     
    
    def _parse_stock_data(self, msg: Dict) -> Dict[str, Any]:
        try:
            time_ms = int(msg.get("time", datetime.datetime.now().timestamp() * 1000))
            dt = datetime.datetime.fromtimestamp(time_ms / 1000)

            data = {
                "ticker": msg.get("id", ""),
                "price": msg.get("price", 0.0),
                "time_ms": time_ms,
                "time_str": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "exchange": msg.get("exchange", ""),
                "quote_type": msg.get("quote_type", 0),
                "change_percent": msg.get("change_percent", 0.0),
                "change": msg.get("change", 0.0),
                "price_hint": msg.get("price_hint", 0),
            }
        except Exception as e:
            self.logger.error(f"Failed to parse: {e}")
            return None


# Usage
if __name__ == "__main__":
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)
    output_path = "outputs/stock_data.csv"
    
    equity_tickers = ["NVDA","MSFT","AAPL","GOOGL","AMZN","META","AVGO","TSLA"]
    connector = YFinanceStockConnector(
        tickers=equity_tickers[:5],
        logger_name="YFinanceStockConnector"
    )

    stock_table = pw.io.python.read(
        subject=connector, 
        schema=YFinanceEquitySchema,
        autocommit_duration_ms=1000,
        name="YFinanceEquityConnector",
        max_backlog_size=1000
    )
    
    pw.io.csv.write(table=stock_table, filename=output_path)
    
    pw.run()