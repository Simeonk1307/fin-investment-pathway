import pathway as pw
import asyncio
from typing import List, Dict, Any
import datetime
from ..logger_config import get_module_logger
import os

import yfinance as yf
from ..schema.stock_schema import StockSchema

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
        self.logger.info(f"Initialized YFinanceStockConnector with tickers: {tickers}")
    
    
    def run(self):
        asyncio.run(self._async_run())
    
    #----HELPER FUNCTIONS ----#
    async def _async_run(self):
        count = 1
        while count < 5:
            try:
                async with yf.AsyncWebSocket() as ws:
                    self.logger.info("yFinance WebSocket connection established")
                    await ws.subscribe(self.tickers)
                    self.logger.info("yFinance WebSocket subscribed to tickers: {self.tickers}")

                    await ws.listen(self._message_handler)
                    self.logger.info("Listening on yFinance WebSocket")
                    
            except Exception as e:
                self.logger.error(f"yFinance WebSocket error: {e}")
                await asyncio.sleep(5)
                self.logger.info(f"Attempting to reconnect to yFinance for {count}th time...")
                count += 1
        
    def _message_handler(self, message: Dict[str, Any]):
        self.logger.info(f"Received message: {message}")
        parsed = self._parse_stock_data(message)
        if parsed:
            self.next(**parsed)
        else:
            self.logger.warning(f"Parsed data is None")     
    
    def _parse_stock_data(self, message: Dict) -> Dict[str, Any]:
        try:
            timestamp_ms = int(message.get("time", datetime.datetime.now().timestamp() * 1000 ))
            timestamp = pw.DateTimeNaive.fromtimestamp(timestamp_ms/1000)

            return {
                "timestamp": timestamp,
                "update_time":timestamp.strftime("%H:%M:%S"),
                "date": timestamp.strftime("%d-%m-%Y"),
                "symbol": message.get("id", ""),
                "volume": int(message.get("day_volume", 0)),
                "price": float(message.get("price", 0.0)),
                "change": float(message.get("change", 0.0)),
                "change_percent": float(message.get("change_percent", 0.0)),
            }
        except Exception as e:
            self.logger.error(f"Failed to parse: {e}")
            return None


# Usage
if __name__ == "__main__":
    output_path = "outputs/stock_data.csv"
    
    tickers = ["NVDA","MSFT","AAPL","GOOGL","AMZN","META","AVGO","TSLA",]
    connector = YFinanceStockConnector(
        tickers=tickers[:5]
    )

    stock_table = pw.io.python.read(
        connector, 
        schema=StockSchema,
        autocommit_duration_ms=1000
    )
    
    pw.io.csv.write(table=stock_table, filename=output_path)
    
    pw.run()