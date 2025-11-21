import pathway as pw
import asyncio
from typing import List, Dict, Any
import logging
import datetime


import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YFinanceStockConnector(pw.io.python.ConnectorSubject):
    #----test-version-- try running in separate jupyter notebook cell

    #import yfinance as yf

    # def message_handler(message):
    #     print("Received message:", message)

    # # Synchronous
    # with yf.WebSocket() as ws:
    #     ws.subscribe(["AAPL", "GOOGL"])
    #     ws.listen(message_handler)

    # # Asynchronous
    # async def main():
    #     async with yf.AsyncWebSocket() as ws:
    #         await ws.subscribe(["AAPL", "GOOGL"])
    #         await ws.listen()

    # asyncio.run(main())
    #----test-version-end--
    """
    Asynchronous yFinance WebSocket connector for real-time stock prices
    """
    
    class StockSchema(pw.Schema):
        update_time:str
        timestamp: pw.DateTimeNaive
        date: str
        symbol: str
        price: float
        change: float
        change_percent: float
        volume: int

    
    def __init__(self, symbols: List[str]):
        super().__init__()
        self.symbols = symbols
        logger.info(f"Initialized YFinanceStockConnector with symbols: {symbols}")
    
    def run(self):
        asyncio.run(self._async_run())
    
    async def _async_run(self):
        
        try:
            async with yf.AsyncWebSocket() as ws:
                await ws.subscribe(self.symbols)
                
                # Listen for messages
                await ws.listen(self._on_message)
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
    
    def _on_message(self, message: Dict[str, Any]):

        try:
            # yFinance message structure (example):
            #{'id': 'GOOGL', 'price': 295.02, 'time': '1763739554000', 'exchange': 'NMS', 'quote_type': 8, 
            # 'market_hours': 1, 'change_percent': 1.9243312, 'day_volume': '23172292', 'change': 5.569977,
            # 'last_size': '300', 'price_hint': '2'}
            
            parsed = self._parse_stock_data(message)
            if parsed:
                # Send to Pathway buffer
                self.next(**parsed)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
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
            logger.error(f"Parse error: {e}")
            return None


# Usage
if __name__ == "__main__":
    output_path = "stock_data.csv"
    
    # Create connector
    tickers = ["NVDA","MSFT","AAPL","GOOGL","AMZN","META","AVGO","TSLA",]
    connector = YFinanceStockConnector(
        symbols=tickers[:2]# Using only first 2 for testing purposes
    )
    
    # Create Pathway table
    stock_table = pw.io.python.read(
        connector, 
        schema=connector.StockSchema,
        autocommit_duration_ms=1000  # Commit every 1 second
    )
    
    # Write to CSV
    pw.io.csv.write(table=stock_table, filename=output_path)
    
    # Run the pipeline
    pw.run()
    
    logger.info(f"✅ Stock data written to {output_path}")