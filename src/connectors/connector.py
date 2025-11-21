from src.connectors.news_connector import GNewsConnector, AirbyteNewsConnector
from src.connectors.stock_connector import YFinanceStockConnector 
import pathway as pw


if __name__ == "__main__":

    # interval = 5  # seconds
    # output_path = "news.csv"
    # news_connector = GNewsConnector(max_articles=100,poll_interval=interval) #10 seconds
    # newsapi_table = pw.io.python.read(news_connector, schema=news_connector.NewsSchema,autocommit_duration_ms= interval * 1000)

    # pw.io.csv.write(table=newsapi_table,filename=output_path)

    nvidia_stock_connector = YFinanceStockConnector(symbols=["NVDA"])
    stock_table = pw.io.python.read(
        nvidia_stock_connector, 
        schema=nvidia_stock_connector.StockSchema,
        autocommit_duration_ms=1000  # Commit every interval seconds
    )
    output_path_stock = "stock_data_nvda.csv"
    pw.io.csv.write(table=stock_table,filename=output_path_stock)

    msft_stock_connector = YFinanceStockConnector(symbols=["MSFT"])
    stock_table_msft = pw.io.python.read(
        msft_stock_connector, 
        schema=msft_stock_connector.StockSchema,
        autocommit_duration_ms=1000  # Commit every interval seconds
    )
    output_path_stock_msft = "stock_data_msft.csv"
    pw.io.csv.write(table=stock_table_msft,filename=output_path_stock_msft)

    googl_stock_connector = YFinanceStockConnector(symbols=["GOOGL"])
    stock_table_googl = pw.io.python.read(
        googl_stock_connector, 
        schema=googl_stock_connector.StockSchema,
        autocommit_duration_ms=1000  # Commit every interval seconds
    )
    output_path_stock_googl = "stock_data_googl.csv"
    pw.io.csv.write(table=stock_table_googl,filename=output_path_stock_googl)

    pw.run()


