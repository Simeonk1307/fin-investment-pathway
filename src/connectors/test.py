from .news_connector import GNewsConnector, AirbyteNewsConnector, FinnHubNewsConnector
from .stock_connector import YFinanceStockConnector
from ..schemas.stock_schema import YFinanceSchema
from ..schemas.news_schema import FinnHubNewsSchema
import pathway as pw


if __name__ == "__main__":

    # interval = 5  # seconds
    # output_path = "news.csv"
    # news_connector = GNewsConnector(max_articles=100,poll_interval=interval) #10 seconds
    # newsapi_table = pw.io.python.read(news_connector, schema=news_connector.NewsSchema,autocommit_duration_ms= interval * 1000)

    # pw.io.csv.write(table=newsapi_table,filename=output_path)

    nvidia_stock_connector = YFinanceStockConnector(tickers=["NVDA"], logger_name="NVDA_Connector")
    stock_table = pw.io.python.read(
        nvidia_stock_connector, 
        schema=YFinanceSchema,
        autocommit_duration_ms=1000  # Commit every interval seconds
    )
    output_path_stock = "outputs/stock_data_nvda.csv"
    pw.io.csv.write(table=stock_table,filename=output_path_stock)

    #-----------------------------------------------------------------------------------------

    msft_stock_connector = YFinanceStockConnector(tickers=["MSFT"], logger_name="MSFT_Connector")
    stock_table_msft = pw.io.python.read(
        msft_stock_connector, 
        schema=YFinanceSchema,
        autocommit_duration_ms=1000  # Commit every interval seconds
    )
    output_path_stock_msft = "outputs/stock_data_msft.csv"
    pw.io.csv.write(table=stock_table_msft,filename=output_path_stock_msft)

    #-----------------------------------------------------------------------------------------

    googl_stock_connector = YFinanceStockConnector(tickers=["GOOGL"], logger_name="GOOGL_Connector")
    stock_table_googl = pw.io.python.read(
        googl_stock_connector, 
        schema=YFinanceSchema,
        autocommit_duration_ms=1000  # Commit every interval seconds
    )
    output_path_stock_googl = "outputs/stock_data_googl.csv"
    pw.io.csv.write(table=stock_table_googl,filename=output_path_stock_googl)

    #-----------------------------------------------------------------------------------------

    apple_stock_connector = YFinanceStockConnector(tickers=["AAPL"], logger_name="AAPL_Connector")
    stock_table_apple = pw.io.python.read(
        apple_stock_connector, 
        schema=YFinanceSchema,
        autocommit_duration_ms=1000  # Commit every interval seconds
    )
    output_path_stock_apple = "outputs/stock_data_apple.csv"
    pw.io.csv.write(table=stock_table_apple,filename=output_path_stock_apple)

    #-----------------------------------------------------------------------------------------


    finnhub_news_connector = FinnHubNewsConnector(symbols=["AAPL","MSFT","GOOGL","NVDA"],poll_interval=300,lookback_days=1) # 5 minutes
    
    news_table_finnhub = pw.io.python.read(
        finnhub_news_connector, 
        schema=FinnHubNewsSchema,
        autocommit_duration_ms= 1000  # Commit every interval seconds
    )
    output_path_news_finnhub = "outputs/finnhub_news.csv"
    pw.io.csv.write(table=news_table_finnhub,filename=output_path_news_finnhub)

    pw.run()


