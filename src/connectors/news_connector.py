import os
import pathway as pw
import requests
from datetime import date, datetime, timedelta
import time
from typing import Literal, Optional, Dict, Any,Callable,List
from abc import ABC, abstractmethod
import yaml
from config.settings import Settings
import pandas as pd

import finnhub
from ..schemas.news_schema import FinnHubNewsSchema,GNewsSchema
from ..logger_config import get_module_logger

import json



class BaseNewsConnector(pw.io.python.ConnectorSubject, ABC):

    def __init__(self, logger_name, poll_interval:int = 5):
        super().__init__()
        self.poll_interval = poll_interval
        self.logger = get_module_logger(logger_name)
        self.last_fetch_time = None
        self.seen_ids = set()

        self.logger.info(f"Initialized {self.__class__.__name__} with poll interval: {poll_interval} seconds")

    @abstractmethod
    def _fetch_articles(self) -> List[Dict[str, Any]]:
        """Fetch articles from API - to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _parse_article(self, article: Dict) -> Dict[str, str]:
        """Parse article into unified format - to be implemented by subclasses"""
        pass
    
    def run(self):
        self.logger.info(f"Starting {self.__class__.__name__}...")
        while True:
            try:
                articles = self._fetch_articles()
                if articles:
                    for article in articles:
                        parsed = self._parse_article(article)
                        article_id = parsed['id']
                        if article_id in self.seen_ids:
                            continue
                        self.seen_ids.add(article_id)

                        # need to verify the news article already present or not

                        self.next(**parsed)
                
                self.last_fetch_time = datetime.now()

                self.logger.info(f"Fetched and processed {len(articles)} articles. Pausing for {self.poll_interval} seconds...")

                time.sleep(self.poll_interval)

                self.logger.info("Resuming fetch...")
            
            except Exception as e:
                self.logger.warning(
                    f" Error:{e}"
                )
                time.sleep(10)  # wait before retrying

class FinnHubNewsConnector(BaseNewsConnector):


    def __init__(self,symbols : List[str],poll_interval: int = 120,lookback_days : int = 1, logger_name: str = "FinnHub_news_Connector"):

        super().__init__(poll_interval=poll_interval, logger_name=logger_name)

        self.api_key = Settings.APIKEYS.get("FINNHUB", "")

        self.finnhub_client = finnhub.Client(api_key=self.api_key)

        self.symbols = symbols

        self.lookback_days = lookback_days

        self.logger.info(f"Initialized FinnHubConnector for symbols: {symbols}")


    def _fetch_articles(self) -> List[Dict[str, Any]]:

        self.logger.info(f"Fetching articles for symbols: {self.symbols}")
        try:
            current_date = date.today()
            from_date = current_date - timedelta(days=self.lookback_days)
            all_articles = []
            for symbol in self.symbols:
                articles = self.finnhub_client.company_news(symbol, _from=from_date.strftime("%Y-%m-%d"), to=current_date.strftime("%Y-%m-%d"))
                self.logger.info(f"Fetched {len(articles)} articles for symbol: {symbol}")
                all_articles.extend(articles)
            return all_articles 
        except Exception as e:
            self.logger.error(f"Error fetching articles from FinnHub: {e}")
            return []
        
    def _parse_article(self, article: Dict) -> Dict[str, str]:

        # self.logger.info(f"Parsing article: {article.get('id', 'N/A')}")
        try:
            return {
                "id" : int(article.get('id', 0)),
                'headline': article.get('headline', 'N/A'),
                'description': article.get('summary', 'N/A'),
                'url': article.get('url', ''),
                'source': article.get('source', 'Unknown'),
                'published_at': datetime.fromtimestamp(article.get('datetime', 0)).strftime("%Y-%m-%dT%H:%M:%S"),
                'category': article.get('category', 'N/A'),
                'company': article.get('related', 'N/A'),
            }
        except Exception as e:
            self.logger.error(f"Error parsing article: {e}")
            return {}
    
    def get_past_news(self, output_df: bool = False) -> pd.DataFrame | pw.Table:# 1year past news
        current_date = date.today()
        from_date = current_date - timedelta(days=365) 
        all_articles = []
        for symbol in self.symbols:
            articles = self.finnhub_client.company_news(symbol, _from=from_date.strftime("%Y-%m-%d"), to=current_date.strftime("%Y-%m-%d"))
            all_articles.extend(articles)
            time.sleep(1)  # to avoid hitting rate limits

        parsed_articles = [self._parse_article(article) for article in all_articles]

        news_df = pd.DataFrame(parsed_articles)
        news_df.sort_values(by='published_at', ascending=False, inplace=True)
        # return news_table
        if output_df:
            return news_df
        else:
            news_table = pw.debug.table_from_pandas(news_df)
            return news_table


class GNewsConnector(BaseNewsConnector):

    def __init__(self,max_articles: int = 100, poll_interval: int = 300):
        super().__init__(poll_interval=poll_interval)
        
        self.base_url = "https://gnews.io/api/v4"

        self.config = {
            'language': 'en',
            'category': 'business',
            'lang': 'en',
            'country': 'us',
            'max': max_articles,
            'from': None,# can be set later
            'to': datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),#ISO 8601 format.
            'apikey': Settings.APIKEYS.get("GNEWS", ""),
        }

        self.max_articles = max_articles
    def _fetch_articles(self) -> List[Dict[str, Any]]:

        try:
            endpoint = f"{self.base_url}/top-headlines"
            params = self.config
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data.get('articles', [])
        except Exception as e:
            self.logger.error(f"Error fetching articles from GNews: {e}")
            return []
        
    def _parse_article(self, article: Dict) -> Dict[str, str]:

        source = article.get("source", {})
        article_id = article.get('id') or article.get('url', '').split('/')[-1]
        return {
            "id" : str(article_id),
            'headline': article.get('title', 'N/A'),
            'description': article.get('description', 'N/A'),
            'content': article.get('content', 'N/A'),
            'url': article.get('url', ''),
            'published_at': article.get('publishedAt', ''),
            'language': article.get('lang', 'en'),
            'source': source.get('name', 'Unknown'),
        }
    
    def get_past_news(self) -> List[Dict[str, str]]:
        "Gnews only give past 30 day news, 100 requests per day limit and  `max` ten articles per request for free API plan"
        pass


class AirbyteNewsConnector:
    NEWS_APIS = ['NEWSAPI', 'GNEWS']

    NewsSchema={
        "article_id":str,
        "title":str,
        "description": str,
        "content": str,
        "url": str,
        "published_at": str,
        "language": str,
        "source_name": str,
        "source_url": str,
    }

    def __init__(self, logger_name, api: Literal['NEWSAPI', 'GNEWS']):
        "Initialize news connector"
        self.api = api
        self.logger = get_module_logger(logger_name)
        self.config_path = self.temp_config_modifier()
        self.schema = self.NewsSchema

    def _normalize(self,article:dict)->dict:
        return {
            "article_id" : article.get("id",""),
            "title":article.get("title",""),
            "description": article.get("description", ""),
            "content": article.get("content", ""),
            "url": article.get("url", ""),
            "published_at": article.get("publishedAt", ""),
            "language": article.get("lang", "en"),
            "source_name": article.get("source", {}).get("name", ""),
            "source_url": article.get("source", {}).get("url", ""),
        }

    def temp_config_modifier(self):
        "Modify config file to insert API key"
        try:
            api_key = Settings.APIKEYS.get(self.api, "")

            config = Settings.load_api_config()
            
            api_config = config.get(self.api, {})
            api_config['source']['config']['api_key'] = api_key

            new_path = Settings.CONFIG_DIR / f"temp/temp_{self.api.lower()}_config.yaml"
            self.config_path = new_path

            with open(new_path, 'w') as f:
                yaml.dump(api_config, f)

            
            self.logger.info("✅ Config file updated with API key.")
            return new_path

        except Exception as e:
            self.logger.error(f"Error updating config file: {e}")
    
    @staticmethod
    @pw.udf
    def parse_newsapi_article(self, data : pw.Json)->dict:
        try:
            article :dict = json.loads(json.loads(str(data)))
            #example output 
                # dict_keys(['id', 'title', 'description', 'content', 'url', 'image', 'publishedAt', 'lang', 'source'])
                # 'source': {'id': 'd8ebc62fd4b923bed9968347f7c07cd8', 'name': 'FOX Sports', 'url': 'https://www.foxsports.com'}


            return {
                "article_id" : article.get("id",""),
                "title":article.get("title",""),
                "description": article.get("description", ""),
                "content": article.get("content", ""),
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", ""),
                "language": article.get("lang", "en"),
                "source_name": article.get("source", {}).get("name", ""),
                "source_url": article.get("source", {}).get("url", ""),
            }


        except Exception as e:
            self.logger.error(f"Error parsing NewsAPI article: {e}")

    def fetch_news(self, stream_name:str = "top_headlines",mode : str = 'static') ->pw.Table:

        news_table = pw.io.airbyte.read(
            config_file_path= self.config_path,
            streams = [stream_name],
            mode=mode,
            enforce_method="docker",
            refresh_interval_ms=50000# 5 seconds
        )


        parser= self.parse_newsapi_article

        parsed_table = news_table.with_columns(**{k:parser(pw.this.data)[k] for k in self.schema.keys()})

        return parsed_table
        
    
if __name__ == "__main__":
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)

    # connector = AirbyteNewsConnector(api="GNEWS")

    # newsapi_table = connector.fetch_news(stream_name="top_headlines",mode='static')

    # gnews_connector = GNewsConnector(max_articles=100,poll_interval=5) #10 seconds
    # newsapi_table = pw.io.python.read(gnews_connector, schema=GNewsSchema,autocommit_duration_ms=  1000)

    # pw.io.csv.write(table=newsapi_table,filename="outputs/gnews_data.csv")

    finn_connector = FinnHubNewsConnector(symbols=["AAPL","MSFT","GOOGL"],poll_interval=60,lookback_days=1) # 5 minutes


    finnhub_table = pw.io.python.read(
        finn_connector, 
        schema=FinnHubNewsSchema,
        autocommit_duration_ms=1000  # Commit every 1 second    
    )
    pw.io.csv.write(table=finnhub_table,filename="outputs/finnhub_news.csv")
    pw.run()

    # connector = FinnHubNewsConnector(symbols=["AAPL","MSFT","GOOGL"],poll_interval=60,lookback_days=1) # 5 minutes
    # past_news = connector.get_past_news(output_df = True)
    # past_news.to_csv("outputs/past_finnhub_news.csv", index=True)

    
            

