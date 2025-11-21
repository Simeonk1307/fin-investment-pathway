import pathway as pw
import requests
from datetime import datetime
import time
from typing import Literal, Optional, Dict, Any,Callable,List
import logging
from abc import ABC, abstractmethod
import yaml
from config.settings import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import json



class BaseNewsConnector(pw.io.python.ConnectorSubject, ABC):

    class NewsSchema(pw.Schema):
        article_id: str
        title: str
        description: str
        content: str
        url: str
        published_at: str
        language: str
        source_name: str
        source_url: str

    def __init__(self,poll_interval:int = 300,max_articles:int = 100):
        super().__init__()
        self.poll_interval = poll_interval
        self.max_articles = max_articles
        self.last_fetch_time = None
        self.seen_ids = set()

    @abstractmethod
    def _fetch_articles(self) -> List[Dict[str, Any]]:
        """Fetch articles from API - to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _parse_article(self, article: Dict) -> Dict[str, str]:
        """Parse article into unified format - to be implemented by subclasses"""
        pass
    
    def run(self):
        logger.info(f"Starting {self.__class__.__name__}...")
        while True:
            try:
                articles = self._fetch_articles()
                if articles:
                    for article in articles:
                        parsed = self._parse_article(article)
                        article_id = parsed['article_id']
                        if article_id in self.seen_ids:
                            continue
                        self.seen_ids.add(article_id)

                        # need to verify the news article already present or not

                        self.next(**parsed)
                
                self.last_fetch_time = datetime.now()

                time.sleep(self.poll_interval)
            
            except Exception as e:
                logger.warning(
                    f" Error:{e}"
                )


class GNewsConnector(BaseNewsConnector):

    def __init__(self,max_articles: int = 100, poll_interval: int = 300):
        super().__init__(poll_interval=poll_interval, max_articles=max_articles)
        
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

    def _fetch_articles(self) -> List[Dict[str, Any]]:

        try:
            endpoint = f"{self.base_url}/top-headlines"
            params = self.config
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data.get('articles', [])
        except Exception as e:
            logger.error(f"Error fetching articles from GNews: {e}")
            return []
        
    def _parse_article(self, article: Dict) -> Dict[str, str]:

        source = article.get("source", {})
        article_id = article.get('id') or article.get('url', '').split('/')[-1]
        return {
            "article_id" : str(article_id),
            'title': article.get('title', 'N/A'),
            'description': article.get('description', 'N/A'),
            'content': article.get('content', 'N/A'),
            'url': article.get('url', ''),
            'published_at': article.get('publishedAt', ''),
            'language': article.get('lang', 'en'),
            'source_name': source.get('name', 'Unknown'),
            'source_url': source.get('url', ''),
        }
    
    def get_past_news(self) -> List[Dict[str, str]]:
        "Gnews Only give past 30 day news for free api plan also 100 requests per day limit also 100 `max` articles per request"
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

    def __init__(self, api: Literal['NEWSAPI', 'GNEWS']):
        "Initialize news connector"
        self.api = api
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

            
            logger.info("✅ Config file updated with API key.")
            return new_path

        except Exception as e:
            logger.error(f"Error updating config file: {e}")
    
    @staticmethod
    @pw.udf
    def parse_newsapi_article(data : pw.Json)->dict:
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
            logger.error(f"Error parsing NewsAPI article: {e}")

    def fetch_news(self,stream_name:str = "top_headlines",mode : str = 'static') ->pw.Table:

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

    output_path = "news.csv"

    # connector = AirbyteNewsConnector(api="GNEWS")

    # newsapi_table = connector.fetch_news(stream_name="top_headlines",mode='static')
    interval = 5  # seconds

    connector = GNewsConnector(max_articles=100,poll_interval=interval) #10 seconds
    newsapi_table = pw.io.python.read(connector, schema=connector.NewsSchema,autocommit_duration_ms= interval * 1000)

    # pw.debug.compute_and_print(newsapi_table, include_id=False) # only for static mode

    # Write to CSV
    pw.io.csv.write(table=newsapi_table,filename=output_path)

    pw.run()
    logger.info(f"✅ Written structured data to {output_path}")

        

    
            

