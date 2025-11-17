from config.settings import settings

api_conf = settings.load_api_config()
news_key = settings.NEWS_API_KEY

print(f"News API Key: {news_key}")
print(f"API Config: {api_conf}")

# try running python -m src.connectors.news_connector to see output
