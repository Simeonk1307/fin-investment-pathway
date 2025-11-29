"""
Configuration module for Financial Knowledge Graph System
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class Neo4jConfig:
    """Neo4j database configuration"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"

@dataclass
class WikidataConfig:
    """Wikidata SPARQL endpoint configuration"""
    endpoint: str = "https://query.wikidata.org/sparql"
    timeout: int = 30
    max_results: int = 1000

@dataclass
class LLMConfig:
    """LLM configuration for news processing"""
    api_key: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 2000

@dataclass
class AppConfig:
    """Application configuration"""
    neo4j: Neo4jConfig = Neo4jConfig()
    wikidata: WikidataConfig = WikidataConfig()
    llm: LLMConfig = LLMConfig()
    country_filter: str = "India"
    log_level: str = "INFO"
