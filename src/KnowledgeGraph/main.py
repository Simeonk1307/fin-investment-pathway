"""
Main entry point for Financial Knowledge Graph System
"""
import argparse
import logging
from typing import Optional

from config import AppConfig
from kg_builder import WikidataKGBuilder
from news_agent import NewsProcessingAgent
from kg_queries import KGQueryService
from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialKGSystem:
    """Main system orchestrator"""

    def __init__(self, config: AppConfig):
        """Initialize the system"""
        self.config = config
        self.driver = None
        self.query_service: Optional[KGQueryService] = None
        self.news_agent: Optional[NewsProcessingAgent] = None
        self.kg_builder: Optional[WikidataKGBuilder] = None

    def initialize(self) -> None:
        """Initialize all components"""
        logger.info("Initializing Financial KG System...")

        # Connect to Neo4j
        self.driver = GraphDatabase.driver(
            self.config.neo4j.uri,
            auth=(self.config.neo4j.username, self.config.neo4j.password)
        )

        # Initialize services
        self.query_service = KGQueryService(self.driver)
        self.news_agent = NewsProcessingAgent(self.config)
        self.news_agent.driver = self.driver
        self.kg_builder = WikidataKGBuilder(self.config)
        self.kg_builder.driver = self.driver

        logger.info("System initialized successfully")

    def build_initial_kg(self) -> None:
        """Build initial knowledge graph from Wikidata"""
        logger.info("Building initial knowledge graph from Wikidata...")
        self.kg_builder.build_knowledge_graph()
        logger.info("Initial KG construction completed")

    def process_news(self, news_text: str, use_llm: bool = True) -> None:
        """Process a news article and update KG"""
        logger.info("Processing news article...")
        stats = self.news_agent.update_knowledge_graph(news_text, use_llm=use_llm)
        logger.info(f"News processing completed: {stats}")

    def query_example(self) -> None:
        """Run example queries"""
        logger.info("Running example queries...")

        # Example: Get CEOs of a company
        result = self.query_service.get_ceos_and_board_members("Tata Consultancy Services")
        logger.info(f"CEOs and Board Members: {result.data}")

        # Example: Find similar stocks
        result = self.query_service.find_similar_stocks("Infosys")
        logger.info(f"Similar stocks: {result.data}")

    def close(self) -> None:
        """Close all connections"""
        if self.driver:
            self.driver.close()
        if self.news_agent:
            self.news_agent.close()
        if self.kg_builder:
            self.kg_builder.close()
        logger.info("System shut down")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Financial Knowledge Graph System")
    parser.add_argument(
        "--action",
        choices=["build", "update", "query", "full"],
        default="full",
        help="Action to perform"
    )
    parser.add_argument(
        "--news-file",
        type=str,
        help="Path to news article file for update action"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for news processing"
    )

    args = parser.parse_args()

    # Initialize configuration
    config = AppConfig()
    system = FinancialKGSystem(config)

    try:
        system.initialize()

        if args.action in ["build", "full"]:
            system.build_initial_kg()

        if args.action == "update" and args.news_file:
            with open(args.news_file, 'r') as f:
                news_text = f.read()
            system.process_news(news_text, use_llm=args.use_llm)

        if args.action in ["query", "full"]:
            system.query_example()

    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
    finally:
        system.close()

if __name__ == "__main__":
    main()
