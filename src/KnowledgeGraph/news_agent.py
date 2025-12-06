"""
News Processing Agent - Updates KG from news articles using LLM
Extracts entities, relations, and events from financial news
"""
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase, Driver
import logging
import json
import re
from datetime import datetime
from dataclasses import dataclass

from src.KnowledgeGraph.models import Person, Company, Event, EventType, RelationType
from src.KnowledgeGraph.config import AppConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedTriple:
    """Represents an extracted knowledge triple"""
    subject: str
    subject_type: str  # Person or Company
    relation: str
    object: str
    object_type: str  # Person, Company, or Event
    confidence: float = 0.0

@dataclass
class ExtractedEvent:
    """Represents an extracted event"""
    event_type: EventType
    date: str
    description: str
    value: Optional[str] = None
    companies: List[str] = None
    persons: List[str] = None

    def __post_init__(self):
        if self.companies is None:
            self.companies = []
        if self.persons is None:
            self.persons = []

class NewsProcessingAgent:
    """Agent for processing news articles and updating knowledge graph"""

    def __init__(self, config: AppConfig):
        """
        Initialize news processing agent

        Args:
            config: Application configuration
        """
        self.config = config
        self.driver: Optional[Driver] = None
        self.llm_client = None  

    def connect_neo4j(self) -> None:
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j.uri,
                auth=(self.config.neo4j.username, self.config.neo4j.password)
            )
            logger.info("Connected to Neo4j successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self) -> None:
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def _extract_entities_basic(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Basic entity extraction using pattern matching
        Fallback when LLM is not available

        Args:
            text: News article text

        Returns:
            Tuple of (person_names, company_names)
        """
        persons = []
        companies = []

        company_patterns = [
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|Limited|Inc|Corp|Corporation|Pvt)',
            r'[A-Z][A-Z]+'
        ]

        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            companies.extend(matches)

        # Person pattern: Capitalized words (simple heuristic)
        person_pattern = r'(?:[A-Z][a-z]+\s+){1,3}[A-Z][a-z]+'
        potential_persons = re.findall(person_pattern, text)

        # Filter out companies from persons
        persons = [p for p in potential_persons if p not in companies]

        return list(set(persons)), list(set(companies))

    def _call_llm_for_extraction(self, text: str) -> Dict[str, Any]:
        """
        Use LLM to extract structured information from news text

        Args:
            text: News article text

        Returns:
            Dictionary containing extracted entities, relations, and events
        """
        prompt = f"""
You are a financial knowledge extraction system. Extract structured information from the following news article.

News Article:
{text}

Extract the following information in JSON format:
1. "persons": List of person names mentioned (e.g., CEOs, board members)
2. "companies": List of company names mentioned
3. "relations": List of relationships in format {{"person": "name", "relation": "IS_CEO_OF|IS_BOARD_MEMBER_OF|WORKS_FOR|FOUNDED", "company": "name"}}
4. "company_relations": List of company relationships {{"company1": "name", "relation": "IS_SUBSIDIARY_OF|HAS_PARTNERSHIP_WITH|IS_SUPPLIER_OF", "company2": "name"}}
5. "events": List of events {{"type": "Acquisition|Partnership|LeadershipChange|IPO|Merger", "date": "YYYY-MM-DD", "description": "brief desc", "value": "monetary value if mentioned", "companies": ["company names"], "persons": ["person names"]}}

Return only valid JSON. If no information found for a category, return empty list.

Example output:
{{
    "persons": ["John Smith", "Jane Doe"],
    "companies": ["Acme Corp", "TechStart Ltd"],
    "relations": [{{"person": "John Smith", "relation": "IS_CEO_OF", "company": "Acme Corp"}}],
    "company_relations": [{{"company1": "Acme Corp", "relation": "HAS_PARTNERSHIP_WITH", "company2": "TechStart Ltd"}}],
    "events": [{{"type": "Partnership", "date": "2025-11-21", "description": "Strategic partnership announced", "value": "100M USD", "companies": ["Acme Corp", "TechStart Ltd"], "persons": []}}]
}}

JSON:
"""

        try:
            # Placeholder for actual LLM API call
            # In production, use OpenAI, Anthropic, or local LLM
            # Example: response = openai.ChatCompletion.create(...)

            logger.info("Calling LLM for information extraction...")

            # Mock response for demonstration
            # In production, parse actual LLM response
            response_text = prompt  # Placeholder

            # Parse JSON from LLM response
            # json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            # if json_match:
            #     return json.loads(json_match.group())

            # Return empty structure as fallback
            return {
                "persons": [],
                "companies": [],
                "relations": [],
                "company_relations": [],
                "events": []
            }

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return {
                "persons": [],
                "companies": [],
                "relations": [],
                "company_relations": [],
                "events": []
            }

    def extract_information(self, text: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        Extract entities, relations, and events from news text

        Args:
            text: News article text
            use_llm: Whether to use LLM for extraction

        Returns:
            Dictionary with extracted information
        """
        if use_llm and self.config.llm.api_key:
            return self._call_llm_for_extraction(text)
        else:
            persons, companies = self._extract_entities_basic(text)
            return {
                "persons": persons,
                "companies": companies,
                "relations": [],
                "company_relations": [],
                "events": []
            }

    def _create_or_update_person(self, tx, name: str, title: Optional[str] = None) -> None:
        """Create or update person node"""
        query = """
        MERGE (p:Person {name: $name})
        """
        if title:
            query += " SET p.title = $title"
        tx.run(query, name=name, title=title)

    def _create_or_update_company(self, tx, name: str) -> None:
        """Create or update company node"""
        query = """
        MERGE (c:Company {name: $name})
        SET c.last_updated = datetime()
        """
        tx.run(query, name=name)

    def _create_event_node(self, tx, event: ExtractedEvent) -> int:
        """
        Create event node and return its ID

        Args:
            tx: Neo4j transaction
            event: Extracted event object

        Returns:
            Neo4j internal ID of created event
        """
        query = """
        CREATE (e:Event {
            type: $event_type,
            date: $date,
            description: $description,
            value: $value,
            created_at: datetime()
        })
        RETURN id(e) AS event_id
        """
        result = tx.run(query,
                       event_type=event.event_type.value,
                       date=event.date,
                       description=event.description,
                       value=event.value)
        record = result.single()
        return record["event_id"] if record else None

    def _link_person_company(self, tx, person_name: str, company_name: str, relation: str) -> None:
        """Create or update person-company relationship"""
        query = f"""
        MATCH (p:Person {{name: $person_name}})
        MATCH (c:Company {{name: $company_name}})
        MERGE (p)-[r:{relation}]->(c)
        SET r.last_updated = datetime()
        """
        tx.run(query, person_name=person_name, company_name=company_name)

    def _link_companies(self, tx, comp1_name: str, relation: str, comp2_name: str) -> None:
        """Create or update company-company relationship"""
        query = f"""
        MATCH (c1:Company {{name: $comp1_name}})
        MATCH (c2:Company {{name: $comp2_name}})
        MERGE (c1)-[r:{relation}]->(c2)
        SET r.last_updated = datetime()
        """
        tx.run(query, comp1_name=comp1_name, comp2_name=comp2_name)

    def _link_event_to_entities(self, tx, event_id: int, event: ExtractedEvent) -> None:
        """Link event node to companies and persons"""
        # Determine event-specific relations
        if event.event_type == EventType.ACQUISITION:
            if len(event.companies) >= 2:
                # First company is acquirer, second is target
                query = """
                MATCH (e:Event), (acq:Company {name: $acquirer}), (tgt:Company {name: $target})
                WHERE id(e) = $event_id
                MERGE (e)-[:HAS_ACQUIRER]->(acq)
                MERGE (e)-[:HAS_TARGET]->(tgt)
                """
                tx.run(query, event_id=event_id, 
                      acquirer=event.companies[0], 
                      target=event.companies[1])

        # Link persons involved
        for person in event.persons:
            query = """
            MATCH (e:Event), (p:Person {name: $person_name})
            WHERE id(e) = $event_id
            MERGE (e)-[:INVOLVES_PERSON]->(p)
            """
            tx.run(query, event_id=event_id, person_name=person)

    def update_knowledge_graph(self, news_text: str, use_llm: bool = True) -> Dict[str, int]:
        """
        Process news article and update knowledge graph

        Args:
            news_text: News article text
            use_llm: Whether to use LLM for extraction

        Returns:
            Dictionary with counts of created/updated entities
        """
        if not self.driver:
            self.connect_neo4j()

        logger.info("Processing news article...")

        # Extract information
        extracted = self.extract_information(news_text, use_llm=use_llm)

        stats = {
            "persons_created": 0,
            "companies_created": 0,
            "relations_created": 0,
            "events_created": 0
        }

        with self.driver.session() as session:
            # Create/update persons
            for person_name in extracted["persons"]:
                session.write_transaction(self._create_or_update_person, person_name)
                stats["persons_created"] += 1

            # Create/update companies
            for company_name in extracted["companies"]:
                session.write_transaction(self._create_or_update_company, company_name)
                stats["companies_created"] += 1

            # Create person-company relations
            for rel in extracted["relations"]:
                session.write_transaction(
                    self._link_person_company,
                    rel["person"],
                    rel["company"],
                    rel["relation"]
                )
                stats["relations_created"] += 1

            # Create company-company relations
            for rel in extracted["company_relations"]:
                session.write_transaction(
                    self._link_companies,
                    rel["company1"],
                    rel["relation"],
                    rel["company2"]
                )
                stats["relations_created"] += 1

            # Create events and link to entities
            for event_data in extracted["events"]:
                event = ExtractedEvent(
                    event_type=EventType[event_data["type"].upper()],
                    date=event_data["date"],
                    description=event_data["description"],
                    value=event_data.get("value"),
                    companies=event_data.get("companies", []),
                    persons=event_data.get("persons", [])
                )
                event_id = session.write_transaction(self._create_event_node, event)
                if event_id:
                    session.write_transaction(self._link_event_to_entities, event_id, event)
                    stats["events_created"] += 1

        logger.info(f"Knowledge graph updated: {stats}")
        return stats

    def process_news_batch(self, news_articles: List[str], use_llm: bool = True) -> Dict[str, int]:
        """
        Process multiple news articles in batch

        Args:
            news_articles: List of news article texts
            use_llm: Whether to use LLM

        Returns:
            Aggregated statistics
        """
        total_stats = {
            "persons_created": 0,
            "companies_created": 0,
            "relations_created": 0,
            "events_created": 0,
            "articles_processed": 0
        }

        for article in news_articles:
            try:
                stats = self.update_knowledge_graph(article, use_llm=use_llm)
                for key in stats:
                    total_stats[key] += stats[key]
                total_stats["articles_processed"] += 1
            except Exception as e:
                logger.error(f"Failed to process article: {e}")
                continue

        logger.info(f"Batch processing completed: {total_stats}")
        return total_stats

def main():
    """Main entry point for news agent"""
    config = AppConfig()
    agent = NewsProcessingAgent(config)

    try:
        # Example news article
        sample_news = """
        Mumbai, Nov 21, 2025 - Tata Consultancy Services (TCS) announced a strategic 
        partnership with Infosys Limited today. The deal, valued at $500 million, aims 
        to expand cloud services across India. TCS CEO Rajesh Gopinathan and Infosys 
        CEO Salil Parekh attended the signing ceremony.
        """

        # Process single article
        agent.update_knowledge_graph(sample_news, use_llm=False)

        # Example batch processing
        # news_batch = [article1, article2, article3]
        # agent.process_news_batch(news_batch)

    finally:
        agent.close()

if __name__ == "__main__":
    main()
