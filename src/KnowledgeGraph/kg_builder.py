"""
Knowledge Graph Builder - Constructs initial KG from Wikidata
Focused on Indian companies and executives
"""
from typing import List, Dict, Any, Optional
from SPARQLWrapper import SPARQLWrapper, JSON
from neo4j import GraphDatabase, Driver
import logging
from dataclasses import asdict

from models import Person, Company, RelationType
from config import AppConfig, Neo4jConfig, WikidataConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikidataKGBuilder:
    """Builds initial knowledge graph from Wikidata for Indian companies"""

    def __init__(self, config: AppConfig):
        """
        Initialize builder with configuration

        Args:
            config: Application configuration
        """
        self.config = config
        self.sparql = SPARQLWrapper(config.wikidata.endpoint)
        self.sparql.setTimeout(config.wikidata.timeout)
        self.driver: Optional[Driver] = None

    def connect_neo4j(self) -> None:
        """Establish connection to Neo4j database"""
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

    def _run_sparql_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute SPARQL query against Wikidata

        Args:
            query: SPARQL query string

        Returns:
            List of result bindings
        """
        try:
            self.sparql.setQuery(query)
            self.sparql.setReturnFormat(JSON)
            results = self.sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            return []

    def fetch_indian_companies(self) -> List[Company]:
        """
        Fetch Indian companies from Wikidata

        Returns:
            List of Company objects
        """
        query = f"""
        SELECT DISTINCT ?company ?companyLabel ?ticker ?industryLabel ?headquarters ?foundedDate
        WHERE {{
          ?company wdt:P31/wdt:P279* wd:Q4830453;  # instance of/subclass of business
                   wdt:P17 wd:Q668.                 # country: India

          OPTIONAL {{ ?company wdt:P414 ?ticker. }}     # stock ticker
          OPTIONAL {{ ?company wdt:P452 ?industry. }}   # industry
          OPTIONAL {{ ?company wdt:P159 ?hq. }}         # headquarters
          OPTIONAL {{ ?company wdt:P571 ?foundedDate. }} # inception

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {self.config.wikidata.max_results}
        """

        logger.info("Fetching Indian companies from Wikidata...")
        results = self._run_sparql_query(query)

        companies = []
        for entry in results:
            try:
                company = Company(
                    name=entry.get('companyLabel', {}).get('value', ''),
                    ticker=entry.get('ticker', {}).get('value'),
                    industry=entry.get('industryLabel', {}).get('value'),
                    country="India",
                    wikidata_uri=entry.get('company', {}).get('value'),
                    properties={
                        'headquarters': entry.get('headquarters', {}).get('value'),
                        'founded_date': entry.get('foundedDate', {}).get('value')
                    }
                )
                if company.name:  # Only add if name exists
                    companies.append(company)
            except Exception as e:
                logger.warning(f"Error parsing company entry: {e}")
                continue

        logger.info(f"Fetched {len(companies)} Indian companies")
        return companies

    def fetch_indian_executives(self) -> List[tuple[Person, str, str]]:
        """
        Fetch Indian business executives and their company affiliations

        Returns:
            List of tuples (Person, company_name, relation_type)
        """
        query = f"""
        SELECT DISTINCT ?person ?personLabel ?positionLabel ?companyLabel ?company
        WHERE {{
          ?person wdt:P31 wd:Q5;              # instance of human
                  wdt:P27 wd:Q668;            # citizen of India
                  wdt:P39 ?position.          # position held

          ?person wdt:P108 ?company.          # employer
          ?company wdt:P17 wd:Q668.           # company in India

          # Filter for executive positions
          VALUES ?position {{ wd:Q484876 wd:Q1162163 wd:Q3242115 }}  # CEO, Director, Board Member

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {self.config.wikidata.max_results}
        """

        logger.info("Fetching Indian executives from Wikidata...")
        results = self._run_sparql_query(query)

        executives = []
        for entry in results:
            try:
                person = Person(
                    name=entry.get('personLabel', {}).get('value', ''),
                    title=entry.get('positionLabel', {}).get('value'),
                    wikidata_uri=entry.get('person', {}).get('value')
                )
                company_name = entry.get('companyLabel', {}).get('value', '')
                position = entry.get('positionLabel', {}).get('value', '').lower()

                # Map position to relation type
                if 'ceo' in position or 'chief executive' in position:
                    relation = RelationType.IS_CEO_OF.value
                elif 'board' in position or 'director' in position:
                    relation = RelationType.IS_BOARD_MEMBER_OF.value
                else:
                    relation = RelationType.WORKS_FOR.value

                if person.name and company_name:
                    executives.append((person, company_name, relation))
            except Exception as e:
                logger.warning(f"Error parsing executive entry: {e}")
                continue

        logger.info(f"Fetched {len(executives)} Indian executives")
        return executives

    def fetch_company_relationships(self) -> List[tuple[str, str, str]]:
        """
        Fetch relationships between Indian companies (subsidiaries, partnerships)

        Returns:
            List of tuples (company1_name, relation_type, company2_name)
        """
        query = f"""
        SELECT DISTINCT ?company1Label ?company2Label ?relationLabel
        WHERE {{
          ?company1 wdt:P17 wd:Q668.  # Indian company
          ?company2 wdt:P17 wd:Q668.  # Indian company

          {{
            ?company1 wdt:P749 ?company2.  # parent organization
            BIND("IS_SUBSIDIARY_OF" AS ?relationLabel)
          }}
          UNION
          {{
            ?company1 wdt:P1830 ?company2.  # owner of
            BIND("IS_SUBSIDIARY_OF" AS ?relationLabel)
          }}

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {self.config.wikidata.max_results}
        """

        logger.info("Fetching company relationships from Wikidata...")
        results = self._run_sparql_query(query)

        relationships = []
        for entry in results:
            try:
                comp1 = entry.get('company1Label', {}).get('value', '')
                comp2 = entry.get('company2Label', {}).get('value', '')
                relation = entry.get('relationLabel', {}).get('value', '')

                if comp1 and comp2 and relation:
                    relationships.append((comp1, relation, comp2))
            except Exception as e:
                logger.warning(f"Error parsing relationship entry: {e}")
                continue

        logger.info(f"Fetched {len(relationships)} company relationships")
        return relationships

    def _create_company_node(self, tx, company: Company) -> None:
        """Create or merge a company node in Neo4j"""
        query = """
        MERGE (c:Company {name: $name})
        SET c.ticker = $ticker,
            c.industry = $industry,
            c.country = $country,
            c.wikidata = $wikidata_uri,
            c.headquarters = $headquarters,
            c.founded_date = $founded_date
        """
        tx.run(query, 
               name=company.name,
               ticker=company.ticker,
               industry=company.industry,
               country=company.country,
               wikidata_uri=company.wikidata_uri,
               headquarters=company.properties.get('headquarters'),
               founded_date=company.properties.get('founded_date'))

    def _create_person_node(self, tx, person: Person) -> None:
        """Create or merge a person node in Neo4j"""
        query = """
        MERGE (p:Person {name: $name})
        SET p.title = $title,
            p.wikidata = $wikidata_uri
        """
        tx.run(query,
               name=person.name,
               title=person.title,
               wikidata_uri=person.wikidata_uri)

    def _create_person_company_relation(self, tx, person_name: str, company_name: str, relation: str) -> None:
        """Create relationship between person and company"""
        query = f"""
        MATCH (p:Person {{name: $person_name}})
        MATCH (c:Company {{name: $company_name}})
        MERGE (p)-[r:{relation}]->(c)
        """
        tx.run(query, person_name=person_name, company_name=company_name)

    def _create_company_relation(self, tx, comp1_name: str, relation: str, comp2_name: str) -> None:
        """Create relationship between companies"""
        query = f"""
        MATCH (c1:Company {{name: $comp1_name}})
        MATCH (c2:Company {{name: $comp2_name}})
        MERGE (c1)-[r:{relation}]->(c2)
        """
        tx.run(query, comp1_name=comp1_name, comp2_name=comp2_name)

    def build_knowledge_graph(self) -> None:
        """
        Build the complete knowledge graph from Wikidata
        Main orchestration method
        """
        if not self.driver:
            self.connect_neo4j()

        logger.info("Starting KG construction from Wikidata (India focus)...")

        # Step 1: Fetch and create companies
        companies = self.fetch_indian_companies()
        with self.driver.session() as session:
            for company in companies:
                session.write_transaction(self._create_company_node, company)
        logger.info(f"Created {len(companies)} company nodes")

        # Step 2: Fetch and create executives with relationships
        executives = self.fetch_indian_executives()
        with self.driver.session() as session:
            for person, company_name, relation in executives:
                session.write_transaction(self._create_person_node, person)
                session.write_transaction(self._create_person_company_relation, 
                                        person.name, company_name, relation)
        logger.info(f"Created {len(executives)} person-company relationships")

        # Step 3: Fetch and create company relationships
        relationships = self.fetch_company_relationships()
        with self.driver.session() as session:
            for comp1, relation, comp2 in relationships:
                session.write_transaction(self._create_company_relation, comp1, relation, comp2)
        logger.info(f"Created {len(relationships)} company-company relationships")

        logger.info("Knowledge graph construction completed successfully!")

def main():
    """Main entry point for KG builder"""
    # Initialize configuration
    config = AppConfig()
    config.country_filter = "India"

    # Build knowledge graph
    builder = WikidataKGBuilder(config)
    try:
        builder.build_knowledge_graph()
    finally:
        builder.close()

if __name__ == "__main__":
    main()
