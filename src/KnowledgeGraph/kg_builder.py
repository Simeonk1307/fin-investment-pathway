"""
Knowledge Graph Builder - Constructs KG from Wikidata
(Default: US companies and executives)
"""
from typing import List, Dict, Any, Optional
from SPARQLWrapper import SPARQLWrapper, JSON
from neo4j import GraphDatabase, Driver
import logging

from src.KnowledgeGraph.models import Person, Company, RelationType
from src.KnowledgeGraph.config import AppConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default country = United States (Q30)
COUNTRY_QID = "Q30"


class WikidataKGBuilder:
    """Builds initial knowledge graph from Wikidata."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.sparql = SPARQLWrapper(config.wikidata.endpoint)
        self.sparql.setTimeout(config.wikidata.timeout)
        self.driver: Optional[Driver] = None

    # ------------------------------------------------------
    # Neo4j Connection
    # ------------------------------------------------------
    def connect_neo4j(self):
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j.uri,
                auth=(self.config.neo4j.username, self.config.neo4j.password)
            )
            logger.info("Connected to Neo4j.")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()

    # ------------------------------------------------------
    # SPARQL Helper
    # ------------------------------------------------------
    def _run_sparql_query(self, query: str) -> List[Dict[str, Any]]:
        try:
            self.sparql.setQuery(query)
            self.sparql.setReturnFormat(JSON)
            output = self.sparql.query().convert()
            return output["results"]["bindings"]
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            return []

    # ------------------------------------------------------
    # COMPANY FETCHER (Generic Name)
    # ------------------------------------------------------
    def fetch_companies(self) -> List[Company]:
        """
        Fetch companies for configured country (default: US)
        """
        query = f"""
        SELECT DISTINCT ?company ?companyLabel ?ticker ?industryLabel ?headquarters ?foundedDate
        WHERE {{
          ?company wdt:P31/wdt:P279* wd:Q4830453;
                   wdt:P17 wd:{COUNTRY_QID}.

          OPTIONAL {{
            ?company p:P414 ?exchangeStmt.
            ?exchangeStmt ps:P414 ?exchange.
            ?exchangeStmt pq:P249 ?ticker.
          }}

          OPTIONAL {{ ?company wdt:P452 ?industry. }}
          OPTIONAL {{ ?company wdt:P159 ?headquarters. }}
          OPTIONAL {{ ?company wdt:P571 ?foundedDate. }}

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {self.config.wikidata.max_results}
        """

        logger.info("Fetching companies from Wikidata...")
        results = self._run_sparql_query(query)

        companies: List[Company] = []
        for e in results:
            try:
                c = Company(
                    name=e.get("companyLabel", {}).get("value", ""),
                    ticker=e.get("ticker", {}).get("value"),
                    industry=e.get("industryLabel", {}).get("value"),
                    country="United States",
                    wikidata_uri=e.get("company", {}).get("value"),
                    properties={
                        "headquarters": e.get("headquarters", {}).get("value"),
                        "founded_date": e.get("foundedDate", {}).get("value"),
                    }
                )
                if c.name:
                    companies.append(c)
            except Exception:
                continue

        logger.info(f"Fetched {len(companies)} companies.")
        return companies

    # ------------------------------------------------------
    # EXECUTIVE FETCHER (Generic Name)
    # ------------------------------------------------------
    def fetch_executives(self) -> List[tuple[Person, str, str]]:
        """
        Fetch executives and their company associations.
        """
        query = f"""
        SELECT DISTINCT ?person ?personLabel ?positionLabel ?companyLabel ?company
        WHERE {{
          ?person wdt:P31 wd:Q5;
                  wdt:P27 wd:{COUNTRY_QID};
                  wdt:P39 ?position.

          ?person wdt:P108 ?company.
          ?company wdt:P17 wd:{COUNTRY_QID}.

          # Executive-type positions
          VALUES ?position {{ wd:Q484876 wd:Q1162163 wd:Q3242115 }}

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {self.config.wikidata.max_results}
        """

        logger.info("Fetching executives...")
        results = self._run_sparql_query(query)

        execs = []
        for e in results:
            try:
                person = Person(
                    name=e.get("personLabel", {}).get("value", ""),
                    title=e.get("positionLabel", {}).get("value", ""),
                    wikidata_uri=e.get("person", {}).get("value"),
                )

                company_name = e.get("companyLabel", {}).get("value", "")
                pos = person.title.lower()

                # Determine relation
                if "ceo" in pos:
                    relation = RelationType.IS_CEO_OF.value
                elif "board" in pos or "director" in pos:
                    relation = RelationType.IS_BOARD_MEMBER_OF.value
                else:
                    relation = RelationType.WORKS_FOR.value

                execs.append((person, company_name, relation))

            except Exception:
                continue

        logger.info(f"Fetched {len(execs)} executives.")
        return execs

    # ------------------------------------------------------
    # COMPANY RELATIONSHIPS (Generic Name)
    # ------------------------------------------------------
    def fetch_company_relationships(self) -> List[tuple[str, str, str]]:
        """
        Fetch inter-company relationships (subsidiaries etc.)
        """
        query = f"""
        SELECT DISTINCT ?company1Label ?company2Label ?relationLabel
        WHERE {{
          ?company1 wdt:P17 wd:{COUNTRY_QID}.
          ?company2 wdt:P17 wd:{COUNTRY_QID}.

          {{
            ?company1 wdt:P749 ?company2.
            BIND("IS_SUBSIDIARY_OF" AS ?relationLabel)
          }}
          UNION
          {{
            ?company1 wdt:P1830 ?company2.
            BIND("IS_SUBSIDIARY_OF" AS ?relationLabel)
          }}

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {self.config.wikidata.max_results}
        """

        logger.info("Fetching company relationships...")
        results = self._run_sparql_query(query)

        rels = []
        for e in results:
            c1 = e.get("company1Label", {}).get("value", "")
            c2 = e.get("company2Label", {}).get("value", "")
            rel = e.get("relationLabel", {}).get("value", "")
            if c1 and c2:
                rels.append((c1, rel, c2))

        logger.info(f"Fetched {len(rels)} relationships.")
        return rels

    # ------------------------------------------------------
    # Neo4j INSERTION (unchanged)
    # ------------------------------------------------------
    def _create_company_node(self, tx, company: Company):
        tx.run(
            """
            MERGE (c:Company {name: $name})
            SET c.ticker = $ticker,
                c.industry = $industry,
                c.country = $country,
                c.wikidata = $wikidata,
                c.headquarters = $headquarters,
                c.founded_date = $founded_date
            """,
            name=company.name,
            ticker=company.ticker,
            industry=company.industry,
            country=company.country,
            wikidata=company.wikidata_uri,
            headquarters=company.properties.get("headquarters"),
            founded_date=company.properties.get("founded_date")
        )

    def _create_person_node(self, tx, person: Person):
        tx.run(
            """
            MERGE (p:Person {name: $name})
            SET p.title = $title,
                p.wikidata = $wikidata
            """,
            name=person.name,
            title=person.title,
            wikidata=person.wikidata_uri
        )

    def _create_person_company_relation(self, tx, pname, cname, rel):
        tx.run(
            f"""
            MATCH (p:Person {{name: $pname}})
            MATCH (c:Company {{name: $cname}})
            MERGE (p)-[:{rel}]->(c)
            """,
            pname=pname,
            cname=cname
        )

    def _create_company_relation(self, tx, c1, relation, c2):
        tx.run(
            f"""
            MATCH (a:Company {{name: $c1}})
            MATCH (b:Company {{name: $c2}})
            MERGE (a)-[:{relation}]->(b)
            """,
            c1=c1,
            c2=c2
        )

    # ------------------------------------------------------
    # BUILD FINAL GRAPH
    # ------------------------------------------------------
    def build_knowledge_graph(self):
        if not self.driver:
            self.connect_neo4j()

        logger.info("Building Knowledge Graph...")

        companies = self.fetch_companies()
        execs = self.fetch_executives()
        relations = self.fetch_company_relationships()

        with self.driver.session() as session:
            for c in companies:
                session.write_transaction(self._create_company_node, c)

            for p, cname, rel in execs:
                session.write_transaction(self._create_person_node, p)
                session.write_transaction(self._create_person_company_relation, p.name, cname, rel)

            for c1, rel, c2 in relations:
                session.write_transaction(self._create_company_relation, c1, rel, c2)

        logger.info("Knowledge Graph build complete!")

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
