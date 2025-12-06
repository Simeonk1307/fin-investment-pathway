"""
Knowledge Graph Builder - Constructs initial KG from Wikidata
Focused on public companies + executives for a given country (default: USA).
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import logging
import time
import re

from SPARQLWrapper import SPARQLWrapper, JSON
from neo4j import GraphDatabase, Driver

from src.KnowledgeGraph.models import Person, Company, RelationType
from src.KnowledgeGraph.config import AppConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Known country QIDs
COUNTRY_QIDS: Dict[str, str] = {
    "United States": "Q30",
    "India": "Q668",
    "United Kingdom": "Q145",
    "Germany": "Q183",
    "France": "Q142",
    "Japan": "Q17",
    "China": "Q148",
    "Canada": "Q16",
    "Australia": "Q408",
    "Brazil": "Q155",
    "Singapore": "Q334",
    "Switzerland": "Q39",
    "Netherlands": "Q55",
    "South Korea": "Q884",
    "Italy": "Q38",
    "Spain": "Q29",
    "Mexico": "Q96",
    "Israel": "Q801",
    "Sweden": "Q34",
}


class WikidataKGBuilder:
    """
    Builds a knowledge graph of:
      - Public companies in a given country
      - Their key executives (CEO, chairperson, board members, founders)
      - Company–company relationships (subsidiary)
    """

    QID_PATTERN = re.compile(r"^Q\d+$")

    def __init__(self, config: AppConfig, country_name: str = "United States") -> None:
        self.config = config

        if country_name not in COUNTRY_QIDS:
            raise ValueError(
                f"Unknown country '{country_name}'. "
                f"Available: {', '.join(COUNTRY_QIDS.keys())}"
            )
        self.country_name = country_name
        self.country_qid = COUNTRY_QIDS[country_name]

        self.sparql = SPARQLWrapper(config.wikidata.endpoint)
        self.sparql.setTimeout(config.wikidata.timeout)

        self.driver: Optional[Driver] = None

        self._seen_companies: Set[str] = set()
        self._company_name_index: Dict[str, str] = {}
        self._seen_person_company_keys: Set[str] = set()

        logger.info(
            "Initialized WikidataKGBuilder for %s (QID=%s)",
            self.country_name,
            self.country_qid,
        )

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _is_valid_label(self, label: Optional[str]) -> bool:
        if not label:
            return False
        label = label.strip()
        if not label:
            return False
        if self.QID_PATTERN.match(label):
            return False
        if label.isdigit():
            return False
        if len(label) < 2:
            return False
        return True

    def _normalize_name(self, name: str) -> str:
        return re.sub(r"\s+", " ", name.strip()).lower()

    # --------------------------------------------------------------------- #
    # Neo4j connection
    # --------------------------------------------------------------------- #
    def connect_neo4j(self) -> None:
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j.uri,
                auth=(self.config.neo4j.username, self.config.neo4j.password),
            )
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j successfully.")
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", e)
            raise

    def close(self) -> None:
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")

    # --------------------------------------------------------------------- #
    # SPARQL helper with retry
    # --------------------------------------------------------------------- #
    def _run_sparql_query(self, query: str, retries: int = 3) -> List[Dict[str, Any]]:
        for attempt in range(retries + 1):
            try:
                self.sparql.setQuery(query)
                self.sparql.setReturnFormat(JSON)
                results = self.sparql.query().convert()
                return results["results"]["bindings"]
            except Exception as e:
                if attempt < retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "SPARQL query failed (attempt %s/%s): %s. Retrying in %ss",
                        attempt + 1,
                        retries + 1,
                        e,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "SPARQL query failed after %s attempts: %s", retries + 1, e
                    )
                    return []

    # --------------------------------------------------------------------- #
    # COMPANY FETCHER
    # --------------------------------------------------------------------- #
    def fetch_companies(self) -> List[Company]:
        query = f"""
        SELECT DISTINCT ?company ?companyLabel ?ticker ?industryLabel ?foundedDate ?employeeCount
        WHERE {{
          ?company wdt:P31/wdt:P279* wd:Q4830453;
                   wdt:P17 wd:{self.country_qid}.

          ?company p:P414 ?listingStmt.
          ?listingStmt ps:P414 ?exchange;
                       pq:P249 ?ticker.

          OPTIONAL {{ ?company wdt:P452 ?industry. }}
          OPTIONAL {{ ?company wdt:P571 ?foundedDate. }}
          OPTIONAL {{ ?company wdt:P1128 ?employeeCount. }}

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {self.config.wikidata.max_results}
        """

        logger.info(
            "Fetching %s public companies from Wikidata (limit=%s)...",
            self.country_name,
            self.config.wikidata.max_results,
        )
        results = self._run_sparql_query(query)

        companies: List[Company] = []

        for row in results:
            try:
                name = row.get("companyLabel", {}).get("value", "")
                if not self._is_valid_label(name):
                    continue

                norm = self._normalize_name(name)
                if norm in self._seen_companies:
                    continue

                ticker = row.get("ticker", {}).get("value")
                if ticker:
                    ticker = ticker.strip()

                industry = row.get("industryLabel", {}).get("value")
                if industry and not self._is_valid_label(industry):
                    industry = None

                company = Company(
                    name=name,
                    ticker=ticker,
                    industry=industry,
                    country=self.country_name,
                    wikidata_uri=row.get("company", {}).get("value"),
                    properties={
                        "founded_date": row.get("foundedDate", {}).get("value"),
                        "employee_count": row.get("employeeCount", {}).get("value"),
                    },
                )

                self._seen_companies.add(norm)
                self._company_name_index[norm] = name
                companies.append(company)

            except Exception as e:
                logger.warning("Error parsing company row: %s", e)
                continue

        logger.info("Fetched %s companies for %s.", len(companies), self.country_name)
        return companies

    # --------------------------------------------------------------------- #
    # CEO FETCHERS - Multiple approaches for better coverage
    # --------------------------------------------------------------------- #
    def _fetch_ceos_via_p169(self) -> List[Dict[str, Any]]:
        """
        Fetch CEO via P169 (chief executive officer) on company.
        Uses statement-level access to filter for current CEOs (no end date).
        """
        query = f"""
        SELECT DISTINCT ?company ?companyLabel ?person ?personLabel
        WHERE {{
          ?company wdt:P31/wdt:P279* wd:Q4830453;
                   wdt:P17 wd:{self.country_qid};
                   p:P169 ?ceoStmt.

          ?ceoStmt ps:P169 ?person.

          # Only current CEOs - no end date qualifier
          FILTER NOT EXISTS {{ ?ceoStmt pq:P582 ?endDate. }}

          # Ensure person is human
          ?person wdt:P31 wd:Q5.

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {self.config.wikidata.max_results}
        """
        return self._run_sparql_query(query)

    def _fetch_ceos_via_position_held(self) -> List[Dict[str, Any]]:
        """
        Fetch CEO via P39 (position held) on person.
        Person holds position Q484876 (CEO) or subclass, at company (P642 qualifier).
        """
        query = f"""
        SELECT DISTINCT ?company ?companyLabel ?person ?personLabel
        WHERE {{
          ?company wdt:P31/wdt:P279* wd:Q4830453;
                   wdt:P17 wd:{self.country_qid}.

          ?person wdt:P31 wd:Q5;
                  p:P39 ?posStmt.

          # Position is CEO (Q484876) or subclass
          ?posStmt ps:P39/wdt:P279* wd:Q484876;
                   pq:P642 ?company.

          # Current position only
          FILTER NOT EXISTS {{ ?posStmt pq:P582 ?endDate. }}

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {self.config.wikidata.max_results}
        """
        return self._run_sparql_query(query)

    def _fetch_ceos_via_employer(self) -> List[Dict[str, Any]]:
        """
        Fetch CEO via P108 (employer) with P39 position qualifier being CEO.
        """
        query = f"""
        SELECT DISTINCT ?company ?companyLabel ?person ?personLabel
        WHERE {{
          ?company wdt:P31/wdt:P279* wd:Q4830453;
                   wdt:P17 wd:{self.country_qid}.

          ?person wdt:P31 wd:Q5;
                  p:P108 ?empStmt.

          ?empStmt ps:P108 ?company;
                   pq:P39/wdt:P279* wd:Q484876.

          FILTER NOT EXISTS {{ ?empStmt pq:P582 ?endDate. }}

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {self.config.wikidata.max_results}
        """
        return self._run_sparql_query(query)

    # --------------------------------------------------------------------- #
    # CHAIRPERSON FETCHER
    # --------------------------------------------------------------------- #
    def _fetch_chairpersons(self) -> List[Dict[str, Any]]:
        """
        Fetch chairpersons via P488 (chairperson).
        """
        query = f"""
        SELECT DISTINCT ?company ?companyLabel ?person ?personLabel
        WHERE {{
          ?company wdt:P31/wdt:P279* wd:Q4830453;
                   wdt:P17 wd:{self.country_qid};
                   p:P488 ?chairStmt.

          ?chairStmt ps:P488 ?person.

          FILTER NOT EXISTS {{ ?chairStmt pq:P582 ?endDate. }}

          ?person wdt:P31 wd:Q5.

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {self.config.wikidata.max_results}
        """
        return self._run_sparql_query(query)

    # --------------------------------------------------------------------- #
    # BOARD MEMBERS FETCHER
    # --------------------------------------------------------------------- #
    def _fetch_board_members(self) -> List[Dict[str, Any]]:
        """
        Fetch board members via P3320 (board member).
        """
        query = f"""
        SELECT DISTINCT ?company ?companyLabel ?person ?personLabel
        WHERE {{
          ?company wdt:P31/wdt:P279* wd:Q4830453;
                   wdt:P17 wd:{self.country_qid};
                   p:P3320 ?boardStmt.

          ?boardStmt ps:P3320 ?person.

          FILTER NOT EXISTS {{ ?boardStmt pq:P582 ?endDate. }}

          ?person wdt:P31 wd:Q5.

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {self.config.wikidata.max_results}
        """
        return self._run_sparql_query(query)

    # --------------------------------------------------------------------- #
    # FOUNDERS FETCHER
    # --------------------------------------------------------------------- #
    def _fetch_founders(self) -> List[Dict[str, Any]]:
        """
        Fetch founders via P112 (founded by).
        """
        query = f"""
        SELECT DISTINCT ?company ?companyLabel ?person ?personLabel
        WHERE {{
          ?company wdt:P31/wdt:P279* wd:Q4830453;
                   wdt:P17 wd:{self.country_qid};
                   wdt:P112 ?person.

          ?person wdt:P31 wd:Q5.

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {self.config.wikidata.max_results}
        """
        return self._run_sparql_query(query)

    # --------------------------------------------------------------------- #
    # CFO FETCHER
    # --------------------------------------------------------------------- #
    def _fetch_cfos(self) -> List[Dict[str, Any]]:
        """
        Fetch CFO via P169-like approach or position held.
        Q623279 = chief financial officer
        """
        query = f"""
        SELECT DISTINCT ?company ?companyLabel ?person ?personLabel
        WHERE {{
          ?company wdt:P31/wdt:P279* wd:Q4830453;
                   wdt:P17 wd:{self.country_qid}.

          ?person wdt:P31 wd:Q5;
                  p:P39 ?posStmt.

          ?posStmt ps:P39/wdt:P279* wd:Q623279;
                   pq:P642 ?company.

          FILTER NOT EXISTS {{ ?posStmt pq:P582 ?endDate. }}

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {self.config.wikidata.max_results}
        """
        return self._run_sparql_query(query)

    # --------------------------------------------------------------------- #
    # AGGREGATED EXECUTIVE FETCH
    # --------------------------------------------------------------------- #
    def fetch_executives(self) -> List[Tuple[Person, str, str]]:
        """
        Fetch all executives and map them to companies.

        Returns:
            List of (Person, company_name, relation_type)
        """
        logger.info("Fetching executives for %s...", self.country_name)

        executives: List[Tuple[Person, str, str]] = []

        # Define all executive fetchers with their relation types and titles
        fetchers = [
            ("CEO (P169)", self._fetch_ceos_via_p169, "IS_CEO_OF", "CEO"),
            ("CEO (position held)", self._fetch_ceos_via_position_held, "IS_CEO_OF", "CEO"),
            ("CEO (employer)", self._fetch_ceos_via_employer, "IS_CEO_OF", "CEO"),
            ("Chairperson", self._fetch_chairpersons, "IS_CHAIRPERSON_OF", "Chairperson"),
            ("Board Member", self._fetch_board_members, "IS_BOARD_MEMBER_OF", "Board Member"),
            ("Founder", self._fetch_founders, "FOUNDED", "Founder"),
            ("CFO", self._fetch_cfos, "IS_CFO_OF", "CFO"),
        ]

        for name, fetcher, rel_type, title in fetchers:
            logger.info("Fetching %s...", name)
            try:
                rows = fetcher()
                logger.info("  Found %d raw results for %s", len(rows), name)

                for row in rows:
                    try:
                        company_name = row.get("companyLabel", {}).get("value", "")
                        person_name = row.get("personLabel", {}).get("value", "")

                        if not self._is_valid_label(company_name):
                            continue
                        if not self._is_valid_label(person_name):
                            continue

                        norm_company = self._normalize_name(company_name)
                        norm_person = self._normalize_name(person_name)
                        key = f"{norm_person}::{rel_type}::{norm_company}"

                        if key in self._seen_person_company_keys:
                            continue
                        self._seen_person_company_keys.add(key)

                        person = Person(
                            name=person_name,
                            title=title,
                            wikidata_uri=row.get("person", {}).get("value"),
                        )

                        executives.append((person, company_name, rel_type))

                    except Exception as e:
                        logger.warning("Error parsing %s row: %s", name, e)

            except Exception as e:
                logger.warning("Error fetching %s: %s", name, e)

        logger.info(
            "Fetched %s total executive relationships for %s.",
            len(executives),
            self.country_name,
        )

        # Log breakdown by type
        from collections import Counter
        type_counts = Counter(rel for _, _, rel in executives)
        for rel_type, count in type_counts.items():
            logger.info("  - %s: %d", rel_type, count)

        return executives

    # --------------------------------------------------------------------- #
    # COMPANY RELATIONSHIPS (Subsidiaries only, OWNS removed)
    # --------------------------------------------------------------------- #
    def fetch_company_relationships(self) -> List[Tuple[str, str, str]]:
        """
        Fetch company-to-company relationships:
          - IS_SUBSIDIARY_OF: using P749 (subsidiary -> parent)
          - IS_PARENT_OF: inverse of subsidiary
        """
        relationships: List[Tuple[str, str, str]] = []
        seen_keys: Set[str] = set()

        # Subsidiaries
        logger.info("Fetching subsidiary relationships for %s...", self.country_name)
        query_subs = f"""
        SELECT DISTINCT ?subsidiary ?subsidiaryLabel ?parent ?parentLabel
        WHERE {{
          ?subsidiary wdt:P31/wdt:P279* wd:Q4830453;
                      wdt:P17 wd:{self.country_qid};
                      wdt:P749 ?parent.

          ?parent wdt:P31/wdt:P279* wd:Q4830453.

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 1000
        """
        rows = self._run_sparql_query(query_subs)
        for row in rows:
            try:
                sub_name = row.get("subsidiaryLabel", {}).get("value", "")
                parent_name = row.get("parentLabel", {}).get("value", "")

                if not self._is_valid_label(sub_name) or not self._is_valid_label(parent_name):
                    continue

                norm_sub = self._normalize_name(sub_name)
                norm_parent = self._normalize_name(parent_name)

                # IS_SUBSIDIARY_OF: subsidiary -> parent
                key1 = f"{norm_sub}::IS_SUBSIDIARY_OF::{norm_parent}"
                if key1 not in seen_keys:
                    seen_keys.add(key1)
                    relationships.append((sub_name, "IS_SUBSIDIARY_OF", parent_name))

                # IS_PARENT_OF: parent -> subsidiary (inverse)
                key2 = f"{norm_parent}::IS_PARENT_OF::{norm_sub}"
                if key2 not in seen_keys:
                    seen_keys.add(key2)
                    relationships.append((parent_name, "IS_PARENT_OF", sub_name))

            except Exception as e:
                logger.warning("Error parsing subsidiary row: %s", e)

        logger.info("Fetched %s company relationships.", len(relationships))
        return relationships

    # --------------------------------------------------------------------- #
    # HEADQUARTERS / LOCATION FETCHER (bonus)
    # --------------------------------------------------------------------- #
    def fetch_company_locations(self) -> List[Tuple[str, str]]:
        """
        Fetch headquarters locations via P159.
        Returns list of (company_name, location_name)
        """
        query = f"""
        SELECT DISTINCT ?company ?companyLabel ?location ?locationLabel
        WHERE {{
          ?company wdt:P31/wdt:P279* wd:Q4830453;
                   wdt:P17 wd:{self.country_qid};
                   wdt:P159 ?location.

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {self.config.wikidata.max_results}
        """
        logger.info("Fetching headquarters locations...")
        rows = self._run_sparql_query(query)

        locations: List[Tuple[str, str]] = []
        seen: Set[str] = set()

        for row in rows:
            try:
                company_name = row.get("companyLabel", {}).get("value", "")
                location_name = row.get("locationLabel", {}).get("value", "")

                if not self._is_valid_label(company_name) or not self._is_valid_label(location_name):
                    continue

                key = f"{self._normalize_name(company_name)}::{self._normalize_name(location_name)}"
                if key in seen:
                    continue
                seen.add(key)

                locations.append((company_name, location_name))
            except Exception as e:
                logger.warning("Error parsing location row: %s", e)

        logger.info("Fetched %s company-location pairs.", len(locations))
        return locations

    # --------------------------------------------------------------------- #
    # Neo4j helpers
    # --------------------------------------------------------------------- #
    def _create_indexes(self, session) -> None:
        statements = [
            "CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)",
            "CREATE INDEX company_ticker IF NOT EXISTS FOR (c:Company) ON (c.ticker)",
            "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX location_name IF NOT EXISTS FOR (l:Location) ON (l.name)",
        ]
        for stmt in statements:
            try:
                session.run(stmt)
            except Exception as e:
                logger.debug("Index creation failed/exists: %s", e)

    def _create_company_node(self, tx, company: Company) -> None:
        tx.run(
            """
            MERGE (c:Company {name: $name})
            SET c.ticker = $ticker,
                c.industry = $industry,
                c.country = $country,
                c.wikidata = $wikidata_uri,
                c.founded_date = $founded_date,
                c.employee_count = $employee_count
            """,
            name=company.name,
            ticker=company.ticker,
            industry=company.industry,
            country=company.country,
            wikidata_uri=company.wikidata_uri,
            founded_date=company.properties.get("founded_date"),
            employee_count=company.properties.get("employee_count"),
        )

    def _create_person_node(self, tx, person: Person) -> None:
        tx.run(
            """
            MERGE (p:Person {name: $name})
            SET p.title = $title,
                p.wikidata = $wikidata_uri
            """,
            name=person.name,
            title=person.title,
            wikidata_uri=person.wikidata_uri,
        )

    def _create_location_node(self, tx, location_name: str) -> None:
        tx.run(
            """
            MERGE (l:Location {name: $name})
            """,
            name=location_name,
        )

    def _create_person_company_relation(
        self, tx, person_name: str, company_name: str, relation: str
    ) -> None:
        tx.run(
            f"""
            MATCH (p:Person {{name: $pname}})
            MATCH (c:Company {{name: $cname}})
            MERGE (p)-[:{relation}]->(c)
            """,
            pname=person_name,
            cname=company_name,
        )

    def _create_company_relation(
        self, tx, comp1_name: str, relation: str, comp2_name: str
    ) -> None:
        tx.run(
            f"""
            MATCH (c1:Company {{name: $c1}})
            MATCH (c2:Company {{name: $c2}})
            MERGE (c1)-[:{relation}]->(c2)
            """,
            c1=comp1_name,
            c2=comp2_name,
        )

    def _create_company_location_relation(
        self, tx, company_name: str, location_name: str
    ) -> None:
        tx.run(
            """
            MATCH (c:Company {name: $cname})
            MATCH (l:Location {name: $lname})
            MERGE (c)-[:HEADQUARTERED_IN]->(l)
            """,
            cname=company_name,
            lname=location_name,
        )

    # --------------------------------------------------------------------- #
    # Orchestration
    # --------------------------------------------------------------------- #
    def build_knowledge_graph(self) -> Dict[str, int]:
        if not self.driver:
            self.connect_neo4j()

        stats = {
            "companies": 0,
            "persons": 0,
            "locations": 0,
            "person_company_relations": 0,
            "company_relations": 0,
            "location_relations": 0,
        }

        logger.info("Starting KG construction for %s...", self.country_name)

        # 1) Indexes
        with self.driver.session() as session:
            self._create_indexes(session)

        # 2) Companies
        companies = self.fetch_companies()
        with self.driver.session() as session:
            for c in companies:
                session.execute_write(self._create_company_node, c)
        stats["companies"] = len(companies)
        logger.info("Created/merged %s company nodes.", stats["companies"])

        # 3) Executives
        executives = self.fetch_executives()
        unique_persons: Set[str] = set()
        relation_count = 0

        with self.driver.session() as session:
            for person, company_label, rel_type in executives:
                norm_company = self._normalize_name(company_label)
                canonical_company = self._company_name_index.get(norm_company)
                if not canonical_company:
                    continue

                session.execute_write(self._create_person_node, person)
                unique_persons.add(self._normalize_name(person.name))

                session.execute_write(
                    self._create_person_company_relation,
                    person.name,
                    canonical_company,
                    rel_type,
                )
                relation_count += 1

        stats["persons"] = len(unique_persons)
        stats["person_company_relations"] = relation_count
        logger.info("Created/merged %s person nodes.", stats["persons"])
        logger.info("Created %s person-company relationships.", relation_count)

        # 4) Company relationships
        company_rels = self.fetch_company_relationships()
        with self.driver.session() as session:
            for c1, rel, c2 in company_rels:
                session.execute_write(self._create_company_relation, c1, rel, c2)
        stats["company_relations"] = len(company_rels)
        logger.info("Created %s company-company relationships.", stats["company_relations"])

        # 5) Locations (headquarters)
        locations = self.fetch_company_locations()
        unique_locations: Set[str] = set()
        location_rel_count = 0

        with self.driver.session() as session:
            for company_name, location_name in locations:
                norm_company = self._normalize_name(company_name)
                canonical_company = self._company_name_index.get(norm_company)
                if not canonical_company:
                    continue

                session.execute_write(self._create_location_node, location_name)
                unique_locations.add(self._normalize_name(location_name))

                session.execute_write(
                    self._create_company_location_relation,
                    canonical_company,
                    location_name,
                )
                location_rel_count += 1

        stats["locations"] = len(unique_locations)
        stats["location_relations"] = location_rel_count
        logger.info("Created/merged %s location nodes.", stats["locations"])
        logger.info("Created %s company-location relationships.", location_rel_count)

        logger.info("KG construction complete for %s.", self.country_name)
        return stats


def main() -> None:
    config = AppConfig()
    builder = WikidataKGBuilder(config, country_name="United States")
    try:
        stats = builder.build_knowledge_graph()
        print("\n=== KNOWLEDGE GRAPH BUILD COMPLETE ===")
        print(f"Country: {builder.country_name}")
        for k, v in stats.items():
            print(f"- {k}: {v}")
    finally:
        builder.close()


if __name__ == "__main__":
    main()