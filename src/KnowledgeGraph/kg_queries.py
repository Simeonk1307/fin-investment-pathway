"""
Knowledge Graph Query Functions for Financial KG System
All queries are implemented as typed functions with error handling
"""
from typing import List, Dict, Any, Optional, Tuple
from neo4j import Driver, Session, Result
import logging
from datetime import datetime

from src.KnowledgeGraph.models import QueryResult, RelationType

logger = logging.getLogger(__name__)

class KGQueryService:
    """Service class for all knowledge graph queries"""

    def __init__(self, driver: Driver):
        """
        Initialize query service with Neo4j driver

        Args:
            driver: Neo4j driver instance
        """
        self.driver = driver

    def _execute_query(self, query: str, parameters: Dict[str, Any] = None) -> QueryResult:
        """
        Execute a Cypher query and return wrapped results

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            QueryResult with data and metadata
        """
        start_time = datetime.now()
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                data = [dict(record) for record in result]
                query_time = (datetime.now() - start_time).total_seconds()

                return QueryResult(
                    data=data,
                    count=len(data),
                    query_time=query_time
                )
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    # ========== Person-Company Queries ==========

    def get_ceos_and_board_members(self, company_name: str) -> QueryResult:
        """
        Get CEOs and Board Members of a given Company

        Args:
            company_name: Name of the company

        Returns:
            QueryResult containing person names, titles, and relation types
        """
        query = """
        MATCH (p:Person)-[r:IS_CEO_OF|IS_BOARD_MEMBER_OF]->(c:Company {name: $company_name})
        RETURN p.name AS person_name, 
               p.title AS title, 
               type(r) AS relation_type,
               p.wikidata AS wikidata_uri
        ORDER BY relation_type, person_name
        """
        return self._execute_query(query, {"company_name": company_name})

    def get_companies_by_person(self, person_name: str) -> QueryResult:
        """
        Get all Companies a Person is CEO/Board Member/Founder of or works for

        Args:
            person_name: Name of the person

        Returns:
            QueryResult containing company details and relationship types
        """
        query = """
        MATCH (p:Person {name: $person_name})-[r:IS_CEO_OF|IS_BOARD_MEMBER_OF|FOUNDED|WORKS_FOR]->(c:Company)
        RETURN c.name AS company_name,
               c.ticker AS ticker,
               c.industry AS industry,
               type(r) AS relation_type,
               c.wikidata AS wikidata_uri
        ORDER BY relation_type, company_name
        """
        return self._execute_query(query, {"person_name": person_name})

    # ========== Company Relationship Queries ==========

    def get_subsidiaries_and_parents(self, company_name: str) -> QueryResult:
        """
        Get Subsidiaries and Parent Companies of a Company

        Args:
            company_name: Name of the company

        Returns:
            QueryResult with subsidiaries and parent companies
        """
        query = """
        MATCH (c:Company {name: $company_name})
        OPTIONAL MATCH (c)-[:IS_SUBSIDIARY_OF]->(parent:Company)
        OPTIONAL MATCH (subsidiary:Company)-[:IS_SUBSIDIARY_OF]->(c)
        RETURN collect(DISTINCT parent.name) AS parent_companies,
               collect(DISTINCT subsidiary.name) AS subsidiaries,
               parent.ticker AS parent_ticker,
               parent.industry AS parent_industry
        """
        return self._execute_query(query, {"company_name": company_name})

    def get_partnerships(self, company_name: str) -> QueryResult:
        """
        Get Partnership relations between Companies

        Args:
            company_name: Name of the company

        Returns:
            QueryResult containing partner companies
        """
        query = """
        MATCH (c1:Company {name: $company_name})-[:HAS_PARTNERSHIP_WITH]-(c2:Company)
        RETURN DISTINCT c2.name AS partner_name,
               c2.ticker AS ticker,
               c2.industry AS industry,
               c2.country AS country
        ORDER BY partner_name
        """
        return self._execute_query(query, {"company_name": company_name})

    def get_suppliers_and_clients(self, company_name: str) -> QueryResult:
        """
        Get Supplier and client Company relationships

        Args:
            company_name: Name of the company

        Returns:
            QueryResult with suppliers and clients
        """
        query = """
        MATCH (c:Company {name: $company_name})
        OPTIONAL MATCH (supplier:Company)-[:IS_SUPPLIER_OF]->(c)
        OPTIONAL MATCH (c)-[:IS_SUPPLIER_OF]->(client:Company)
        RETURN collect(DISTINCT {name: supplier.name, ticker: supplier.ticker, industry: supplier.industry}) AS suppliers,
               collect(DISTINCT {name: client.name, ticker: client.ticker, industry: client.industry}) AS clients
        """
        return self._execute_query(query, {"company_name": company_name})

    # ========== Event Queries ==========

    def get_acquisition_events(self, company_name: str) -> QueryResult:
        """
        Get Acquisition events involving specific Companies

        Args:
            company_name: Name of the company

        Returns:
            QueryResult containing acquisition events
        """
        query = """
        MATCH (e:Event {type: 'Acquisition'})
        OPTIONAL MATCH (e)-[:HAS_ACQUIRER]->(acquirer:Company {name: $company_name})
        OPTIONAL MATCH (e)-[:HAS_TARGET]->(target:Company {name: $company_name})
        WHERE acquirer IS NOT NULL OR target IS NOT NULL
        OPTIONAL MATCH (e)-[:HAS_ACQUIRER]->(acq:Company)
        OPTIONAL MATCH (e)-[:HAS_TARGET]->(tgt:Company)
        RETURN e.date AS event_date,
               e.value AS deal_value,
               e.description AS description,
               acq.name AS acquirer_name,
               tgt.name AS target_name
        ORDER BY e.date DESC
        """
        return self._execute_query(query, {"company_name": company_name})

    def get_companies_in_event(self, event_id: int) -> QueryResult:
        """
        Get Companies involved in a specified Event (acquirer, target)

        Args:
            event_id: Internal Neo4j ID of the event

        Returns:
            QueryResult with companies and their roles
        """
        query = """
        MATCH (e:Event) WHERE id(e) = $event_id
        OPTIONAL MATCH (e)-[:HAS_ACQUIRER]->(acquirer:Company)
        OPTIONAL MATCH (e)-[:HAS_TARGET]->(target:Company)
        RETURN e.type AS event_type,
               e.date AS event_date,
               e.value AS value,
               acquirer.name AS acquirer_name,
               acquirer.ticker AS acquirer_ticker,
               target.name AS target_name,
               target.ticker AS target_ticker
        """
        return self._execute_query(query, {"event_id": event_id})

    def get_events_by_person(self, person_name: str) -> QueryResult:
        """
        Get Events involving a given Person

        Args:
            person_name: Name of the person

        Returns:
            QueryResult containing events
        """
        query = """
        MATCH (e:Event)-[:INVOLVES_PERSON]->(p:Person {name: $person_name})
        OPTIONAL MATCH (e)-[:HAS_ACQUIRER|HAS_TARGET]-(c:Company)
        RETURN e.type AS event_type,
               e.date AS event_date,
               e.description AS description,
               e.value AS value,
               collect(DISTINCT c.name) AS related_companies
        ORDER BY e.date DESC
        """
        return self._execute_query(query, {"person_name": person_name})

    # ========== Similarity and Clustering Queries ==========

    def find_similar_stocks(
        self, 
        stock_name: str, 
        similarity_factors: List[str] = None
    ) -> QueryResult:
        """
        Find Stocks similar to a given stock based on industry, leadership, or partnerships

        Args:
            stock_name: Name of the stock/company
            similarity_factors: List of factors to consider ['industry', 'leadership', 'partnerships']

        Returns:
            QueryResult with similar stocks and similarity scores
        """
        if similarity_factors is None:
            similarity_factors = ['industry', 'leadership', 'partnerships']

        # Build dynamic query based on factors
        conditions = []

        if 'industry' in similarity_factors:
            conditions.append("c1.industry = c2.industry")

        query_parts = ["MATCH (c1:Company {name: $stock_name})"]

        if 'leadership' in similarity_factors:
            query_parts.append("""
            OPTIONAL MATCH (c1)<-[:IS_CEO_OF|IS_BOARD_MEMBER_OF]-(p:Person)-[:IS_CEO_OF|IS_BOARD_MEMBER_OF]->(c2:Company)
            WHERE c1 <> c2
            """)

        if 'partnerships' in similarity_factors:
            query_parts.append("""
            OPTIONAL MATCH (c1)-[:HAS_PARTNERSHIP_WITH]-(c3:Company)
            """)

        query_parts.append("""
        WITH c1, c2, c3
        MATCH (c_similar:Company)
        WHERE c_similar <> c1 
        AND (c_similar = c2 OR c_similar = c3 OR c1.industry = c_similar.industry)
        RETURN DISTINCT c_similar.name AS similar_stock,
               c_similar.ticker AS ticker,
               c_similar.industry AS industry,
               c_similar.country AS country,
               CASE 
                   WHEN c1.industry = c_similar.industry THEN 1 ELSE 0 
               END AS industry_match
        ORDER BY industry_match DESC, similar_stock
        LIMIT 20
        """)

        query = " ".join(query_parts)
        return self._execute_query(query, {"stock_name": stock_name})

    def cluster_user_stocks(self, stock_names: List[str]) -> QueryResult:
        """
        Cluster/group stocks held by a user based on graph connectivity

        Args:
            stock_names: List of stock/company names

        Returns:
            QueryResult with clustered stocks and connections
        """
        query = """
        MATCH (c:Company)
        WHERE c.name IN $stock_names
        OPTIONAL MATCH path = (c)-[r:HAS_PARTNERSHIP_WITH|IS_SUBSIDIARY_OF|IS_SUPPLIER_OF*1..2]-(related:Company)
        WHERE related.name IN $stock_names
        RETURN c.name AS stock,
               c.industry AS industry,
               collect(DISTINCT {
                   related: related.name, 
                   relationship: [rel in relationships(path) | type(rel)],
                   path_length: length(path)
               }) AS connections
        ORDER BY industry, stock
        """
        return self._execute_query(query, {"stock_names": stock_names})

    # ========== Impact Analysis Queries ==========

    def get_affected_entities_by_news(
        self, 
        company_name: Optional[str] = None,
        person_name: Optional[str] = None,
        max_hops: int = 2
    ) -> QueryResult:
        """
        Get Keywords and entities affected by a news Event on a Company or Person

        Args:
            company_name: Name of the company in news
            person_name: Name of the person in news
            max_hops: Maximum graph traversal distance

        Returns:
            QueryResult with affected entities and impact paths
        """
        if company_name:
            query = f"""
            MATCH (c:Company {{name: $entity_name}})
            MATCH path = (c)-[*1..{max_hops}]-(affected)
            WHERE affected:Company OR affected:Person
            RETURN DISTINCT 
                   labels(affected)[0] AS entity_type,
                   affected.name AS entity_name,
                   affected.ticker AS ticker,
                   affected.industry AS industry,
                   length(path) AS distance,
                   [rel in relationships(path) | type(rel)] AS relationship_path
            ORDER BY distance, entity_name
            """
            params = {"entity_name": company_name}
        elif person_name:
            query = f"""
            MATCH (p:Person {{name: $entity_name}})
            MATCH path = (p)-[*1..{max_hops}]-(affected)
            WHERE affected:Company OR affected:Person
            RETURN DISTINCT 
                   labels(affected)[0] AS entity_type,
                   affected.name AS entity_name,
                   affected.ticker AS ticker,
                   length(path) AS distance,
                   [rel in relationships(path) | type(rel)] AS relationship_path
            ORDER BY distance, entity_name
            """
            params = {"entity_name": person_name}
        else:
            raise ValueError("Either company_name or person_name must be provided")

        return self._execute_query(query, params)

    def get_executive_networks(self, company_name: str) -> QueryResult:
        """
        Get Networks of companies connected through shared executives or partnerships

        Args:
            company_name: Name of the company

        Returns:
            QueryResult with network of connected companies
        """
        query = """
        MATCH (c1:Company {name: $company_name})
        MATCH path = (c1)<-[:IS_CEO_OF|IS_BOARD_MEMBER_OF]-(p:Person)-[:IS_CEO_OF|IS_BOARD_MEMBER_OF]->(c2:Company)
        WHERE c1 <> c2
        RETURN DISTINCT c2.name AS connected_company,
               c2.ticker AS ticker,
               c2.industry AS industry,
               collect(DISTINCT p.name) AS shared_executives,
               count(DISTINCT p) AS executive_count
        ORDER BY executive_count DESC, connected_company
        """
        return self._execute_query(query, {"company_name": company_name})

    def get_temporal_events(
        self, 
        company_name: Optional[str] = None,
        industry: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> QueryResult:
        """
        Get Temporal sequences of Events related to specific Companies or industries

        Args:
            company_name: Filter by company name
            industry: Filter by industry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            QueryResult with temporal event sequence
        """
        query_parts = ["MATCH (e:Event)"]
        conditions = []
        params = {}

        if company_name:
            query_parts.append("""
            MATCH (e)-[:HAS_ACQUIRER|HAS_TARGET|INVOLVES_PERSON*1..2]-(c:Company {name: $company_name})
            """)
            params["company_name"] = company_name
        elif industry:
            query_parts.append("""
            MATCH (e)-[:HAS_ACQUIRER|HAS_TARGET]-(c:Company {industry: $industry})
            """)
            params["industry"] = industry

        if start_date:
            conditions.append("e.date >= $start_date")
            params["start_date"] = start_date

        if end_date:
            conditions.append("e.date <= $end_date")
            params["end_date"] = end_date

        where_clause = " AND ".join(conditions) if conditions else "true"
        query_parts.append(f"WHERE {where_clause}")

        query_parts.append("""
        OPTIONAL MATCH (e)-[:HAS_ACQUIRER]->(acq:Company)
        OPTIONAL MATCH (e)-[:HAS_TARGET]->(tgt:Company)
        OPTIONAL MATCH (e)-[:INVOLVES_PERSON]->(p:Person)
        RETURN e.type AS event_type,
               e.date AS event_date,
               e.value AS value,
               e.description AS description,
               acq.name AS acquirer,
               tgt.name AS target,
               collect(DISTINCT p.name) AS persons_involved
        ORDER BY e.date DESC
        """)

        query = " ".join(query_parts)
        return self._execute_query(query, params)

    def calculate_impact_scores(
        self,
        source_company: str,
        sentiment_score: float = 0.0
    ) -> QueryResult:
        """
        Calculate Impact scores propagated from news Events through related Companies and Persons

        Args:
            source_company: Company at the center of the news
            sentiment_score: Sentiment score (-1.0 to 1.0) of the news

        Returns:
            QueryResult with impact scores for related entities
        """
        query = """
        MATCH (source:Company {name: $source_company})
        MATCH path = (source)-[*1..3]-(affected:Company)
        WHERE source <> affected
        WITH affected, 
             path,
             length(path) AS distance,
             $sentiment_score AS sentiment
        RETURN DISTINCT affected.name AS affected_company,
               affected.ticker AS ticker,
               affected.industry AS industry,
               distance,
               ROUND(sentiment / distance * 100) / 100.0 AS impact_score,
               [rel in relationships(path) | type(rel)] AS impact_path
        ORDER BY impact_score DESC, distance ASC
        LIMIT 50
        """
        return self._execute_query(query, {
            "source_company": source_company,
            "sentiment_score": sentiment_score
        })
