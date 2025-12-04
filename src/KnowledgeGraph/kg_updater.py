from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import logging

import spacy
from neo4j import GraphDatabase, Driver, Session

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str


class KGNewsUpdater:
    """
    Updates the financial knowledge graph in Neo4j
    from Finnhub-style news records (no LLM, spaCy-based).
    """

    def __init__(self, neo4j_config: Neo4jConfig) -> None:
        self.config = neo4j_config
        self.driver: Driver = GraphDatabase.driver(
            neo4j_config.uri,
            auth=(neo4j_config.user, neo4j_config.password),
        )
        self.nlp = spacy.load("en_core_web_sm")

    def close(self) -> None:
        self.driver.close()


    def update_kg(self, news: Dict[str, Any]) -> None:
        """
        news dict must contain at least:
        category: str
        datetime: int (Unix timestamp)
        headline: str
        news_id: int
        image: str
        related: str (comma-separated tickers)
        source: str
        summary: str
        url: str
        """
        try:
            text = f"{news.get('headline', '')}. {news.get('summary', '')}"
            doc = self.nlp(text)

            persons, orgs = self._extract_entities(doc)
            tickers = self._extract_tickers_from_related(news.get("related", ""))

            with self.driver.session() as session:
                # ensure companies for all tickers, even if not mentioned by name
                if tickers:
                    self._ensure_companies_from_tickers(session, tickers)

                lowered = text.lower()

                # simple pattern: CEO appointment
                if "appointed as ceo" in lowered or "appointed ceo" in lowered:
                    self._handle_ceo_appointment(session, news, persons, orgs)

                # simple pattern: acquisition / merger
                if any(word in lowered for word in ["acquires", "acquired", "acquisition", "merger"]):
                    self._handle_acquisition(session, news, orgs)

                # generic NewsUpdate event linking persons/orgs
                self._handle_generic_news_event(session, news, persons, orgs)

        except Exception as exc:
            logger.exception("Failed to update KG from news_id=%s: %s", news.get("news_id"), exc)

    # ---------- helpers: entity & ticker extraction ----------

    def _extract_entities(self, doc) -> Tuple[List[str], List[str]]:
        persons: List[str] = []
        orgs: List[str] = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if ent.text not in persons:
                    persons.append(ent.text)
            elif ent.label_ in ("ORG", "GPE"):
                if ent.text not in orgs:
                    orgs.append(ent.text)
        return persons, orgs

    def _extract_tickers_from_related(self, related: str) -> List[str]:
        """
        Finnhub 'related' looks like: "TCS,INFY,RELIANCE"
        """
        if not related:
            return []
        parts = [t.strip() for t in related.split(",") if t.strip()]
        # unique, order-preserving
        return list(dict.fromkeys(parts))

    # ---------- KG operations ----------

    def _ensure_companies_from_tickers(self, session: Session, tickers: List[str]) -> None:
        """
        Ensure Company nodes exist for each ticker.
        Name/industry can be enriched later by another job.
        """
        session.execute_write(self._cypher_ensure_companies_from_tickers, tickers)

    @staticmethod
    def _cypher_ensure_companies_from_tickers(tx, tickers: List[str]) -> None:
        query = """
        UNWIND $tickers AS t
        MERGE (c:Company {ticker: t})
        """
        tx.run(query, tickers=tickers)

    # ---------- specific event handlers ----------

    def _handle_ceo_appointment(
        self,
        session: Session,
        news: Dict[str, Any],
        persons: List[str],
        orgs: List[str],
    ) -> None:
        """
        Heuristic:
        - first PERSON -> CEO
        - first ORG -> company
        """
        if not persons or not orgs:
            return

        ceo_name = persons[0]
        company_name = orgs[0]
        logger.info(
            "Detected CEO appointment: %s as CEO of %s (news_id=%s)",
            ceo_name,
            company_name,
            news.get("news_id"),
        )

        session.execute_write(
            self._cypher_ceo_appointment,
            ceo_name,
            company_name,
            news,
        )

    @staticmethod
    def _cypher_ceo_appointment(
        tx,
        ceo_name: str,
        company_name: str,
        news: Dict[str, Any],
    ) -> None:
        query = """
        MERGE (p:Person {name: $ceo_name})
        MERGE (c:Company {name: $company_name})
        MERGE (p)-[:IS_CEO_OF]->(c)

        MERGE (e:Event {news_id: $news_id})
        ON CREATE SET e.type = 'LeadershipChange',
                      e.datetime = $datetime,
                      e.headline = $headline,
                      e.source = $source,
                      e.url = $url
        MERGE (e)-[:INVOLVES_PERSON]->(p)
        MERGE (e)-[:HAS_TARGET]->(c)
        """
        tx.run(
            query,
            ceo_name=ceo_name,
            company_name=company_name,
            news_id=int(news["news_id"]),
            datetime=int(news["datetime"]),
            headline=str(news.get("headline", "")),
            source=str(news.get("source", "")),
            url=str(news.get("url", "")),
        )

    def _handle_acquisition(
        self,
        session: Session,
        news: Dict[str, Any],
        orgs: List[str],
    ) -> None:
        """
        Heuristic:
        - if 2+ orgs: orgs[0] acquires orgs[1]
        """
        if len(orgs) < 2:
            return

        acquirer = orgs[0]
        target = orgs[1]

        logger.info(
            "Detected acquisition: %s -> %s (news_id=%s)",
            acquirer,
            target,
            news.get("news_id"),
        )

        session.execute_write(
            self._cypher_acquisition,
            acquirer,
            target,
            news,
        )

    @staticmethod
    def _cypher_acquisition(
        tx,
        acquirer_name: str,
        target_name: str,
        news: Dict[str, Any],
    ) -> None:
        query = """
        MERGE (acq:Company {name: $acquirer_name})
        MERGE (tgt:Company {name: $target_name})

        MERGE (e:Event {news_id: $news_id})
        ON CREATE SET e.type = 'Acquisition',
                      e.datetime = $datetime,
                      e.headline = $headline,
                      e.source = $source,
                      e.url = $url

        MERGE (e)-[:HAS_ACQUIRER]->(acq)
        MERGE (e)-[:HAS_TARGET]->(tgt)
        """
        tx.run(
            query,
            acquirer_name=acquirer_name,
            target_name=target_name,
            news_id=int(news["news_id"]),
            datetime=int(news["datetime"]),
            headline=str(news.get("headline", "")),
            source=str(news.get("source", "")),
            url=str(news.get("url", "")),
        )

    def _handle_generic_news_event(
        self,
        session: Session,
        news: Dict[str, Any],
        persons: List[str],
        orgs: List[str],
    ) -> None:
        """
        Generic NewsUpdate event, linking all mentioned persons/orgs.
        """
        session.execute_write(
            self._cypher_generic_news_event,
            news,
            persons,
            orgs,
        )

    @staticmethod
    def _cypher_generic_news_event(
        tx,
        news: Dict[str, Any],
        persons: List[str],
        orgs: List[str],
    ) -> None:
        query = """
        MERGE (e:Event {news_id: $news_id})
        ON CREATE SET e.type = 'NewsUpdate',
                      e.datetime = $datetime,
                      e.headline = $headline,
                      e.category = $category,
                      e.source = $source,
                      e.url = $url

        WITH e
        UNWIND $persons AS person_name
        MERGE (p:Person {name: person_name})
        MERGE (e)-[:INVOLVES_PERSON]->(p)

        WITH e
        UNWIND $orgs AS org_name
        MERGE (c:Company {name: org_name})
        MERGE (e)-[:HAS_TARGET]->(c)
        """
        tx.run(
            query,
            news_id=int(news["news_id"]),
            datetime=int(news["datetime"]),
            headline=str(news.get("headline", "")),
            category=str(news.get("category", "")),
            source=str(news.get("source", "")),
            url=str(news.get("url", "")),
            persons=persons,
            orgs=orgs,
        )
