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
    def __init__(self, neo4j_config: Neo4jConfig) -> None:
        self.config = neo4j_config
        self.driver: Driver = GraphDatabase.driver(
            neo4j_config.uri,
            auth=(neo4j_config.user, neo4j_config.password),
        )
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

    def close(self) -> None:
        self.driver.close()

    def update_kg_from_news(self, news: Dict[str, Any]) -> None:
        try:
            text = f"{news.get('title', '')}. {news.get('content', '')}"
            doc = self.nlp(text)

            persons, orgs = self._extract_entities(doc)
            symbol = news.get("symbol", "")
            tickers = [t.strip() for t in symbol.split(",") if t.strip()] if symbol else []

            with self.driver.session() as session:
                if tickers:
                    self._ensure_companies_from_tickers(session, tickers)

                lowered = text.lower()

                if "appointed as ceo" in lowered or "appointed ceo" in lowered:
                    self._handle_ceo_appointment(session, news, persons, orgs, tickers)

                if any(word in lowered for word in ["acquires", "acquired", "acquisition", "merger"]):
                    self._handle_acquisition(session, news, orgs)

                self._handle_generic_news_event(session, news, persons, orgs, tickers)

        except Exception as exc:
            logger.exception("Failed to update KG from news_id=%s: %s", news.get("news_id"), exc)

    def _extract_entities(self, doc) -> Tuple[List[str], List[str]]:
        persons, orgs = [], []
        for ent in doc.ents:
            if ent.label_ == "PERSON" and ent.text not in persons:
                persons.append(ent.text)
            elif ent.label_ in ("ORG", "GPE") and ent.text not in orgs:
                orgs.append(ent.text)
        return persons, orgs

    def _ensure_companies_from_tickers(self, session: Session, tickers: List[str]) -> None:
        session.execute_write(lambda tx: tx.run(
            "UNWIND $tickers AS t MERGE (c:Company {ticker: t})",
            tickers=tickers
        ))

    def _handle_ceo_appointment(self, session: Session, news: Dict, persons: List[str], orgs: List[str], tickers: List[str]) -> None:
        if not persons or not orgs:
            return
        session.execute_write(
            lambda tx: tx.run("""
                MERGE (p:Person {name: $ceo})
                MERGE (c:Company {name: $company})
                MERGE (p)-[:IS_CEO_OF]->(c)
                MERGE (e:Event {news_id: $news_id})
                ON CREATE SET e.type='LeadershipChange', e.timestamp=$timestamp, e.title=$title
                MERGE (e)-[:INVOLVES_PERSON]->(p)
                MERGE (e)-[:HAS_TARGET]->(c)
            """, ceo=persons[0], company=orgs[0], news_id=int(news["news_id"]),
                timestamp=int(news["timestamp"]), title=str(news.get("title", "")))
        )

    def _handle_acquisition(self, session: Session, news: Dict, orgs: List[str]) -> None:
        if len(orgs) < 2:
            return
        session.execute_write(
            lambda tx: tx.run("""
                MERGE (acq:Company {name: $acquirer})
                MERGE (tgt:Company {name: $target})
                MERGE (e:Event {news_id: $news_id})
                ON CREATE SET e.type='Acquisition', e.timestamp=$timestamp, e.title=$title
                MERGE (e)-[:HAS_ACQUIRER]->(acq)
                MERGE (e)-[:HAS_TARGET]->(tgt)
            """, acquirer=orgs[0], target=orgs[1], news_id=int(news["news_id"]),
                timestamp=int(news["timestamp"]), title=str(news.get("title", "")))
        )

    def _handle_generic_news_event(self, session: Session, news: Dict, persons: List[str], orgs: List[str], tickers: List[str]) -> None:
        session.execute_write(
            lambda tx: tx.run("""
                MERGE (e:Event {news_id: $news_id})
                ON CREATE SET e.type='NewsUpdate', e.timestamp=$timestamp, e.title=$title,
                              e.content=$content, e.category=$category, e.source=$source,
                              e.url=$url, e.symbol=$symbol
                WITH e
                UNWIND CASE WHEN size($persons) > 0 THEN $persons ELSE [null] END AS pname
                FOREACH (_ IN CASE WHEN pname IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (p:Person {name: pname}) MERGE (e)-[:MENTIONS]->(p))
                WITH e
                UNWIND CASE WHEN size($orgs) > 0 THEN $orgs ELSE [null] END AS oname
                FOREACH (_ IN CASE WHEN oname IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (c:Company {name: oname}) MERGE (e)-[:MENTIONS_COMPANY]->(c))
                WITH e
                UNWIND CASE WHEN size($tickers) > 0 THEN $tickers ELSE [null] END AS t
                FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (ct:Company {ticker: t}) MERGE (e)-[:ABOUT]->(ct))
            """, news_id=int(news["news_id"]), timestamp=int(news["timestamp"]),
                title=str(news.get("title", "")), content=str(news.get("content", ""))[:500],
                category=str(news.get("category", "")), source=str(news.get("source", "")),
                url=str(news.get("url", "")), symbol=str(news.get("symbol", "")),
                persons=persons, orgs=orgs, tickers=tickers)
        )