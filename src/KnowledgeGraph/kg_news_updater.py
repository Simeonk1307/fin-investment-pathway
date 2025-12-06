from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import logging
import re
import spacy
from neo4j import GraphDatabase, Driver, Session

logger = logging.getLogger(__name__)


@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str


class KGNewsUpdater:
    """Updates Neo4j KG from streaming news."""

    class RelationType(str, Enum):
        IS_CEO_OF = "IS_CEO_OF"
        IS_CFO_OF = "IS_CFO_OF"
        IS_CHAIRPERSON_OF = "IS_CHAIRPERSON_OF"
        IS_BOARD_MEMBER_OF = "IS_BOARD_MEMBER_OF"
        IS_SUBSIDIARY_OF = "IS_SUBSIDIARY_OF"
        IS_PARENT_OF = "IS_PARENT_OF"

    LEADERSHIP_PATTERNS = [
        # CEO – appointments
        (r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+(?:has been\s+)?"
         r"(?:named|appointed|becomes?|hired as|promoted to)\s+(?:the\s+)?(?:new\s+)?CEO",
         "IS_CEO_OF", "CEO", False),
        (r"(?:named|appointed|hires?)\s+(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"
         r"\s+(?:as\s+)?(?:the\s+)?(?:new\s+)?CEO",
         "IS_CEO_OF", "CEO", False),
        (r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+will\s+(?:become|serve as)\s+CEO",
         "IS_CEO_OF", "CEO", False),

        # CEO – departures
        (r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+"
         r"(?:steps?\s+down|resigns?|leaves?|retires?)\s+(?:as\s+)?CEO",
         "IS_CEO_OF", "CEO", True),
        (r"CEO\s+(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+"
         r"(?:steps?\s+down|resigns?|leaves?|retires?)",
         "IS_CEO_OF", "CEO", True),

        # CFO
        (r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+"
         r"(?:has been\s+)?(?:named|appointed|becomes?|hired as|promoted to)\s+"
         r"(?:the\s+)?(?:new\s+)?CFO",
         "IS_CFO_OF", "CFO", False),
        (r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+"
         r"(?:steps?\s+down|resigns?|leaves?)\s+(?:as\s+)?CFO",
         "IS_CFO_OF", "CFO", True),

        # Chairperson
        (r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+"
         r"(?:named|appointed|elected)\s+(?:the\s+)?(?:new\s+)?"
         r"chair(?:man|woman|person)?",
         "IS_CHAIRPERSON_OF", "Chairperson", False),

        # Board members
        (r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+"
         r"(?:joins?|appointed to|elected to)\s+(?:the\s+)?board",
         "IS_BOARD_MEMBER_OF", "Board Member", False),
    ]

    ACQUISITION_PATTERNS = [
        r"(?P<acquirer>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\s+"
        r"(?:to\s+)?(?:acquire[sd]?|purchase[sd]?|b[uo]ys?)\s+"
        r"(?P<target>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})",

        r"(?P<target>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\s+"
        r"(?:to\s+be\s+)?acquired\s+by\s+"
        r"(?P<acquirer>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})",
    ]

    SKIP_ENTITIES = frozenset({
        "us", "u.s.", "usa", "united states", "america",
        "congress", "senate", "sec", "fda", "ftc", "fed",
        "eu", "european union", "uk", "china",
        "reuters", "bloomberg", "ap", "nasdaq", "nyse",
    })

    SUFFIX_RE = re.compile(
        r"\s*[,.]?\s*\b(inc|corp|corporation|ltd|llc|plc|company|co|group|holdings?)\b\.?\s*$",
        re.I,
    )

    def __init__(self, config: Neo4jConfig, verbose: bool = False) -> None:
        self._verbose = verbose
        self.driver: Driver = GraphDatabase.driver(
            config.uri, auth=(config.user, config.password)
        )

        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            import subprocess
            subprocess.run(
                ["python", "-m", "spacy", "download", "en_core_web_md"],
                check=True,
            )
            self.nlp = spacy.load("en_core_web_md")

        self._companies: Dict[str, str] = {}
        self._tickers: Dict[str, str] = {}
        self._persons: Set[str] = set()
        self._refresh_cache()

    def close(self) -> None:
        self.driver.close()

    def _log(self, msg: str) -> None:
        if self._verbose:
            logger.info(msg)

    def _norm(self, s: str) -> str:
        return re.sub(r"\s+", " ", s.strip()).lower() if s else ""

    def _norm_company(self, s: str) -> str:
        return self._norm(self.SUFFIX_RE.sub("", s)) if s else ""

    def _valid_person(self, name: str) -> bool:
        if not name or len(name) < 4:
            return False
        parts = name.split()
        if not (2 <= len(parts) <= 4):
            return False
        if self._norm(name) in self.SKIP_ENTITIES:
            return False
        for p in parts:
            if len(p) <= 1:
                continue
            if not p[0].isupper():
                return False
        return True

    def _valid_company(self, name: str) -> bool:
        if not name or len(name) < 2 or len(name) > 80:
            return False
        norm = self._norm(name)
        if norm in self.SKIP_ENTITIES:
            return False
        compact = name.replace(" ", "").replace(".", "")
        if compact.isdigit():
            return False
        return True

    def _parse_tickers(self, symbol: str) -> List[str]:
        if not symbol:
            return []
        items = []
        for raw in symbol.split(","):
            t = raw.strip().strip('"\'').upper()
            if t and re.match(r"^[A-Z]{1,5}$", t):
                items.append(t)
        return items

    def _refresh_cache(self) -> None:
        self._companies.clear()
        self._tickers.clear()
        self._persons.clear()
        try:
            with self.driver.session() as s:
                for r in s.run(
                    "MATCH (c:Company) "
                    "WHERE c.name IS NOT NULL "
                    "RETURN c.name AS n, c.ticker AS t"
                ):
                    n = r.get("n")
                    t = r.get("t")
                    if n:
                        self._companies[self._norm_company(n)] = n
                    if n and t:
                        clean = t.strip().strip('"\'').upper()
                        if re.match(r"^[A-Z]{1,5}$", clean):
                            self._tickers[clean] = n
                for r in s.run("MATCH (p:Person) RETURN p.name AS n"):
                    n = r.get("n")
                    if n:
                        self._persons.add(self._norm(n))
            self._log(
                f"[KG] Cache: {len(self._companies)} companies, "
                f"{len(self._tickers)} tickers, {len(self._persons)} persons"
            )
        except Exception as e:
            self._log(f"[KG] Cache refresh failed: {e}")

    def _resolve_company(self, name: str) -> Optional[str]:
        if not name or not self._valid_company(name):
            return None
        upper = name.upper().strip()
        if len(upper) <= 5 and upper in self._tickers:
            return self._tickers[upper]
        return self._companies.get(self._norm_company(name))

    def _get_company_for_tickers(self, tickers: List[str]) -> Optional[str]:
        for t in tickers:
            if c := self._tickers.get(t):
                return c
        return None

    def update_kg_from_news(self, news: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            "news_id": news.get("news_id"),
            "events_detected": [],
            "relationships_created": [],
            "success": True,
            "error": None,
        }

        try:
            news_id = int(news.get("news_id", 0))
            if news_id <= 0:
                result["success"] = False
                result["error"] = "Invalid news_id"
                return result
        except (TypeError, ValueError):
            result["success"] = False
            result["error"] = "Invalid news_id"
            return result

        try:
            title = str(news.get("title", "") or "")
            content = str(news.get("content", "") or "")
            text = f"{title}. {content}".strip()
            if len(text) < 10:
                return result

            doc = self.nlp(text)

            persons: List[tuple[str, int]] = []
            orgs: List[tuple[str, int]] = []

            for ent in doc.ents:
                txt = ent.text.strip()
                if ent.label_ == "PERSON" and self._valid_person(txt):
                    persons.append((txt, ent.start_char))
                elif ent.label_ == "ORG" and self._valid_company(txt):
                    orgs.append((txt, ent.start_char))

            tickers = self._parse_tickers(str(news.get("symbol", "")))
            primary_company = self._get_company_for_tickers(tickers)
            ticker = tickers[0] if tickers else None

            with self.driver.session() as session:
                # leadership events
                for raw_pattern, relation, title_role, is_departure in self.LEADERSHIP_PATTERNS:
                    pattern = re.compile(raw_pattern, flags=re.IGNORECASE)
                    for m in pattern.finditer(text):
                        person = m.groupdict().get("person")
                        if not person or not self._valid_person(person):
                            continue

                        company = primary_company
                        if not company:
                            for org, pos in orgs:
                                if abs(pos - m.start()) < 200:
                                    company = self._resolve_company(org) or org
                                    break
                        if not company:
                            continue

                        if is_departure:
                            self._end_relationship(
                                session, person, company, ticker, relation, news, result
                            )
                        else:
                            self._create_relationship(
                                session,
                                person,
                                company,
                                ticker,
                                relation,
                                title_role,
                                news,
                                result,
                            )

                # acquisitions
                for raw_pattern in self.ACQUISITION_PATTERNS:
                    pattern = re.compile(raw_pattern, flags=re.IGNORECASE)
                    for m in pattern.finditer(text):
                        acq = m.group("acquirer").strip()
                        tgt = m.group("target").strip()
                        if (
                            self._valid_company(acq)
                            and self._valid_company(tgt)
                            and self._norm_company(acq) != self._norm_company(tgt)
                        ):
                            self._create_acquisition(session, acq, tgt, news, result)

                self._create_news_event(session, news, tickers)

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)

        return result

    def _create_relationship(
        self,
        session: Session,
        person: str,
        company: str,
        ticker: Optional[str],
        relation: str,
        title: str,
        news: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        rel_map = {
            "IS_CEO_OF": "IS_CEO_OF",
            "IS_CFO_OF": "IS_CFO_OF",
            "IS_CHAIRPERSON_OF": "IS_CHAIRPERSON_OF",
            "IS_BOARD_MEMBER_OF": "IS_BOARD_MEMBER_OF",
        }
        rel = rel_map.get(relation)
        if not rel:
            return

        if ticker:
            company_clause = (
                "MERGE (c:Company {ticker: $ticker}) "
                "ON CREATE SET c.name = coalesce($company, c.name)"
            )
        else:
            company_clause = "MERGE (c:Company {name: $company})"

        query = f"""
            MERGE (p:Person {{name: $person}})
            ON CREATE SET p.title = $title
            {company_clause}
            MERGE (p)-[r:{rel}]->(c)
            ON CREATE SET r.start_date = $ts, r.source_news_id = $nid
            RETURN c.name AS company
        """

        try:
            rows = session.execute_write(
                lambda tx: list(
                    tx.run(
                        query,
                        person=person,
                        title=title,
                        company=company,
                        ticker=ticker or "",
                        ts=int(news.get("timestamp", 0) or 0),
                        nid=int(news["news_id"]),
                    )
                )
            )
            if rows:
                result["events_detected"].append(f"appointment_{rel}")
                result["relationships_created"].append(f"{person} -{rel}-> {company}")
        except Exception:
            pass

    def _end_relationship(
        self,
        session: Session,
        person: str,
        company: str,
        ticker: Optional[str],
        relation: str,
        news: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        match_map = {
            "IS_CEO_OF": "IS_CEO_OF",
            "IS_CFO_OF": "IS_CFO_OF",
            "IS_CHAIRPERSON_OF": "IS_CHAIRPERSON_OF",
            "IS_BOARD_MEMBER_OF": "IS_BOARD_MEMBER_OF",
        }
        rel = match_map.get(relation)
        if not rel:
            return

        cond = "c.ticker = $ticker" if ticker else "c.name = $company"
        query = f"""
            MATCH (p:Person {{name: $person}})-[r:{rel}]->(c:Company)
            WHERE {cond}
            SET r.end_date = $ts, r.ended_by_news_id = $nid
            RETURN c.name AS company
        """
        try:
            rows = session.execute_write(
                lambda tx: list(
                    tx.run(
                        query,
                        person=person,
                        company=company,
                        ticker=ticker or "",
                        ts=int(news.get("timestamp", 0) or 0),
                        nid=int(news["news_id"]),
                    )
                )
            )
            if rows:
                result["events_detected"].append(f"departure_{rel}")
        except Exception:
            pass

    def _create_acquisition(
        self,
        session: Session,
        acquirer: str,
        target: str,
        news: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        acq = self._resolve_company(acquirer) or acquirer
        tgt = self._resolve_company(target) or target
        if self._norm_company(acq) == self._norm_company(tgt):
            return

        try:
            session.execute_write(
                lambda tx: tx.run(
                    """
                    MERGE (a:Company {name: $acq})
                    MERGE (t:Company {name: $tgt})
                    MERGE (a)-[:IS_PARENT_OF]->(t)
                    MERGE (t)-[:IS_SUBSIDIARY_OF]->(a)
                    """,
                    acq=acq,
                    tgt=tgt,
                )
            )
            result["events_detected"].append("acquisition")
            result["relationships_created"].append(f"{acq} -IS_PARENT_OF-> {tgt}")
            self._companies[self._norm_company(acq)] = acq
            self._companies[self._norm_company(tgt)] = tgt
        except Exception:
            pass

    def _create_news_event(
        self,
        session: Session,
        news: Dict[str, Any],
        tickers: List[str],
    ) -> None:
        try:
            session.execute_write(
                lambda tx: tx.run(
                    """
                    MERGE (e:Event {news_id: $nid})
                    ON CREATE SET 
                        e.type = 'NewsMention',
                        e.timestamp = $ts,
                        e.title = $title,
                        e.source = $source,
                        e.url = $url,
                        e.category = $category,
                        e.symbol = $symbol
                    WITH e
                    UNWIND $tickers AS t
                    MERGE (c:Company {ticker: t})
                    MERGE (e)-[:ABOUT]->(c)
                    """,
                    nid=int(news["news_id"]),
                    ts=int(news.get("timestamp", 0) or 0),
                    title=str(news.get("title", ""))[:500],
                    source=str(news.get("source", "")),
                    url=str(news.get("url", "")),
                    category=str(news.get("category", "")),
                    symbol=str(news.get("symbol", "")),
                    tickers=tickers[:5] or [],
                )
            )
        except Exception:
            pass

    def refresh_cache(self) -> None:
        self._refresh_cache()
