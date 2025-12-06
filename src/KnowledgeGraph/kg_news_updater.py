from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Set
from enum import Enum
import logging
import re
import spacy
from spacy.tokens import Doc, Span
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
    Updates Knowledge Graph based on news articles.
    
    Consistent with WikidataKGBuilder schema:
    - Nodes: Company, Person, Location, Event
    - Relations: IS_CEO_OF, IS_CFO_OF, IS_CHAIRPERSON_OF, IS_BOARD_MEMBER_OF, 
                 FOUNDED, IS_SUBSIDIARY_OF, IS_PARENT_OF, HEADQUARTERED_IN
    """

    # ------------------------------------------------------------------ #
    # Relation types (matching KG builder) - Use Enum for safety
    # ------------------------------------------------------------------ #
    class RelationType(str, Enum):
        IS_CEO_OF = "IS_CEO_OF"
        IS_CFO_OF = "IS_CFO_OF"
        IS_CHAIRPERSON_OF = "IS_CHAIRPERSON_OF"
        IS_BOARD_MEMBER_OF = "IS_BOARD_MEMBER_OF"
        FOUNDED = "FOUNDED"
        IS_SUBSIDIARY_OF = "IS_SUBSIDIARY_OF"
        IS_PARENT_OF = "IS_PARENT_OF"
        HEADQUARTERED_IN = "HEADQUARTERED_IN"

    # ------------------------------------------------------------------ #
    # Confidence thresholds
    # ------------------------------------------------------------------ #
    MIN_CONFIDENCE_THRESHOLD = 0.6
    HIGH_CONFIDENCE_THRESHOLD = 0.8

    # ------------------------------------------------------------------ #
    # Leadership patterns with named groups for person extraction
    # ------------------------------------------------------------------ #
    LEADERSHIP_CONFIG = {
        "ceo": {
            "relation": RelationType.IS_CEO_OF,
            "title": "CEO",
            "appointment_patterns": [
                # Pattern captures person name before/after role mention
                r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:has been\s+)?(?:named|appointed|becomes?|hired as|promoted to)\s+(?:the\s+)?(?:new\s+)?CEO",
                r"(?:named|appointed|hires?)\s+(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:as\s+)?(?:the\s+)?(?:new\s+)?CEO",
                r"(?:new\s+)?CEO[,:\s]+(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:takes?\s+over|assumes?)\s+(?:as\s+)?(?:the\s+)?CEO",
            ],
            "departure_patterns": [
                r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:steps?\s+down|resigns?|leaves?|departs?|exits?|retires?)\s+(?:as\s+)?(?:from\s+)?(?:the\s+)?CEO",
                r"CEO\s+(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:steps?\s+down|resigns?|leaves?|departs?)",
            ],
        },
        "cfo": {
            "relation": RelationType.IS_CFO_OF,
            "title": "CFO",
            "appointment_patterns": [
                r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:has been\s+)?(?:named|appointed|becomes?|hired as|promoted to)\s+(?:the\s+)?(?:new\s+)?CFO",
                r"(?:named|appointed|hires?)\s+(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:as\s+)?(?:the\s+)?(?:new\s+)?CFO",
                r"(?:new\s+)?CFO[,:\s]+(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
            ],
            "departure_patterns": [
                r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:steps?\s+down|resigns?|leaves?)\s+(?:as\s+)?(?:the\s+)?CFO",
                r"CFO\s+(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:steps?\s+down|resigns?|leaves?)",
            ],
        },
        "chairperson": {
            "relation": RelationType.IS_CHAIRPERSON_OF,
            "title": "Chairperson",
            "appointment_patterns": [
                r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:has been\s+)?(?:named|appointed|elected|becomes?)\s+(?:the\s+)?(?:new\s+)?(?:board\s+)?chair(?:man|woman|person)?",
                r"(?:named|appointed|elects?)\s+(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:as\s+)?(?:the\s+)?chair(?:man|woman|person)?",
            ],
            "departure_patterns": [
                r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:steps?\s+down|resigns?|leaves?)\s+(?:as\s+)?chair(?:man|woman|person)?",
            ],
        },
        "board_member": {
            "relation": RelationType.IS_BOARD_MEMBER_OF,
            "title": "Board Member",
            "appointment_patterns": [
                r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:joins?|appointed to|elected to)\s+(?:the\s+)?board",
                r"(?:appoints?|elects?)\s+(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:to\s+)?(?:the\s+)?board",
            ],
            "departure_patterns": [
                r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:leaves?|departs?|resigns?\s+from)\s+(?:the\s+)?board",
            ],
        },
    }

    # ------------------------------------------------------------------ #
    # Acquisition patterns with entity extraction
    # ------------------------------------------------------------------ #
    ACQUISITION_PATTERNS = [
        # acquirer acquires target
        (r"(?P<acquirer>[A-Z][A-Za-z\s&]+?)\s+(?:to\s+)?(?:acquires?|acquired|purchases?|purchased|buys?|bought)\s+(?P<target>[A-Z][A-Za-z\s&]+?)(?:\s+for|\s+in\s+a|\.|,)", 0.85),
        # target acquired by acquirer  
        (r"(?P<target>[A-Z][A-Za-z\s&]+?)\s+(?:to\s+be\s+)?(?:acquired|purchased|bought)\s+by\s+(?P<acquirer>[A-Z][A-Za-z\s&]+?)(?:\s+for|\.|,)", 0.85),
        # acquirer completes acquisition of target
        (r"(?P<acquirer>[A-Z][A-Za-z\s&]+?)\s+(?:completes?|announces?|closes?)\s+(?:the\s+)?acquisition\s+of\s+(?P<target>[A-Z][A-Za-z\s&]+?)(?:\.|,)", 0.9),
    ]

    # ------------------------------------------------------------------ #
    # Entities to skip (not companies)
    # ------------------------------------------------------------------ #
    SKIP_ENTITIES: Set[str] = {
        # Government/regulatory
        "us", "u.s.", "usa", "u.s.a.", "united states", "america", "american",
        "congress", "senate", "house", "white house", "supreme court",
        "sec", "fda", "ftc", "fcc", "doj", "epa", "irs", "fed", "federal reserve",
        "eu", "european union", "uk", "united kingdom", "china", "india",
        # Common false positives
        "reuters", "bloomberg", "associated press", "ap", "afp",
        "wall street", "nasdaq", "nyse", "s&p", "dow jones",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june", 
        "july", "august", "september", "october", "november", "december",
    }

    # Common corporate suffixes for normalization
    CORPORATE_SUFFIXES = re.compile(
        r"\s*[,.]?\s*\b(inc\.?|corp\.?|corporation|company|co\.?|ltd\.?|llc|plc|limited|lp|l\.p\.|group|holdings?)\b\.?\s*$",
        re.IGNORECASE
    )

    def __init__(self, neo4j_config: Neo4jConfig) -> None:
        self.config = neo4j_config
        self.driver: Driver = GraphDatabase.driver(
            neo4j_config.uri,
            auth=(neo4j_config.user, neo4j_config.password),
        )
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"], check=True)
            self.nlp = spacy.load("en_core_web_md")

        # Entity cache for linking
        self._company_name_cache: Dict[str, str] = {}  # normalized -> canonical
        self._company_name_variants: Dict[str, str] = {}  # variants -> canonical
        self._ticker_to_company: Dict[str, str] = {}   # ticker -> canonical name
        self._person_cache: Dict[str, str] = {}  # normalized -> canonical
        
        self._refresh_cache()

    def close(self) -> None:
        self.driver.close()

    # ------------------------------------------------------------------ #
    # Normalization (improved)
    # ------------------------------------------------------------------ #
    def _normalize_name(self, name: str) -> str:
        """Basic normalization: lowercase, single spaces."""
        if not name:
            return ""
        return re.sub(r"\s+", " ", name.strip()).lower()

    def _normalize_company_name(self, name: str) -> str:
        """
        Normalize company name for matching.
        Removes corporate suffixes and normalizes spacing.
        """
        if not name:
            return ""
        # Remove corporate suffixes
        cleaned = self.CORPORATE_SUFFIXES.sub("", name)
        return self._normalize_name(cleaned)

    def _is_valid_person_name(self, name: str) -> bool:
        """Validate that a name looks like a real person name."""
        if not name or len(name) < 3:
            return False
        
        parts = name.split()
        # Should have at least first and last name
        if len(parts) < 2:
            return False
        
        # Each part should start with capital (not all caps unless short)
        for part in parts:
            if len(part) < 2:
                continue
            if part.isupper() and len(part) > 3:
                return False  # Likely acronym
            if not part[0].isupper():
                return False
                
        # Reject if it's a known non-person
        normalized = self._normalize_name(name)
        if normalized in self.SKIP_ENTITIES:
            return False
            
        return True

    def _is_valid_company_name(self, name: str) -> bool:
        """Validate that a name could be a company."""
        if not name or len(name) < 2:
            return False
            
        normalized = self._normalize_name(name)
        
        # Skip known non-companies
        if normalized in self.SKIP_ENTITIES:
            return False
        
        # Skip if purely numeric
        if name.replace(" ", "").replace(".", "").isdigit():
            return False
            
        # Skip very short all-caps (likely acronyms for non-companies)
        if name.isupper() and len(name) <= 3 and name not in self._ticker_to_company:
            return False
            
        return True

    # ------------------------------------------------------------------ #
    # Cache management (improved matching)
    # ------------------------------------------------------------------ #
    def _refresh_cache(self) -> None:
        """Load existing entities from KG for linking."""
        self._company_name_cache.clear()
        self._company_name_variants.clear()
        self._ticker_to_company.clear()
        self._person_cache.clear()
        
        try:
            with self.driver.session() as session:
                # Companies - load name and ticker
                result = session.run("""
                    MATCH (c:Company)
                    WHERE c.name IS NOT NULL
                    RETURN c.name AS name, c.ticker AS ticker
                """)
                for record in result:
                    name = record.get("name")
                    ticker = record.get("ticker")
                    if name:
                        normalized = self._normalize_company_name(name)
                        self._company_name_cache[normalized] = name
                        
                        # Also store without suffix as variant
                        base_normalized = self._normalize_name(name)
                        if base_normalized != normalized:
                            self._company_name_variants[base_normalized] = name
                            
                    if name and ticker:
                        self._ticker_to_company[ticker.upper()] = name

                # Persons
                result = session.run("MATCH (p:Person) RETURN p.name AS name")
                for record in result:
                    if name := record.get("name"):
                        self._person_cache[self._normalize_name(name)] = name

            logger.info(
                "Cache loaded: %d companies, %d tickers, %d persons",
                len(self._company_name_cache),
                len(self._ticker_to_company),
                len(self._person_cache),
            )
        except Exception as e:
            logger.warning("Failed to refresh cache: %s", e)

    def _resolve_company(self, name: str) -> Optional[str]:
        """
        Resolve extracted company name to canonical name in KG.
        Uses strict matching to avoid false positives.
        """
        if not name or not self._is_valid_company_name(name):
            return None

        # Try as ticker first (exact match only)
        upper = name.upper().strip()
        if len(upper) <= 5 and upper in self._ticker_to_company:
            return self._ticker_to_company[upper]

        # Normalize and try exact match
        normalized = self._normalize_company_name(name)
        
        if canonical := self._company_name_cache.get(normalized):
            return canonical
            
        # Try full name normalized
        full_normalized = self._normalize_name(name)
        if canonical := self._company_name_variants.get(full_normalized):
            return canonical

        # Try word-boundary prefix/suffix match (strict)
        for cached_norm, canonical in self._company_name_cache.items():
            if self._is_valid_name_match(normalized, cached_norm):
                return canonical

        return None

    def _is_valid_name_match(self, query: str, cached: str) -> bool:
        """
        Check if query matches cached name with word boundaries.
        Avoids 'apple' matching 'pineapple'.
        """
        if query == cached:
            return True
            
        # Must be at word boundary
        query_words = set(query.split())
        cached_words = set(cached.split())
        
        # Significant word overlap (not just common words)
        common_words = {"the", "a", "an", "and", "&", "of", "in", "for"}
        query_significant = query_words - common_words
        cached_significant = cached_words - common_words
        
        if not query_significant or not cached_significant:
            return False
            
        # At least 50% overlap of significant words
        overlap = len(query_significant & cached_significant)
        min_words = min(len(query_significant), len(cached_significant))
        
        return overlap >= min_words * 0.5 and overlap >= 1

    def _get_company_from_tickers(self, tickers: List[str]) -> Optional[str]:
        """Get canonical company name from ticker list."""
        for ticker in tickers:
            if company := self._ticker_to_company.get(ticker.upper()):
                return company
        return None

    def _resolve_person(self, name: str) -> Optional[str]:
        """Resolve person name to canonical form if exists."""
        if not name or not self._is_valid_person_name(name):
            return None
        normalized = self._normalize_name(name)
        return self._person_cache.get(normalized)

    # ------------------------------------------------------------------ #
    # Entity extraction (improved)
    # ------------------------------------------------------------------ #
    @dataclass
    class ExtractedEntity:
        """Entity with position information for context matching."""
        text: str
        label: str
        start: int
        end: int
        sentence_idx: int

    def _extract_entities(self, doc: Doc) -> Tuple[List["KGNewsUpdater.ExtractedEntity"], List["KGNewsUpdater.ExtractedEntity"]]:
        """Extract validated PERSON and ORG entities with positions."""
        persons: List[KGNewsUpdater.ExtractedEntity] = []
        orgs: List[KGNewsUpdater.ExtractedEntity] = []
        seen_persons: Set[str] = set()
        seen_orgs: Set[str] = set()

        # Build sentence index mapping
        sent_starts = {sent.start: i for i, sent in enumerate(doc.sents)}
        
        def get_sentence_idx(token_idx: int) -> int:
            for start, idx in sorted(sent_starts.items(), reverse=True):
                if token_idx >= start:
                    return idx
            return 0

        for ent in doc.ents:
            text = ent.text.strip()
            normalized = self._normalize_name(text)
            sent_idx = get_sentence_idx(ent.start)

            if ent.label_ == "PERSON":
                if normalized not in seen_persons and self._is_valid_person_name(text):
                    seen_persons.add(normalized)
                    persons.append(self.ExtractedEntity(
                        text=text,
                        label="PERSON",
                        start=ent.start_char,
                        end=ent.end_char,
                        sentence_idx=sent_idx,
                    ))

            elif ent.label_ == "ORG":
                # Only ORG, not GPE (locations are not companies)
                if normalized not in seen_orgs and self._is_valid_company_name(text):
                    seen_orgs.add(normalized)
                    orgs.append(self.ExtractedEntity(
                        text=text,
                        label="ORG",
                        start=ent.start_char,
                        end=ent.end_char,
                        sentence_idx=sent_idx,
                    ))

        return persons, orgs

    # ------------------------------------------------------------------ #
    # Event detection (improved with confidence and entity extraction)
    # ------------------------------------------------------------------ #
    @dataclass
    class DetectedEvent:
        """Detected event with confidence and extracted entities."""
        event_type: str
        confidence: float
        person: Optional[str] = None
        company: Optional[str] = None
        relation: Optional["KGNewsUpdater.RelationType"] = None
        title: Optional[str] = None
        acquirer: Optional[str] = None
        target: Optional[str] = None
        match_start: int = 0
        match_end: int = 0

    def _detect_leadership_events(
        self, 
        text: str, 
        persons: List[ExtractedEntity],
        orgs: List[ExtractedEntity],
    ) -> List[DetectedEvent]:
        """Detect leadership changes with person extraction from pattern."""
        events: List[KGNewsUpdater.DetectedEvent] = []

        for pos_key, config in self.LEADERSHIP_CONFIG.items():
            relation = config["relation"]
            title = config["title"]
            
            # Check appointment patterns
            for pattern in config.get("appointment_patterns", []):
                for match in re.finditer(pattern, text):
                    person_from_pattern = match.group("person") if "person" in match.groupdict() else None
                    
                    # Validate extracted person
                    if person_from_pattern and self._is_valid_person_name(person_from_pattern):
                        events.append(self.DetectedEvent(
                            event_type=f"appointment_{pos_key}",
                            confidence=0.85,
                            person=person_from_pattern.strip(),
                            relation=relation,
                            title=title,
                            match_start=match.start(),
                            match_end=match.end(),
                        ))
                    else:
                        # Try to find person near the match
                        person = self._find_person_near_match(
                            match.start(), match.end(), persons
                        )
                        if person:
                            events.append(self.DetectedEvent(
                                event_type=f"appointment_{pos_key}",
                                confidence=0.65,  # Lower confidence when inferring
                                person=person.text,
                                relation=relation,
                                title=title,
                                match_start=match.start(),
                                match_end=match.end(),
                            ))

            # Check departure patterns
            for pattern in config.get("departure_patterns", []):
                for match in re.finditer(pattern, text):
                    person_from_pattern = match.group("person") if "person" in match.groupdict() else None
                    
                    if person_from_pattern and self._is_valid_person_name(person_from_pattern):
                        events.append(self.DetectedEvent(
                            event_type=f"departure_{pos_key}",
                            confidence=0.85,
                            person=person_from_pattern.strip(),
                            relation=relation,
                            title=title,
                            match_start=match.start(),
                            match_end=match.end(),
                        ))
                    else:
                        person = self._find_person_near_match(
                            match.start(), match.end(), persons
                        )
                        if person:
                            events.append(self.DetectedEvent(
                                event_type=f"departure_{pos_key}",
                                confidence=0.65,
                                person=person.text,
                                relation=relation,
                                title=title,
                                match_start=match.start(),
                                match_end=match.end(),
                            ))

        return events

    def _detect_acquisition_events(
        self,
        text: str,
        orgs: List[ExtractedEntity],
    ) -> List[DetectedEvent]:
        """Detect acquisition events with acquirer/target extraction."""
        events: List[KGNewsUpdater.DetectedEvent] = []

        for pattern, base_confidence in self.ACQUISITION_PATTERNS:
            for match in re.finditer(pattern, text):
                groups = match.groupdict()
                acquirer_text = groups.get("acquirer", "").strip()
                target_text = groups.get("target", "").strip()

                # Validate both entities
                if (self._is_valid_company_name(acquirer_text) and 
                    self._is_valid_company_name(target_text) and
                    acquirer_text.lower() != target_text.lower()):
                    
                    # Boost confidence if entities match known companies
                    confidence = base_confidence
                    if self._resolve_company(acquirer_text):
                        confidence += 0.05
                    if self._resolve_company(target_text):
                        confidence += 0.05

                    events.append(self.DetectedEvent(
                        event_type="acquisition",
                        confidence=min(confidence, 0.95),
                        acquirer=acquirer_text,
                        target=target_text,
                        match_start=match.start(),
                        match_end=match.end(),
                    ))

        return events

    def _find_person_near_match(
        self,
        match_start: int,
        match_end: int,
        persons: List[ExtractedEntity],
        max_distance: int = 100,
    ) -> Optional[ExtractedEntity]:
        """Find the closest person entity to a pattern match."""
        if not persons:
            return None

        best_person = None
        best_distance = float("inf")

        for person in persons:
            # Calculate distance
            if person.end <= match_start:
                distance = match_start - person.end
            elif person.start >= match_end:
                distance = person.start - match_end
            else:
                distance = 0  # Overlapping

            if distance < best_distance and distance <= max_distance:
                best_distance = distance
                best_person = person

        return best_person

    # ------------------------------------------------------------------ #
    # Main update method (interface unchanged)
    # ------------------------------------------------------------------ #
    def update_kg_from_news(self, news: Dict[str, Any]) -> Dict[str, Any]:
        """Update KG from a news article."""
        result = {
            "news_id": news.get("news_id"),
            "events_detected": [],
            "entities_created": [],
            "relationships_created": [],
            "warnings": [],
            "success": True,
            "error": None,
        }

        # Validate news_id
        try:
            news_id = int(news.get("news_id", 0))
            if news_id <= 0:
                result["success"] = False
                result["error"] = "Invalid or missing news_id"
                return result
        except (TypeError, ValueError):
            result["success"] = False
            result["error"] = "news_id must be a valid integer"
            return result

        try:
            title = str(news.get("title", ""))
            content = str(news.get("content", ""))
            text = f"{title}. {content}"
            
            if len(text.strip()) < 10:
                result["warnings"].append("Article text too short for meaningful extraction")
                return result

            doc = self.nlp(text)

            persons, orgs = self._extract_entities(doc)
            symbol = news.get("symbol", "")
            tickers = [t.strip().upper() for t in symbol.split(",") if t.strip()] if symbol else []

            # Detect events with confidence
            leadership_events = self._detect_leadership_events(text, persons, orgs)
            acquisition_events = self._detect_acquisition_events(text, orgs)

            with self.driver.session() as session:
                # Resolve primary company from tickers
                primary_company = self._get_company_from_tickers(tickers)
                primary_ticker = tickers[0] if tickers else None

                # Handle leadership events (filtered by confidence)
                for event in leadership_events:
                    if event.confidence < self.MIN_CONFIDENCE_THRESHOLD:
                        result["warnings"].append(
                            f"Skipped low-confidence event: {event.event_type} ({event.confidence:.2f})"
                        )
                        continue

                    target_company = primary_company
                    if not target_company and orgs:
                        # Find org near the match
                        for org in orgs:
                            if abs(org.start - event.match_start) < 200:
                                target_company = self._resolve_company(org.text) or org.text
                                break

                    if not target_company:
                        result["warnings"].append(f"No target company for: {event.event_type}")
                        continue

                    if event.event_type.startswith("appointment_"):
                        self._handle_leadership_appointment(
                            session, news, event, target_company, primary_ticker, result
                        )
                    elif event.event_type.startswith("departure_"):
                        self._handle_leadership_departure(
                            session, news, event, target_company, primary_ticker, result
                        )

                # Handle acquisitions (filtered by confidence)
                for event in acquisition_events:
                    if event.confidence < self.MIN_CONFIDENCE_THRESHOLD:
                        result["warnings"].append(
                            f"Skipped low-confidence acquisition ({event.confidence:.2f})"
                        )
                        continue

                    self._handle_acquisition(session, news, event, result)

                # Create base news event
                self._create_news_event(
                    session, news, 
                    [p.text for p in persons], 
                    [o.text for o in orgs], 
                    tickers
                )

        except Exception as exc:
            logger.exception("Failed to update KG from news_id=%s", news.get("news_id"))
            result["success"] = False
            result["error"] = str(exc)

        return result

    # ------------------------------------------------------------------ #
    # Neo4j write operations (fixed - no Cypher injection)
    # ------------------------------------------------------------------ #
    
    # Separate queries for each relation type to avoid f-string injection
    _LEADERSHIP_QUERIES = {
        RelationType.IS_CEO_OF: """
            MERGE (p:Person {name: $person})
            ON CREATE SET p.title = $title, p.created_at = timestamp()
            WITH p
            MATCH (c:Company) WHERE c.name = $company OR c.ticker = $ticker
            WITH p, c LIMIT 1
            WHERE c IS NOT NULL
            MERGE (p)-[r:IS_CEO_OF]->(c)
            ON CREATE SET r.start_date = $timestamp, r.source_news_id = $news_id
            RETURN p.name AS person, c.name AS company
        """,
        RelationType.IS_CFO_OF: """
            MERGE (p:Person {name: $person})
            ON CREATE SET p.title = $title, p.created_at = timestamp()
            WITH p
            MATCH (c:Company) WHERE c.name = $company OR c.ticker = $ticker
            WITH p, c LIMIT 1
            WHERE c IS NOT NULL
            MERGE (p)-[r:IS_CFO_OF]->(c)
            ON CREATE SET r.start_date = $timestamp, r.source_news_id = $news_id
            RETURN p.name AS person, c.name AS company
        """,
        RelationType.IS_CHAIRPERSON_OF: """
            MERGE (p:Person {name: $person})
            ON CREATE SET p.title = $title, p.created_at = timestamp()
            WITH p
            MATCH (c:Company) WHERE c.name = $company OR c.ticker = $ticker
            WITH p, c LIMIT 1
            WHERE c IS NOT NULL
            MERGE (p)-[r:IS_CHAIRPERSON_OF]->(c)
            ON CREATE SET r.start_date = $timestamp, r.source_news_id = $news_id
            RETURN p.name AS person, c.name AS company
        """,
        RelationType.IS_BOARD_MEMBER_OF: """
            MERGE (p:Person {name: $person})
            ON CREATE SET p.title = $title, p.created_at = timestamp()
            WITH p
            MATCH (c:Company) WHERE c.name = $company OR c.ticker = $ticker
            WITH p, c LIMIT 1
            WHERE c IS NOT NULL
            MERGE (p)-[r:IS_BOARD_MEMBER_OF]->(c)
            ON CREATE SET r.start_date = $timestamp, r.source_news_id = $news_id
            RETURN p.name AS person, c.name AS company
        """,
    }

    _DEPARTURE_QUERIES = {
        RelationType.IS_CEO_OF: """
            MATCH (p:Person {name: $person})-[r:IS_CEO_OF]->(c:Company)
            WHERE c.name = $company OR c.ticker = $ticker
            SET r.end_date = $timestamp, r.ended_by_news_id = $news_id
            RETURN p.name AS person, c.name AS company
        """,
        RelationType.IS_CFO_OF: """
            MATCH (p:Person {name: $person})-[r:IS_CFO_OF]->(c:Company)
            WHERE c.name = $company OR c.ticker = $ticker
            SET r.end_date = $timestamp, r.ended_by_news_id = $news_id
            RETURN p.name AS person, c.name AS company
        """,
        RelationType.IS_CHAIRPERSON_OF: """
            MATCH (p:Person {name: $person})-[r:IS_CHAIRPERSON_OF]->(c:Company)
            WHERE c.name = $company OR c.ticker = $ticker
            SET r.end_date = $timestamp, r.ended_by_news_id = $news_id
            RETURN p.name AS person, c.name AS company
        """,
        RelationType.IS_BOARD_MEMBER_OF: """
            MATCH (p:Person {name: $person})-[r:IS_BOARD_MEMBER_OF]->(c:Company)
            WHERE c.name = $company OR c.ticker = $ticker
            SET r.end_date = $timestamp, r.ended_by_news_id = $news_id
            RETURN p.name AS person, c.name AS company
        """,
    }

    def _handle_leadership_appointment(
        self,
        session: Session,
        news: Dict,
        event: DetectedEvent,
        target_company: str,
        ticker: Optional[str],
        result: Dict,
    ) -> None:
        """Handle leadership appointment with proper relation creation."""
        if not event.person or not event.relation:
            return

        query = self._LEADERSHIP_QUERIES.get(event.relation)
        if not query:
            result["warnings"].append(f"No query for relation: {event.relation}")
            return

        try:
            records = session.execute_write(
                lambda tx: list(tx.run(
                    query,
                    person=event.person,
                    title=event.title or "",
                    company=target_company,
                    ticker=ticker or "",
                    news_id=int(news["news_id"]),
                    timestamp=int(news.get("timestamp", 0)),
                ))
            )

            if records:
                result["events_detected"].append(event.event_type)
                result["relationships_created"].append(
                    f"{event.person} -{event.relation.value}-> {target_company}"
                )
                self._person_cache[self._normalize_name(event.person)] = event.person
            else:
                result["warnings"].append(
                    f"Company not found for appointment: {target_company}"
                )

        except Exception as e:
            result["warnings"].append(f"Failed to create appointment: {e}")
            logger.error("Appointment creation failed: %s", e)

    def _handle_leadership_departure(
        self,
        session: Session,
        news: Dict,
        event: DetectedEvent,
        target_company: str,
        ticker: Optional[str],
        result: Dict,
    ) -> None:
        """Handle leadership departure by setting end_date on relationship."""
        if not event.person or not event.relation:
            return

        query = self._DEPARTURE_QUERIES.get(event.relation)
        if not query:
            result["warnings"].append(f"No departure query for relation: {event.relation}")
            return

        try:
            records = session.execute_write(
                lambda tx: list(tx.run(
                    query,
                    person=event.person,
                    company=target_company,
                    ticker=ticker or "",
                    news_id=int(news["news_id"]),
                    timestamp=int(news.get("timestamp", 0)),
                ))
            )

            if records:
                result["events_detected"].append(event.event_type)
                logger.info(
                    "Marked departure: %s from %s as %s",
                    event.person, target_company, event.relation.value
                )
            else:
                # No existing relationship found - just log, don't create Event for non-existent departure
                result["warnings"].append(
                    f"No existing {event.relation.value} relationship found for {event.person}"
                )

        except Exception as e:
            result["warnings"].append(f"Failed to record departure: {e}")
            logger.error("Departure recording failed: %s", e)

    def _handle_acquisition(
        self,
        session: Session,
        news: Dict,
        event: DetectedEvent,
        result: Dict,
    ) -> None:
        """Handle acquisition with IS_PARENT_OF and IS_SUBSIDIARY_OF."""
        if not event.acquirer or not event.target:
            return

        # Resolve to canonical names
        acquirer = self._resolve_company(event.acquirer) or event.acquirer
        target = self._resolve_company(event.target) or event.target

        if self._normalize_company_name(acquirer) == self._normalize_company_name(target):
            result["warnings"].append(f"Acquirer same as target: {acquirer}")
            return

        try:
            session.execute_write(
                lambda tx: tx.run(
                    """
                    MERGE (acq:Company {name: $acquirer})
                    ON CREATE SET acq.created_at = timestamp(), acq.created_from_news = $news_id
                    MERGE (tgt:Company {name: $target})
                    ON CREATE SET tgt.created_at = timestamp(), tgt.created_from_news = $news_id
                    
                    MERGE (acq)-[r1:IS_PARENT_OF]->(tgt)
                    ON CREATE SET r1.acquisition_date = $timestamp, r1.source_news_id = $news_id
                    
                    MERGE (tgt)-[r2:IS_SUBSIDIARY_OF]->(acq)
                    ON CREATE SET r2.acquisition_date = $timestamp, r2.source_news_id = $news_id
                    
                    MERGE (e:Event {news_id: $news_id})
                    ON CREATE SET 
                        e.type = 'Acquisition',
                        e.timestamp = $timestamp,
                        e.title = $news_title
                    MERGE (e)-[:INVOLVES_COMPANY]->(acq)
                    MERGE (e)-[:INVOLVES_COMPANY]->(tgt)
                    """,
                    acquirer=acquirer,
                    target=target,
                    news_id=int(news["news_id"]),
                    timestamp=int(news.get("timestamp", 0)),
                    news_title=str(news.get("title", "")),
                )
            )

            result["events_detected"].append("acquisition")
            result["relationships_created"].append(f"{acquirer} -IS_PARENT_OF-> {target}")
            
            # Update cache
            self._company_name_cache[self._normalize_company_name(acquirer)] = acquirer
            self._company_name_cache[self._normalize_company_name(target)] = target

        except Exception as e:
            result["warnings"].append(f"Failed to create acquisition: {e}")
            logger.error("Acquisition creation failed: %s", e)

    def _create_news_event(
        self,
        session: Session,
        news: Dict,
        persons: List[str],
        orgs: List[str],
        tickers: List[str],
    ) -> None:
        """Create base news event with validated mentions."""
        # Filter to only valid entities
        valid_persons = [p for p in persons if self._is_valid_person_name(p)]
        valid_orgs = [o for o in orgs if self._is_valid_company_name(o)]

        try:
            session.execute_write(
                lambda tx: tx.run(
                    """
                    MERGE (e:Event {news_id: $news_id})
                    ON CREATE SET 
                        e.type = COALESCE(e.type, 'NewsMention'),
                        e.timestamp = $timestamp,
                        e.title = $title,
                        e.category = $category,
                        e.source = $source,
                        e.url = $url,
                        e.symbol = $symbol
                    
                    WITH e
                    UNWIND CASE WHEN size($persons) > 0 THEN $persons ELSE [null] END AS pname
                    FOREACH (_ IN CASE WHEN pname IS NOT NULL THEN [1] ELSE [] END |
                        MERGE (p:Person {name: pname})
                        MERGE (e)-[:MENTIONS]->(p)
                    )
                    
                    WITH e
                    UNWIND CASE WHEN size($tickers) > 0 THEN $tickers ELSE [null] END AS t
                    FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
                        MERGE (c:Company {ticker: t})
                        MERGE (e)-[:ABOUT]->(c)
                    )
                    """,
                    news_id=int(news["news_id"]),
                    timestamp=int(news.get("timestamp", 0)),
                    title=str(news.get("title", ""))[:500],
                    category=str(news.get("category", "")),
                    source=str(news.get("source", "")),
                    url=str(news.get("url", "")),
                    symbol=str(news.get("symbol", "")),
                    persons=valid_persons[:10],  # Limit to prevent huge queries
                    tickers=tickers[:5],
                )
            )
        except Exception as e:
            logger.warning("Failed to create news event: %s", e)

    def refresh_cache(self) -> None:
        """Manually refresh entity cache."""
        self._refresh_cache()