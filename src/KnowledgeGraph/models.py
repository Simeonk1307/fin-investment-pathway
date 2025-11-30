"""
Data models for Financial Knowledge Graph System
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class RelationType(Enum):
    """Types of relationships in the knowledge graph"""
    IS_CEO_OF = "IS_CEO_OF"
    IS_BOARD_MEMBER_OF = "IS_BOARD_MEMBER_OF"
    WORKS_FOR = "WORKS_FOR"
    FOUNDED = "FOUNDED"
    IS_SUBSIDIARY_OF = "IS_SUBSIDIARY_OF"
    IS_SUPPLIER_OF = "IS_SUPPLIER_OF"
    HAS_PARTNERSHIP_WITH = "HAS_PARTNERSHIP_WITH"
    HAS_ACQUIRER = "HAS_ACQUIRER"
    HAS_TARGET = "HAS_TARGET"
    INVOLVES_PERSON = "INVOLVES_PERSON"

class EventType(Enum):
    """Types of events"""
    ACQUISITION = "Acquisition"
    PARTNERSHIP = "Partnership"
    LEADERSHIP_CHANGE = "LeadershipChange"
    NEWS_UPDATE = "NewsUpdate"
    IPO = "IPO"
    MERGER = "Merger"

@dataclass
class Person:
    """Person entity model"""
    name: str
    title: Optional[str] = None
    wikidata_uri: Optional[str] = None
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

@dataclass
class Company:
    """Company entity model"""
    name: str
    ticker: Optional[str] = None
    industry: Optional[str] = None
    net_worth: Optional[str] = None
    country: Optional[str] = None
    wikidata_uri: Optional[str] = None
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

@dataclass
class Event:
    """Event entity model"""
    event_type: EventType
    date: str
    description: Optional[str] = None
    value: Optional[str] = None
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

@dataclass
class QueryResult:
    """Generic query result wrapper"""
    data: List[Dict[str, Any]]
    count: int
    query_time: Optional[float] = None
