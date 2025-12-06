"""
KG Query Tools — Neo4j Knowledge Graph Tools for LangChain Agents
Automatically wraps each method of KGQueryService as an LLM-callable tool.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from neo4j import Driver

from src.KnowledgeGraph.models import QueryResult, RelationType
from src.KnowledgeGraph.kg_queries import KGQueryService

logger = logging.getLogger(__name__)

class CompanyInput(BaseModel):
    company_name: str = Field(..., description="Name of the company.")


class PersonInput(BaseModel):
    person_name: str = Field(..., description="Name of the person.")


class EventIdInput(BaseModel):
    event_id: int = Field(..., description="Internal Neo4j Event ID.")


class SimilarityInput(BaseModel):
    stock_name: str
    similarity_factors: Optional[List[str]] = Field(
        default=None,
        description="List of similarity factors: ['industry', 'leadership', 'partnerships']"
    )


class ImpactInput(BaseModel):
    source_company: str
    sentiment_score: float = Field(0.0, description="Sentiment score (-1 to 1).")


class TemporalEventInput(BaseModel):
    company_name: Optional[str] = None
    industry: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class AffectedEntityInput(BaseModel):
    company_name: Optional[str] = None
    person_name: Optional[str] = None
    max_hops: int = 2


class ClusterInput(BaseModel):
    stock_names: List[str]


class KGQueryTool(BaseTool):
    """
    Generic wrapper that allows ANY KGQueryService method to be used as a LangChain Tool.
    """
    name: str
    description: str
    args_schema: type

    def __init__(self, kg: KGQueryService, method_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kg = kg
        self.method_name = method_name

    def _run(self, **kwargs):
        method = getattr(self.kg, self.method_name)
        result = method(**kwargs)

        if isinstance(result, QueryResult):
            return result.model_dump()
        return result

    async def _arun(self, **kwargs):
        return self._run(**kwargs)


def build_kg_tools(driver: Driver) -> List[BaseTool]:
    """
    Automatically wraps all query methods from KGQueryService into tools.

    Returns:
        List[BaseTool] → pass directly to LangChain initialize_agent().
    """

    kg = KGQueryService(driver)

    TOOL_CONFIG = [
        # Person–Company relationships
        ("get_ceos_and_board_members", CompanyInput, "Get CEOs and board members of a company."),
        ("get_companies_by_person", PersonInput, "Get all companies associated with a person."),

        # Company relationships
        ("get_subsidiaries_and_parents", CompanyInput, "Get subsidiaries and parent companies."),
        ("get_partnerships", CompanyInput, "Get partnership relations."),
        ("get_suppliers_and_clients", CompanyInput, "Get suppliers and clients."),

        # Event-related
        ("get_acquisition_events", CompanyInput, "Get acquisition events involving a company."),
        ("get_companies_in_event", EventIdInput, "Get companies involved in an event."),
        ("get_events_by_person", PersonInput, "Get events involving a person."),

        # Similarity & Clustering
        ("find_similar_stocks", SimilarityInput, "Find similar stocks based on various factors."),
        ("cluster_user_stocks", ClusterInput, "Cluster user's stock list via graph connectivity."),

        # Impact & Reasoning
        ("get_affected_entities_by_news", AffectedEntityInput, "Get affected entities around a news event."),
        ("get_executive_networks", CompanyInput, "Get companies connected through shared executives."),
        ("calculate_impact_scores", ImpactInput, "Calculate graph-propagated impact scores."),

        # Temporal analysis
        ("get_temporal_events", TemporalEventInput, "Get temporal event sequence filtered by company or industry."),
    ]

    tools = []
    for name, schema, description in TOOL_CONFIG:
        tool = KGQueryTool(
            kg=kg,
            method_name=name,
            name=name,
            description=description,
            args_schema=schema
        )
        tools.append(tool)

    return tools


def load_kg_tools(uri: str, user: str, password: str):
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return build_kg_tools(driver)
