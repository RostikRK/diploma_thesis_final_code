from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class Node(BaseModel):
    """Represents a node in the knowledge graph."""
    id: str
    type: str
    properties: Dict[str, Any] = {}


class Relationship(BaseModel):
    """Represents a relationship in the knowledge graph."""
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = {}


class Graph(BaseModel):
    """Represents a subgraph with nodes and relationships."""
    nodes: List[Node]
    relationships: List[Relationship]


class QueryRequest(BaseModel):
    """Request model for processing a user query."""
    query: str
    chat_history: Optional[List[Dict[str, str]]] = None


class QueryResponse(BaseModel):
    """Response model for a processed query."""
    content: str
    references: List[Dict[str, Any]] = []
    image_urls: List[str] = []
    error: Optional[str] = None


class LLMRequest(BaseModel):
    """Request model for LLM inference."""
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.4
    max_tokens: int = 1024
    top_p: float = 0.9
    repetition_penalty: float = 1.1


class LLMResponse(BaseModel):
    """Response model from LLM inference."""
    text: str
    usage: Dict[str, int] = Field(default_factory=dict)


class RouterResponse(BaseModel):
    """Response model from query router."""
    query_type: str
    parameters: Dict[str, Any] = {}