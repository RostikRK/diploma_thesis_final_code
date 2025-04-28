import os
from typing import Dict, Any

# Set up start
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password1234")

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")

IMAGE_API_URL = os.environ.get("IMAGE_API_URL", "http://localhost:8082/image")

LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://localhost:5000")
# LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit")
# LLM_LOCAL_MODE = os.environ.get("LLM_LOCAL_MODE", "False").lower() in ("true", "1", "t")

API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# Set up end

# System prompts
SYSTEM_PROMPTS: Dict[str, str] = {
    "router": """You are an expert system that analyzes user queries and determines their intent.
Your job is to classify queries related to electronic components and extract relevant parameters.""",

    "cross_reference": """You are a technical assistant specialized in electronic component cross-references.
Your task is to present alternative components clearly and concisely.""",

    "entity_search": """You are a knowledge graph assistant specialized in electronic components.
Your task is to present factual information about components, manufacturers, and specifications.""",

    "general_search": """You are a technical documentation assistant specialized in electronic components.
Your task is to provide helpful information from datasheets and technical documentation."""
}