from pydantic import BaseModel, Field
from typing import List
from rag_system.rag_pipeline import CulinaryRAG

rag = CulinaryRAG()
rag.load_index()

class CulinaryQuery(BaseModel):
    question: str = Field(..., description="Culinary description to analyze")

def culinary_rag_tool(input: CulinaryQuery) -> str:
    """Agent action to perform culinary origin retrieval."""
    docs = rag.query(input.question)
    context = "\n".join([d.page_content for d in docs])
    return f"Retrieved context:\n{context}"
