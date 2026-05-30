"""Backend that fuses RAG retrieval with agentic actions behind one endpoint."""
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Reuse components from earlier projects:
#   from rag_pipeline import RAGPipeline, openai_llm
#   from agent import Agent; from tools import make_registry

app = FastAPI(title="GenAI Product API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class Query(BaseModel):
    message: str
    use_agent: bool = False


# rag = RAGPipeline(); rag.index_corpus("./corpus")
# agent = Agent(OpenAI(), make_registry())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(q: Query):
    """Route to RAG for knowledge questions, or the agent for action tasks."""
    if q.use_agent:
        # return agent.run(q.message)
        return {"mode": "agent", "answer": f"[agent stub] {q.message}"}
    # return rag.answer(q.message, openai_llm)
    return {"mode": "rag", "answer": f"[rag stub] {q.message}",
            "sources": ["doc_a.txt", "doc_b.txt"]}
