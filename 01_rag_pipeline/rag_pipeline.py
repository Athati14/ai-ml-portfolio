"""Minimal but production-shaped RAG pipeline with a pluggable LLM + evaluator."""
from __future__ import annotations

import os, glob, json
from dataclasses import dataclass, field
from typing import List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    doc_id: str
    text: str
    embedding: np.ndarray | None = None


@dataclass
class RAGConfig:
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 4


class VectorStore:
    """Thin FAISS wrapper that keeps the chunk metadata alongside the index."""
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)   # cosine sim on normalized vectors
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk]) -> None:
        mat = np.vstack([c.embedding for c in chunks]).astype("float32")
        faiss.normalize_L2(mat)
        self.index.add(mat)
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int) -> List[Chunk]:
        q = query_vec.astype("float32").reshape(1, -1)
        faiss.normalize_L2(q)
        _, idx = self.index.search(q, k)
        return [self.chunks[i] for i in idx[0] if i != -1]


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    words, out, step = text.split(), [], max(1, size - overlap)
    for i in range(0, len(words), step):
        out.append(" ".join(words[i:i + size]))
    return out


class RAGPipeline:
    def __init__(self, cfg: RAGConfig = RAGConfig()):
        self.cfg = cfg
        self.embedder = SentenceTransformer(cfg.embed_model)
        self.store: VectorStore | None = None

    def index_corpus(self, folder: str) -> None:
        chunks: List[Chunk] = []
        for path in glob.glob(os.path.join(folder, "**/*.txt"), recursive=True):
            with open(path, encoding="utf-8") as fh:
                for piece in chunk_text(fh.read(), self.cfg.chunk_size, self.cfg.chunk_overlap):
                    chunks.append(Chunk(doc_id=os.path.basename(path), text=piece))
        vecs = self.embedder.encode([c.text for c in chunks], show_progress_bar=True)
        for c, v in zip(chunks, vecs):
            c.embedding = v
        self.store = VectorStore(dim=vecs.shape[1])
        self.store.add(chunks)

    def retrieve(self, question: str) -> List[Chunk]:
        assert self.store, "Call index_corpus() first."
        qv = self.embedder.encode([question])[0]
        return self.store.search(qv, self.cfg.top_k)

    def answer(self, question: str, llm) -> dict:
        ctx = self.retrieve(question)
        context_block = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in ctx)
        prompt = (
            "Answer using ONLY the context. If unsupported, say you don't know.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
        )
        return {"question": question, "answer": llm(prompt),
                "contexts": [c.text for c in ctx],
                "sources": sorted({c.doc_id for c in ctx})}


def openai_llm(prompt: str) -> str:
    """Swap for any local model; kept optional so the repo runs without a key."""
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    rag = RAGPipeline()
    rag.index_corpus("./corpus")
    out = rag.answer("What is the maximum allowable beam deflection?", openai_llm)
    print(json.dumps(out, indent=2))
