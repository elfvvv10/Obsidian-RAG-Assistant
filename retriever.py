"""Retrieve relevant note chunks for a user query."""

from __future__ import annotations

from config import AppConfig
from embeddings import OllamaEmbeddingClient
from reranker import rerank_chunks
from utils import RetrievalFilters, RetrievalOptions, RetrievedChunk
from vector_store import VectorStore


class Retriever:
    """Coordinates query embedding and vector search."""

    def __init__(
        self,
        config: AppConfig,
        embedding_client: OllamaEmbeddingClient,
        vector_store: VectorStore,
    ) -> None:
        self.config = config
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
        options: RetrievalOptions | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k relevant chunks for a question."""
        if self.vector_store.count() == 0:
            raise RuntimeError("The vector store is empty. Run `python main.py index` first.")

        top_k = options.top_k if options and options.top_k is not None else self.config.top_k_results
        candidate_count = (
            options.candidate_count
            if options and options.candidate_count is not None
            else max(top_k, top_k * self.config.retrieval_candidate_multiplier)
        )
        rerank_enabled = (
            options.rerank
            if options and options.rerank is not None
            else self.config.enable_reranking
        )
        boost_tags = options.boost_tags if options else ()

        query_embedding = self.embedding_client.embed_text(query)
        chunks = self.vector_store.query(query_embedding, candidate_count, filters=filters)
        if rerank_enabled or boost_tags:
            chunks = rerank_chunks(
                query,
                chunks,
                boost_tags=boost_tags,
                tag_boost_weight=self.config.tag_boost_weight,
            )
        return chunks[:top_k]
