# src/ingestion/musique_qa_adapter.py
import os
import json
import random
from typing import Optional, Any, List, Tuple

import numpy as np
from openai import AsyncOpenAI


class MusiqueQAAdapter:
    """Async adapter for the Musique QA dataset.

    Uses the same Ollama-hosted, OpenAI-compatible embedding endpoint as the rest
    of the pipeline, so no torch/transformers dependency is required.
    """

    DOWNLOAD_URL = (
        "https://drive.google.com/file/d/"
        "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing"
    )

    def __init__(
        self,
        embedding_model: str = "bge-large:latest",
        embedding_endpoint_url: str = "http://localhost:11434/v1",
        embedding_api_key: str = "not-needed",
        dataset_path: Optional[str] = None,
        embedding_batch_size: int = 64,
        dedup_threshold: float = 0.99,
    ):
        """Initialize the adapter with embedding endpoint and dedup settings."""
        self.client = AsyncOpenAI(
            api_key=embedding_api_key, base_url=embedding_endpoint_url
        )
        self.embedding_model = embedding_model
        self.embedding_batch_size = embedding_batch_size
        self.dedup_threshold = dedup_threshold
        self.dataset_path = dataset_path or "./datasets/musique_data.jsonl"

    # ---- Async batched embedding (Ollama /v1/embeddings) ---------------
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts in a single API call, preserving input order."""
        if not texts:
            return []
        resp = await self.client.embeddings.create(
            model=self.embedding_model, input=texts
        )
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    async def _embed_all(self, texts: List[str]) -> List[np.ndarray]:
        """Embed all texts in batches of `embedding_batch_size`."""
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i : i + self.embedding_batch_size]
            for vec in await self._embed_batch(batch):
                out.append(np.asarray(vec, dtype=np.float32))
        return out

    # ---- Pure-CPU helpers (unchanged from original) -------------------
    def _get_golden_context(self, item: dict[str, Any]) -> str:
        """Build the golden supporting context for a Musique item."""
        golden_context = []
        paragraphs = item.get("paragraphs", [])
        for step in item.get("question_decomposition", []):
            support_idx = step.get("paragraph_support_idx")
            if isinstance(support_idx, int) and 0 <= support_idx < len(paragraphs):
                para = paragraphs[support_idx]
                golden_context.append(f"{para['title']}:  {para['paragraph_text']}")
            golden_context.append(f"Q: {step['question']}")
            golden_context.append(f"A: {step['answer']}")
            golden_context.append("")
        return "\n".join(golden_context)

    def _process_context(self, golden_context: str) -> str:
        """Filter golden context to keep only question and answer lines."""
        context = golden_context.split("\n")
        new_context = []
        for p in context:
            if not p:
                continue
            if p[:2] in {"A:", "Q:"}:
                new_context.append(p)
        return "\n".join(new_context)

    def _extract_entities(self, new_context: str) -> str:
        """Extract answer entities from processed context, joined by '|'."""
        context = new_context.split("\n")
        entities = []
        for p in context:
            if p.startswith("A:"):
                entities.append(p[2:].strip())
        return " | ".join(entities)

    def _get_question_answer_pair(
        self, item: dict[str, Any], load_golden_context: bool = False
    ) -> dict[str, Any]:
        """Extract a question-answer pair from a raw dataset item.

        Optionally includes golden context and entities.
        """
        qa_pair = {
            "id": item.get("id", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", "").lower()
            if isinstance(item.get("answer"), str)
            else item.get("answer"),
        }
        if load_golden_context:
            golden_context = self._get_golden_context(item)
            new_context = self._process_context(golden_context)
            entities = self._extract_entities(new_context)
            qa_pair["golden_context"] = golden_context
            qa_pair["new_context"] = new_context
            qa_pair["entities"] = entities
        return qa_pair

    # ---- Raw corpus loading (file I/O, sync is fine here) -------------
    def _get_raw_corpus(self) -> List[dict[str, Any]]:
        """Load raw Musique corpus from the local dataset file."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset file not found at {self.dataset_path}. "
                f"Please download it manually from: {self.DOWNLOAD_URL} "
                f"and save it to {self.dataset_path}"
            )
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    # ---- Corpus extraction with batched, in-memory cosine dedup --------
    async def _build_corpus(self, raw_corpus: List[dict[str, Any]]) -> List[str]:
        """Build a deduplicated corpus from raw items via cosine-similarity.

        Removes paragraphs with the same title whose embedding similarity
        exceeds `dedup_threshold`.
        """
        # Gather all (title, text) pairs from every item, preserving order.
        flat: List[Tuple[str, str]] = []
        for item in raw_corpus:
            for para in item.get("paragraphs", []):
                flat.append((para["title"], para["paragraph_text"]))

        # One batched embedding pass over ALL paragraphs (same count the
        # original embeds — it embeds every paragraph to store in
        # encountered_docs, including ones later deduped away).
        texts = [t for (_, t) in flat]
        embeddings = await self._embed_all(texts)

        encountered: dict[str, List[np.ndarray]] = {}
        corpus_list: List[str] = []
        for (title, text), emb in zip(flat, embeddings):
            emb_unit = emb / np.linalg.norm(emb)
            if title not in encountered:
                corpus_list.append(text)
                encountered[title] = [emb_unit]
                continue
            stored = np.stack(encountered[title])  # (k, dim), unit-norm rows
            sims = stored @ emb_unit               # cosine similarities
            if sims.max() < self.dedup_threshold:
                corpus_list.append(text)
                encountered[title].append(emb_unit)
        return corpus_list

    # ---- Public entry point (async) -----------------------------------
    async def aload_corpus(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
        load_golden_context: bool = True,
    ) -> Tuple[List[str], List[dict[str, Any]]]:
        """Load and return a (corpus, qa_pairs) tuple, optionally sampling `limit` items."""
        raw_corpus = self._get_raw_corpus()
        if limit is not None and 0 < limit < len(raw_corpus):
            random.seed(seed)
            raw_corpus = random.sample(raw_corpus, limit)

        corpus_list = await self._build_corpus(raw_corpus)
        question_answer_pairs = [
            self._get_question_answer_pair(item, load_golden_context)
            for item in raw_corpus
        ]
        return corpus_list, question_answer_pairs