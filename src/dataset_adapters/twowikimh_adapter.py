# src/ingestion/two_wiki_adapter.py
import os
import json
import random
from typing import Optional, Any, List, Tuple

import httpx
import numpy as np
from openai import AsyncOpenAI


class TwoWikiMultihopAdapter:
    """Async adapter for the 2WikiMultihopQA dev set.

    Embeddings are produced via the same Ollama-hosted OpenAI-compatible
    endpoint used by the rest of the pipeline, so no torch/transformers
    dependency is required.
    """

    DOWNLOAD_URL = (
        "https://huggingface.co/datasets/voidful/2WikiMultihopQA/"
        "resolve/main/dev.json"
    )

    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        embedding_endpoint_url: str = "http://localhost:11434/v1",
        embedding_api_key: str = "not-needed",
        dataset_path: Optional[str] = None,
        embedding_batch_size: int = 64,
        dedup_threshold: float = 0.99,
    ):
        self.client = AsyncOpenAI(
            api_key=embedding_api_key, base_url=embedding_endpoint_url
        )
        self.embedding_model = embedding_model
        self.embedding_batch_size = embedding_batch_size
        self.dedup_threshold = dedup_threshold
        self.dataset_path = dataset_path or "./datasets/2wikimultihop_dev.json"

    # ---- Async batched embedding (Ollama /v1/embeddings) --------------
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = await self.client.embeddings.create(
            model=self.embedding_model, input=texts
        )
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    async def _embed_all(self, texts: List[str]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i : i + self.embedding_batch_size]
            for vec in await self._embed_batch(batch):
                out.append(np.asarray(vec, dtype=np.float32))
        return out

    # ---- Raw corpus loading (file preferred, async HTTP fallback) -----
    async def _get_raw_corpus(self) -> List[dict[str, Any]]:
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Fallback: download from HuggingFace and persist locally.
        async with httpx.AsyncClient(timeout=120.0) as http:
            resp = await http.get(self.DOWNLOAD_URL)
            resp.raise_for_status()
            corpus_json = resp.json()

        os.makedirs(os.path.dirname(self.dataset_path) or ".", exist_ok=True)
        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump(corpus_json, f, ensure_ascii=False, indent=4)
        return corpus_json

    # ---- Corpus extraction with batched, in-memory cosine dedup --------
    async def _build_corpus(self, raw_corpus: List[dict[str, Any]]) -> List[str]:
        flat: List[Tuple[str, str]] = []
        for item in raw_corpus:
            for paragraph in item.get("context", []):
                title = paragraph[0]
                sentences = " ".join(paragraph[1])
                flat.append((title, sentences))

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
            stored = np.stack(encountered[title])  # (k, dim)
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
    ) -> Tuple[List[str], List[dict[str, Any]]]:
        raw_corpus = await self._get_raw_corpus()
        if limit is not None and 0 < limit < len(raw_corpus):
            random.seed(seed)
            raw_corpus = random.sample(raw_corpus, limit)

        corpus_list = await self._build_corpus(raw_corpus)
        question_answer_pairs = [
            {
                "question": item["question"],
                "answer": item["answer"].lower(),
                "type": item["type"],
            }
            for item in raw_corpus
        ]
        return corpus_list, question_answer_pairs