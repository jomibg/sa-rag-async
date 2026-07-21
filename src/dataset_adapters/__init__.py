from typing import Optional
from .musique_adapter import MusiqueQAAdapter
from .twowikimh_adapter import TwoWikiMultihopAdapter

VALID_ADAPTERS = {
    "TwoWikiMultiHop": TwoWikiMultihopAdapter,
    "MuSiQuE": MusiqueQAAdapter,
}

async def load_corpus(
    adapter_name: str,
    embedding_model: str,
    embedding_endpoint_url: str,
    embedding_api_key: str,
    dataset_path: str,
    limit: Optional[int] = None,
):
    if adapter_name not in VALID_ADAPTERS:
        raise ValueError(f"Not a valid adapter name: {adapter_name}")
    adapter = VALID_ADAPTERS[adapter_name](
        embedding_model=embedding_model,
        embedding_endpoint_url=embedding_endpoint_url,
        embedding_api_key=embedding_api_key,
        dataset_path=dataset_path,
    )
    return await adapter.aload_corpus(limit=limit)


__all__ = ["MusiqueQAAdapter", "TwoWikiMultihopAdapter", "load_corpus"]