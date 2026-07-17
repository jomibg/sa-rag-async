import asyncio
import time
from pathlib import Path

from ingestion.kg_ingestion import AdvancedKGIngestor


async def amain():
    folder_path = Path("./test_data")
    text_list = [f.read_text(encoding="utf-8") for f in folder_path.glob("*.txt")]

    ingestion_pipeline = AdvancedKGIngestor(
        neo4j_url="bolt://3.239.36.253",
        neo4j_user="neo4j",
        neo4j_password="accruals-chamber-standardization",
        ner_model="phi4-mini:latest",
        re_model="phi4-mini:latest",
        llm_endpoint_url="http://localhost:11434/v1",
        llm_api_key="not-needed",
        template_re_loc="../prompts/re.txt",
        template_ner_loc="../prompts/ner.txt",
    )

    # Tune `concurrency` to the capacity of your Ollama/model server.
    # Start low (2-4) and raise it while watching GPU/CPU utilization.
    await ingestion_pipeline.ingest(text_list, concurrency=4)


if __name__ == "__main__":
    start_time = time.perf_counter()
    asyncio.run(amain())
    duration = time.perf_counter() - start_time
    print(f"Script finished in {duration:.4f} seconds")