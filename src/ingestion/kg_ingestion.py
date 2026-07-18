import asyncio
from tqdm import tqdm
from typing import List, Optional
from neo4j import AsyncGraphDatabase

from .chunking import chunk_documents
from .knowledge_graph_construction import AdvancedKGConstructor

_SENTINEL = object()  # end-of-stream marker pushed into the queues


class AdvancedKGIngestor:
    def __init__(
        self,
        neo4j_url: str,
        neo4j_user: str,
        neo4j_password: str,
        template_re_loc: str,
        template_ner_loc: str,
        ner_model: str = "qwen2.5:7b",
        re_model: str = "qwen2.5:7b",
        embedding_model: str = "bge-large:latest",
        chunking_method: str = "word_based",
        chunk_size: int = 500,
        overlap_size: int = 200,
        llm_endpoint_url: str = "",
        llm_api_key: str = "",
    ):
        self.chunking_method = chunking_method
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.driver = AsyncGraphDatabase.driver(
            neo4j_url, auth=(neo4j_user, neo4j_password),
            notifications_disabled_categories=["DEPRECATION"],
        )
        self.kg_pipeline = AdvancedKGConstructor(
            template_re_loc=template_re_loc,
            template_ner_loc=template_ner_loc,
            ner_model=ner_model,
            re_model=re_model,
            llm_endpoint_url=llm_endpoint_url,
            llm_api_key=llm_api_key,
            embedding_model=embedding_model
        )

    # ---- Neo4j write helpers (run serially on one async session) ----

    async def create_entity_description(self, session, e_description, e_name, embedding):
        query = """
            MERGE (d:Description {text: $e_description})
            SET d.embedding = $e_embedding
            WITH d
            MATCH (e:Entity {name: $e_name})
            MERGE (d)-[r:DESCRIBES]->(e)
            RETURN r
        """
        result = await session.run(
            query, e_description=e_description, e_name=e_name, e_embedding=embedding
        )
        await result.consume()

    async def create_entity_node(self, session, e_name, e_type, e_aliases):
        # Don't mutate the caller's list; build a fresh one.
        aliases = list(e_aliases) + [e_name]
        query = """
        OPTIONAL MATCH (e:Entity)
        WHERE e.name = $e_name
        OR $e_name IN e.aliases
        OR any(alias IN $e_aliases WHERE alias = e.name)
        WITH e
        CALL apoc.do.when(
            e IS NULL,
            'CREATE (newEntity:Entity {name: $e_name, e_type: $e_type, aliases: $e_aliases})
             RETURN newEntity AS resultEntity',
            'SET entity.aliases = entity.aliases + [x IN $e_aliases WHERE NOT x IN entity.aliases],
                entity.e_type = $e_type
             RETURN entity AS resultEntity',
            {entity: e, e_name: $e_name, e_type: $e_type, e_aliases: $e_aliases}
        ) YIELD value
        RETURN value.resultEntity.name AS name
        """
        result = await session.run(
            query, e_name=e_name, e_type=e_type, e_aliases=aliases
        )
        name = None
        async for record in result:
            name = record["name"]
            break
        await result.consume()
        return name

    async def create_relationship_entity_entity(self, session, h_name, t_name, rel_name, rel_embedding):
        query = """
            MATCH (e1:Entity {name: $h_name})
            WITH e1
            MATCH (e2:Entity {name: $t_name})
            MERGE (e1)-[r:RELATED_TO {name: $rel_name}]->(e2)
            ON CREATE SET r.embedding = $rel_embedding
        """
        result = await session.run(
            query, h_name=h_name, t_name=t_name, rel_name=rel_name, rel_embedding=rel_embedding
        )
        await result.consume()

    async def create_chunk_node(self, session, chunk, index, embedding):
        query = """
        MERGE (d:Document {doc_id: $doc_id, text: $text})
        SET d.embedding = $embedding
        RETURN d
        """
        result = await session.run(query, doc_id=index, text=chunk, embedding=embedding)
        await result.consume()

    async def connect_chunk_entity(self, session, chunk_id, entity_name):
        query = """
        MATCH (d:Document {doc_id: $doc_id})
        WITH d
        MATCH (e:Entity {name: $e_name})
        MERGE (d)-[r:DESCRIBES]->(e)
        RETURN r
        """
        result = await session.run(query, e_name=entity_name, doc_id=chunk_id)
        await result.consume()

    # ---- Per-chunk DB write step ----
    async def _write_chunk(
        self,
        session,
        index,
        chunk,
        entities,
        relationships,
        chunk_embedding,
        desc_embeddings,
        rel_embeddings,
    ):
        await self.create_chunk_node(session, chunk, index, chunk_embedding)
        for e, e_emb in zip(entities, desc_embeddings):
            retrieved_name = await self.create_entity_node(
                session, e.name, e.type, e.aliases
            )
            if retrieved_name is None:
                continue
            await self.create_entity_description(
                session, e.entity_information, retrieved_name, e_emb
            )
            await self.connect_chunk_entity(session, index, retrieved_name)
        for (h, rel, t), r_emb in zip(relationships, rel_embeddings):
            await self.create_relationship_entity_entity(
                session, h, t, rel, r_emb
            )

    async def create_vector_index(self, session):
        query = (
            "CREATE VECTOR INDEX textEmbedding IF NOT EXISTS "
            "FOR (n:Document) ON n.embedding"
        )
        result = await session.run(query)
        await result.consume()

    # ---- Async pipeline: producer / K extractors / 1 writer ----
    async def ingest(self, corpus_list: List[str], concurrency: int = 4):
        chunks = list(
            chunk_documents(
                corpus_list,
                self.chunking_method,
                self.chunk_size,
                self.overlap_size,
            )
        )
        total_chunks = len(chunks)

        extract_queue: "asyncio.Queue" = asyncio.Queue()
        writer_queue: "asyncio.Queue" = asyncio.Queue()

        async def producer():
            for index, chunk in chunks:
                await extract_queue.put((index, chunk))
            for _ in range(concurrency):
                await extract_queue.put(_SENTINEL)

        async def extractor():
            while True:
                item = await extract_queue.get()
                try:
                    if item is _SENTINEL:
                        break
                    index, chunk = item
                    (
                        entities,
                        relationships,
                        chunk_embedding,
                        desc_embeddings,
                        rel_embeddings,
                    ) = await self.kg_pipeline.extract_entities_and_relationships(chunk)
                    await writer_queue.put(
                        (
                            index,
                            chunk,
                            entities,
                            relationships,
                            chunk_embedding,
                            desc_embeddings,
                            rel_embeddings,
                        )
                    )
                finally:
                    extract_queue.task_done()
            await writer_queue.put(_SENTINEL)

        async def writer():
            finished = 0
            async with self.driver.session() as session:
                # close() is guaranteed on exception via the `with`, so the bar
                # never leaves a stale line if _write_chunk raises.
                with tqdm(
                    total=total_chunks,
                    unit="chunk",
                    desc="ingest",
                    unit_scale=True,      # shows "5.2k/10k" on big corpora
                    dynamic_ncols=True,  # adapts to terminal width
                    smoothing=0.0,        # show true counts, not running average
                ) as pbar:
                    while finished < concurrency:
                        item = await writer_queue.get()
                        try:
                            if item is _SENTINEL:
                                finished += 1
                                continue
                            (
                                index, chunk, entities, relationships,
                                chunk_embedding, desc_embeddings, rel_embeddings,
                            ) = item
                            await self._write_chunk(
                                session, index, chunk, entities, relationships,
                                chunk_embedding, desc_embeddings, rel_embeddings,
                            )
                            pbar.set_postfix_str(f"chunk={index}")
                            pbar.update(1)
                        finally:
                            writer_queue.task_done()

        await asyncio.gather(
            producer(),
            *[extractor() for _ in range(concurrency)],
            writer(),
        )

        # ---- Final step: create vector index for Document embeddings ----
        async with self.driver.session() as session:
            await self.create_vector_index(session)
            
        await self.driver.close()