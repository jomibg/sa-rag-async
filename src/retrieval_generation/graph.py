from collections import deque
from typing import List, Dict, Tuple

async def run_sa(
	adj_dict: Dict[str, List[Tuple]],
    initially_activated: List[str],
    activation_threshold: float,
    pruning_threshold: float) -> Tuple[set, set]:
    """Spreading activation algorithm

    Args:
        adj_dict: Adjacency dictionary mapping entity names to their connections.
        initially_activated: List of initially activated entity names.
        activation_threshold: Threshold for entity activation.
        pruning_threshold: Threshold for pruning relationships.

    Returns:
        A tuple containing (set of relevant triplet indices, set of relevant entity names).
    """
    entity_score = {e: 0 for e in adj_dict.keys()}
    for e in initially_activated:
        entity_score[e] = max(1, entity_score[e])
        visited = set()
        queue = deque([e])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for arc in adj_dict[node]: 
                target, arc_index, prob = arc
                entity_score[target] = min(1, entity_score[target] + prob * entity_score[node])
                if target not in visited:
                    queue.append(target)
    activated_entities = {k for k, v in entity_score.items() if v > activation_threshold}
    relevant_triplets = {a[1] for e, arcs in adj_dict.items() if e in activated_entities
                             for a in arcs if a[0] in activated_entities and a[2] >= pruning_threshold}
    return relevant_triplets, activated_entities

async def create_adj_dict(normalization_parameter, arc_list: List[Dict], initially_activated: List[str]) -> Dict[str, List[Tuple]]:
    """Creates an adjacency dictionary from the list of arcs.

    Args:
        normalization_parameter: Value used to normalize relation similarity.
        arc_list: List of dictionaries containing relation information.
        initially_activated: List of initially activated entity names.

    Returns:
        A dictionary mapping entity names to lists of tuples containing
        (connected entity, arc index, similarity score).
    """
    adj_dict = dict()
    for j, a in enumerate(arc_list):
        normalized_similarity = max(0, (a['similarity'] - normalization_parameter) / 
                                   (1 - normalization_parameter))
        if a['head_entity_name'] not in adj_dict:
            adj_dict[a['head_entity_name']] = []
        if a['tail_entity_name'] not in adj_dict:
            adj_dict[a['tail_entity_name']] = []
        adj_dict[a['head_entity_name']].append(
            (a['tail_entity_name'], j, normalized_similarity))
        adj_dict[a['tail_entity_name']].append(
            (a['head_entity_name'], j, normalized_similarity))
    for n in initially_activated:
        if n not in adj_dict: 
            adj_dict[n] = []
    return adj_dict


class GraphRetrievalMixin:
    """Mixin providing common graph retrieval and diffusion operations (async).

    Assumes the inheriting class has:
    - self.driver (Neo4j AsyncGraphDatabase driver)
    - self.k_hop (int)
    - self.normalization_parameter (float)
    """

    async def _retrieve_k_hop_neighbours(self, seed_names: List[str], k_hop: int) -> List[str]:
        """Retrieve neighbors up to K hops away from seed entities.

        Args:
            seed_names: Initial list of entity names.
            k_hop: Maximum number of hops in the graph.

        Returns:
            Unique list of neighbor entity names within k_hop distance.
        """
        query = f"""
        UNWIND $seed_names AS seedName
        MATCH (s:Entity {{name: seedName}})
        OPTIONAL MATCH (s)-[:RELATED_TO*1..{k_hop}]->(neighbor:Entity)
        RETURN DISTINCT neighbor.name AS name
        """
        async with self.driver.session() as session:
            results = await session.run(query, seed_names=seed_names)
            return [r.data()['name'] async for r in results]

    async def _retrieve_relations(self, entity_names: List[str], query_embedding: List[float]) -> List[Dict]:
        """Retrieve relations among given entities, scored by similarity.

        Args:
            entity_names:  Entities to consider as heads/tails.
            query_embedding: Embedding vector of the query.

        Returns:
            List of relation dicts with head, tail, name, and similarity score.
        """
        query = """
        WITH $entity_names AS e_names, $query_embedding AS queryEmbedding
        MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity)
        WHERE e1.name IN e_names AND e2.name IN e_names
        RETURN DISTINCT {
        head_entity_name: e1.name,
        tail_entity_name: e2.name,
        relationship_name: r.name,
        similarity: gds.similarity.cosine(queryEmbedding, r.embedding)
        } AS relation
        """
        async with self.driver.session() as session:
            results = await session.run(
                query,
                query_embedding=query_embedding,
                entity_names=entity_names,
            )
            return [r.data()['relation'] async for r in results]

    async def _retrieve_activated_entities(self, query_embedding: List[float], top_k: int) -> List[str]:
        """Retrieve top-k entity names whose descriptions match the query embedding.

        Args:
            query_embedding: Embedding vector of the query.
            top_k: Number of top entities to return.

        Returns:
            List of entity names.
        """
        query = """
                WITH $query_embedding AS queryEmbedding
                MATCH (d:Description)
                WITH d, gds.similarity.cosine(d.embedding, queryEmbedding) AS similarity
                ORDER BY similarity DESC
                LIMIT $top_k
                MATCH (d)-[:DESCRIBES]->(e:Entity)
                RETURN DISTINCT e.name AS name
            """
        async with self.driver.session() as s:
            results = await s.run(query, query_embedding=query_embedding, top_k=top_k)
            return [r.data()['name'] async for r in results]

    async def _retrieve_entities(self, query_embedding: List[float], top_k: int) -> List[str]:
        """Retrieve seed entities (top_k by description similarity) plus their k-hop neighbours.

        Returns:
            List of entity names.
        """
        seed_names = await self._retrieve_activated_entities(query_embedding, top_k)
        neighbor_names = await self._retrieve_k_hop_neighbours(seed_names, self.k_hop)
        return list(set(seed_names + neighbor_names))

    async def _retrieve_relevant_documents(
        self,
        entity_names: List[str],
        query_embedding: List[float],
        pruning_threshold: float,
    ) -> List[str]:
        """Retrieves relevant descriptions for the given entities.

        Args:
            entity_names: List of entity names to retrieve descriptions for.
            query_embedding: The embedding vector of the query.
            pruning_threshold: Similarity threshold for document filtering.

        Returns:
            A list of text strings containing entity descriptions.
        """
        query = """
            MATCH (e:Entity)<-[:DESCRIBES]-(d:Document)
            WHERE e.name IN $entity_names
            WITH DISTINCT d, gds.similarity.cosine(d.embedding, $query_embedding) AS similarity
            WHERE similarity >= $threshold
            RETURN d.text as text, similarity
            ORDER BY similarity DESC
        """
        async with self.driver.session() as s:
            results = await s.run(
                query,
                entity_names=entity_names,
                query_embedding=query_embedding,
                threshold=pruning_threshold,
            )
            context = [r.data()['text'] async for r in results]
            return context

    async def _knowledge_acquisition_step(
        self,
        query_embedding: List[float],
        retrieve_k: int,
        activating_descriptions: int,
        activation_threshold: float,
        pruning_threshold: float,
    ) -> str:
        """Run one step of retrieval and diffusion to assemble context.

        Args:
            query_embedding:  Embedding vector of the query.
            retrieve_k: Number of seed entities to retrieve.
            activating_descriptions: Number of descriptions for initial activation.
            activation_threshold: Entity score threshold.
            pruning_threshold:  Relation similarity threshold.

        Returns:
            A formatted context string with descriptions and relationships.
        """

        retrieved_entities = await self._retrieve_entities(query_embedding, retrieve_k)
        activated_entities = await self._retrieve_activated_entities(query_embedding, activating_descriptions)
        retrieved_rels = await self._retrieve_relations(retrieved_entities, query_embedding)

        adj_dict = await create_adj_dict(
            normalization_parameter=self.normalization_parameter,
            arc_list=retrieved_rels,
            initially_activated=activated_entities,
        )
        relevant_triplets, relevant_entities = await run_sa(
            adj_dict=adj_dict,
            initially_activated=activated_entities,
            activation_threshold=activation_threshold,
            pruning_threshold=pruning_threshold,
        )

        documents = await self._retrieve_relevant_documents(
            list(relevant_entities), query_embedding, pruning_threshold
        )
        triplets_context = [
            ' '.join(
                (
                    retrieved_rels[i]['head_entity_name'],
                    retrieved_rels[i]['relationship_name'],
                    retrieved_rels[i]['tail_entity_name'],
                )
            )
            for i in relevant_triplets
        ]
        context = (
            "### Context " + '\n' + '\n\n'.join(documents)
            + '\n' + '**Key relationships**' + '\n' + '\n'.join(triplets_context)
        )
        return context