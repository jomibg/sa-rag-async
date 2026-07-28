from typing import Any, Dict, List, Annotated
from pydantic import BaseModel, RootModel, Field
from openai import AsyncOpenAI

class NEREntity(BaseModel):
    """A named entity with type, aliases, and descriptive info."""

    name: str
    type: str
    aliases: List[str] = Field(default_factory=list)
    entity_information: str = ""


class NEREntityList(RootModel[List[NEREntity]]):
    """Root model wrapping a list of NER entities."""

    pass


Triple = Annotated[List[str], Field(min_length=3, max_length=3)]

class RelationTriples(BaseModel):
    """Structured LLM response containing relationship triplets."""

    triples: List[Triple]


struct_out_ner = NEREntityList
struct_out_re = RelationTriples


class AdvancedKGConstructor:
    """Extracts entities and relationship triplets from text to build a knowledge graph."""

    def __init__(self, template_ner_loc: str, template_re_loc: str, #embedding_pipeline: Any,
                 llm_endpoint_url: str,
                 llm_api_key: str,
                 ner_model: str = "qwen2.5:3b",
                 re_model: str = "hermes3",
                 embedding_model: str = "bge-large:latest"):
        """Initialize the constructor with NER/RE templates and model settings.

        Args:
            template_ner_loc: File path to the NER template.
            template_re_loc: File path to the RE template.
            llm_endpoint_url: URL for the LLM API endpoint.
            llm_api_key: API key for the LLM endpoint.
            ner_model: Language model identifier for NER.
            re_model: Language model identifier for RE.
            embedding_model: Name of the embedding model.
        """
        self.client = AsyncOpenAI(api_key=llm_api_key, base_url=llm_endpoint_url)
        #self.embedding_pipeline = embedding_pipeline
        with open(template_ner_loc, 'r') as file:
            self.template_ner = file.read()
        with open(template_re_loc, 'r') as file:
            self.template_re = file.read()
        self.ner_model = ner_model
        self.re_model = re_model
        self.embedding_model = embedding_model


    async def _embed(self, texts: List[str]) -> List[List[float]]:
        """Batched async embedding call via the OpenAI-compatible endpoint.

        Returns an empty list when `texts` is empty so callers can zip safely.
        """
        if not texts:
            return []
        resp = await self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        # Newer openai-python versions expose `.data` sorted by index; sort defensively.
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


    async def _extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from the text using the NER template.

        Args:
            text: The input text from which to extract entities.

        Returns:
            List of NEREntity objects.
        """
        messages = [{"role": "system", "content": self.template_ner}, {"role": "user", "content": text}]
        result = await self.client.beta.chat.completions.parse(
            model=self.ner_model, 
            messages=messages,
            response_format=NEREntityList)

        parsed = result.choices[0].message.parsed
        
        return parsed.root


    async def _extract_relationships(self, text: str, ner_list: str):
        """Extract relationship triplets from text appended with named entities.

        Args:
            text: The input text from which to extract relationships.
            ner_list: Named entities appended to the text, as a string.

        Returns:
            List of relationship triplets, each a list of three strings.
        """
        new_text = text + '\nnamed_entities: ' + ner_list
        messages = [{"role": "system", "content": self.template_re}, {"role": "user", "content": new_text}]
        result = await self.client.beta.chat.completions.parse(
            model=self.re_model,
            messages=messages,
            response_format=RelationTriples)

        response = result.choices[0].message.parsed

        relations = [t for t in response.triples if len(t) == 3]
        #embeddings = self._create_relation_embeddings(relations)
        return relations#, embeddings

    async def extract_entities_and_relationships(self, text: str):
        """Run the full extraction pipeline and compute embeddings for entities/relations.

        Args:
            text: The input text from which to extract entities and relationships.

        Returns:
            Tuple of (entities, relationships, chunk_embedding,
            entity_descriptor_embeddings, relation_embeddings).
        """
        entities = await self._extract_entities(text)
        entity_names = [e.name for e in entities]
        relationships = await self._extract_relationships(text, ner_list=str(entity_names))
        
        # EMBEDDINGS 
        # Textualization you will embed. Keep ordering stable; we zip by index below.
        doc_text = text
        desc_texts = [e.entity_information or "" for e in entities]
        rel_texts = [f"{h} {rel} {t}" for (h, rel, t) in relationships]

        # One batched embedding call per chunk — amortizes Ollama overhead.
        # Order: [doc, *descs, *rels]
        all_texts = [doc_text, *desc_texts, *rel_texts]
        vectors = await self._embed(all_texts)

        chunk_embedding = vectors[0]
        n_desc = len(desc_texts)
        entity_descriptor_embeddings = vectors[1 : 1 + n_desc]
        relation_embeddings = vectors[1 + n_desc :]

        return (
            entities,
            relationships,
            chunk_embedding,
            entity_descriptor_embeddings,
            relation_embeddings,
        )
