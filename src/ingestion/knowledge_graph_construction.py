from typing import Any, Dict, List, Annotated
from pydantic import BaseModel, RootModel, Field
from openai import AsyncOpenAI

class NEREntity(BaseModel):
    name: str
    type: str
    aliases: List[str] = Field(default_factory=list)
    entity_information: str = ""


class NEREntityList(RootModel[List[NEREntity]]):
    pass


Triple = Annotated[List[str], Field(min_length=3, max_length=3)]

class RelationTriples(BaseModel):
    triples: List[Triple]


struct_out_ner = NEREntityList
struct_out_re = RelationTriples


class AdvancedKGConstructor:
    """Constructs a knowledge graph by extracting entities and relationships from text.

    This class leverages language models and embedding pipelines to extract named entities and
    relationship triplets from a given text, compute embeddings for the relations, remove duplicates
    based on cosine similarity, and extract frequent relations based on occurrence counts.
    """

    def __init__(self, template_ner_loc: str, template_re_loc: str, #embedding_pipeline: Any,
                 llm_endpoint_url: str,
                 llm_api_key: str,
                 ner_model: str = "qwen2.5:3b", re_model: str = "hermes3"):
        """Initializes the KnowledgeGraphConstructor.

        Loads the templates for named entity recognition (NER) and relationship extraction (RE)
        from the specified file locations. Also, sets up a client for interacting with a language
        model server and stores the provided embedding pipeline and NER model.

        Args:
            template_ner_loc: The file path to the NER template.
            template_re_loc: The file path to the RE template.
            embedding_pipeline: An instance of an embedding pipeline for creating
                embeddings from textual data.
            llm_endpoint_url: URL for the LLM API endpoint.
            ner_model: The language model identifier to be used for NER.
                Defaults to "qwen2.5:3b".
            re_model: The language model identifier to be used for RE.
                Defaults to "hermes3".
        """
        self.client = AsyncOpenAI(api_key=llm_api_key, base_url=llm_endpoint_url)
        #self.embedding_pipeline = embedding_pipeline
        with open(template_ner_loc, 'r') as file:
            self.template_ner = file.read()
        with open(template_re_loc, 'r') as file:
            self.template_re = file.read()
        self.ner_model = ner_model
        self.re_model = re_model

    async def _extract_entities(self, text: str) -> List[Dict]:
        """Extracts named entities from the given text using the NER template.

        This method sends a prompt to the language model client with the NER template and the
        provided text, and returns the model's response containing the extracted entities.

        Args:
            text: The input text from which to extract entities.

        Returns:
            A list of dictionaries representing the extracted entities with their properties.
        """
        messages = [{"role": "system", "content": self.template_ner}, {"role": "user", "content": text}]
        result = await self.client.beta.chat.completions.parse(
            model=self.ner_model, 
            messages=messages,
            response_format=NEREntityList)

        parsed = result.choices[0].message.parsed
        
        return parsed.root

#TODO: implement relationship and entity embeddings

    async def _extract_relationships(self, text: str, ner_list: str):
        """Extracts relationships from the given text by leveraging NER and RE templates.

        First, it appends the named entities to the text and sends the augmented text to the RE
        template for relationship extraction. The response is validated against a JSON schema,
        and only triplets with exactly three elements are considered. Finally, embeddings for
        these triplets are created.

        Args:
            text: The input text from which to extract relationships.
            ner_list: List of named entities as a string.

        Returns:
            tuple: A tuple containing:
                - A list of relationship triplets, where each triplet is a list of three strings.
                - A list of embeddings corresponding to the triplets.
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
        """Extracts entities and relationships from the given text.

        This method performs the full knowledge graph extraction pipeline:
        1. Extract named entities from the text
        2. Create embeddings for each entity
        3. Extract relationships between entities
        4. Create embeddings for each relationship

        Args:
            text: The input text from which to extract entities and relationships.

        Returns:
            tuple: A tuple containing:
                - A list of entities with their embeddings
                - A list of relationship triplets
                - A list of embeddings for the relationships
        """
        entities = await self._extract_entities(text)
        entity_names = [e.name for e in entities]
        #embedded_entities = self._create_entity_embeddings(entities)
        relationships = await self._extract_relationships(text, ner_list=str(entity_names))
        return entities, relationships#, relation_embeddings
