# SA-RAG

Spreading-Activation Retrieval-Augmented Generation pipeline with knowledge-graph
ingestion, benchmark evaluation, and multiple RAG strategies (vector, CoT,
decomposition, and SA variants).

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and the Docker Compose plugin
- [uv](https://docs.astral.sh/uv/) for local Python dependency management
- An [Ollama](https://ollama.com/) server hosting the models used by the pipeline
  (default `phi4-mini:latest` for the LLM and `bge-large:latest` for embeddings)
- The benchmark datasets placed under `datasets/` (see `configs.py` for filenames)

## Environment variables

The application reads the following from the environment (set them in your shell
or in a `.env` file loaded by `python-dotenv`):

| Variable        | Description                                  | Example                      |
|-----------------|----------------------------------------------|------------------------------|
| `LLM_BASE_URL`  | OpenAI-compatible LLM endpoint (Ollama /v1)  | `http://localhost:11434/v1`  |
| `LLM_API_KEY`   | API key for the LLM endpoint                 | `not-needed`                 |
| `NEO4J_URL`     | Bolt URL of the Neo4j instance                | `bolt://localhost:7687`      |
| `NEO4J_USER`    | Neo4j username                               | `neo4j`                      |
| `NEO4J_PASSWORD`| Neo4j password                               | `test1234`                   |

## Running with Docker

The `docker-compose.yml` brings up two Neo4j instances (with the Graph Data
Science and APOC plugins) and the application container.

### 1. Prepare Ollama

Start Ollama and pull the required models on the host:

```bash
ollama serve &
ollama pull phi4-mini:latest
ollama pull bge-large:latest
```

Compose is configured to reach the host's Ollama at `http://172.17.0.1:11434/v1`
(the default Docker bridge gateway). Adjust `LLM_BASE_URL` in
`docker-compose.yml` if your setup differs.

### 2. Build and launch the stack

From the repository root:

```bash
docker compose build
docker compose up
```

The app container will start once both Neo4j databases report healthy. Results
are written to the mounted `./results` directory.

### 3. Run a single command (optional)

To run the pipeline without leaving the compose stack up, override the default
command:

```bash
docker compose run --rm app python run.py
```

> **Note:** The `Dockerfile` default `CMD` references `run_eval.py`; the actual
> entrypoint is `src/run.py`. Override it as shown above until the Dockerfile is
> updated.

## Running locally with uv

`uv` is the recommended way to manage dependencies outside of Docker.

### 1. Install dependencies

From the repository root:

```bash
uv sync
```

This creates a virtual environment using the locked versions in `uv.lock`.

### 2. Start the backing services

Neo4j must be running with the Graph Data Science and APOC plugins. You can
start just the databases with Docker Compose:

```bash
docker compose up -d neo4j_db1
```

Make sure Ollama is serving the required models (see "Prepare Ollama" above).

### 3. Set environment variables

Export the variables listed in the table above, for example:

```bash
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_API_KEY="not-needed"
export NEO4J_URL="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="test1234"
```

Alternatively, place them in a `.env` file at the repository root; the app loads
it automatically via `python-dotenv`.

### 4. Run the pipeline

```bash
uv run python src/run.py
```

Pipeline behaviour is controlled by `RunConfigs` in `src/configs.py` (which
benchmark to use, number of questions, which RAG pipelines to evaluate, etc.).
Outputs (sampled corpus, answers, evaluation metrics, and dashboards) are
written under `results/`.