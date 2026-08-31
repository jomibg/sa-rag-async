# SA-RAG

Spreading-Activation Retrieval-Augmented Generation pipeline with knowledge-graph
ingestion, benchmark evaluation, and multiple RAG strategies (vector, CoT,
decomposition, and SA variants).

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and the Docker Compose plugin
- [uv](https://docs.astral.sh/uv/) for local Python dependency management
- An [Ollama](https://ollama.com/) server hosting the models used by the pipeline
  (default `phi4:latest` for the LLM and `qwen3-embedding:0.6b` for embeddings)
- The benchmark datasets placed under `datasets/` (see `src/configs.py` for filenames)

## Environment variables

The application reads the following from the environment. Export them in your
shell before running locally, or provide them in the container configuration:

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
ollama pull phi4:latest
ollama pull qwen3-embedding:0.6b
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

The app container starts once both Neo4j databases report healthy and runs
`run_eval.py`. Configure `RunConfigs` in `src/configs.py` first; its defaults
expect sampled corpus and question files that are not included in the repository.
Results are written to the mounted `./results` directory.

### 3. Run a single command (optional)

To run the pipeline without leaving the compose stack up, override the default
command:

```bash
docker compose run --rm app python run_eval.py
```

The image works in `/app/src`, so `run_eval.py` is the correct command. Before
running the evaluation, configure `RunConfigs` in `src/configs.py` and make sure
the expected sampled corpus and question files exist under `results/`.

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

The dependency `python-dotenv` is installed, but the current entrypoint does not
load a `.env` file automatically.

### 4. Run the pipeline

```bash
cd src
uv run --project .. python run_eval.py
```

Pipeline behaviour is controlled by `RunConfigs` in `src/configs.py` (which
benchmark to use, number of questions, which RAG pipelines to evaluate, etc.).
For a first run, set `sample_data = True` and `ingest_corpus = True` in
`RunConfigs` to create the sampled files and ingest the graph; subsequent runs
can set those options back to `False`.
Outputs (sampled corpus, answers, evaluation metrics, and dashboards) are
written under `results/`.

## Running Hyperparameter Tuning with uv

`src/run_tunning.py` runs an Optuna study for the SA-CoT pipeline. It has its own
configuration and does not read the environment variables used by `run_eval.py`.
Before starting it, edit the values in the `LOCAL CONFIGURATION` section of
`src/run_tunning.py`, especially:

- `LLM_ENDPOINT`, `LLM_API_KEY`, `LLM_MODEL`, and `EMBEDDING_MODEL`
- `NEO4J_URL`, `NEO4J_USER`, and `NEO4J_PW`
- `BENCHMARK` if using `TwoWikiMultiHop` instead of `MuSiQuE`

For the local services from the commands above, the relevant values can be:

```python
LLM_ENDPOINT = "http://localhost:11434/v1"
LLM_API_KEY = "not-needed"
LLM_MODEL = "phi4:latest"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PW = "test1234"
```

Start Neo4j and Ollama, pull the configured Ollama models, and then run the
script from `src/` because its dataset, prompt, and output paths are relative to
that directory:

```bash
uv sync
cd src
uv run --project .. python run_tunning.py
```

The script samples and ingests 200 questions by default, evaluates 50 Optuna
trials, and writes the best-parameter summary to
`results/tunning/params.txt`.
