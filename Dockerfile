FROM python:3.12-slim

WORKDIR /app
# Install system dependencies including gcc
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN pip install uv

RUN uv pip install --system -r requirements.txt


COPY ./src /app/src
COPY ./prompts /app/prompts
COPY ./datasets /app/datasets
COPY ./results /app/results
COPY README.md /app/

WORKDIR /app/src

CMD ["python", "run_eval.py"]