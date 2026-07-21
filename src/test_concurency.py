"""
test_ollama_concurrency.py

Probe whether a local Ollama server actually processes concurrent
AsyncOpenAI requests in parallel, or silently serializes them.

Usage:
    python test_ollama_concurrency.py
    python test_ollama_concurrency.py --endpoint http://localhost:11434/v1 \
        --model phi4-mini:latest --embeddings bge-large:latest --n 4

What it does:
  1. Fires N sequential chat-completion calls and times them -> baseline.
  2. Fires the same N calls concurrently with asyncio.gather and times the wall-clock.
  3. Repeats for the embeddings endpoint.
  4. Prints a verdict: true parallelism vs serial queueing.

Interpretation of the ratio `concurrent_total / sum_of_sequential`:
  - ~1.0  -> Ollama queued them serially (set OLLAMA_NUM_PARALLEL>1)
  - ~1/N  -> perfect parallelism
  - in between -> partial parallelism
"""

import argparse
import asyncio
import time
import statistics
from openai import AsyncOpenAI


PROMPTS = [
    "List three primary colors.",
    "Name a noble gas and explain one use.",
    "What is the capital of Australia?",
    "Give a one-line definition of recursion.",
    "Translate 'good morning' to French.",
    "What is 17 multiplied by 23?",
    "Name one moon of Jupiter.",
    "What does 'HTTP' stand for?",
]

EMBED_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Ollama runs large language models locally.",
    "Asynchronous IO overlaps waiting times productively.",
    "Neo4j stores embeddings in vector indices.",
    "Retrieval-augmented generation grounds answers in evidence.",
    "Cosine similarity measures vector alignment.",
    "Knowledge graphs connect entities via relationships.",
    "Chunking splits long documents for embedding.",
]


async def time_chat_call(client, model, prompt):
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=20,
    )
    dur = time.perf_counter() - t0
    # touch the response so it materializes
    _ = resp.choices[0].message.content
    return dur


async def time_embed_call(client, model, text):
    t0 = time.perf_counter()
    resp = await client.embeddings.create(model=model, input=[text])
    dur = time.perf_counter() - t0
    _ = resp.data[0].embedding
    return dur


async def run_sequential(client, model, kind, items):
    durations = []
    for item in items:
        if kind == "chat":
            d = await time_chat_call(client, model, item)
        else:
            d = await time_embed_call(client, model, item)
        durations.append(d)
        print(f"  seq  {kind:4s}  {d:6.2f}s  ({truncate(item, 40)})")
    return durations


async def run_concurrent(client, model, kind, items, log_starts=False):
    async def one(i, item):
        if log_starts:
            print(f"  [{time.strftime('%H:%M:%S')}.{int(time.perf_counter()*1000)%1000:03d}] "
                  f"START {kind} #{i}")
        if kind == "chat":
            d = await time_chat_call(client, model, item)
        else:
            d = await time_embed_call(client, model, item)
        if log_starts:
            print(f"  [{time.strftime('%H:%M:%S')}.{int(time.perf_counter()*1000)%1000:03d}] "
                  f"END   {kind} #{i}  {d:.2f}s")
        return d

    t0 = time.perf_counter()
    durations = await asyncio.gather(*(one(i, x) for i, x in enumerate(items)))
    total = time.perf_counter() - t0
    return list(durations), total


def truncate(s, n):
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def report(label, seq_durations, conc_durations, conc_total):
    sum_seq = sum(seq_durations)
    sum_conc = sum(conc_durations)
    n = len(seq_durations)
    ratio = conc_total / sum_seq if sum_seq else float("nan")
    speedup = sum_seq / conc_total if conc_total else float("nan")
    print(f"\n=== {label} ===")
    print(f"  sequential per-call : {', '.join(f'{d:.2f}' for d in seq_durations)}")
    print(f"  concurrent per-call : {', '.join(f'{d:.2f}' for d in conc_durations)}")
    print(f"  sum(sequential)     : {sum_seq:6.2f}s")
    print(f"  sum(concurrent CPU) : {sum_conc:6.2f}s  (work done)")
    print(f"  wall-clock concurrent: {conc_total:6.2f}s")
    print(f"  ratio (wall / sum)  : {ratio:.2f}   "
          f"(1.0 = serial, {1/n:.2f} = perfect parallel for n={n})")
    print(f"  speedup             : {speedup:.2f}x")
    if ratio > 0.85:
        verdict = "LIKELY SERIALIZED — Ollama queued requests one after another."
    elif ratio < (1.0 / n) + 0.10:
        verdict = "TRUE PARALLELISM — Ollama processed requests concurrently."
    else:
        verdict = "PARTIAL PARALLELISM — some overlap, some queueing."
    print(f"  VERDICT: {verdict}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--model", default="phi4-mini:latest")
    parser.add_argument("--embeddings", default="bge-large:latest")
    parser.add_argument("--n", type=int, default=3, help="number of calls")
    parser.add_argument("--skip-chat", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    args = parser.parse_args()

    n = args.n
    if n > len(PROMPTS):
        # just reuse prompts cyclically
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n)]
    else:
        prompts = PROMPTS[:n]
    if n > len(EMBED_TEXTS):
        embeds = [EMBED_TEXTS[i % len(EMBED_TEXTS)] for i in range(n)]
    else:
        embeds = EMBED_TEXTS[:n]

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.endpoint)

    # Warm up: make sure the model is resident so cold-load time doesn't pollute results
    print(f"Warming up models at {args.endpoint} ...")
    if not args.skip_chat:
        await time_chat_call(client, args.model, "hi")
    if not args.skip_embed:
        await time_embed_call(client, args.embeddings, EMBED_TEXTS[0])
    print("Warm-up complete.\n")

    if not args.skip_chat:
        print(f"--- CHAT COMPLETIONS ({args.model}) n={n} ---")
        print("Sequential:")
        seq_chat = await run_sequential(client, args.model, "chat", prompts)
        print("\nConcurrent (timestamps shown so you can visually confirm overlap):")
        conc_chat, total_chat = await run_concurrent(
            client, args.model, "chat", prompts, log_starts=True
        )
        report(f"CHAT  {args.model}", seq_chat, conc_chat, total_chat)

    if not args.skip_embed:
        print(f"\n--- EMBEDDINGS ({args.embeddings}) n={n} ---")
        print("Sequential:")
        seq_emb = await run_sequential(client, args.embeddings, "embed", embeds)
        print("\nConcurrent:")
        conc_emb, total_emb = await run_concurrent(
            client, args.embeddings, "embed", embeds, log_starts=True
        )
        report(f"EMBED {args.embeddings}", seq_emb, conc_emb, total_emb)

    # Bonus: mixed workload (chat + embed concurrently, like ingestion)
    if not args.skip_chat and not args.skip_embed:
        print(f"\n--- MIXED (one {args.model} chat + one {args.embeddings} embed) "
              f"both kinds in parallel ---")
        await time_chat_call(client, args.model, "hi")         # warm
        await time_embed_call(client, args.embeddings, "x")    # warm
        t0 = time.perf_counter()
        d_chat, d_emb = await asyncio.gather(
            time_chat_call(client, args.model, "List three programming languages."),
            time_embed_call(client, args.embeddings, "vector retrieval demo"),
        )
        mixed_total = time.perf_counter() - t0
        print(f"  chat call dur   : {d_chat:.2f}s")
        print(f"  embed call dur  : {d_emb:.2f}s")
        print(f"  wall-clock both : {mixed_total:.2f}s")
        if mixed_total < max(d_chat, d_emb) * 1.2:
            print("  -> Both models served concurrently (both resident in VRAM).")
        else:
            print("  -> Likely serialized / model-swap overhead between the two models.")


if __name__ == "__main__":
    asyncio.run(main())