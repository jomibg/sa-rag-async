import asyncio
from typing import Sequence, List

from .base import RetrievalPipeline
from tqdm.asyncio import tqdm_asyncio

async def retrieve_generate(
    pipeline: RetrievalPipeline,
    questions: Sequence[str],
    max_concurrency: int = 4,
    show_progress: bool = True,
    return_exceptions: bool = False,
) -> List:
    """Answer a batch of questions with one retrieval-generation pipeline.

    The pipeline's `run()` is invoked concurrently for every question, but
    in-flight calls are capped at `max_concurrency` so the model/embedding
    server is not overwhelmed. Answers are returned in the SAME order as
    `questions` regardless of completion order.

    Args:
        pipeline: any concrete RetrievalPipeline (vector / pre / post).
        questions: ordered iterable of user questions.
        top_k: forwarded to `pipeline.run`.
        max_concurrency: maximum number of questions processed at once.
            Tune to your Ollama/model server capacity (start at 2–4 on a
            single-GPU box and raise while watching utilization).
        show_progress: if True, wrap gather in tqdm for a live bar.
        return_exceptions: forwarded to asyncio.gather. When True, a failed
            question yields its Exception object in the result list instead
            of aborting the whole batch.

    Returns:
        List of answers (or Exception objects) aligned 1:1 with `questions`.
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")

    sem = asyncio.Semaphore(max_concurrency)

    async def run_one(q: str):
        # `async with sem` is what actually bounds concurrency: the coroutine
        # pauses here until a slot frees up, instead of running eagerly the
        # moment we create the task below.
        async with sem:
            return await pipeline.run(q)

    # Build tasks once. Because run_one awaits the semaphore before any I/O,
    # at most `max_concurrency` of them will be doing real work at a time.
    tasks = [asyncio.create_task(run_one(q)) for q in questions]

    try:
        if show_progress:
            answers = await tqdm_asyncio.gather(
                *tasks, desc="retrieve+generate", unit="q",
            )
        else:
            answers = await asyncio.gather(
                *tasks, return_exceptions=return_exceptions
            )
    except BaseException:
        # If the caller cancels, or something outside return_exceptions raises,
        # make sure we don't leave dangling tasks around.
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    return list(answers)