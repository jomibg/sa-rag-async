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

    The pipeline's `run()` is invoked concurrently, with in-flight calls
    capped at `max_concurrency`. Answers are returned in input order.

    Args:
        pipeline: A concrete RetrievalPipeline instance.
        questions: Ordered iterable of user questions.
        max_concurrency: Maximum number of questions processed at once.
        show_progress: If True, show a tqdm progress bar.
        return_exceptions: If True, failed questions yield Exception objects
            instead of aborting the batch.

    Returns:
        List of answers aligned 1:1 with `questions`.
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