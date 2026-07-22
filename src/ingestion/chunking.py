from typing import List, Generator, Tuple


def size_based_chunking(text: str, chunk_size: int, overlap_size: int) -> Generator[str, None, None]:
    """Yield fixed-size character windows with overlap between consecutive windows.

    A window of length `chunk_size` is taken, then the start index advances by
    `chunk_size - overlap_size`. The last window may be shorter than `chunk_size`.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap_size < 0:
        raise ValueError("overlap_size must be non-negative.")
    if overlap_size >= chunk_size:
        raise ValueError("overlap_size must be smaller than chunk_size.")
    if not text:
        return

    step = chunk_size - overlap_size
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield text[start:end]
        if end == len(text):          # reached the end -> stop, no padding
            break
        start += step


def word_based_chunking(text: str, chunk_size: int, overlap_size: int) -> Generator[str, None, None]:
    """Yield word-count windows with overlap between consecutive windows.

    Each chunk contains `chunk_size` words; the start index advances by
    `chunk_size - overlap_size`. The last chunk may be shorter.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap_size < 0:
        raise ValueError("overlap_size must be non-negative.")
    if overlap_size >= chunk_size:
        raise ValueError("overlap_size must be smaller than chunk_size.")
    if not text:
        return

    words = text.strip().split()
    start = 0
    while start <= len(words) - chunk_size:
        chunk = words[start:start + chunk_size]
        yield ' '.join(chunk)
        start += (chunk_size - overlap_size)
    # Handle the last chunk if there are remaining words
    if start < len(words):
        yield ' '.join(words[start:])


def chunk_documents(
        docs: List[str],
        chunk_method: str,
        chunk_size: int,
        overlap_size: int
) -> Generator[Tuple[int, str], None, None]:
    """Chunk each document according to the specified method.

    Args:
        docs: List of document strings.
        chunk_method: Either "size_based" or "word_based".
        chunk_size: Size of each chunk.
        overlap_size: Overlap between consecutive chunks.

    Yields:
        Tuples of (chunk_index, chunk_text).
    """
    chunk_index=0
    for i, doc in enumerate(docs):
        if chunk_method == "size_based":
            for j, chunk in enumerate(size_based_chunking(doc, chunk_size, overlap_size)):
                yield chunk_index, chunk
                chunk_index+=1
        elif chunk_method == "word_based":
            for j, chunk in enumerate(word_based_chunking(doc, chunk_size, overlap_size)):
                yield chunk_index, chunk
                chunk_index+=1
        else:
            raise ValueError(f"Unknown chunking method: {chunk_method}")
