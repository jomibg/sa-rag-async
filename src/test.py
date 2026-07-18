import inspect
from tqdm.asyncio import tqdm_asyncio
print(inspect.signature(tqdm_asyncio.gather))
# (*awaitables, total=None, tqdm_class=None, **tqdm_kwargs)
