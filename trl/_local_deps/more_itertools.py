def unique_everseen(iterable, key=None):
    """Simple implementation for unique_everseen"""
    seen = set()
    for item in iterable:
        k = item if key is None else key(item)
        if k not in seen:
            seen.add(k)
            yield item

def chunked(iterable, n):
    """Simple implementation for chunked"""
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, n))
        if not chunk:
            break
        yield chunk

from itertools import islice
