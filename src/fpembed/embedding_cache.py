"""Bounded, externally fillable LRU cache (a leaf module).

This module provides :class:`BoundedCache`, a small least-recently-used cache
that the caller fills explicitly. Unlike :func:`functools.lru_cache`, key
derivation and value computation live in different places: the caller derives
the key, looks it up, and — only on a miss — computes and stores the value.
That split is what lets a SMILES canonicalisation produce the key while the
already-parsed molecule is reused for the embedding, so a miss never costs a
redundant parse.

The method names (:meth:`~BoundedCache.cache_info`,
:meth:`~BoundedCache.cache_clear`) deliberately mirror the interface of
``functools.lru_cache`` so consumers can swap one for the other with minimal
change.

Thread safety: a :class:`threading.Lock` guards all bookkeeping (the ordered
map plus the hit/miss counters). Only benign races are possible for the
intended use — values are deterministic per key, so a race recomputes an entry
rather than corrupting one. As with ``lru_cache``, the hit/miss counters are
approximate under concurrency.

Standard library only — no NumPy, no RDKit.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, namedtuple
from typing import Any, Hashable

CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])

# Miss sentinel distinct from ``None`` so a cached ``None`` value (e.g. the
# result of parsing an invalid SMILES) is not mistaken for an absent entry.
_MISS = object()


class BoundedCache:
    """Bounded LRU that the caller fills.

    Key derivation and value computation can live in different places: the
    caller looks a key up with :meth:`get`, and on a miss computes the value
    itself and stores it with :meth:`put`.

    LRU policy is ``OrderedDict.move_to_end`` on a hit and
    ``popitem(last=False)`` on overflow — there is nothing else to it.

    Hit/miss counters are approximate under concurrency, as
    ``functools.lru_cache``'s already are.
    """

    def __init__(self, maxsize: int) -> None:
        """Create a cache holding at most ``maxsize`` entries.

        ``maxsize`` is expected to be a positive integer. Callers that want to
        disable caching should simply not construct a ``BoundedCache``; a
        non-positive ``maxsize`` is coerced to ``1`` defensively so the cache
        never grows unbounded.
        """
        self._maxsize = maxsize if maxsize and maxsize > 0 else 1
        self._data: "OrderedDict[Hashable, Any]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Any:
        """Return the value for ``key``, or the ``_MISS`` sentinel if absent.

        Never returns ``None`` to signal a miss: ``None`` is a storable value
        distinct from an absent entry. On a hit the key is promoted to
        most-recently-used.
        """
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._hits += 1
                return self._data[key]
            self._misses += 1
            return _MISS

    def put(self, key: Hashable, value: Any) -> None:
        """Store ``value`` under ``key``, evicting the LRU entry on overflow.

        An existing key is refreshed to most-recently-used. When the cache
        exceeds its bound, the least-recently-used entry is discarded.
        """
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def cache_info(self) -> CacheInfo:
        """Return ``CacheInfo(hits, misses, maxsize, currsize)``."""
        with self._lock:
            return CacheInfo(
                hits=self._hits,
                misses=self._misses,
                maxsize=self._maxsize,
                currsize=len(self._data),
            )

    def cache_clear(self) -> None:
        """Drop all entries and reset the hit/miss counters."""
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0
