"""Byte-bounded LRU cache for random-projection matrices (a leaf module).

The random-projection compressor needs a large ``(L, D)`` float64 matrix per
``(fp_length, output_dim, seed, sparse)`` parameter set. A single dense draw is
134 MB at L=16384, D=1024, so an entry-count bound (as ``functools.lru_cache``
uses) ignores the axis that actually matters: total bytes retained. This module
provides a small purpose-built LRU keyed on the parameter tuple and evicted by
accumulated bytes, with a small entry-count floor as a safety net so a single
oversized entry can never starve the cache to nothing.

Only the canonical C-contiguous ``(L, D)`` orientation is stored. The ``(D, L)``
orientation is a non-owning ``.T`` view of the same buffer, so no second owning
copy exists. Cached arrays are marked read-only before sharing; the ``.T`` view
inherits that.

Standard library plus NumPy only — no RDKit.

See ``compression_projection.py`` (split out of it per the File Size Limits
rule; history follows via ``hg cp -A``).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable, Hashable

import numpy as np
import numpy.typing as npt

# Default upper bound, in bytes, on the total memory retained by the projection
# matrix cache across all live entries. This is the tunable budget: lower it to
# cap resident memory at the cost of more deterministic rebuilds, raise it to
# retain more distinct parameter sets. 512 MiB holds several large matrices
# (a 134 MB dense L=16384/D=1024 draw fits ~3x over) while staying a small
# fraction of a typical 16 GB developer machine.
RP_CACHE_BYTE_BUDGET = 512 * 1024 * 1024

# Entry-count safety floor: keep at least this many most-recently-used entries
# even if their combined size exceeds the byte budget, so a single matrix larger
# than the budget does not evict itself immediately and thrash on every call.
RP_CACHE_MIN_ENTRIES = 1


class _ByteBoundedArrayCache:
    """LRU cache of NumPy arrays evicted by accumulated bytes.

    Reuses the ``OrderedDict`` move-to-end / popitem-first LRU pattern of
    :class:`embedding_cache.BoundedCache`, but stores arrays and bounds by the
    sum of ``array.nbytes`` rather than by entry count. An entry-count floor
    (:attr:`_min_entries`) guarantees the most-recently-used entries survive
    even when a single array exceeds the byte budget.

    Thread safety mirrors ``BoundedCache``: a lock guards the ordered map and
    the accumulated-byte counter. Values are deterministic per key, so a race
    only recomputes an entry rather than corrupting one.
    """

    def __init__(self, byte_budget: int, min_entries: int) -> None:
        self._byte_budget = byte_budget
        self._min_entries = max(1, min_entries)
        self._data: "OrderedDict[Hashable, npt.NDArray[np.float64]]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get_or_build(
        self, key: Hashable, build: Callable[[], npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        """Return the cached array for ``key``, building and storing it on a miss.

        On a hit the key is promoted to most-recently-used. On a miss ``build``
        is called (outside the lock so a slow draw does not block other keys),
        the result is marked read-only, stored, and the least-recently-used
        entries are evicted until the accumulated bytes fit the budget or only
        the entry-count floor remains.
        """
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]

        arr = build()
        arr.setflags(write=False)

        with self._lock:
            if key in self._data:
                # Another thread built the same key while we were drawing; keep
                # the existing entry so all callers share one buffer.
                self._data.move_to_end(key)
                return self._data[key]
            self._data[key] = arr
            self._bytes += arr.nbytes
            self._evict()
            return arr

    def _evict(self) -> None:
        """Drop least-recently-used entries past the budget, honoring the floor."""
        while self._bytes > self._byte_budget and len(self._data) > self._min_entries:
            _, evicted = self._data.popitem(last=False)
            self._bytes -= evicted.nbytes

    def cache_clear(self) -> None:
        """Drop all entries and reset the accumulated-byte counter."""
        with self._lock:
            self._data.clear()
            self._bytes = 0

    def currsize(self) -> int:
        """Return the number of entries currently held."""
        with self._lock:
            return len(self._data)

    def currbytes(self) -> int:
        """Return the total bytes currently retained."""
        with self._lock:
            return self._bytes


_RP_MATRIX_T_CACHE = _ByteBoundedArrayCache(
    RP_CACHE_BYTE_BUDGET, RP_CACHE_MIN_ENTRIES
)


def get_rp_matrix_t(
    fp_length: int,
    output_dim: int,
    seed: int,
    sparse: bool,
    draw: Callable[[int, int, int, bool], npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Return the cached C-contiguous ``(fp_length, output_dim)`` matrix.

    Keyed on ``(fp_length, output_dim, seed, sparse)``. On a miss the ``(D, L)``
    matrix is drawn by ``draw`` (identical ``RandomState`` sequence to before)
    and transposed into a C-contiguous ``(L, D)`` owning array, which is cached
    read-only. The ``(D, L)`` orientation is served elsewhere as a ``.T`` view
    of this buffer.
    """
    key = (fp_length, output_dim, seed, sparse)

    def _build() -> npt.NDArray[np.float64]:
        return np.ascontiguousarray(draw(fp_length, output_dim, seed, sparse).T)

    return _RP_MATRIX_T_CACHE.get_or_build(key, _build)


def clear_rp_matrix_cache() -> None:
    """Clear the projection-matrix cache (used by determinism tests)."""
    _RP_MATRIX_T_CACHE.cache_clear()
