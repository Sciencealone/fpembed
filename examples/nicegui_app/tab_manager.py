"""Primary / secondary tab management.

Tracks which browser tab is the *primary* (full-control) tab and which
are *secondary* (read-only).  Handles stale-primary detection, graceful
promotion after a configurable grace period, and ordered secondary
tracking.

NiceGUI API used:
    ``from nicegui import Client``
    - ``Client.instances``  — ``dict[str, Client]`` of all active clients
    - ``client.has_socket_connection`` — ``True`` when WebSocket is live
"""

from __future__ import annotations

import asyncio
import logging
import time

from nicegui import Client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_primary_client_id: str | None = None
_promotion_task: asyncio.Task | None = None
_secondary_connect_times: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def get_primary_client_id() -> str | None:
    """Return the current primary client ID."""
    return _primary_client_id


def is_primary(client_id: str) -> bool:
    """Check if the given client ID is the primary tab."""
    return _primary_client_id is not None and _primary_client_id == client_id


def assign_primary(client_id: str) -> None:
    """Set the given client ID as primary."""
    global _primary_client_id
    _primary_client_id = client_id
    # Remove from secondaries if it was tracked there
    _secondary_connect_times.pop(client_id, None)
    logger.info("Assigned primary tab: %s", client_id)


def clear_primary() -> None:
    """Clear the primary client ID."""
    global _primary_client_id
    _primary_client_id = None
    logger.info("Primary tab cleared")


# ---------------------------------------------------------------------------
# Connection / disconnection handlers
# ---------------------------------------------------------------------------
def _is_client_connected(client_id: str) -> bool:
    """Return ``True`` if *client_id* exists in NiceGUI and has a live socket."""
    client = Client.instances.get(client_id)
    return client is not None and client.has_socket_connection


def on_client_connect(
    client_id: str,
    grace_period: float,
    from_page_handler: bool = False,
) -> bool:
    """Handle a new client connection.

    Returns ``True`` if the client was assigned as primary.

    Stale-primary detection: if ``_primary_client_id`` is set but the
    referenced client no longer has an active WebSocket, the stale
    primary is cleared and the new client becomes primary.

    If the primary reconnects within the grace period, the pending
    promotion task is cancelled.

    When *from_page_handler* is ``True`` and no secondaries are tracked,
    a lingering-socket primary is replaced (page-refresh heuristic).
    """
    global _primary_client_id, _promotion_task

    # If the reconnecting client IS the current primary, cancel any
    # pending promotion and keep it as primary.
    if _primary_client_id == client_id:
        if _promotion_task is not None and not _promotion_task.done():
            _promotion_task.cancel()
            _promotion_task = None
            logger.info("Primary %s reconnected — promotion cancelled", client_id)
        return True

    # No primary yet → assign immediately
    if _primary_client_id is None:
        assign_primary(client_id)
        return True

    # Primary is set but stale (no live socket) → replace
    if not _is_client_connected(_primary_client_id):
        logger.info(
            "Stale primary %s detected — replacing with %s",
            _primary_client_id,
            client_id,
        )
        clear_primary()
        assign_primary(client_id)
        return True

    # Primary appears connected but no secondaries tracked, and this call
    # is from a page handler → likely a page refresh with lingering socket.
    if from_page_handler and not _secondary_connect_times:
        logger.info(
            "Refresh detected — replacing lingering primary %s with %s",
            _primary_client_id,
            client_id,
        )
        clear_primary()
        assign_primary(client_id)
        return True

    # Primary exists and is connected → new client is secondary
    _secondary_connect_times[client_id] = time.time()
    logger.info("Assigned secondary tab: %s", client_id)
    return False


def on_client_disconnect(client_id: str, grace_period: float) -> None:
    """Handle client disconnect.

    If the *primary* disconnects, schedule an ``asyncio`` delayed
    promotion task that fires after *grace_period* seconds.  If a
    *secondary* disconnects, simply remove it from tracking.
    """
    global _primary_client_id, _promotion_task

    if client_id == _primary_client_id:
        logger.info(
            "Primary %s disconnected — scheduling promotion in %.1fs",
            client_id,
            grace_period,
        )
        # Schedule delayed promotion
        _promotion_task = asyncio.get_event_loop().create_task(
            _delayed_promotion(grace_period)
        )
    else:
        # Secondary disconnected — remove from tracking
        _secondary_connect_times.pop(client_id, None)
        logger.info("Secondary %s disconnected — removed from tracking", client_id)


def get_secondary_client_ids() -> list[str]:
    """Return connected secondaries ordered by connection time (oldest first)."""
    return sorted(_secondary_connect_times, key=_secondary_connect_times.get)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Internal promotion coroutine
# ---------------------------------------------------------------------------
async def _delayed_promotion(grace_period: float) -> None:
    """Wait *grace_period* seconds, then promote the oldest secondary.

    If the primary reconnects before the grace period expires, this
    task is cancelled via ``on_client_connect``.
    """
    global _primary_client_id

    try:
        await asyncio.sleep(grace_period)
    except asyncio.CancelledError:
        logger.info("Promotion task cancelled (primary reconnected)")
        return

    secondaries = get_secondary_client_ids()
    if secondaries:
        promoted = secondaries[0]
        assign_primary(promoted)
        logger.info("Promoted secondary %s to primary after grace period", promoted)
    else:
        clear_primary()
        logger.info("No secondaries to promote — primary cleared")
