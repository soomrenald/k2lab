from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque


class SlidingWindowRateLimiter:
    """Small process-local limiter for protecting an individual service instance."""

    def __init__(self) -> None:
        self._entries: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def allow(
        self, key: str, *, limit: int, window_seconds: float = 60.0
    ) -> tuple[bool, int]:
        if limit < 1:
            return False, max(1, int(window_seconds))
        now = time.monotonic()
        cutoff = now - window_seconds
        async with self._lock:
            entries = self._entries[key]
            while entries and entries[0] <= cutoff:
                entries.popleft()
            if len(entries) >= limit:
                retry_after = max(1, int(window_seconds - (now - entries[0])) + 1)
                return False, retry_after
            entries.append(now)
            return True, 0
