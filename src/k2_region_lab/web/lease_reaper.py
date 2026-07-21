from __future__ import annotations

import asyncio
from contextlib import suppress

from k2_region_lab.web.domain import utc_now
from k2_region_lab.web.runpod_backend import RunPodPersistentPodBackend


class WorkspaceLeaseReaper:
    """Stops provider compute whose durable control-plane lease has expired."""

    def __init__(
        self,
        backend: RunPodPersistentPodBackend,
        *,
        interval_seconds: float = 30.0,
    ) -> None:
        self._backend = backend
        self._interval_seconds = interval_seconds

    async def run_once(self) -> list[str]:
        workspace_ids = await self._backend.state_store.expired_workspace_ids(utc_now())
        stopped: list[str] = []
        for workspace_id in workspace_ids:
            try:
                await self._backend.stop_workspace(workspace_id)
            except Exception as error:
                await self._backend.state_store.append_audit(
                    action="workspace.lease.reap",
                    result="failure",
                    workspace_id=workspace_id,
                    context={"error_type": type(error).__name__},
                )
                continue
            await self._backend.state_store.append_audit(
                action="workspace.lease.reap",
                result="success",
                workspace_id=workspace_id,
            )
            stopped.append(workspace_id)
        return stopped

    async def run_forever(self) -> None:
        while True:
            await self.run_once()
            await asyncio.sleep(self._interval_seconds)

    @staticmethod
    async def cancel(task: asyncio.Task[None]) -> None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
