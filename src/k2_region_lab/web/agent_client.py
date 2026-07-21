from __future__ import annotations

import re
from typing import Protocol

import httpx

from k2_region_lab.agent.domain import AgentCapabilities, AgentHealth, StorageStatus
from k2_region_lab.web.domain import WorkspaceError


class WorkspaceAgentApi(Protocol):
    async def health(self) -> AgentHealth: ...

    async def capabilities(self) -> AgentCapabilities: ...

    async def storage(self) -> StorageStatus: ...


class WorkspaceAgentClient:
    """Authenticated client for one Pod agent through RunPod's HTTPS proxy."""

    _POD_ID = re.compile(r"^[a-zA-Z0-9-]{1,191}$")

    def __init__(
        self,
        pod_id: str,
        session_token: str,
        *,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not self._POD_ID.fullmatch(pod_id):
            raise ValueError("invalid RunPod Pod ID")
        self._base_url = f"https://{pod_id}-8080.proxy.runpod.net"
        self._session_token = session_token
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def health(self) -> AgentHealth:
        return AgentHealth.model_validate(await self._request("/v1/health"))

    async def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities.model_validate(await self._request("/v1/capabilities"))

    async def storage(self) -> StorageStatus:
        return StorageStatus.model_validate(await self._request("/v1/storage"))

    async def _request(self, path: str) -> dict:
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.get(
                    path,
                    headers={"Authorization": f"Bearer {self._session_token}"},
                )
        except httpx.TimeoutException as error:
            raise WorkspaceError(
                "agent_timeout",
                "The workspace agent is not responding yet.",
                status_code=504,
            ) from error
        except httpx.HTTPError as error:
            raise WorkspaceError(
                "agent_unavailable",
                "The workspace agent is currently unreachable.",
                status_code=502,
            ) from error
        if response.status_code == 401:
            raise WorkspaceError(
                "agent_authentication_failed",
                "The workspace agent rejected its session credential.",
                status_code=502,
            )
        if response.status_code != 200:
            raise WorkspaceError(
                "agent_unavailable",
                "The workspace agent could not complete its health request.",
                status_code=502,
            )
        try:
            payload = response.json()
        except ValueError as error:
            raise WorkspaceError(
                "agent_response_invalid",
                "The workspace agent returned invalid JSON.",
                status_code=502,
            ) from error
        if not isinstance(payload, dict):
            raise WorkspaceError(
                "agent_response_invalid",
                "The workspace agent returned an unexpected response.",
                status_code=502,
            )
        return payload
