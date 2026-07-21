from __future__ import annotations

import re
from typing import Any, Protocol

import httpx

from k2_region_lab.agent.domain import (
    AgentCapabilities,
    AgentHealth,
    ChunkReceipt,
    FileKind,
    FilePage,
    StorageStatus,
    UploadCompleteResponse,
    UploadCreateRequest,
    UploadSession,
)
from k2_region_lab.web.domain import WorkspaceError


class WorkspaceAgentApi(Protocol):
    async def health(self) -> AgentHealth: ...

    async def capabilities(self) -> AgentCapabilities: ...

    async def storage(self) -> StorageStatus: ...

    async def inventory(
        self, kind: FileKind, *, cursor: str | None = None
    ) -> FilePage: ...

    async def create_upload(self, request: UploadCreateRequest) -> UploadSession: ...

    async def upload_status(self, upload_id: str) -> UploadSession: ...

    async def write_chunk(
        self, upload_id: str, index: int, content: bytes, sha256: str
    ) -> ChunkReceipt: ...

    async def complete_upload(self, upload_id: str) -> UploadCompleteResponse: ...

    async def cancel_upload(self, upload_id: str) -> None: ...


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

    async def inventory(
        self, kind: FileKind, *, cursor: str | None = None
    ) -> FilePage:
        params = {"kind": kind.value}
        if cursor:
            params["cursor"] = cursor
        return FilePage.model_validate(await self._request("/v1/files", params=params))

    async def create_upload(self, request: UploadCreateRequest) -> UploadSession:
        return UploadSession.model_validate(
            await self._request("/v1/uploads", method="POST", json=request.model_dump(mode="json"))
        )

    async def upload_status(self, upload_id: str) -> UploadSession:
        return UploadSession.model_validate(await self._request(f"/v1/uploads/{upload_id}"))

    async def write_chunk(
        self, upload_id: str, index: int, content: bytes, sha256: str
    ) -> ChunkReceipt:
        return ChunkReceipt.model_validate(
            await self._request(
                f"/v1/uploads/{upload_id}/chunks/{index}",
                method="PUT",
                content=content,
                extra_headers={"X-Chunk-SHA256": sha256},
            )
        )

    async def complete_upload(self, upload_id: str) -> UploadCompleteResponse:
        return UploadCompleteResponse.model_validate(
            await self._request(f"/v1/uploads/{upload_id}/complete", method="POST")
        )

    async def cancel_upload(self, upload_id: str) -> None:
        await self._request(f"/v1/uploads/{upload_id}", method="DELETE")

    async def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        extra_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict:
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.request(
                    method,
                    path,
                    headers={
                        "Authorization": f"Bearer {self._session_token}",
                        **(extra_headers or {}),
                    },
                    **kwargs,
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
        if response.status_code == 204:
            return {}
        if response.status_code < 200 or response.status_code >= 300:
            try:
                error_body = response.json()
            except ValueError:
                error_body = {}
            if isinstance(error_body, dict):
                code = error_body.get("code")
                message = error_body.get("message")
                if isinstance(code, str) and isinstance(message, str):
                    raise WorkspaceError(code, message, status_code=response.status_code)
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
