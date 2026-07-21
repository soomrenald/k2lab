from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from k2_region_lab.web.domain import WorkspaceError


class RunPodPrice(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    stock_status: str | None = Field(default=None, alias="stockStatus")
    uninterruptible_price: float | None = Field(default=None, alias="uninterruptablePrice")
    available_gpu_counts: list[int] = Field(default_factory=list, alias="availableGpuCounts")

    @property
    def one_gpu_available(self) -> bool:
        return self.stock_status not in {None, "None"} and 1 in self.available_gpu_counts


class RunPodGpuType(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    display_name: str = Field(alias="displayName")
    memory_gb: int = Field(alias="memoryInGb")
    secure_cloud: bool = Field(alias="secureCloud")
    community_cloud: bool = Field(alias="communityCloud")
    secure_price: RunPodPrice | None = Field(default=None, alias="securePrice")
    community_price: RunPodPrice | None = Field(default=None, alias="communityPrice")


class RunPodGpuAvailability(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    gpu_type_id: str = Field(alias="gpuTypeId")
    display_name: str = Field(alias="displayName")
    stock_status: str = Field(alias="stockStatus")


class RunPodDatacenter(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    name: str
    location: str
    gpu_availability: list[RunPodGpuAvailability] = Field(
        default_factory=list, alias="gpuAvailability"
    )


class RunPodNetworkVolume(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    name: str
    size_gb: int = Field(alias="size")
    datacenter_id: str = Field(alias="dataCenterId")


class RunPodApi(Protocol):
    async def validate_credentials(self) -> None: ...

    async def list_gpu_types(self) -> list[RunPodGpuType]: ...

    async def list_datacenters(self) -> list[RunPodDatacenter]: ...

    async def list_network_volumes(self) -> list[RunPodNetworkVolume]: ...

    async def get_network_volume(self, volume_id: str) -> RunPodNetworkVolume: ...

    async def create_network_volume(
        self, *, name: str, size_gb: int, datacenter_id: str
    ) -> RunPodNetworkVolume: ...

    async def delete_network_volume(self, volume_id: str) -> None: ...

    async def create_pod(self, request: Mapping[str, Any]) -> dict[str, Any]: ...

    async def get_pod(self, pod_id: str) -> dict[str, Any]: ...

    async def start_pod(self, pod_id: str) -> dict[str, Any]: ...

    async def stop_pod(self, pod_id: str) -> dict[str, Any]: ...

    async def delete_pod(self, pod_id: str) -> None: ...


class RunPodApiClient:
    """Small RunPod REST/GraphQL adapter with stable, redacted failures."""

    REST_BASE_URL = "https://rest.runpod.io/v1"
    GRAPHQL_URL = "https://api.runpod.io/graphql"
    GPU_INVENTORY_QUERY = """
        query K2LabGpuInventory {
          gpuTypes {
            id
            displayName
            memoryInGb
            secureCloud
            communityCloud
            securePrice: lowestPrice(input: {gpuCount: 1, secureCloud: true}) {
              stockStatus
              uninterruptablePrice
              availableGpuCounts
            }
            communityPrice: lowestPrice(input: {gpuCount: 1, secureCloud: false}) {
              stockStatus
              uninterruptablePrice
              availableGpuCounts
            }
          }
        }
    """
    DATACENTER_QUERY = """
        query K2LabDatacenterInventory {
          dataCenters {
            id
            name
            location
            gpuAvailability {
              gpuTypeId
              displayName
              stockStatus
            }
          }
        }
    """

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = SecretStr(api_key)
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def validate_credentials(self) -> None:
        await self._rest_request("GET", "/pods", params={"includeWorkers": "false"})

    async def list_gpu_types(self) -> list[RunPodGpuType]:
        payload = await self._request(
            "POST",
            self.GRAPHQL_URL,
            json={"query": self.GPU_INVENTORY_QUERY},
        )
        if not isinstance(payload, dict):
            raise WorkspaceError(
                "provider_response_invalid",
                "RunPod returned an unexpected GPU inventory response.",
                status_code=502,
            )
        errors = payload.get("errors")
        if errors:
            raise WorkspaceError(
                "provider_inventory_failed",
                "RunPod could not return GPU inventory for this API key.",
                status_code=502,
            )
        try:
            items = payload["data"]["gpuTypes"]
            return [RunPodGpuType.model_validate(item) for item in items]
        except (KeyError, TypeError, ValueError) as error:
            raise WorkspaceError(
                "provider_response_invalid",
                "RunPod returned an unexpected GPU inventory response.",
                status_code=502,
            ) from error

    async def list_datacenters(self) -> list[RunPodDatacenter]:
        payload = await self._request(
            "POST",
            self.GRAPHQL_URL,
            json={"query": self.DATACENTER_QUERY},
        )
        if not isinstance(payload, dict) or payload.get("errors"):
            raise WorkspaceError(
                "provider_inventory_failed",
                "RunPod could not return datacenter inventory for this API key.",
                status_code=502,
            )
        try:
            return [
                RunPodDatacenter.model_validate(item) for item in payload["data"]["dataCenters"]
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise WorkspaceError(
                "provider_response_invalid",
                "RunPod returned an unexpected datacenter inventory response.",
                status_code=502,
            ) from error

    async def list_network_volumes(self) -> list[RunPodNetworkVolume]:
        payload = await self._rest_request("GET", "/networkvolumes")
        if not isinstance(payload, list):
            raise WorkspaceError(
                "provider_response_invalid",
                "RunPod returned an unexpected network-volume response.",
                status_code=502,
            )
        try:
            return [RunPodNetworkVolume.model_validate(item) for item in payload]
        except ValueError as error:
            raise WorkspaceError(
                "provider_response_invalid",
                "RunPod returned an unexpected network-volume response.",
                status_code=502,
            ) from error

    async def get_network_volume(self, volume_id: str) -> RunPodNetworkVolume:
        return RunPodNetworkVolume.model_validate(
            await self._rest_request("GET", f"/networkvolumes/{volume_id}")
        )

    async def create_network_volume(
        self, *, name: str, size_gb: int, datacenter_id: str
    ) -> RunPodNetworkVolume:
        payload = await self._rest_request(
            "POST",
            "/networkvolumes",
            json={"name": name, "size": size_gb, "dataCenterId": datacenter_id},
        )
        return RunPodNetworkVolume.model_validate(payload)

    async def delete_network_volume(self, volume_id: str) -> None:
        await self._rest_request("DELETE", f"/networkvolumes/{volume_id}", expected={204})

    async def create_pod(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = await self._rest_request("POST", "/pods", json=dict(request), expected={201})
        return self._object_response(payload)

    async def get_pod(self, pod_id: str) -> dict[str, Any]:
        return self._object_response(await self._rest_request("GET", f"/pods/{pod_id}"))

    async def start_pod(self, pod_id: str) -> dict[str, Any]:
        return self._object_response(await self._rest_request("POST", f"/pods/{pod_id}/start"))

    async def stop_pod(self, pod_id: str) -> dict[str, Any]:
        return self._object_response(await self._rest_request("POST", f"/pods/{pod_id}/stop"))

    async def delete_pod(self, pod_id: str) -> None:
        await self._rest_request("DELETE", f"/pods/{pod_id}", expected={204})

    async def _rest_request(
        self,
        method: str,
        path: str,
        *,
        expected: set[int] | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self._request(
            method,
            f"{self.REST_BASE_URL}{path}",
            expected=expected,
            **kwargs,
        )

    async def _request(
        self,
        method: str,
        url: str,
        *,
        expected: set[int] | None = None,
        **kwargs: Any,
    ) -> Any:
        headers = {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.request(method, url, headers=headers, **kwargs)
        except httpx.TimeoutException as error:
            raise WorkspaceError(
                "provider_timeout",
                "RunPod did not respond before the provider timeout.",
                status_code=504,
            ) from error
        except httpx.HTTPError as error:
            raise WorkspaceError(
                "provider_unavailable",
                "The RunPod API is currently unreachable.",
                status_code=502,
            ) from error

        allowed = expected or {200}
        if response.status_code not in allowed:
            self._raise_provider_error(response.status_code)
        if response.status_code == 204 or not response.content:
            return {}
        try:
            payload = response.json()
        except ValueError as error:
            raise WorkspaceError(
                "provider_response_invalid",
                "RunPod returned a response that was not valid JSON.",
                status_code=502,
            ) from error
        if not isinstance(payload, (dict, list)):
            raise WorkspaceError(
                "provider_response_invalid",
                "RunPod returned an unexpected response shape.",
                status_code=502,
            )
        return payload

    @staticmethod
    def _object_response(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise WorkspaceError(
                "provider_response_invalid",
                "RunPod returned an unexpected response shape.",
                status_code=502,
            )
        return payload

    @staticmethod
    def _raise_provider_error(status_code: int) -> None:
        if status_code == 401:
            code, message, client_status = (
                "invalid_api_key",
                "RunPod rejected the API key.",
                401,
            )
        elif status_code == 403:
            code, message, client_status = (
                "insufficient_api_permissions",
                "The RunPod API key does not have the required permissions.",
                403,
            )
        elif status_code == 402:
            code, message, client_status = (
                "insufficient_runpod_credit",
                "RunPod reports insufficient account credit.",
                409,
            )
        elif status_code in {404, 409}:
            code, message, client_status = (
                "provider_resource_unavailable",
                "The requested RunPod resource is unavailable or changed state.",
                409,
            )
        elif status_code >= 500:
            code, message, client_status = (
                "provider_unavailable",
                "RunPod could not complete the request.",
                502,
            )
        else:
            code, message, client_status = (
                "provider_request_rejected",
                "RunPod rejected the provider request.",
                400,
            )
        raise WorkspaceError(code, message, status_code=client_status)
