export type WorkspaceState =
  | "provisioning"
  | "starting"
  | "ready"
  | "stopping"
  | "stopped"
  | "deleting"
  | "deleted"
  | "error";

export interface CapabilityManifest {
  api_version: string;
  project_schema: string;
  project_schema_version: number;
  minimum_gpu_memory_gb: number;
  workspace_modes: string[];
  development_backend: boolean;
}

export interface CredentialStatus {
  configured: boolean;
  key_hint: string | null;
  validated_at: string | null;
  development_only: boolean;
}

export interface GpuOption {
  id: string;
  display_name: string;
  memory_gb: number;
  secure_available: boolean;
  community_available: boolean;
  on_demand_price_per_hour: number;
  interruptible_price_per_hour: number | null;
  available: boolean;
}

export interface WorkspacePlanRequest {
  mode: "persistent_pod";
  gpu_priority_ids: string[];
  cloud_type: "secure" | "community";
  interruptible: boolean;
  container_disk_gb: number;
  workspace_disk_gb: number;
  idle_timeout_seconds: number;
  hard_deadline_seconds: number;
}

export interface WorkspacePlan {
  id: string;
  request: WorkspacePlanRequest;
  selected_gpu: GpuOption;
  estimated_compute_per_hour: number;
  estimated_storage_per_month: number;
  image_digest: string;
  warnings: string[];
  created_at: string;
}

export interface WorkspaceRecord {
  id: string;
  name: string;
  mode: "persistent_pod";
  state: WorkspaceState;
  gpu: GpuOption;
  cloud_type: "secure" | "community";
  interruptible: boolean;
  container_disk_gb: number;
  workspace_disk_gb: number;
  estimated_compute_per_hour: number;
  estimated_storage_per_month: number;
  idle_timeout_seconds: number;
  hard_deadline_seconds: number;
  lease_expires_at: string;
  hard_expires_at: string;
  created_at: string;
  updated_at: string;
  provider_resource_id: string | null;
  readiness: Record<string, boolean>;
  error_code: string | null;
  error_message: string | null;
}

export interface ApiErrorBody {
  code: string;
  message: string;
}

export type FileKind = "diffusion_models" | "text_encoders" | "vae" | "loras" | "upscale_models" | "face_detection" | "projects" | "inputs" | "outputs";

export interface FileRecord {
  id: string;
  kind: FileKind;
  display_name: string;
  size_bytes: number;
  sha256: string;
  modified_at: string;
}

export interface FilePage {
  items: FileRecord[];
  next_cursor: string | null;
}

export interface UploadSession {
  id: string;
  filename: string;
  display_name: string;
  destination_kind: FileKind;
  size_bytes: number;
  sha256: string;
  chunk_size_bytes: number;
  chunk_count: number;
  completed_chunks: number[];
  state: string;
}

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(status: number, body: ApiErrorBody) {
    super(body.message);
    this.name = "ApiError";
    this.status = status;
    this.code = body.code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (response.status === 204) return undefined as T;
  const body = (await response.json()) as T | ApiErrorBody;
  if (!response.ok) {
    throw new ApiError(response.status, body as ApiErrorBody);
  }
  return body as T;
}

export const controlPlane = {
  capabilities: () => request<CapabilityManifest>("/api/v1/capabilities"),
  credentialStatus: () =>
    request<CredentialStatus>("/api/v1/credentials/runpod"),
  connectRunPod: (apiKey: string) =>
    request<CredentialStatus>("/api/v1/credentials/runpod", {
      method: "POST",
      body: JSON.stringify({ api_key: apiKey }),
    }),
  disconnectRunPod: () =>
    request<CredentialStatus>("/api/v1/credentials/runpod", {
      method: "DELETE",
    }),
  gpus: () => request<GpuOption[]>("/api/v1/gpus"),
  workspaces: () => request<WorkspaceRecord[]>("/api/v1/workspaces"),
  workspace: (workspaceId: string) =>
    request<WorkspaceRecord>(`/api/v1/workspaces/${workspaceId}`),
  planWorkspace: (payload: WorkspacePlanRequest) =>
    request<WorkspacePlan>("/api/v1/workspace-plans", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  createWorkspace: (planId: string, name: string) =>
    request<WorkspaceRecord>("/api/v1/workspaces", {
      method: "POST",
      body: JSON.stringify({ plan_id: planId, name }),
    }),
  startWorkspace: (workspaceId: string) =>
    request<WorkspaceRecord>(`/api/v1/workspaces/${workspaceId}/start`, {
      method: "POST",
    }),
  stopWorkspace: (workspaceId: string) =>
    request<WorkspaceRecord>(`/api/v1/workspaces/${workspaceId}/stop`, {
      method: "POST",
    }),
  extendLease: (workspaceId: string) =>
    request<WorkspaceRecord>(`/api/v1/workspaces/${workspaceId}/lease`, {
      method: "POST",
    }),
  terminateWorkspace: (workspaceId: string, confirmation: string) =>
    request<WorkspaceRecord>(`/api/v1/workspaces/${workspaceId}/terminate`, {
      method: "POST",
      body: JSON.stringify({ confirmation }),
    }),
  files: (workspaceId: string, kind: FileKind, cursor?: string) =>
    request<FilePage>(`/api/v1/workspaces/${workspaceId}/files?kind=${kind}${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ""}`),
  createUpload: (workspaceId: string, payload: {
    filename: string; destination_kind: FileKind; size_bytes: number; sha256: string; chunk_size_bytes: number;
  }) => request<UploadSession>(`/api/v1/workspaces/${workspaceId}/uploads`, {
    method: "POST", body: JSON.stringify(payload),
  }),
  uploadStatus: (workspaceId: string, uploadId: string) =>
    request<UploadSession>(`/api/v1/workspaces/${workspaceId}/uploads/${uploadId}`),
  uploadChunk: (workspaceId: string, uploadId: string, index: number, content: ArrayBuffer, sha256: string) =>
    request<{ upload_id: string; index: number }>(`/api/v1/workspaces/${workspaceId}/uploads/${uploadId}/chunks/${index}`, {
      method: "PUT",
      headers: { "Content-Type": "application/octet-stream", "X-Chunk-SHA256": sha256 },
      body: content,
    }),
  completeUpload: (workspaceId: string, uploadId: string) =>
    request<{ file: FileRecord; duplicate: boolean }>(`/api/v1/workspaces/${workspaceId}/uploads/${uploadId}/complete`, { method: "POST" }),
  cancelUpload: (workspaceId: string, uploadId: string) =>
    request<void>(`/api/v1/workspaces/${workspaceId}/uploads/${uploadId}`, { method: "DELETE" }),
};
