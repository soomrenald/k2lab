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
  mode: "persistent_pod" | "portable_workspace";
  gpu_priority_ids: string[];
  cloud_type: "secure" | "community";
  interruptible: boolean;
  container_disk_gb: number;
  workspace_disk_gb: number;
  idle_timeout_seconds: number;
  hard_deadline_seconds: number;
  network_volume_id: string | null;
  datacenter_priority_ids: string[];
}

export interface GpuAvailability {
  gpu_type_id: string;
  display_name: string;
  stock_status: string;
}

export interface DatacenterOption {
  id: string;
  name: string;
  location: string;
  gpu_availability: GpuAvailability[];
}

export interface NetworkVolumeOption {
  id: string;
  name: string;
  size_gb: number;
  datacenter_id: string;
}

export interface WorkspacePlan {
  id: string;
  request: WorkspacePlanRequest;
  selected_gpu: GpuOption;
  estimated_compute_per_hour: number;
  estimated_storage_per_month: number;
  image_digest: string;
  selected_datacenter_id: string | null;
  selected_network_volume: NetworkVolumeOption | null;
  create_network_volume: boolean;
  warnings: string[];
  created_at: string;
}

export interface WorkspaceRecord {
  id: string;
  name: string;
  mode: "persistent_pod" | "portable_workspace";
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
  gpu_priority_ids: string[];
  network_volume_id: string | null;
  datacenter_id: string | null;
  owns_network_volume: boolean;
  storage_tier: "pod_volume" | "network_volume";
  workspace_layout_version: number;
  retained_original_provider_resource_id: string | null;
}

export type MigrationState = "preparing" | "copying" | "verifying" | "awaiting_confirmation" | "completed" | "failed";

export interface WorkspaceMigrationRecord {
  id: string;
  workspace_id: string;
  state: MigrationState;
  source_provider_resource_id: string;
  target_provider_resource_id: string | null;
  target_network_volume_id: string | null;
  target_datacenter_id: string | null;
  target_workspace_disk_gb: number;
  owns_target_volume: boolean;
  current_file_index: number;
  current_file_offset: number;
  bytes_copied: number;
  bytes_total: number;
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface ApiErrorBody {
  code: string;
  message: string;
}

export interface BrowserSession {
  authenticated: boolean;
  subject: string;
  mfa_verified: boolean;
  expires_at: string;
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

export type RemoteProvider = "civitai" | "huggingface";
export type TransferState = "pending" | "resolving" | "downloading" | "verifying" | "paused" | "completed" | "cancelled" | "failed";

export interface CivitaiFilePreview {
  id: string;
  filename: string;
  size_bytes: number | null;
  format: string | null;
  sha256: string | null;
  pickle_scan: string | null;
  virus_scan: string | null;
  preferred: boolean;
  requires_unsafe_confirmation: boolean;
}

export interface CivitaiPreview {
  model_id: string;
  version_id: string;
  model_name: string;
  version_name: string;
  model_type: string | null;
  base_model: string | null;
  training_words: string[];
  files: CivitaiFilePreview[];
}

export interface HuggingFacePreview {
  repo_id: string;
  repo_type: string;
  revision: string;
  filename: string | null;
  mirror_repository: boolean;
  files: { filename: string; size_bytes: number; cached: boolean }[];
  required_bytes: number;
}

export interface RemoteTransfer {
  id: string;
  provider: RemoteProvider;
  source_url: string;
  destination_kind: FileKind;
  state: TransferState;
  filename: string | null;
  bytes_total: number | null;
  bytes_complete: number;
  sha256: string | null;
  files: FileRecord[];
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export type JobKind = "generate" | "edit_image" | "refine_faces";
export type JobState = "queued" | "starting" | "running" | "completed" | "cancelled" | "failed";

export interface GenerationJob {
  id: string;
  command_id: string;
  kind: JobKind;
  project_id: string;
  state: JobState;
  progress_current: number;
  progress_total: number;
  output_file_ids: string[];
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface JobEvent {
  sequence: number;
  state: string;
  message: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface JobEventPage {
  items: JobEvent[];
  next_cursor: string;
}

export interface UnifiedPromptPreview {
  prompt: string;
  regions: { id: string; name: string; spatial_role: string; clause: string }[];
}

export interface DetectedFaceRecord {
  index: number;
  box: [number, number, number, number];
  score: number;
}

export interface FaceDetectionResult {
  width: number;
  height: number;
  execution_provider: string;
  faces: DetectedFaceRecord[];
}

export interface WorkerReleaseResult {
  released: boolean;
  cancelled_job_ids: string[];
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
  const method = (init?.method ?? "GET").toUpperCase();
  const csrfToken = document.cookie
    .split("; ")
    .find((item) => item.startsWith("k2lab-csrf="))
    ?.slice("k2lab-csrf=".length);
  const response = await fetch(path, {
    ...init,
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
      ...(csrfToken && !["GET", "HEAD", "OPTIONS"].includes(method)
        ? { "X-CSRF-Token": decodeURIComponent(csrfToken) }
        : {}),
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
  openSession: () => request<BrowserSession>("/api/v1/auth/session", { method: "POST" }),
  capabilities: () => request<CapabilityManifest>("/api/v1/capabilities"),
  previewUnifiedPrompt: (project: Record<string, unknown>) =>
    request<UnifiedPromptPreview>("/api/v1/projects/unified-prompt-preview", {
      method: "POST",
      body: JSON.stringify({ project }),
    }),
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
  datacenters: () => request<DatacenterOption[]>("/api/v1/datacenters"),
  networkVolumes: () => request<NetworkVolumeOption[]>("/api/v1/network-volumes"),
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
  migrations: (workspaceId: string) =>
    request<WorkspaceMigrationRecord[]>(`/api/v1/workspaces/${workspaceId}/migrations`),
  createMigration: (workspaceId: string, payload: {
    network_volume_id?: string | null; workspace_disk_gb?: number; datacenter_priority_ids?: string[];
  }) => request<WorkspaceMigrationRecord>(`/api/v1/workspaces/${workspaceId}/migrations`, {
    method: "POST", body: JSON.stringify(payload),
  }),
  resumeMigration: (workspaceId: string, migrationId: string) =>
    request<WorkspaceMigrationRecord>(`/api/v1/workspaces/${workspaceId}/migrations/${migrationId}/resume`, { method: "POST" }),
  confirmMigration: (workspaceId: string, migrationId: string, confirmation: string) =>
    request<WorkspaceMigrationRecord>(`/api/v1/workspaces/${workspaceId}/migrations/${migrationId}/confirm`, {
      method: "POST", body: JSON.stringify({ confirmation }),
    }),
  files: (workspaceId: string, kind: FileKind, cursor?: string) =>
    request<FilePage>(`/api/v1/workspaces/${workspaceId}/files?kind=${kind}${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ""}`),
  saveProject: (workspaceId: string, filename: string, project: Record<string, unknown>) =>
    request<FileRecord>(`/api/v1/workspaces/${workspaceId}/projects/${encodeURIComponent(filename)}`, {
      method: "PUT", body: JSON.stringify({ project }),
    }),
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
  downloadCredential: (provider: RemoteProvider) =>
    request<CredentialStatus>(`/api/v1/credentials/downloads/${provider}`),
  storeDownloadCredential: (provider: RemoteProvider, token: string) =>
    request<CredentialStatus>(`/api/v1/credentials/downloads/${provider}`, {
      method: "POST", body: JSON.stringify({ token }),
    }),
  clearDownloadCredential: (provider: RemoteProvider) =>
    request<CredentialStatus>(`/api/v1/credentials/downloads/${provider}`, { method: "DELETE" }),
  previewCivitai: (workspaceId: string, sourceUrl: string) =>
    request<CivitaiPreview>(`/api/v1/workspaces/${workspaceId}/downloads/civitai/preview`, {
      method: "POST", body: JSON.stringify({ source_url: sourceUrl }),
    }),
  startCivitai: (workspaceId: string, payload: {
    source_url: string; file_id: string; destination_kind: FileKind; allow_unsafe_format: boolean; resume_transfer_id?: string;
  }) => request<RemoteTransfer>(`/api/v1/workspaces/${workspaceId}/downloads/civitai`, {
    method: "POST", body: JSON.stringify(payload),
  }),
  previewHuggingFace: (workspaceId: string, sourceUrl: string, allowPatterns: string[]) =>
    request<HuggingFacePreview>(`/api/v1/workspaces/${workspaceId}/downloads/huggingface/preview`, {
      method: "POST", body: JSON.stringify({ source_url: sourceUrl, allow_patterns: allowPatterns }),
    }),
  startHuggingFace: (workspaceId: string, payload: {
    source_url: string; destination_kind: FileKind; allow_patterns: string[]; allow_unsafe_format: boolean; resume_transfer_id?: string;
  }) => request<RemoteTransfer>(`/api/v1/workspaces/${workspaceId}/downloads/huggingface`, {
    method: "POST", body: JSON.stringify(payload),
  }),
  transfer: (workspaceId: string, transferId: string) =>
    request<RemoteTransfer>(`/api/v1/workspaces/${workspaceId}/transfers/${transferId}`),
  cancelTransfer: (workspaceId: string, transferId: string) =>
    request<RemoteTransfer>(`/api/v1/workspaces/${workspaceId}/transfers/${transferId}/cancel`, { method: "POST" }),
  submitJob: (workspaceId: string, payload: {
    command_id: string; kind: JobKind; project_id: string; project: Record<string, unknown>; input_file_id?: string;
    diffusion_model_file_id?: string; text_encoder_file_id?: string; vae_file_id?: string;
    face_detector_file_id?: string; filename_prefix: string;
    lora_file_ids?: string[]; upscale_model_file_id?: string; selected_face_indices?: number[];
    manual_face_paths?: number[][][];
  }) => request<GenerationJob>(`/api/v1/workspaces/${workspaceId}/jobs`, {
    method: "POST", body: JSON.stringify(payload),
  }),
  job: (workspaceId: string, jobId: string) =>
    request<GenerationJob>(`/api/v1/workspaces/${workspaceId}/jobs/${jobId}`),
  jobEvents: (workspaceId: string, jobId: string, cursor?: string) =>
    request<JobEventPage>(`/api/v1/workspaces/${workspaceId}/jobs/${jobId}/events${cursor ? `?cursor=${encodeURIComponent(cursor)}` : ""}`),
  cancelJob: (workspaceId: string, jobId: string) =>
    request<GenerationJob>(`/api/v1/workspaces/${workspaceId}/jobs/${jobId}/cancel`, { method: "POST" }),
  detectFaces: (workspaceId: string, payload: { input_file_id: string; face_detector_file_id?: string; threshold: number; provider: "auto" | "cpu" | "cuda" }) =>
    request<FaceDetectionResult>(`/api/v1/workspaces/${workspaceId}/faces/detect`, {
      method: "POST", body: JSON.stringify(payload),
    }),
  releaseWorkerMemory: (workspaceId: string) =>
    request<WorkerReleaseResult>(`/api/v1/workspaces/${workspaceId}/worker/release`, { method: "POST" }),
  outputUrl: (workspaceId: string, fileId: string) =>
    `/api/v1/workspaces/${workspaceId}/outputs/${fileId}`,
  fileUrl: (workspaceId: string, fileId: string) =>
    `/api/v1/workspaces/${workspaceId}/files/${fileId}/content`,
};
