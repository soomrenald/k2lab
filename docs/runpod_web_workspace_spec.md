# RunPod Web Workspace Implementation Specification

Status: approved future implementation plan  
Audience: product, frontend, backend, infrastructure, security, and QA engineers  
Scope: non-explicit K2 Region Lab workloads using infrastructure owned and billed by each user

## 1. Purpose

This document specifies a user-friendly web application that provisions and controls a
RunPod GPU workspace using a RunPod API key supplied by the user. The user must not need
to install Docker, CUDA, Python, ComfyUI, Git, model-management scripts, or command-line
tools. The application must support K2 image generation, regional prompting, regional
LoRA routing, image editing, face refinement, project persistence, local file transfer,
and direct model downloads from Civitai and Hugging Face.

Implementation is divided into two phases:

1. **Persistent-Pod mode:** one RunPod Pod and its regular `/workspace` volume are kept as
   the user's cloud workspace. The Pod is stopped when idle and restarted later. This is
   the recommended initial implementation.
2. **Portable-workspace mode:** the application offers a choice between the original
   persistent Pod and a RunPod network volume attached to an ephemeral Pod selected from
   the user's GPU priority list for each session.

The second phase must not require redesigning the first-phase client, project schema,
remote-agent API, download manager, or generation protocol.

## 2. Product goals

- A first-time user can connect a RunPod account, select compute and storage settings,
  provision a workspace, download required files, and generate an image without using a
  terminal.
- The user pays RunPod directly. The application never pools users onto an
  application-owned endpoint.
- Stopping a persistent Pod releases its GPU. A prominent lifecycle indicator prevents
  accidental compute charges.
- Models, LoRAs, projects, inputs, and outputs survive normal stop/start cycles.
- The same project document can execute through a local worker or a remote Pod worker.
- Large uploads and downloads are resumable, observable, cancellable, and verified.
- Provider credentials are encrypted, least-privileged, redacted from logs, and never
  written into project JSON or generated PNG metadata.
- A client crash, lost browser connection, or sleeping computer does not leave compute
  running indefinitely.

## 3. Non-goals

- Phase one does not move a regular Pod volume between machines or regions.
- Phase one does not guarantee that a stopped Pod will regain a GPU immediately.
- The application does not redistribute third-party model weights without an applicable
  license.
- The application does not accept arbitrary shell commands or arbitrary download hosts.
- The application is not a general RunPod administration console.
- Network volumes are not treated as permanent backup. Users must be able to export
  critical data to local or dedicated object storage.

## 4. Required architecture

```text
Web client
  |
  | HTTPS + authenticated session
  v
Application control plane
  |- user/account database
  |- encrypted credential vault
  |- workspace state machine and lease watchdog
  |- RunPod REST/GraphQL adapter
  `- audit and job event store
          |
          | RunPod API with the user's decrypted key
          v
RunPod Pod
  |- versioned K2 CUDA container
  |- authenticated workspace agent
  |- K2 worker process
  `- /workspace persistent data
```

Recommended implementation stack:

- Web client: TypeScript, React, and Vite, with a canvas/SVG region overlay.
- Control plane: Python 3.12, FastAPI, Pydantic, PostgreSQL, and an encrypted secret
  provider backed by a cloud KMS.
- Remote agent: Python 3.12 and FastAPI/Uvicorn inside the K2 CUDA image.
- GPU execution: the existing line-delimited K2 worker protocol behind an agent adapter.
- Builds: CI builds, scans, signs, and publishes immutable container-image digests.

The client must call application APIs, not RunPod directly. This prevents exposing the
RunPod key to browser JavaScript and gives the lease watchdog authority to clean up
abandoned compute.

## 5. Shared domain interfaces

The UI must target these backend interfaces rather than provider-specific endpoints:

```text
WorkspaceBackend
  validate_credentials()
  list_gpu_options()
  list_storage_options()
  plan_workspace(request)
  start_workspace(plan)
  stop_workspace(workspace_id)
  terminate_workspace(workspace_id)
  get_workspace_status(workspace_id)
  get_cost_snapshot(workspace_id)
  get_file_inventory(workspace_id)
  upload_file(workspace_id, upload)
  start_remote_download(workspace_id, request)
  submit_job(workspace_id, command)
  cancel_job(workspace_id, job_id)

LocalWorkerBackend implements the same job-facing subset.
RunPodPersistentPodBackend implements phase one.
RunPodPortableWorkspaceBackend implements phase two.
```

The canonical K2 project document remains the job source of truth. Remote execution may
add transport fields, but it must not maintain a divergent prompt/region/LoRA model.

## 6. Credential model

### 6.1 RunPod

The onboarding screen directs the user to create a restricted RunPod API key with only
the Pod, template, GPU/datacenter inventory, volume, and billing-read permissions needed
by the selected phase. The control plane validates the key with a read-only identity or
inventory request before saving it.

The plaintext key:

- exists only during TLS request handling and provider calls;
- is envelope-encrypted before database storage;
- is never returned to the web client after initial submission;
- is never placed in Pod environment variables, logs, URLs, project files, or metadata;
- can be replaced or revoked from the application;
- is automatically redacted by structured logging and error reporting.

### 6.2 Hugging Face and Civitai

Hugging Face tokens must be read-only or fine-grained. Civitai tokens are download-only.
They use the same encrypted-vault treatment. The control plane passes a token to the
workspace agent only for one authenticated download job. The agent holds it in memory,
never includes it in a URL, and clears references after the request completes.

### 6.3 Agent authentication

Each Pod session receives a random 256-bit agent-session secret distinct from all
provider keys. It may be supplied during Pod creation because it is replaceable and
scoped to that Pod. All proxy requests require it. Rotate it whenever a Pod is recreated,
and invalidate it in the control plane when the workspace stops or terminates.

## 7. Phase one: persistent-Pod mode

### 7.1 Workspace resource model

Each application workspace owns:

- one RunPod Pod ID;
- one regular Pod volume mounted at `/workspace`;
- a selected immutable container image digest;
- one ordered GPU preference list used for initial provisioning;
- container disk, volume disk, and optional resource constraints;
- an inactivity timeout and maximum-session deadline;
- a durable application database record containing lifecycle and reconciliation state.

The regular volume survives Pod stop/start but is deleted by RunPod when the Pod is
terminated. The product must consistently distinguish **Stop** from **Delete cloud
workspace**. Delete requires typed confirmation plus a backup warning.

### 7.2 Onboarding workflow

1. User submits a RunPod API key.
2. Control plane validates it and retrieves the account's visible GPU/datacenter data.
3. UI presents supported GPUs only. Initially require one GPU and at least 24 GiB VRAM;
   exact compatibility must be driven by a versioned image capability manifest.
4. User orders a GPU priority list and chooses Secure or Community Cloud where supported.
5. User chooses on-demand by default. Spot/interruptible compute is an Advanced option
   with a clear preemption warning.
6. User selects container-disk size and persistent `/workspace` size. Validate both
   against the runtime plus model manifest before provisioning.
7. UI shows estimated hourly compute cost, persistent stopped-volume cost, region/cloud
   limitations, idle timeout, and destructive lifecycle semantics.
8. User confirms **Create cloud workspace**.

### 7.3 Pod creation

Create the Pod through the RunPod REST API with a request equivalent to:

```json
{
  "name": "k2lab-<workspace-short-id>",
  "imageName": "<registry>/<image>@sha256:<digest>",
  "gpuTypeIds": ["<ordered GPU IDs>"],
  "gpuTypePriority": "custom",
  "gpuCount": 1,
  "containerDiskInGb": 50,
  "volumeInGb": 200,
  "volumeMountPath": "/workspace",
  "interruptible": false,
  "locked": false,
  "ports": ["8080/http"],
  "env": {
    "K2LAB_AGENT_SESSION_TOKEN": "<ephemeral secret>",
    "K2LAB_WORKSPACE_ID": "<opaque id>",
    "K2LAB_IMAGE_VERSION": "<version>"
  }
}
```

Provider availability is inherently racy. A failed creation must produce a fresh plan
from the remaining GPU preference list instead of blindly retrying. Store the provider
request ID, response, Pod ID, selected GPU, datacenter, price, and image digest, with all
secrets redacted.

### 7.4 Startup and readiness

The container entrypoint must not install the runtime. It performs only:

1. validate required GPU/CUDA compatibility;
2. create/migrate the `/workspace/k2lab` directory structure;
3. verify writable space and minimum free capacity;
4. start the workspace agent;
5. start or lazily start the isolated K2 worker;
6. report staged readiness: `container`, `agent`, `storage`, `models`, `worker`.

The web client displays these stages. A Pod reporting `RUNNING` is not considered ready
until the agent's authenticated `/v1/health` response reports its expected workspace ID
and image version.

### 7.5 Stop and restart

Stopping a workspace calls the RunPod stop API. It releases GPU compute, destroys
container-disk contents, and retains `/workspace`. The application retains the Pod ID.

Starting calls the RunPod start API and then performs readiness checks. Because a regular
volume is tied to the Pod's physical machine, restart can yield no GPU if the original
capacity has been rented. The UI must then offer:

- retry later;
- start CPU-only for file export if RunPod exposes the option;
- use RunPod's supported migration flow;
- migrate to phase-two portable storage after that feature ships.

GPU priority affects initial provisioning; the UI must not imply that a stopped
persistent Pod is freely reselected from the entire priority list on every restart.

### 7.6 Cost and abandonment safety

The system must not rely only on browser-close events. The control plane owns a lease:

- browser sessions send activity/heartbeat events;
- active generation and transfer jobs extend the lease;
- ordinary UI activity may extend it up to a hard session deadline;
- warnings appear before idle termination;
- **Extend session** requires an explicit user action;
- when the lease expires, the control plane stops the persistent Pod;
- a scheduled reaper reconciles every supposedly active workspace against RunPod;
- startup reconciliation detects orphaned running Pods and immediately shows their cost;
- a global **Stop GPU now** control remains visible in every workspace screen.

Stopping eliminates GPU idle cost, not storage cost. The UI must show stopped-volume cost
and must not use the phrase "free while stopped."

## 8. Remote workspace-agent API

The agent is publicly reachable through RunPod's HTTPS proxy and therefore requires
authentication, strict schemas, rate limits, and path isolation. Requests expected to
exceed the proxy timeout return a job ID immediately.

Minimum API:

```text
GET    /v1/health
GET    /v1/capabilities
GET    /v1/storage
GET    /v1/files?kind=<kind>&cursor=<cursor>
POST   /v1/uploads
PUT    /v1/uploads/{upload_id}/chunks/{index}
POST   /v1/uploads/{upload_id}/complete
DELETE /v1/uploads/{upload_id}
POST   /v1/downloads/civitai
POST   /v1/downloads/huggingface
GET    /v1/transfers/{job_id}
POST   /v1/transfers/{job_id}/cancel
POST   /v1/jobs
GET    /v1/jobs/{job_id}
GET    /v1/jobs/{job_id}/events?cursor=<cursor>
POST   /v1/jobs/{job_id}/cancel
GET    /v1/outputs/{file_id}
```

All file references are opaque IDs. The agent maps them to paths under an allowlisted
root. It must reject `..`, absolute paths, symlink escapes, device files, FIFOs, and
unsupported destinations.

### 8.1 Workspace layout

```text
/workspace/k2lab/
  models/
    diffusion_models/
    text_encoders/
    vae/
    loras/
    upscale_models/
    face_detection/
  projects/
  inputs/
  outputs/
  downloads/
    incomplete/
  cache/
    huggingface/
  state/
    migrations/
    inventory/
```

No durable credentials are stored in this tree.

## 9. File upload and retrieval

Browser uploads are chunked and resumable. The control plane creates an upload session;
the browser may upload directly to the authenticated Pod agent using a short-lived scoped
transfer token. Each chunk records length and checksum. Completion verifies total size
and SHA-256 before an atomic rename from `downloads/incomplete`.

Required behavior:

- configurable concurrency and chunk size;
- progress, speed, elapsed time, and estimated remaining time;
- pause, resume, retry, and cancellation;
- duplicate detection by SHA-256;
- free-space validation before acceptance;
- safe filename normalization while preserving a display name;
- explicit destination kinds rather than arbitrary paths;
- output download with HTTP range support;
- local deletion and cloud deletion treated as separate actions.

## 10. Civitai download workflow

The user pastes a Civitai model or model-version download URL. The agent must not execute
the URL through a shell.

1. Parse and validate HTTPS plus an exact `civitai.com` host allowlist.
2. Resolve the model-version ID through the Civitai API.
3. Retrieve metadata and present model name, version, type, filename, size, training
   words, file format, hashes, and available scan results.
4. Require the user to choose a destination category when it cannot be inferred safely.
5. Download with the token in an authorization header, following only approved redirects.
6. Stream to an incomplete file, support retry/range where the server permits it, and
   continuously report progress.
7. Verify expected size and available hash; reject HTML/error payloads masquerading as
   model files.
8. Prefer safetensors. Warn and require confirmation for pickle-based formats.
9. Atomically install and refresh model inventory.

The event and error models must never include the supplied token or an authenticated URL.

## 11. Hugging Face download workflow

Accept canonical Hugging Face repository, blob, resolve, and file URLs. Parse them into
`repo_id`, `repo_type`, `filename`, and `revision`; do not pass arbitrary URLs to a shell.
Use `huggingface_hub.hf_hub_download` or `snapshot_download`, with:

- a one-operation read/fine-grained token;
- `/workspace/k2lab/cache/huggingface` as cache;
- an explicit final model destination;
- dry-run metadata when available to display required bytes;
- support for pinned revisions and gated repositories;
- a clear 401/403 explanation instructing the user to accept repository access terms;
- hash/size reporting and atomic installation.

The UI distinguishes downloading one file from mirroring a repository and estimates the
space required before the latter.

## 12. Generation and image-edit jobs

The control plane sends the same typed commands used by the local worker. The agent:

1. validates a submitted project and command against its advertised schema version;
2. resolves opaque file IDs into allowlisted paths;
3. starts or reuses the isolated GPU worker;
4. forwards progress events into its durable in-memory job event buffer;
5. writes output PNG and embedded project metadata to `/workspace/k2lab/outputs`;
6. returns an opaque output-file record, never a raw filesystem path;
7. supports cancellation by terminating and safely restarting the isolated worker.

Client reconnects resume from an event cursor. Job records in the control plane retain
summary, status, timestamps, image version, project ID, output IDs, and redacted failure
information.

## 13. Versioning and upgrades

- Containers use semantic application versions and immutable image digests.
- `/v1/capabilities` returns agent, worker-protocol, project-schema, CUDA, PyTorch, and
  model-layout versions.
- Persistent storage migrations are forward-only, idempotent, journaled, and backed up
  before destructive transformation.
- The control plane never silently resets a running Pod to update it. It asks the user to
  stop, displays the change, then starts from the new image.
- A rollback can select the previous compatible image without downgrading storage unless
  that downgrade is explicitly supported.

## 14. Phase two: portable-workspace mode

### 14.1 User-visible choice

Add a workspace-type selection:

- **Persistent Pod:** faster and simpler stop/start on the same machine; regular volume
  disappears if the Pod is terminated.
- **Portable workspace:** independent network volume survives Pod termination and can be
  attached to a newly selected Pod in the same datacenter.

Do not automatically convert existing storage. Migration is a separate, resumable job.

### 14.2 Portable resource model

Each portable workspace owns:

- one RunPod network-volume ID and datacenter ID;
- zero or one active ephemeral Pod ID;
- storage tier, capacity, and workspace-layout version;
- the same GPU priorities, lifecycle lease, and container image settings as phase one.

Network volumes for Pods require Secure Cloud and replace the ordinary `/workspace`
volume. They must be attached when the Pod is created.

### 14.3 Existing network volume

1. List the user's network volumes.
2. Read each volume's datacenter and capacity.
3. Validate or initialize the K2 directory marker after attaching it to a Pod.
4. Filter the GPU priority list to capacity visible in that datacenter.
5. Show unavailable preferences rather than silently discarding them.
6. Create a new Pod pinned to the volume's datacenter and attach `networkVolumeId`.

### 14.4 New network volume

1. Query datacenters and GPU availability.
2. Rank `(GPU, datacenter)` candidates by the user's ordered GPU preferences, then region
   preferences, availability, and configured maximum price.
3. Show the proposed selection and persistent storage price.
4. Create the volume in the chosen datacenter.
5. Create the Pod attached to it.
6. Journal both operations. If Pod allocation loses a race, retain the empty volume and
   re-plan within that datacenter by default. Never delete a non-empty volume
   automatically.

### 14.5 Portable shutdown

At idle timeout or explicit stop, cancel/finish active work, flush agent state, verify
outputs, and terminate the ephemeral Pod. The network volume remains. On the next session,
create a new Pod from the current priority list restricted to that volume's datacenter.

Network volume persistence removes GPU idle cost but not storage cost.

### 14.6 Region changes and multi-region fallback

A single network volume is datacenter-bound. Moving a workspace requires a second volume
and an explicit copy operation through RunPod's S3-compatible interface or temporary
compute. The application must show that this is a copy, not an instantaneous attachment.

Phase two may later support one replica per preferred datacenter, but replicas are not
automatically coherent. A manifest, generation number, conflict policy, and exclusive
writer lease are required before multi-region writes are enabled.

### 14.7 Migration from persistent Pod

Migration steps:

1. stop generation and transfers;
2. calculate required bytes and create/select a compatible network volume;
3. start temporary migration compute if required;
4. copy `/workspace/k2lab` with resumable verification;
5. compare file inventory and SHA-256 manifests;
6. switch the application's workspace record only after verification;
7. retain the stopped original Pod until the user confirms success;
8. warn that terminating the original deletes its regular volume;
9. require explicit deletion confirmation.

## 15. Persistence schema

Minimum control-plane entities:

```text
User
ProviderCredential(id, user_id, provider, encrypted_secret, key_hint, permissions,
                   created_at, last_validated_at, revoked_at)
Workspace(id, user_id, mode, state, image_digest, idle_timeout_seconds,
          hard_deadline_seconds, created_at, updated_at)
RunPodResource(workspace_id, pod_id, volume_kind, volume_id, datacenter_id,
               gpu_type_id, cloud_type, container_disk_gb, workspace_disk_gb,
               cost_per_hour, desired_state, observed_state)
Transfer(id, workspace_id, kind, source, destination_kind, state, bytes_total,
         bytes_complete, sha256, error_code, created_at, updated_at)
GenerationJob(id, workspace_id, command_id, command_kind, project_id, state,
              progress_current, progress_total, output_file_ids, error_code,
              created_at, updated_at)
Lease(workspace_id, expires_at, hard_expires_at, last_activity_at, reason)
AuditEvent(user_id, workspace_id, action, result, redacted_context, created_at)
```

State transitions must use compare-and-swap or database row locking so duplicate browser
requests cannot create two Pods.

## 16. Failure handling

Define stable user-facing error codes for:

- invalid/insufficient API key;
- insufficient RunPod credit;
- requested GPU unavailable;
- Pod returned zero GPUs;
- volume/datacenter mismatch;
- image pull or readiness failure;
- incompatible CUDA/GPU/image manifest;
- storage full;
- proxy timeout or agent authentication failure;
- download unauthorized, gated, missing, unsafe, or hash mismatch;
- worker OOM and deterministic retry outcome;
- abandoned resource discovered during reconciliation;
- attempted deletion of non-empty persistent storage.

Every multi-resource operation is journaled and reconciled after process restart. Users
must never have to inspect raw container logs to understand a common failure.

## 17. Security requirements

- TLS for every browser, control-plane, provider, and Pod-agent connection.
- CSRF protection, secure cookies, session rotation, and MFA support for hosted accounts.
- Strict origin policy and no provider token in browser storage.
- Agent session tokens scoped to one workspace and one Pod lifetime.
- Host allowlists for remote downloads and redirect validation on every hop.
- No shell interpolation of URLs, filenames, prompts, or model names.
- File type, maximum size, free-space, path, and checksum validation.
- Safetensors header inspection before a downloaded LoRA/model is selectable.
- Rate limiting on provisioning, upload, download, and job endpoints.
- Signed container images, dependency scanning, and a published software bill of
  materials.
- Structured audit entries for credential changes, provisioning, lifecycle changes,
  downloads, and deletions.

## 18. Observability

Collect:

- provisioning duration by stage;
- Pod start/readiness/failure counts by GPU and datacenter;
- orphan reaper actions;
- active versus stopped duration and estimated cost;
- worker load time, generation time, OOM/retry data, and free VRAM;
- transfer throughput, retries, cancellations, and checksum failures;
- agent and image versions.

Do not collect prompt text, uploaded image contents, model tokens, or provider keys in
ordinary telemetry.

## 19. Test strategy

### Unit tests

- RunPod response parsing and GPU-priority planning;
- lifecycle state machine and idempotency;
- cost and timeout calculations;
- URL parsers and redirect allowlists;
- path traversal and symlink escape rejection;
- chunk assembly, resume, and hash verification;
- credential redaction;
- project/worker protocol compatibility.

### Contract tests

- generated client against RunPod's OpenAPI schema;
- mocked provider behavior for races, zero-GPU restart, insufficient funds, and partial
  failures;
- Civitai and Hugging Face metadata/download fixtures without real secrets;
- agent API backward/forward compatibility.

### Integration tests

- create, ready, stop, restart, and delete a disposable persistent Pod;
- confirm `/workspace` survives stop/start and container disk does not;
- upload, cancel, resume, and download a large file;
- direct Civitai/Hugging Face download into the correct model directory;
- load a downloaded LoRA and complete a generation;
- client disconnect followed by lease-reaper stop;
- phase two: terminate/recreate around one network volume and verify the manifest.

### Acceptance scenarios for phase one

- A clean account reaches a ready workspace using only the web UI.
- Closing all clients causes the configured idle stop even if no close event arrives.
- Reopening starts the same Pod and finds all prior `/workspace` files.
- An unavailable restart is explained without data loss.
- No credential appears in browser storage, logs, project JSON, PNG metadata, or agent
  job history.
- Deleting the cloud workspace cannot occur through a single accidental click.

### Acceptance scenarios for phase two

- User can choose Persistent Pod or Portable Workspace during creation.
- A portable Pod terminates while its network-volume inventory remains intact.
- A later Pod is selected from compatible priority GPUs in the volume's datacenter.
- Existing and newly created network volumes both work.
- Persistent-to-portable migration verifies a complete manifest before switchover.

## 20. Implementation order

1. Extract provider-neutral workspace/job interfaces and formalize the worker transport.
2. Build RunPod API adapter, encrypted credentials, planning, and reconciliation.
3. Publish the versioned CUDA container with agent and existing K2 worker.
4. Implement persistent-Pod onboarding, create/start/stop, readiness, lease, and cost UI.
5. Implement file inventory plus resumable local upload/download.
6. Implement Civitai and Hugging Face download jobs.
7. Implement generation/edit/refinement job submission and reconnectable events.
8. Harden security, audit, backups, failure UX, and disposable-account integration tests.
9. Ship persistent-Pod mode.
10. Add network-volume inventory/planning and ephemeral Pod lifecycle.
11. Add verified persistent-to-portable migration.
12. Ship the two-mode workspace selector.

## 21. Authoritative platform references

- RunPod REST API overview: <https://docs.runpod.io/api-reference/overview>
- RunPod Pod creation: <https://docs.runpod.io/api-reference/pods/POST/pods>
- RunPod Pod lifecycle: <https://docs.runpod.io/pods/manage-pods>
- RunPod storage types: <https://docs.runpod.io/pods/storage/types>
- RunPod network volumes: <https://docs.runpod.io/storage/network-volumes>
- RunPod port/proxy behavior: <https://docs.runpod.io/pods/configuration/expose-ports>
- RunPod API-key permissions: <https://docs.runpod.io/get-started/api-keys>
- Hugging Face downloads: <https://huggingface.co/docs/huggingface_hub/guides/download>
- Hugging Face token scopes: <https://huggingface.co/docs/hub/en/security-tokens>
- Civitai REST API reference:
  <https://github.com/civitai/civitai/wiki/REST-API-Reference/a1f328d15e27c0149c4627473008c99f110f1a61>

