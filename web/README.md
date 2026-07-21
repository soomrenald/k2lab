# K2 Region Lab Web

This directory contains the browser interface for the provider-neutral control plane in
`src/k2_region_lab/web`. It is the first implementation milestone from
`docs/runpod_web_workspace_spec.md`.

## Current milestone

Implemented:

- FastAPI workspace contracts and lifecycle endpoints;
- a safe in-memory development backend that never calls or bills RunPod;
- account connection, ordered GPU preferences, cloud type, storage, idle timeout, cost
  review, workspace creation, stop/start, lease extension, and typed deletion UX;
- the Generate, Edit, and Faces workspace shell;
- local PNG/JPEG/WebP loading for interface development;
- separate image-edit reference and target layers;
- SVG region drawing, selection, movement, and live eight-direction resizing;
- prompt, region, LoRA, and advanced inspector layouts;
- responsive desktop/mobile styling with locally bundled fonts.
- a production RunPod REST/GraphQL adapter with redacted provider errors;
- encrypted process-local credential storage and explicit production backend selection;
- live GPU inventory/pricing plans and persistent-Pod create/start/stop/delete requests.
- a separate versioned workspace-agent image with authenticated health, capabilities,
  storage validation, and idempotent persistent-layout initialization.
- durable cloud-file inventory, checksum-verified resumable uploads, duplicate detection,
  and authenticated ranged output retrieval;
- an Assets panel with streaming SHA-256 hashing, pause/resume/retry/cancel controls,
  transfer progress, throughput, and ETA.
- provider-side Civitai and Hugging Face inspection/download jobs with encrypted
  least-privilege tokens, strict URL/redirect validation, resumable Civitai ranges,
  Hugging Face cache reuse, unsafe-format confirmation, and safetensors validation.
- generation, image-edit, and face-refinement jobs using the canonical project document,
  durable summaries/events, cursor-based reconnect, cancellation, progress display, and
  authenticated output retrieval through opaque file IDs.
- persistent-Pod and portable-workspace onboarding, RunPod datacenter/network-volume
  inventory, compatible GPU filtering, independent storage pricing, and ephemeral Pod
  termination/recreation around a retained network volume.
- sealed allowlisted workspace manifests, resumable checksum-verified persistent-to-portable
  copy, durable reconnectable migration progress, manifest-gated switchover, and separate
  typed confirmation before deleting the stopped original Pod.

Not yet implemented:

- a cloud-KMS adapter (the production vault currently uses a persisted Fernet root key);
- a published and signed CUDA workspace image (the build definition and agent are present);
- multi-account tenancy or a hosted deployment.

The development backend is labelled throughout the UI. Its generation buttons are
disabled so it cannot be confused with a connected GPU worker.

## Experimental RunPod backend

The default remains the non-billing development backend. The RunPod backend must be
selected explicitly and requires an immutable runtime image plus a Fernet encryption key:

```bash
export K2LAB_WEB_BACKEND=runpod
export K2LAB_CREDENTIAL_FERNET_KEY="<persisted-secret-from-your-KMS-bootstrap>"
export K2LAB_DATABASE_URL="postgresql+asyncpg://k2lab:<password>@<host>/k2lab"
export K2LAB_ALLOWED_ORIGINS="https://studio.example.com"
export K2LAB_AUTH_ALLOWED_SUBJECT="<stable-subject-from-your-identity-provider>"
export K2LAB_TRUSTED_PROXY_SECRET="<random-secret-at-least-32-characters>"
uv run k2lab-web
```

Generate the Fernet value once with `Fernet.generate_key()`, store it in a secret manager,
and reuse it after every restart. Rotating or losing it makes existing provider credentials
unreadable. SQLite (`sqlite+aiosqlite:////absolute/path`) is supported for isolated local
tests, while PostgreSQL is the production store.

The RunPod backend refuses to start without the hosted security settings above. Deploy it
behind a TLS authentication proxy that removes all inbound `X-K2-*` headers, completes
identity-provider login/MFA, and then injects `X-K2-Authenticated-User`,
`X-K2-Authenticated-MFA: true`, and `X-K2-Proxy-Secret` on the session-bootstrap request.
The allowed subject makes this deployment single-account by design. The control plane then
uses a rotating opaque `Secure`, `HttpOnly`, `SameSite=Strict` browser session plus a
double-submit CSRF token, strict Origin checks, request bounds, and endpoint-class rate
limits. Do not expose the Uvicorn port directly to the internet.

This mode can create billable Pods. Workspace records, leases, encrypted credentials,
provider-resource mappings, the operation-journal schema, and redacted audit events are
durable. Startup reconciliation refreshes known Pod state, and a background reaper stops
compute after lease expiry. The Pod remains in `starting` until the versioned workspace
image and authenticated agent are available.

Browser uploads currently pass through the authenticated control plane to the workspace
agent. Upload manifests and completed chunks live on the persistent workspace volume, so
an interrupted browser transfer can query the session and send only missing chunks.
Provider downloads run inside the workspace, so large model files do not pass through the
browser or control plane. Provider tokens are encrypted at rest and are forwarded in an
agent request header for one operation; they are not written into transfer records.

Remote runs start an isolated line-delimited worker process and send the same versioned
project data used by the local application. The agent persists redacted job events on the
workspace volume, while the control plane persists job summaries and reconnectable event
cursors. Raw filesystem paths, prompts, and credentials are excluded from those events;
completed images are returned through authenticated opaque output URLs with HTTP Range
support.

Migration temporarily bills both source and target compute plus both storage resources.
Closing the browser does not discard progress: reopen the workspace, choose the migration
action, and resume from the last accepted chunk. Explicit **Stop GPU now** aborts an active
migration, terminates its temporary target Pod, unseals the source, and retains the target
network volume. A manifest mismatch also leaves the original workspace authoritative and
retains the network volume for inspection. The application never automatically deletes a
network volume. See [`../docs/runpod_workspace_operations.md`](../docs/runpod_workspace_operations.md)
for the lifecycle and recovery runbook.

## Local development

Install the Python and browser dependencies:

```bash
uv sync --extra dev --extra web
cd web/client
npm install
```

Run the control plane from the repository root:

```bash
uv run k2lab-web --reload
```

In a second terminal, run the Vite client:

```bash
cd web/client
npm run dev
```

Open <http://127.0.0.1:5173>. Vite proxies `/api` to
`http://127.0.0.1:8000`. In development mode, any test key containing at least eight
characters passes the simulated credential check. Do not use a real provider credential
with the development backend.

## Validation

```bash
uv run pytest -q tests/test_web_control_plane.py
cd web/client
npm run build
```

The production bundle is written to the ignored `web/client/dist/` directory.

The ordinary suite never provisions a Pod. A destructive live acceptance test exists for
a dedicated disposable RunPod account and remains skipped unless every required variable
and this exact sentinel are provided:

```bash
uv run pytest -q tests/test_runpod_live_acceptance.py
```

The live suite covers both a billable persistent Pod and a portable network-volume workspace.
It verifies upload persistence across persistent stop/start and verifies an allowlisted SHA-256
manifest across portable Pod termination/recreation. Cleanup permanently deletes every test Pod
and the explicitly tracked disposable network volume.
