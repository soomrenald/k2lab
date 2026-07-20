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

Not yet implemented:

- durable PostgreSQL persistence, a KMS-backed credential repository, and the lease reaper;
- a published CUDA workspace image or authenticated Pod agent;
- cloud file inventory, resumable transfer, Civitai, or Hugging Face downloads;
- remote generation/image-edit/face-refinement job submission and event streaming;
- production authentication, authorization, CSRF protection, or hosted deployment.

The development backend is labelled throughout the UI. Its generation buttons are
disabled so it cannot be confused with a connected GPU worker.

## Experimental RunPod backend

The default remains the non-billing development backend. The RunPod backend must be
selected explicitly and requires an immutable runtime image plus a Fernet encryption key:

```bash
export K2LAB_WEB_BACKEND=runpod
export K2LAB_CREDENTIAL_FERNET_KEY="$(python -c \
  'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
uv run k2lab-web
```

This mode can create billable Pods. It is an integration milestone, not a deployable
hosted control plane: workspace records and encrypted credentials are process-local until
the PostgreSQL/KMS repository and startup reconciler are implemented. The Pod also remains
in `starting` until the versioned workspace image and authenticated agent are available.

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
