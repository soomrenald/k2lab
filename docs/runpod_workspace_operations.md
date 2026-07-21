# RunPod workspace operations

This runbook covers the production lifecycle implemented from
`runpod_web_workspace_spec.md`. The development backend simulates these operations and never
contacts RunPod.

## Deployment prerequisites

- Use PostgreSQL and a persisted Fernet root delivered by the deployment secret manager.
- Put the control plane behind TLS and the trusted authentication proxy described in
  `web/README.md`; do not expose Uvicorn directly.
- Use a restricted, user-owned RunPod API key with only the Pod, volume, and inventory
  permissions needed by this application.
- Run the opt-in disposable-account acceptance test before enabling production provisioning.

## Workspace lifecycle

Persistent-Pod **Stop** calls the provider stop operation and retains the regular Pod volume.
**Delete cloud workspace** terminates the Pod and permanently deletes that regular volume.

Portable-workspace **Stop** terminates the ephemeral Pod and revokes its session token. The
network volume stays in its datacenter and remains billable. **Start** selects compatible
compute in that datacenter, creates a new Pod, attaches the same volume, and rotates the agent
token. Deleting the application workspace still retains the network volume. Delete a network
volume separately in RunPod only after inspecting it and confirming a backup.

## Persistent-to-portable migration

1. Start the persistent workspace and wait until every readiness stage required for file
   access is healthy.
2. Choose an existing compatible network volume or a new volume size/datacenter.
3. Start migration. The source agent seals writes, stops generation/download work, creates a
   generation-numbered allowlisted SHA-256 manifest, and the control plane creates the target
   volume/Pod.
4. The browser may be closed. Reopen the workspace and choose **Resume verified copy**; each
   accepted chunk and file offset is durable and retries are idempotent.
5. Switchover occurs only after source and target layout version, file inventory, byte total,
   per-file SHA-256 values, and root SHA-256 all match.
6. The original Pod is stopped but retained. Test the portable workspace before typing the
   workspace name to delete the original Pod and regular volume.

During steps 3–5, both Pods and both storage resources may be billable. After verification,
the target Pod plus both storage resources are billable until original deletion is confirmed.
The cost endpoint includes these overlapping resources.

## Recovery

- Agent or browser disconnect: retry **Resume verified copy**. No already verified chunk is
  recopied.
- Explicit stop or lease expiry: migration is marked failed, the target Pod is terminated,
  the source is unsealed/stopped, and the network volume is retained.
- Manifest mismatch: the target Pod is terminated, the source stays authoritative and is
  unsealed, and the target volume is retained for inspection.
- Target allocation failure: the source is unsealed. A newly created empty volume is retained
  so the user can re-plan within its datacenter.
- Control-plane restart: migration, provider IDs, manifests, offsets, audit events, and the
  operation journal are read from the durable database. Resume from the UI.

Never manually delete the original Pod before verified switchover, and never delete a retained
network volume merely because a migration record failed.
