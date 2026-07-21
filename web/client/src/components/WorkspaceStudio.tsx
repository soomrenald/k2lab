import { useEffect, useMemo, useRef, useState } from "react";
import type { DatacenterOption, FileRecord, GenerationJob, JobKind, NetworkVolumeOption, UnifiedPromptPreview, WorkspaceMigrationRecord, WorkspaceRecord } from "../api";
import { controlPlane } from "../api";
import { Icon, type IconName } from "./Icon";
import { Inspector } from "./Inspector";
import { AssetPanel } from "./AssetPanel";
import { TransferPanel } from "./TransferPanel";
import {
  buildProjectDocument,
  createStudioLora,
  createStudioSettings,
  type StudioLora,
} from "../studioProject";
import {
  RegionCanvas,
  type RegionBox,
  type RegionLayer,
  type StudioMode,
} from "./RegionCanvas";

interface Props {
  workspace: WorkspaceRecord;
  developmentBackend: boolean;
  datacenters: DatacenterOption[];
  networkVolumes: NetworkVolumeOption[];
  onWorkspace: (workspace: WorkspaceRecord) => void;
  onDelete: () => void;
}

const starterRegions: RegionBox[] = [
  { id: "region-a", name: "Primary subject", layer: "generation", x: 165, y: 180, width: 310, height: 620, prompt: "", faceIdentityPrompt: "", spatialRole: "auto", enabled: true },
  { id: "region-b", name: "Secondary subject", layer: "generation", x: 565, y: 220, width: 270, height: 560, prompt: "", faceIdentityPrompt: "", spatialRole: "auto", enabled: true },
];

export function WorkspaceStudio({ workspace, developmentBackend, datacenters, networkVolumes, onWorkspace, onDelete }: Props) {
  const [mode, setMode] = useState<StudioMode>("generation");
  const [activeLayer, setActiveLayer] = useState<RegionLayer>("generation");
  const [regions, setRegions] = useState<RegionBox[]>(starterRegions);
  const [selectedId, setSelectedId] = useState<string | null>("region-a");
  const [drawMode, setDrawMode] = useState(false);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [sourceName, setSourceName] = useState("");
  const [cloudSource, setCloudSource] = useState<FileRecord | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [comparePosition, setComparePosition] = useState(0.5);
  const [globalPrompts, setGlobalPrompts] = useState<Record<RegionLayer, string>>({
    generation: "",
    reference: "",
    targets: "",
  });
  const [studioSettings, setStudioSettings] = useState(createStudioSettings);
  const [loras, setLoras] = useState<StudioLora[]>([]);
  const [assetPurpose, setAssetPurpose] = useState<"source" | "lora" | "upscale">("source");
  const [showCloud, setShowCloud] = useState(false);
  const [showDelete, setShowDelete] = useState(false);
  const [showAssets, setShowAssets] = useState(false);
  const [showTransfers, setShowTransfers] = useState(false);
  const [showMigration, setShowMigration] = useState(false);
  const [migration, setMigration] = useState<WorkspaceMigrationRecord | null>(null);
  const [migrationConfirmation, setMigrationConfirmation] = useState("");
  const [migrationVolumeId, setMigrationVolumeId] = useState("");
  const [migrationDatacenterId, setMigrationDatacenterId] = useState(datacenters[0]?.id ?? "");
  const [migrationDiskGb, setMigrationDiskGb] = useState(workspace.workspace_disk_gb);
  const [deleteConfirmation, setDeleteConfirmation] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [job, setJob] = useState<GenerationJob | null>(null);
  const [queuedJobs, setQueuedJobs] = useState<GenerationJob[]>([]);
  const [promptPreview, setPromptPreview] = useState<UnifiedPromptPreview | null>(null);
  const eventCursor = useRef<string | undefined>(undefined);

  useEffect(() => () => { if (sourceUrl) URL.revokeObjectURL(sourceUrl); }, [sourceUrl]);

  useEffect(() => {
    if (developmentBackend || workspace.state === "deleted") return undefined;
    let cancelled = false;
    const interval = window.setInterval(async () => {
      try {
        const refreshed = await controlPlane.workspace(workspace.id);
        if (!cancelled) onWorkspace(refreshed);
      } catch (caught) {
        if (!cancelled) {
          setMessage(caught instanceof Error ? caught.message : "Could not refresh workspace status");
        }
      }
    }, 5_000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [developmentBackend, onWorkspace, workspace.id, workspace.state]);

  useEffect(() => {
    if (!job || ["completed", "failed", "cancelled"].includes(job.state)) return undefined;
    const interval = window.setInterval(async () => {
      try {
        const [next, events] = await Promise.all([
          controlPlane.job(workspace.id, job.id),
          controlPlane.jobEvents(workspace.id, job.id, eventCursor.current),
        ]);
        eventCursor.current = events.next_cursor;
        if (events.items.length) setMessage(events.items[events.items.length - 1].message);
        setJob(next);
        if (next.state === "completed" && next.output_file_ids[0]) {
          setResultUrl(controlPlane.outputUrl(workspace.id, next.output_file_ids[0]));
          setMessage(queuedJobs.length ? `Batch image complete. ${queuedJobs.length} queued run(s) remain.` : "Remote job complete. The verified output is stored in cloud files.");
        } else if (next.error_message) {
          setMessage(next.error_message);
        }
        if (["completed", "failed", "cancelled"].includes(next.state) && queuedJobs.length) {
          const [following, ...remaining] = queuedJobs;
          eventCursor.current = undefined;
          setQueuedJobs(remaining);
          setJob(following);
        }
      } catch (caught) {
        setMessage(caught instanceof Error ? caught.message : "Could not refresh remote job");
      }
    }, 1000);
    return () => window.clearInterval(interval);
  }, [job, queuedJobs, workspace.id]);

  useEffect(() => {
    if (!showCloud && !showMigration) return undefined;
    let cancelled = false;
    void controlPlane.migrations(workspace.id).then((items) => {
      if (!cancelled && items.length) setMigration(items[items.length - 1]);
    }).catch(() => undefined);
    return () => { cancelled = true; };
  }, [showCloud, showMigration, workspace.id]);

  const running = workspace.state === "ready";
  const activeCompute = ["provisioning", "starting", "ready", "stopping"].includes(workspace.state);
  const canExtend = workspace.state === "starting" || workspace.state === "ready";
  const canStart = workspace.state === "stopped" || workspace.state === "error";
  const leaseMinutes = Math.max(0, Math.round((new Date(workspace.lease_expires_at).getTime() - Date.now()) / 60_000));
  const readiness = useMemo(() => Object.entries(workspace.readiness), [workspace.readiness]);

  function switchMode(next: StudioMode) {
    setMode(next);
    setDrawMode(false);
    setSelectedId(null);
    setActiveLayer(next === "edit" ? "targets" : "generation");
  }

  function loadImage(file: File) {
    if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    setSourceUrl(URL.createObjectURL(file));
    setSourceName(file.name);
    setCloudSource(null);
    if (mode === "edit") {
      setRegions((items) => items.filter((item) => item.layer === "generation"));
      setActiveLayer("targets");
    }
  }

  async function lifecycle(action: "start" | "stop" | "extend") {
    setBusy(true);
    setMessage("");
    try {
      const next = action === "start"
        ? await controlPlane.startWorkspace(workspace.id)
        : action === "stop"
          ? await controlPlane.stopWorkspace(workspace.id)
          : await controlPlane.extendLease(workspace.id);
      onWorkspace(next);
    } catch (caught) {
      setMessage(caught instanceof Error ? caught.message : "Workspace action failed");
    } finally {
      setBusy(false);
    }
  }

  async function terminate() {
    setBusy(true);
    setMessage("");
    try {
      await controlPlane.terminateWorkspace(workspace.id, deleteConfirmation);
      onDelete();
    } catch (caught) {
      setMessage(caught instanceof Error ? caught.message : "Workspace deletion failed");
    } finally {
      setBusy(false);
    }
  }

  async function advanceMigration(initial: WorkspaceMigrationRecord) {
    let next = initial;
    while (["preparing", "copying", "verifying"].includes(next.state)) {
      next = await controlPlane.resumeMigration(workspace.id, next.id);
      setMigration(next);
    }
    if (next.state === "awaiting_confirmation") {
      onWorkspace(await controlPlane.workspace(workspace.id));
      setMessage("Manifest verification succeeded. Test the portable workspace, then explicitly delete the retained original Pod.");
    } else if (next.error_message) {
      setMessage(next.error_message);
    }
  }

  async function beginMigration() {
    setBusy(true);
    setMessage("");
    try {
      const created = await controlPlane.createMigration(workspace.id, {
        network_volume_id: migrationVolumeId || null,
        workspace_disk_gb: migrationVolumeId
          ? networkVolumes.find((item) => item.id === migrationVolumeId)?.size_gb
          : migrationDiskGb,
        datacenter_priority_ids: migrationVolumeId || !migrationDatacenterId
          ? [] : [migrationDatacenterId],
      });
      setMigration(created);
      await advanceMigration(created);
    } catch (caught) {
      setMessage(caught instanceof Error ? caught.message : "Workspace migration failed");
    } finally {
      setBusy(false);
    }
  }

  async function resumeMigration() {
    if (!migration) return;
    setBusy(true);
    try {
      await advanceMigration(migration);
    } catch (caught) {
      setMessage(caught instanceof Error ? caught.message : "Could not resume migration");
    } finally {
      setBusy(false);
    }
  }

  async function confirmMigration() {
    if (!migration) return;
    setBusy(true);
    try {
      const completed = await controlPlane.confirmMigration(
        workspace.id, migration.id, migrationConfirmation,
      );
      setMigration(completed);
      setMigrationConfirmation("");
      setShowMigration(false);
      onWorkspace(await controlPlane.workspace(workspace.id));
      setMessage("Migration complete. The original Pod and its regular volume were deleted.");
    } catch (caught) {
      setMessage(caught instanceof Error ? caught.message : "Could not confirm migration");
    } finally {
      setBusy(false);
    }
  }

  async function runRemoteJob() {
    if (mode !== "generation" && !cloudSource) {
      setMessage("Choose an uploaded input or prior output from Cloud files first.");
      setShowAssets(true);
      return;
    }
    setBusy(true);
    setMessage("");
    setResultUrl(null);
    eventCursor.current = undefined;
    try {
      await controlPlane.previewUnifiedPrompt(
        buildProjectDocument(regions, globalPrompts, studioSettings, loras),
      );
      const kind: JobKind = mode === "generation" ? "generate" : mode === "edit" ? "edit_image" : "refine_faces";
      const runCount = mode === "generation" && studioSettings.generation.batchMode
        ? studioSettings.generation.batchCount : 1;
      const submitted: GenerationJob[] = [];
      let lastSeed = studioSettings.generation.seed;
      for (let index = 0; index < runCount; index += 1) {
        let seed = studioSettings.generation.seed;
        if (mode === "generation" && studioSettings.generation.seedMode === "random") {
          seed = crypto.getRandomValues(new Uint32Array(1))[0] & 0x7fffffff;
        } else if (mode === "generation" && studioSettings.generation.seedMode === "increment") {
          seed = (studioSettings.generation.seed + index) % 2147483648;
        }
        lastSeed = seed;
        const jobSettings = mode === "generation"
          ? { ...studioSettings, generation: { ...studioSettings.generation, seed } }
          : studioSettings;
        submitted.push(await controlPlane.submitJob(workspace.id, {
          command_id: crypto.randomUUID(),
          kind,
          project_id: `studio-${workspace.id}`,
          project: buildProjectDocument(regions, globalPrompts, jobSettings, loras),
          input_file_id: cloudSource?.id,
          lora_file_ids: loras.map((lora) => lora.fileId),
          upscale_model_file_id: studioSettings.generation.upscaleModelFileId || undefined,
        }));
      }
      if (mode === "generation") {
        const nextSeed = studioSettings.generation.seedMode === "increment"
          ? (studioSettings.generation.seed + runCount) % 2147483648 : lastSeed;
        setStudioSettings({ ...studioSettings, generation: { ...studioSettings.generation, seed: nextSeed } });
      }
      setJob(submitted[0]);
      setQueuedJobs(submitted.slice(1));
      setMessage(runCount > 1 ? `${runCount} remote batch runs queued.` : "Remote job queued.");
    } catch (caught) {
      setMessage(caught instanceof Error ? caught.message : "Could not submit remote job");
    } finally {
      setBusy(false);
    }
  }

  async function cancelRemoteJob() {
    if (!job) return;
    setBusy(true);
    try {
      const [cancelled] = await Promise.all([
        controlPlane.cancelJob(workspace.id, job.id),
        ...queuedJobs.map((queued) => controlPlane.cancelJob(workspace.id, queued.id)),
      ]);
      setJob(cancelled);
      setQueuedJobs([]);
      setMessage("Remote job queue cancelled; worker memory was released.");
    } catch (caught) {
      setMessage(caught instanceof Error ? caught.message : "Could not cancel remote job");
    } finally {
      setBusy(false);
    }
  }

  async function previewUnifiedPrompt() {
    setBusy(true);
    setMessage("");
    try {
      setPromptPreview(await controlPlane.previewUnifiedPrompt(
        buildProjectDocument(regions, globalPrompts, studioSettings, loras),
      ));
    } catch (caught) {
      setMessage(caught instanceof Error ? caught.message : "Could not compile the unified prompt");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="studio-shell">
      <header className="studio-topbar">
        <div className="brand-lockup"><span className="brand-mark">K2</span><span><strong>Region Lab</strong><small>Cloud studio</small></span></div>
        <div className="project-actions">
          <button>New</button><button>Open</button><button>Import PNG</button><button>Save</button>
        </div>
        <div className="workspace-status">
          {developmentBackend && <span className="preview-chip">Preview backend</span>}
          <button className="workspace-chip" onClick={() => setShowCloud(!showCloud)}>
            <span className={`status-dot ${activeCompute ? "online" : "stopped"}`} />
            <span><strong>{workspace.name}</strong><small>{workspace.gpu.display_name} · {workspace.state}</small></span>
            <Icon name="chevronDown" />
          </button>
          {activeCompute && <button className="stop-gpu" disabled={busy || workspace.state === "stopping"} onClick={() => lifecycle("stop")}><Icon name="stop" /> Stop GPU now</button>}
        </div>
      </header>

      {showCloud && (
        <div className="cloud-popover glass-card">
          <div className="cloud-popover-head"><div><p className="kicker">Cloud workspace</p><h3>{workspace.name}</h3></div><span className={`state-badge ${workspace.state}`}>{workspace.state}</span></div>
          <dl className="summary-list compact-summary">
            <div><dt>Compute now</dt><dd>{activeCompute ? `$${workspace.estimated_compute_per_hour.toFixed(2)}/hr` : "$0.00/hr"}</dd></div>
            <div><dt>Storage</dt><dd>${workspace.estimated_storage_per_month.toFixed(2)}/mo</dd></div>
            <div><dt>Lease</dt><dd>{activeCompute ? `${leaseMinutes} min remaining` : "No active lease"}</dd></div>
          </dl>
          <div className="readiness-grid">{readiness.map(([name, ready]) => <span key={name} className={ready ? "ready" : "pending"}><Icon name={ready ? "check" : "clock"} /> {name}</span>)}</div>
          {workspace.error_message && <div className="error-banner">{workspace.error_message}</div>}
          <div className="popover-actions">
            {canExtend
              ? <button className="quiet-button" onClick={() => lifecycle("extend")}>Extend session</button>
              : canStart
                ? <button className="primary-button" onClick={() => lifecycle("start")}>Start GPU</button>
                : null}
            {workspace.mode === "persistent_pod" && (
              <button className="quiet-button" onClick={() => setShowMigration(true)}>Migrate to portable storage</button>
            )}
            {workspace.retained_original_provider_resource_id && (
              <button className="quiet-button" onClick={() => setShowMigration(true)}>Confirm verified migration</button>
            )}
            <button className="danger-text-button" disabled={Boolean(workspace.retained_original_provider_resource_id)} title={workspace.retained_original_provider_resource_id ? "Confirm the verified migration first" : undefined} onClick={() => setShowDelete(true)}>Delete cloud workspace</button>
          </div>
          <p className="field-help">{workspace.mode === "portable_workspace"
            ? "Stopping terminates the Pod and retains the network volume. Deleting this workspace also retains that volume for safety."
            : "Stopping retains the attached volume. Deleting permanently removes it."}</p>
        </div>
      )}

      <aside className="studio-rail">
        <div className="mode-rail">
          <RailButton icon="spark" label="Generate" active={mode === "generation"} onClick={() => switchMode("generation")} />
          <RailButton icon="edit" label="Edit" active={mode === "edit"} onClick={() => switchMode("edit")} />
          <RailButton icon="face" label="Faces" active={mode === "face"} onClick={() => switchMode("face")} />
        </div>
        <div className="utility-rail">
          <RailButton icon="folder" label="Assets" active={showAssets} onClick={() => { setAssetPurpose("source"); setShowAssets(true); }} />
          <RailButton icon="transfer" label="Transfers" active={showTransfers} onClick={() => setShowTransfers(true)} />
          <RailButton icon="settings" label="Setup" active={false} onClick={() => setShowCloud(true)} />
        </div>
      </aside>

      <main className="studio-main">
        <div className="mode-context-bar">
          <div><p className="kicker">Workspace</p><h1>{mode === "generation" ? "Image generation" : mode === "edit" ? "Image editing" : "Face refinement"}</h1></div>
          {mode === "edit" && (
            <div className="layer-switcher">
              <button className={activeLayer === "reference" ? "active" : ""} onClick={() => { setActiveLayer("reference"); setSelectedId(null); }}><Icon name="layers" /> Reference layer</button>
              <button className={activeLayer === "targets" ? "active" : ""} onClick={() => { setActiveLayer("targets"); setSelectedId(null); }}><Icon name="edit" /> Edit targets</button>
            </div>
          )}
        </div>
        <div className="workspace-grid">
          <RegionCanvas
            mode={mode}
            activeLayer={activeLayer}
            sourceUrl={sourceUrl}
            sourceName={sourceName}
            resultUrl={resultUrl}
            regions={regions}
            selectedId={selectedId}
            drawMode={drawMode}
            comparePosition={comparePosition}
            canvasWidth={studioSettings.generation.width}
            canvasHeight={studioSettings.generation.height}
            onComparePosition={setComparePosition}
            onSelect={setSelectedId}
            onRegions={setRegions}
            onDrawMode={setDrawMode}
            onLoadImage={loadImage}
          />
          <Inspector
            mode={mode}
            activeLayer={activeLayer}
            regions={regions}
            selectedId={selectedId}
            globalPrompt={globalPrompts[activeLayer]}
            settings={studioSettings}
            loras={loras}
            onGlobalPrompt={(value) => setGlobalPrompts({ ...globalPrompts, [activeLayer]: value })}
            onSettings={setStudioSettings}
            onLoras={setLoras}
            onChooseLora={() => { setAssetPurpose("lora"); setShowAssets(true); }}
            onChooseUpscaleModel={() => { setAssetPurpose("upscale"); setShowAssets(true); }}
            onPreviewUnifiedPrompt={() => void previewUnifiedPrompt()}
            onRegions={setRegions}
            onSelect={setSelectedId}
          />
        </div>
      </main>

      <footer className="action-bar">
        <div className="action-status"><span className={`status-dot ${activeCompute ? "online" : "stopped"}`} /><span><strong>{job && !["completed", "failed", "cancelled"].includes(job.state) ? `Remote job ${job.state}` : running ? "Workspace ready" : activeCompute ? `Workspace ${workspace.state}` : "GPU stopped"}</strong><small>{message || workspace.error_message || (developmentBackend ? "Interface preview · remote jobs are disabled" : cloudSource ? `Cloud source: ${cloudSource.display_name}` : "Ready")}</small></span></div>
        <div className="memory-meter"><span>Job</span><div><i style={{ width: job?.progress_total ? `${Math.min(100, job.progress_current / job.progress_total * 100)}%` : "0%" }} /></div><small>{job?.progress_total ? `${job.progress_current}/${job.progress_total}` : running ? "Idle" : "Released"}</small></div>
        <button className="run-button" disabled={!running || developmentBackend || busy} title={developmentBackend ? "Remote generation jobs are disabled in preview mode" : undefined} onClick={() => void (job && !["completed", "failed", "cancelled"].includes(job.state) ? cancelRemoteJob() : runRemoteJob())}>
          <Icon name={job && !["completed", "failed", "cancelled"].includes(job.state) ? "stop" : mode === "face" ? "face" : mode === "edit" ? "wand" : "play"} />
          {job && !["completed", "failed", "cancelled"].includes(job.state) ? "Cancel remote job" : mode === "generation" ? "Generate image" : mode === "edit" ? "Run image edit" : "Refine faces"}
        </button>
      </footer>

      {showDelete && (
        <div className="modal-backdrop" role="presentation">
          <section className="confirm-modal" role="dialog" aria-modal="true" aria-labelledby="delete-title">
            <div className="danger-icon"><Icon name="trash" /></div>
            <p className="kicker">Permanent action</p>
            <h2 id="delete-title">Delete cloud workspace?</h2>
            <p>{workspace.mode === "portable_workspace"
              ? "This removes the workspace and any active ephemeral Pod. The network volume is retained to prevent accidental data loss and continues to incur storage cost."
              : "This removes the Pod and its regular persistent volume. Models, projects, inputs, and outputs on that volume cannot be recovered."}</p>
            <label className="field-label" htmlFor="delete-confirmation">Type <strong>{workspace.name}</strong> to confirm</label>
            <input id="delete-confirmation" className="text-input" value={deleteConfirmation} onChange={(event) => setDeleteConfirmation(event.target.value)} />
            {message && <div className="error-banner">{message}</div>}
            <div className="modal-actions"><button className="quiet-button" onClick={() => { setShowDelete(false); setDeleteConfirmation(""); }}>Cancel</button><button className="danger-button" disabled={busy || deleteConfirmation !== workspace.name} onClick={terminate}>{workspace.mode === "portable_workspace" ? "Delete workspace; retain volume" : "Delete workspace and files"}</button></div>
          </section>
        </div>
      )}
      {promptPreview && (
        <div className="modal-backdrop" role="presentation">
          <section className="confirm-modal prompt-preview-modal" role="dialog" aria-modal="true" aria-labelledby="prompt-preview-title">
            <p className="kicker">Legacy compiler output</p>
            <h2 id="prompt-preview-title">Unified spatial prompt</h2>
            <p>{promptPreview.regions.length} regional clause{promptPreview.regions.length === 1 ? "" : "s"} in front-to-back order. Pixel boxes are applied separately as a hidden soft attention grid.</p>
            <textarea className="prompt-area prompt-preview-text" readOnly value={promptPreview.prompt} />
            <div className="preview-region-order">
              {promptPreview.regions.map((region, index) => <div key={region.id}><strong>{index + 1}. {region.name}</strong><span>{region.spatial_role}</span></div>)}
            </div>
            <div className="modal-actions"><button className="primary-button" onClick={() => setPromptPreview(null)}>Close</button></div>
          </section>
        </div>
      )}
      {showMigration && (
        <div className="modal-backdrop" role="presentation">
          <section className="confirm-modal" role="dialog" aria-modal="true" aria-labelledby="migration-title">
            <div className="danger-icon"><Icon name="transfer" /></div>
            <p className="kicker">Verified storage migration</p>
            <h2 id="migration-title">{migration?.state === "awaiting_confirmation" ? "Confirm the portable copy" : "Migrate to a network volume?"}</h2>
            {migration?.state === "awaiting_confirmation" ? (
              <>
                <p>The source and target SHA-256 manifests match. The original Pod is stopped and retained so you can test the portable workspace. Confirming permanently deletes its regular volume.</p>
                <label className="field-label" htmlFor="migration-confirmation">Type <strong>{workspace.name}</strong> to delete the original Pod</label>
                <input id="migration-confirmation" className="text-input" value={migrationConfirmation} onChange={(event) => setMigrationConfirmation(event.target.value)} />
              </>
            ) : (
              <>
                <p>Generation and transfers will stop while durable models, projects, inputs, outputs, and job state are copied. Switchover occurs only after file inventory and SHA-256 manifests match. The original Pod remains stopped until a separate confirmation.</p>
                {!migration && (
                  <div className="two-fields">
                    <label className="number-field">
                      <span>Target network volume</span>
                      <select className="text-input" value={migrationVolumeId} onChange={(event) => {
                        const volume = networkVolumes.find((item) => item.id === event.target.value);
                        setMigrationVolumeId(event.target.value);
                        if (volume) {
                          setMigrationDatacenterId(volume.datacenter_id);
                          setMigrationDiskGb(volume.size_gb);
                        }
                      }}>
                        <option value="">Create a new network volume</option>
                        {networkVolumes.map((volume) => <option value={volume.id} key={volume.id}>{volume.name} · {volume.size_gb} GB · {volume.datacenter_id}</option>)}
                      </select>
                    </label>
                    <label className="number-field">
                      <span>Datacenter</span>
                      <select className="text-input" disabled={Boolean(migrationVolumeId)} value={migrationDatacenterId} onChange={(event) => setMigrationDatacenterId(event.target.value)}>
                        {datacenters.map((datacenter) => <option value={datacenter.id} key={datacenter.id}>{datacenter.name} · {datacenter.location}</option>)}
                      </select>
                    </label>
                    {!migrationVolumeId && (
                      <label className="number-field">
                        <span>Target capacity</span>
                        <span className="number-input-wrap"><input type="number" min={50} max={4000} value={migrationDiskGb} onChange={(event) => setMigrationDiskGb(Math.max(50, Math.min(4000, Number(event.target.value))))} /><small>GB</small></span>
                      </label>
                    )}
                  </div>
                )}
              </>
            )}
            {migration && migration.bytes_total > 0 && (
              <p className="field-help">Copied {(migration.bytes_copied / 1_048_576).toFixed(1)} of {(migration.bytes_total / 1_048_576).toFixed(1)} MiB · {migration.state}</p>
            )}
            {message && <div className="error-banner">{message}</div>}
            <div className="modal-actions">
              <button className="quiet-button" onClick={() => setShowMigration(false)}>Close</button>
              {migration?.state === "awaiting_confirmation" ? (
                <button className="danger-button" disabled={busy || migrationConfirmation !== workspace.name} onClick={confirmMigration}>Delete original Pod and volume</button>
              ) : migration && ["preparing", "copying", "verifying"].includes(migration.state) ? (
                <button className="primary-button" disabled={busy} onClick={resumeMigration}>{busy ? "Migrating…" : "Resume verified copy"}</button>
              ) : (
                <button className="primary-button" disabled={busy || workspace.mode !== "persistent_pod"} onClick={beginMigration}>{busy ? "Preparing migration…" : "Create volume and begin copy"}</button>
              )}
            </div>
          </section>
        </div>
      )}
      {showAssets && <AssetPanel workspaceId={workspace.id} initialKind={assetPurpose === "lora" ? "loras" : assetPurpose === "upscale" ? "upscale_models" : "inputs"} onClose={() => setShowAssets(false)} onSelect={(file) => {
        if (assetPurpose === "lora") {
          if (file.kind === "loras" && !loras.some((lora) => lora.fileId === file.id)) setLoras([...loras, createStudioLora(file.id, file.display_name)]);
          return;
        }
        if (assetPurpose === "upscale") {
          if (file.kind === "upscale_models") setStudioSettings({ ...studioSettings, generation: { ...studioSettings.generation, upscaleModelFileId: file.id, upscaleModelName: file.display_name } });
          return;
        }
        if (file.kind !== "inputs" && file.kind !== "outputs") return;
        setCloudSource(file);
        setSourceName(file.display_name);
        if (file.kind === "outputs") setSourceUrl(controlPlane.outputUrl(workspace.id, file.id));
      }} />}
      {showTransfers && <TransferPanel workspaceId={workspace.id} onClose={() => setShowTransfers(false)} />}
    </div>
  );
}

function RailButton({ icon, label, active, onClick }: { icon: IconName; label: string; active: boolean; onClick: () => void }) {
  return <button className={`rail-button ${active ? "active" : ""}`} onClick={onClick}><Icon name={icon} /><span>{label}</span></button>;
}
