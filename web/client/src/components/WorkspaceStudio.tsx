import { useEffect, useMemo, useRef, useState } from "react";
import type { FileRecord, GenerationJob, JobKind, WorkspaceRecord } from "../api";
import { controlPlane } from "../api";
import { Icon, type IconName } from "./Icon";
import { Inspector } from "./Inspector";
import { AssetPanel } from "./AssetPanel";
import { TransferPanel } from "./TransferPanel";
import {
  RegionCanvas,
  type RegionBox,
  type RegionLayer,
  type StudioMode,
} from "./RegionCanvas";

interface Props {
  workspace: WorkspaceRecord;
  developmentBackend: boolean;
  onWorkspace: (workspace: WorkspaceRecord) => void;
  onDelete: () => void;
}

const starterRegions: RegionBox[] = [
  { id: "region-a", name: "Primary subject", layer: "generation", x: 165, y: 180, width: 310, height: 620, prompt: "", enabled: true },
  { id: "region-b", name: "Secondary subject", layer: "generation", x: 565, y: 220, width: 270, height: 560, prompt: "", enabled: true },
];

export function WorkspaceStudio({ workspace, developmentBackend, onWorkspace, onDelete }: Props) {
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
  const [showCloud, setShowCloud] = useState(false);
  const [showDelete, setShowDelete] = useState(false);
  const [showAssets, setShowAssets] = useState(false);
  const [showTransfers, setShowTransfers] = useState(false);
  const [deleteConfirmation, setDeleteConfirmation] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [job, setJob] = useState<GenerationJob | null>(null);
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
          setMessage("Remote job complete. The verified output is stored in cloud files.");
        } else if (next.error_message) {
          setMessage(next.error_message);
        }
      } catch (caught) {
        setMessage(caught instanceof Error ? caught.message : "Could not refresh remote job");
      }
    }, 1000);
    return () => window.clearInterval(interval);
  }, [job, workspace.id]);

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
      const kind: JobKind = mode === "generation" ? "generate" : mode === "edit" ? "edit_image" : "refine_faces";
      const next = await controlPlane.submitJob(workspace.id, {
        command_id: crypto.randomUUID(),
        kind,
        project_id: `studio-${workspace.id}`,
        project: buildProjectDocument(regions, globalPrompts),
        input_file_id: cloudSource?.id,
        lora_file_ids: [],
      });
      setJob(next);
      setMessage("Remote job queued.");
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
      setJob(await controlPlane.cancelJob(workspace.id, job.id));
      setMessage("Remote job cancelled; worker memory was released.");
    } catch (caught) {
      setMessage(caught instanceof Error ? caught.message : "Could not cancel remote job");
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
            <button className="danger-text-button" onClick={() => setShowDelete(true)}>Delete cloud workspace</button>
          </div>
          <p className="field-help">Stopping retains the attached volume. Deleting permanently removes it.</p>
        </div>
      )}

      <aside className="studio-rail">
        <div className="mode-rail">
          <RailButton icon="spark" label="Generate" active={mode === "generation"} onClick={() => switchMode("generation")} />
          <RailButton icon="edit" label="Edit" active={mode === "edit"} onClick={() => switchMode("edit")} />
          <RailButton icon="face" label="Faces" active={mode === "face"} onClick={() => switchMode("face")} />
        </div>
        <div className="utility-rail">
          <RailButton icon="folder" label="Assets" active={showAssets} onClick={() => setShowAssets(true)} />
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
            onGlobalPrompt={(value) => setGlobalPrompts({ ...globalPrompts, [activeLayer]: value })}
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
            <p>This removes the Pod and its regular persistent volume. Models, projects, inputs, and outputs on that volume cannot be recovered.</p>
            <label className="field-label" htmlFor="delete-confirmation">Type <strong>{workspace.name}</strong> to confirm</label>
            <input id="delete-confirmation" className="text-input" value={deleteConfirmation} onChange={(event) => setDeleteConfirmation(event.target.value)} />
            {message && <div className="error-banner">{message}</div>}
            <div className="modal-actions"><button className="quiet-button" onClick={() => { setShowDelete(false); setDeleteConfirmation(""); }}>Cancel</button><button className="danger-button" disabled={busy || deleteConfirmation !== workspace.name} onClick={terminate}>Delete workspace and files</button></div>
          </section>
        </div>
      )}
      {showAssets && <AssetPanel workspaceId={workspace.id} onClose={() => setShowAssets(false)} onSelect={(file) => { setCloudSource(file); setSourceName(file.display_name); if (file.kind === "outputs") setSourceUrl(controlPlane.outputUrl(workspace.id, file.id)); }} />}
      {showTransfers && <TransferPanel workspaceId={workspace.id} onClose={() => setShowTransfers(false)} />}
    </div>
  );
}

function RailButton({ icon, label, active, onClick }: { icon: IconName; label: string; active: boolean; onClick: () => void }) {
  return <button className={`rail-button ${active ? "active" : ""}`} onClick={onClick}><Icon name={icon} /><span>{label}</span></button>;
}

function buildProjectDocument(regions: RegionBox[], prompts: Record<RegionLayer, string>): Record<string, unknown> {
  return {
    schema: "k2-region-lab-project",
    version: 18,
    canvas: { width: 1024, height: 1024 },
    generation: { global_prompt: prompts.generation, steps: 8, sampler: "euler", scheduler: "simple", seed: 0 },
    regions: regions.filter((region) => region.layer === "generation").map(regionDocument),
    loras: [],
    image_edit: {
      width: 1024,
      height: 1024,
      global_prompt: prompts.targets,
      reference_global_prompt: prompts.reference,
      regions: regions.filter((region) => region.layer === "targets").map(regionDocument),
      reference_regions: regions.filter((region) => region.layer === "reference").map(regionDocument),
    },
    runtime: {},
  };
}

function regionDocument(region: RegionBox) {
  return {
    id: region.id,
    name: region.name,
    box: { x0: region.x, y0: region.y, x1: region.x + region.width, y1: region.y + region.height },
    prompt: region.prompt,
    enabled: region.enabled,
    priority: 0,
    spatial_role: "auto",
  };
}
