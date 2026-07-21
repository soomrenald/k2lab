import { useEffect, useMemo, useState } from "react";
import type { WorkspaceRecord } from "../api";
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
            resultUrl={null}
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
        <div className="action-status"><span className={`status-dot ${activeCompute ? "online" : "stopped"}`} /><span><strong>{running ? "Workspace ready" : activeCompute ? `Workspace ${workspace.state}` : "GPU stopped"}</strong><small>{message || workspace.error_message || (developmentBackend ? "Interface preview · remote jobs are not connected" : "Ready")}</small></span></div>
        <div className="memory-meter"><span>VRAM</span><div><i style={{ width: running ? "18%" : "0%" }} /></div><small>{running ? "Waiting for worker" : "Released"}</small></div>
        <button className="run-button" disabled={!running || developmentBackend} title={developmentBackend ? "Remote generation jobs are not connected yet" : undefined}>
          <Icon name={mode === "face" ? "face" : mode === "edit" ? "wand" : "play"} />
          {mode === "generation" ? "Generate image" : mode === "edit" ? "Run image edit" : "Refine selected faces"}
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
      {showAssets && <AssetPanel workspaceId={workspace.id} onClose={() => setShowAssets(false)} />}
      {showTransfers && <TransferPanel workspaceId={workspace.id} onClose={() => setShowTransfers(false)} />}
    </div>
  );
}

function RailButton({ icon, label, active, onClick }: { icon: IconName; label: string; active: boolean; onClick: () => void }) {
  return <button className={`rail-button ${active ? "active" : ""}`} onClick={onClick}><Icon name={icon} /><span>{label}</span></button>;
}
