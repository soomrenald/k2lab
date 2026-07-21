import { useMemo, useState } from "react";
import type {
  CapabilityManifest,
  CredentialStatus,
  DatacenterOption,
  GpuOption,
  NetworkVolumeOption,
  WorkspacePlan,
  WorkspacePlanRequest,
  WorkspaceRecord,
} from "../api";
import { controlPlane } from "../api";
import { Icon } from "./Icon";

interface Props {
  capabilities: CapabilityManifest;
  credential: CredentialStatus;
  gpus: GpuOption[];
  datacenters: DatacenterOption[];
  networkVolumes: NetworkVolumeOption[];
  onCredential: (
    credential: CredentialStatus,
    gpus: GpuOption[],
    datacenters: DatacenterOption[],
    networkVolumes: NetworkVolumeOption[],
  ) => void;
  onWorkspace: (workspace: WorkspaceRecord) => void;
}

const defaultRequest: WorkspacePlanRequest = {
  mode: "persistent_pod",
  gpu_priority_ids: [],
  cloud_type: "secure",
  interruptible: false,
  container_disk_gb: 50,
  workspace_disk_gb: 200,
  idle_timeout_seconds: 900,
  hard_deadline_seconds: 28_800,
  network_volume_id: null,
  datacenter_priority_ids: [],
};

export function CloudOnboarding({
  capabilities,
  credential,
  gpus,
  datacenters,
  networkVolumes,
  onCredential,
  onWorkspace,
}: Props) {
  const [apiKey, setApiKey] = useState("");
  const [request, setRequest] = useState<WorkspacePlanRequest>(() => ({
    ...defaultRequest,
    gpu_priority_ids: gpus.filter((gpu) => gpu.secure_available).slice(0, 2).map((gpu) => gpu.id),
  }));
  const [plan, setPlan] = useState<WorkspacePlan | null>(null);
  const [workspaceName, setWorkspaceName] = useState("My K2 workspace");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const selectedGpus = useMemo(
    () => request.gpu_priority_ids
      .map((id) => gpus.find((gpu) => gpu.id === id))
      .filter((gpu): gpu is GpuOption => Boolean(gpu)),
    [gpus, request.gpu_priority_ids],
  );
  const portable = request.mode === "portable_workspace";
  const selectedVolume = networkVolumes.find(
    (item) => item.id === request.network_volume_id,
  );
  const selectedDatacenterId =
    selectedVolume?.datacenter_id ?? request.datacenter_priority_ids[0];
  const selectedDatacenter = datacenters.find(
    (item) => item.id === selectedDatacenterId,
  );
  const portableGpuIds = new Set(
    selectedDatacenter?.gpu_availability
      .filter(
        (item) =>
          !["", "none", "unavailable"].includes(item.stock_status.toLowerCase()),
      )
      .map((item) => item.gpu_type_id) ?? [],
  );

  async function connect() {
    setBusy(true);
    setError("");
    try {
      const nextCredential = await controlPlane.connectRunPod(apiKey);
      const [nextGpus, nextDatacenters, nextVolumes] = await Promise.all([
        controlPlane.gpus(),
        controlPlane.datacenters(),
        controlPlane.networkVolumes(),
      ]);
      setRequest((current) => ({
        ...current,
        gpu_priority_ids: nextGpus
          .filter((gpu) => gpu.secure_available)
          .slice(0, 2)
          .map((gpu) => gpu.id),
      }));
      onCredential(nextCredential, nextGpus, nextDatacenters, nextVolumes);
      setApiKey("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not connect account");
    } finally {
      setBusy(false);
    }
  }

  function toggleGpu(gpu: GpuOption) {
    setPlan(null);
    setRequest((current) => {
      const selected = current.gpu_priority_ids.includes(gpu.id);
      return {
        ...current,
        gpu_priority_ids: selected
          ? current.gpu_priority_ids.filter((id) => id !== gpu.id)
          : [...current.gpu_priority_ids, gpu.id],
      };
    });
  }

  function moveGpu(index: number, offset: number) {
    const target = index + offset;
    if (target < 0 || target >= selectedGpus.length) return;
    setPlan(null);
    setRequest((current) => {
      const ids = [...current.gpu_priority_ids];
      [ids[index], ids[target]] = [ids[target], ids[index]];
      return { ...current, gpu_priority_ids: ids };
    });
  }

  async function reviewPlan() {
    setBusy(true);
    setError("");
    try {
      setPlan(await controlPlane.planWorkspace(request));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not plan workspace");
    } finally {
      setBusy(false);
    }
  }

  async function createWorkspace() {
    if (!plan) return;
    setBusy(true);
    setError("");
    try {
      onWorkspace(await controlPlane.createWorkspace(plan.id, workspaceName));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not create workspace");
    } finally {
      setBusy(false);
    }
  }

  if (!credential.configured) {
    return (
      <main className="onboarding-shell">
        <section className="onboarding-copy">
          <div className="eyebrow"><span className="pulse-dot" /> Cloud workspace</div>
          <h1>Your GPU studio,<br />ready when you are.</h1>
          <p>
            Connect your own RunPod account. K2 Region Lab will manage compute, storage,
            models, and idle shutdown without exposing your provider key to the browser.
          </p>
          <div className="feature-row">
            <Feature icon="gpu" title="Priority GPUs" detail="Choose fallbacks in order" />
            <Feature icon="clock" title="Idle protection" detail="Stops compute automatically" />
            <Feature icon="folder" title="Persistent files" detail="Models and outputs survive stops" />
          </div>
        </section>
        <section className="connect-card glass-card">
          <div className="step-index">01</div>
          <div>
            <p className="kicker">Connect provider</p>
            <h2>RunPod account</h2>
          </div>
          <label className="field-label" htmlFor="runpod-key">RunPod API key</label>
          <input
            id="runpod-key"
            className="text-input secret-input"
            type="password"
            autoComplete="off"
            placeholder="Paste a restricted API key"
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
          />
          <p className="field-help">
            Use Pod, inventory, and billing-read permissions only. The key is never written
            into a project or image.
          </p>
          {capabilities.development_backend && (
            <div className="dev-notice">
              Development backend — use any eight-character test value. No cloud resource
              will be created or billed.
            </div>
          )}
          {error && <div className="error-banner">{error}</div>}
          <button
            className="primary-button full-button"
            disabled={busy || apiKey.trim().length < 8}
            onClick={connect}
          >
            {busy ? "Validating…" : "Validate and continue"}
          </button>
          <div className="security-line">
            <span className="security-mark"><Icon name="check" /></span>
            TLS transport · encrypted vault in production · revocable anytime
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="provision-shell">
      <header className="provision-heading">
        <div>
          <p className="kicker">Cloud workspace storage</p>
          <h1>Configure your workspace</h1>
          <p>{portable
            ? "Each session gets a fresh Pod attached to one datacenter-bound network volume."
            : "GPU priority is used for initial provisioning. Storage remains attached to this Pod."}</p>
        </div>
        <span className="account-chip"><span className="status-dot online" /> RunPod {credential.key_hint}</span>
      </header>

      <div className="provision-grid">
        <section className="config-column">
          <div className="setup-section glass-card">
            <div className="section-title-row">
              <span className="section-number">0</span>
              <div><p className="kicker">Workspace type</p><h2>Choose storage behavior</h2></div>
            </div>
            <div className="segmented-control">
              <button
                className={!portable ? "active" : ""}
                onClick={() => {
                  setPlan(null);
                  setRequest({
                    ...request,
                    mode: "persistent_pod",
                    network_volume_id: null,
                    datacenter_priority_ids: [],
                  });
                }}
              >Persistent Pod</button>
              <button
                className={portable ? "active" : ""}
                onClick={() => {
                  setPlan(null);
                  setRequest({
                    ...request,
                    mode: "portable_workspace",
                    cloud_type: "secure",
                    datacenter_priority_ids: datacenters[0] ? [datacenters[0].id] : [],
                  });
                }}
              >Portable workspace</button>
            </div>
            <p className="field-help">Persistent Pods keep a regular volume tied to one Pod. Portable workspaces retain an independent network volume and reselect compatible compute each session.</p>
          </div>

          <div className="setup-section glass-card">
            <div className="section-title-row">
              <span className="section-number">1</span>
              <div><p className="kicker">Compute</p><h2>GPU priority</h2></div>
            </div>
            <div className="segmented-control">
              <button
                className={request.cloud_type === "secure" ? "active" : ""}
                onClick={() => { setPlan(null); setRequest({ ...request, cloud_type: "secure" }); }}
              >Secure Cloud</button>
              <button
                className={request.cloud_type === "community" ? "active" : ""}
                disabled={portable}
                onClick={() => { setPlan(null); setRequest({ ...request, cloud_type: "community" }); }}
              >Community Cloud</button>
            </div>
            <div className="gpu-list">
              {gpus.map((gpu) => {
                const rank = request.gpu_priority_ids.indexOf(gpu.id);
                const cloudAvailable = request.cloud_type === "secure"
                  ? gpu.secure_available : gpu.community_available;
                const datacenterAvailable = !portable || portableGpuIds.has(gpu.id);
                return (
                  <div className={`gpu-row ${rank >= 0 ? "selected" : ""} ${!cloudAvailable || !datacenterAvailable ? "unavailable" : ""}`} key={gpu.id}>
                    <button
                      className="gpu-select"
                      disabled={rank < 0 && (!cloudAvailable || !datacenterAvailable)}
                      onClick={() => toggleGpu(gpu)}
                    >
                      <span className="rank-box">{rank >= 0 ? rank + 1 : "+"}</span>
                      <span><strong>{gpu.display_name}</strong><small>{gpu.memory_gb} GB VRAM</small></span>
                    </button>
                    <span className="gpu-price">${gpu.on_demand_price_per_hour.toFixed(2)}<small>/hr</small></span>
                    {rank >= 0 && (
                      <span className="reorder-buttons">
                        <button aria-label="Move GPU up" onClick={() => moveGpu(rank, -1)}><Icon name="chevronUp" /></button>
                        <button aria-label="Move GPU down" onClick={() => moveGpu(rank, 1)}><Icon name="chevronDown" /></button>
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
            <label className="check-row">
              <input
                type="checkbox"
                checked={request.interruptible}
                onChange={(event) => { setPlan(null); setRequest({ ...request, interruptible: event.target.checked }); }}
              />
              <span><strong>Use interruptible compute</strong><small>Lower cost, but the Pod may stop without notice.</small></span>
            </label>
          </div>

          <div className="setup-section glass-card">
            <div className="section-title-row">
              <span className="section-number">2</span>
              <div><p className="kicker">Storage & safety</p><h2>Persistent workspace</h2></div>
            </div>
            <div className="two-fields">
              <NumberField label="Container disk" suffix="GB" value={request.container_disk_gb} min={30} max={500}
                onChange={(value) => { setPlan(null); setRequest({ ...request, container_disk_gb: value }); }} />
              <NumberField label="Workspace volume" suffix="GB" value={request.workspace_disk_gb} min={50} max={4000}
                onChange={(value) => { setPlan(null); setRequest({ ...request, workspace_disk_gb: value }); }} />
            </div>
            {portable && (
              <div className="two-fields">
                <label className="number-field">
                  <span>Network volume</span>
                  <select
                    className="text-input"
                    value={request.network_volume_id ?? ""}
                    onChange={(event) => {
                      const volume = networkVolumes.find((item) => item.id === event.target.value);
                      setPlan(null);
                      setRequest({
                        ...request,
                        network_volume_id: volume?.id ?? null,
                        workspace_disk_gb: volume?.size_gb ?? request.workspace_disk_gb,
                        datacenter_priority_ids: volume ? [volume.datacenter_id] : request.datacenter_priority_ids,
                      });
                    }}
                  >
                    <option value="">Create a new network volume</option>
                    {networkVolumes.map((volume) => (
                      <option key={volume.id} value={volume.id}>
                        {volume.name} · {volume.size_gb} GB · {volume.datacenter_id}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="number-field">
                  <span>Datacenter</span>
                  <select
                    className="text-input"
                    disabled={Boolean(selectedVolume)}
                    value={selectedDatacenterId ?? ""}
                    onChange={(event) => {
                      setPlan(null);
                      setRequest({ ...request, datacenter_priority_ids: [event.target.value] });
                    }}
                  >
                    {datacenters.map((datacenter) => (
                      <option key={datacenter.id} value={datacenter.id}>
                        {datacenter.name} · {datacenter.location}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            )}
            <div className="two-fields">
              <NumberField label="Idle stop" suffix="min" value={request.idle_timeout_seconds / 60} min={5} max={1440}
                onChange={(value) => { setPlan(null); setRequest({ ...request, idle_timeout_seconds: value * 60 }); }} />
              <NumberField label="Hard session limit" suffix="hr" value={request.hard_deadline_seconds / 3600} min={1} max={168}
                onChange={(value) => { setPlan(null); setRequest({ ...request, hard_deadline_seconds: value * 3600 }); }} />
            </div>
            <div className="storage-note"><Icon name="folder" />
              <span><strong>{portable ? "Stop terminates the Pod and keeps the network volume." : "Stop keeps files. Delete removes them."}</strong> Storage continues to incur cost while the GPU is stopped.</span>
            </div>
          </div>
        </section>

        <aside className="review-card glass-card">
          <p className="kicker">Review</p>
          <h2>{plan ? "Ready to create" : "Workspace estimate"}</h2>
          <label className="field-label" htmlFor="workspace-name">Workspace name</label>
          <input id="workspace-name" className="text-input" value={workspaceName}
            onChange={(event) => setWorkspaceName(event.target.value)} />
          <dl className="summary-list">
            <div><dt>Mode</dt><dd>{portable ? "Portable workspace" : "Persistent Pod"}</dd></div>
            <div><dt>Preferred GPU</dt><dd>{plan?.selected_gpu.display_name ?? selectedGpus[0]?.display_name ?? "Select one"}</dd></div>
            <div><dt>Compute</dt><dd>{plan ? `$${plan.estimated_compute_per_hour.toFixed(2)}/hr` : "Calculated on review"}</dd></div>
            <div><dt>Persistent storage</dt><dd>{plan ? `$${plan.estimated_storage_per_month.toFixed(2)}/mo` : `${request.workspace_disk_gb} GB`}</dd></div>
            <div><dt>Idle stop</dt><dd>{request.idle_timeout_seconds / 60} minutes</dd></div>
          </dl>
          {plan?.warnings.map((warning) => <div className="warning-line" key={warning}>{warning}</div>)}
          {error && <div className="error-banner">{error}</div>}
          {!plan ? (
            <button className="primary-button full-button" disabled={busy || selectedGpus.length === 0} onClick={reviewPlan}>
              {busy ? "Checking availability…" : "Review availability and cost"}
            </button>
          ) : (
            <button className="primary-button full-button" disabled={busy || !workspaceName.trim()} onClick={createWorkspace}>
              {busy ? "Creating workspace…" : capabilities.development_backend ? "Create preview workspace" : "Create cloud workspace"}
            </button>
          )}
          <p className="destructive-note">{portable
            ? "The network volume stays in one datacenter; stopping terminates only its ephemeral Pod."
            : "A persistent Pod may not regain the same GPU immediately after it is stopped."}</p>
        </aside>
      </div>
    </main>
  );
}

function Feature({ icon, title, detail }: { icon: "gpu" | "clock" | "folder"; title: string; detail: string }) {
  return <div className="feature"><Icon name={icon} /><span><strong>{title}</strong><small>{detail}</small></span></div>;
}

function NumberField({ label, suffix, value, min, max, onChange }: {
  label: string; suffix: string; value: number; min: number; max: number; onChange: (value: number) => void;
}) {
  return (
    <label className="number-field">
      <span>{label}</span>
      <span className="number-input-wrap">
        <input type="number" min={min} max={max} value={value}
          onChange={(event) => onChange(Math.max(min, Math.min(max, Number(event.target.value))))} />
        <small>{suffix}</small>
      </span>
    </label>
  );
}
