import { useEffect, useState } from "react";
import type { FileKind, FileRecord } from "../api";
import { controlPlane } from "../api";
import type { RuntimeSettings, StudioSettings } from "../studioProject";

interface Props {
  workspaceId: string;
  settings: StudioSettings;
  onSettings: (settings: StudioSettings) => void;
  onClose: () => void;
  onManageFiles: () => void;
  onTransfers: () => void;
  onEvent?: (message: string, kind: "info" | "error" | "worker") => void;
}

type ModelKind = "diffusion_models" | "text_encoders" | "vae" | "face_detection";

const modelKinds: { kind: ModelKind; label: string; idKey: keyof RuntimeSettings; nameKey: keyof RuntimeSettings }[] = [
  { kind: "diffusion_models", label: "Diffusion model", idKey: "diffusionModelFileId", nameKey: "diffusionModelName" },
  { kind: "text_encoders", label: "Text encoder", idKey: "textEncoderFileId", nameKey: "textEncoderName" },
  { kind: "vae", label: "VAE", idKey: "vaeFileId", nameKey: "vaeName" },
  { kind: "face_detection", label: "Face detector", idKey: "faceDetectorFileId", nameKey: "faceDetectorName" },
];

export function SetupPanel({ workspaceId, settings, onSettings, onClose, onManageFiles, onTransfers, onEvent }: Props) {
  const [inventory, setInventory] = useState<Record<ModelKind, FileRecord[]>>({
    diffusion_models: [], text_encoders: [], vae: [], face_detection: [],
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    void Promise.all(modelKinds.map(async ({ kind }) => [kind, await allFiles(workspaceId, kind)] as const))
      .then((entries) => {
        if (cancelled) return;
        const next = Object.fromEntries(entries) as Record<ModelKind, FileRecord[]>;
        setInventory(next);
        const runtime = { ...settings.runtime };
        let changed = false;
        for (const item of modelKinds) {
          const files = next[item.kind];
          const wantedName = String(runtime[item.nameKey]);
          const match = files.find((file) => file.display_name.toLocaleLowerCase() === wantedName.toLocaleLowerCase())
            ?? (wantedName ? undefined : files[0]);
          if (match && runtime[item.idKey] !== match.id) {
            Object.assign(runtime, { [item.idKey]: match.id, [item.nameKey]: match.display_name });
            changed = true;
          }
        }
        if (changed) onSettings({ ...settings, runtime });
      })
      .catch((caught) => {
        if (cancelled) return;
        const detail = caught instanceof Error ? caught.message : "Could not load model inventory";
        setError(detail);
        onEvent?.(detail, "error");
      })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [workspaceId]);

  function selectModel(item: (typeof modelKinds)[number], id: string) {
    const file = inventory[item.kind].find((candidate) => candidate.id === id);
    const runtime = { ...settings.runtime };
    Object.assign(runtime, { [item.idKey]: file?.id ?? "", [item.nameKey]: file?.display_name ?? "" });
    onSettings({ ...settings, runtime });
    if (file) onEvent?.(`Selected ${item.label.toLocaleLowerCase()}: ${file.display_name}.`, "info");
  }

  const ready = modelKinds.filter((item) => Boolean(settings.runtime[item.idKey])).length;

  return (
    <div className="asset-backdrop">
      <section className="asset-panel setup-panel glass-card" aria-label="Model and output setup">
        <header><div><p className="kicker">Persistent workspace</p><h2>Model & output setup</h2><small>{ready} / {modelKinds.length} model roles selected</small></div><button className="quiet-button" onClick={onClose}>Close</button></header>
        <p className="field-help">Selections are resolved to opaque workspace files for every run and saved by filename in the project.</p>
        <div className="setup-model-grid">
          {modelKinds.map((item) => {
            const files = inventory[item.kind];
            const selected = String(settings.runtime[item.idKey]);
            const missingName = String(settings.runtime[item.nameKey]);
            return <label className="number-field" key={item.kind}><span>{item.label}</span><select className="select-input" disabled={loading} value={selected} onChange={(event) => selectModel(item, event.target.value)}><option value="">{files.length ? "Automatic discovery" : "No uploaded file"}</option>{files.map((file) => <option key={file.id} value={file.id}>{file.display_name}</option>)}</select><small className={selected ? "setup-ready" : "setup-missing"}>{selected ? "Ready · exact file selected" : missingName ? `Missing · ${missingName}` : "Automatic discovery · upload/select for deterministic runs"}</small></label>;
          })}
        </div>
        <label className="number-field setup-prefix"><span>Generation output filename prefix</span><input className="text-input" maxLength={128} value={settings.runtime.filenamePrefix} onChange={(event) => onSettings({ ...settings, runtime: { ...settings.runtime, filenamePrefix: event.target.value } })} /><small>1–128 characters; cannot contain /, \\, or a null character.</small></label>
        {error && <div className="error-banner">{error}</div>}
        <div className="modal-actions"><button className="quiet-button" onClick={onTransfers}>Download models</button><button className="quiet-button" onClick={onManageFiles}>Manage uploaded files</button><button className="primary-button" onClick={onClose}>Done</button></div>
      </section>
    </div>
  );
}

async function allFiles(workspaceId: string, kind: FileKind) {
  const items: FileRecord[] = [];
  let cursor: string | undefined;
  do {
    const page = await controlPlane.files(workspaceId, kind, cursor);
    items.push(...page.items);
    cursor = page.next_cursor ?? undefined;
  } while (cursor);
  return items;
}
