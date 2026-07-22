import { useEffect, useMemo, useRef, useState } from "react";
import type { CivitaiPreview, CredentialStatus, FileKind, HuggingFacePreview, RemoteProvider, RemoteTransfer } from "../api";
import { controlPlane } from "../api";

const destinations: { value: FileKind; label: string }[] = [
  { value: "diffusion_models", label: "Diffusion models" },
  { value: "text_encoders", label: "Text encoders" },
  { value: "vae", label: "VAE" },
  { value: "loras", label: "LoRAs" },
  { value: "upscale_models", label: "Upscalers" },
  { value: "face_detection", label: "Face detection" },
];

interface Props {
  workspaceId: string;
  onClose: () => void;
  onEvent?: (message: string, kind: "info" | "error" | "worker") => void;
}

export function TransferPanel({ workspaceId, onClose, onEvent }: Props) {
  const [provider, setProvider] = useState<RemoteProvider>("civitai");
  const [credential, setCredential] = useState<CredentialStatus | null>(null);
  const [token, setToken] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [destination, setDestination] = useState<FileKind>("loras");
  const [patterns, setPatterns] = useState("*.safetensors");
  const [civitaiPreview, setCivitaiPreview] = useState<CivitaiPreview | null>(null);
  const [huggingFacePreview, setHuggingFacePreview] = useState<HuggingFacePreview | null>(null);
  const [fileId, setFileId] = useState("");
  const [allowUnsafe, setAllowUnsafe] = useState(false);
  const [transfer, setTransfer] = useState<RemoteTransfer | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [speed, setSpeed] = useState(0);
  const sample = useRef<{ bytes: number; time: number } | null>(null);
  const lastReportedState = useRef<string | null>(null);

  useEffect(() => {
    void controlPlane.downloadCredential(provider).then(setCredential).catch(() => setCredential(null));
    setCivitaiPreview(null); setHuggingFacePreview(null); setFileId(""); setAllowUnsafe(false);
  }, [provider]);

  useEffect(() => {
    if (!transfer || terminal(transfer.state)) return undefined;
    const interval = window.setInterval(async () => {
      try {
        const next = await controlPlane.transfer(workspaceId, transfer.id);
        const now = performance.now();
        const previous = sample.current;
        if (previous && now > previous.time) setSpeed(Math.max(0, (next.bytes_complete - previous.bytes) / ((now - previous.time) / 1000)));
        sample.current = { bytes: next.bytes_complete, time: now };
        setTransfer(next);
        if (next.state !== lastReportedState.current && terminal(next.state)) {
          lastReportedState.current = next.state;
          onEvent?.(next.state === "completed"
            ? `Provider transfer completed with ${next.files.length} verified file(s).`
            : `Provider transfer ${next.state}${next.error_message ? `: ${next.error_message}` : "."}`,
          next.state === "failed" ? "error" : "info");
        }
      } catch (caught) { const detail = message(caught); setError(detail); onEvent?.(detail, "error"); }
    }, 1000);
    return () => window.clearInterval(interval);
  }, [transfer, workspaceId]);

  const selectedFile = useMemo(() => civitaiPreview?.files.find((file) => file.id === fileId) ?? null, [civitaiPreview, fileId]);

  async function saveToken() {
    setBusy(true); setError("");
    try { setCredential(await controlPlane.storeDownloadCredential(provider, token)); setToken(""); }
    catch (caught) { setError(message(caught)); }
    finally { setBusy(false); }
  }

  async function disconnectToken() {
    setBusy(true); setError("");
    try { setCredential(await controlPlane.clearDownloadCredential(provider)); }
    catch (caught) { setError(message(caught)); }
    finally { setBusy(false); }
  }

  async function preview() {
    setBusy(true); setError(""); setTransfer(null);
    try {
      if (provider === "civitai") {
        const result = await controlPlane.previewCivitai(workspaceId, sourceUrl);
        setCivitaiPreview(result); setHuggingFacePreview(null);
        setFileId((result.files.find((file) => file.preferred) ?? result.files[0]).id);
      } else {
        setHuggingFacePreview(await controlPlane.previewHuggingFace(workspaceId, sourceUrl, patternList(patterns)));
        setCivitaiPreview(null);
      }
      setAllowUnsafe(false);
    } catch (caught) { setError(message(caught)); }
    finally { setBusy(false); }
  }

  async function start(resume = false) {
    setBusy(true); setError(""); setSpeed(0); sample.current = null;
    try {
      const resume_transfer_id = resume && transfer ? transfer.id : undefined;
      const next = provider === "civitai"
        ? await controlPlane.startCivitai(workspaceId, { source_url: sourceUrl, file_id: fileId, destination_kind: destination, allow_unsafe_format: allowUnsafe, resume_transfer_id })
        : await controlPlane.startHuggingFace(workspaceId, { source_url: sourceUrl, destination_kind: destination, allow_patterns: patternList(patterns), allow_unsafe_format: allowUnsafe, resume_transfer_id });
      setTransfer(next);
      lastReportedState.current = next.state;
      onEvent?.(`${resume ? "Resumed" : "Started"} ${provider} transfer into ${destination}.`, "info");
    } catch (caught) { const detail = message(caught); setError(detail); onEvent?.(detail, "error"); }
    finally { setBusy(false); }
  }

  async function cancel() {
    if (!transfer) return;
    setBusy(true);
    try { setTransfer(await controlPlane.cancelTransfer(workspaceId, transfer.id)); onEvent?.("Provider transfer cancelled; resumable data was retained.", "info"); }
    catch (caught) { const detail = message(caught); setError(detail); onEvent?.(detail, "error"); }
    finally { setBusy(false); }
  }

  const active = Boolean(transfer && !terminal(transfer.state));
  const progress = transfer?.bytes_total ? Math.min(1, transfer.bytes_complete / transfer.bytes_total) : 0;
  const eta = speed > 0 && transfer?.bytes_total ? Math.max(0, (transfer.bytes_total - transfer.bytes_complete) / speed) : null;
  const canStart = provider === "civitai" ? Boolean(civitaiPreview && fileId) : Boolean(huggingFacePreview);
  const requiresUnsafe = Boolean(selectedFile?.requires_unsafe_confirmation || huggingFacePreview?.files.some((file) => unsafeName(file.filename)));

  return (
    <div className="asset-backdrop">
      <section className="asset-panel transfer-panel glass-card" aria-label="Provider downloads">
        <header><div><p className="kicker">Provider-side transfer</p><h2>Download models</h2></div><button className="quiet-button" onClick={onClose}>Close</button></header>
        <div className="asset-kind-tabs"><button className={provider === "civitai" ? "active" : ""} onClick={() => setProvider("civitai")}>Civitai</button><button className={provider === "huggingface" ? "active" : ""} onClick={() => setProvider("huggingface")}>Hugging Face</button></div>
        <div className="provider-credential">
          <span>{credential?.configured ? `Token connected ${credential.key_hint ?? ""}` : "Public files work without a token. Add one for private or gated files."}</span>
          {credential?.configured ? <button className="danger-text-button" disabled={busy} onClick={() => void disconnectToken()}>Remove token</button> : <><input className="text-input secret-input" type="password" autoComplete="off" placeholder={provider === "civitai" ? "Download-only token" : "Read-only token"} value={token} onChange={(event) => setToken(event.target.value)} /><button className="quiet-button" disabled={busy || token.length < 8} onClick={() => void saveToken()}>Save encrypted token</button></>}
        </div>
        <div className="download-form">
          <label><span>Canonical {provider === "civitai" ? "Civitai model" : "Hugging Face repository or file"} URL</span><input className="text-input" value={sourceUrl} onChange={(event) => setSourceUrl(event.target.value)} placeholder={`https://${provider === "civitai" ? "civitai.com/models/..." : "huggingface.co/owner/repo/..."}`} /></label>
          <label><span>Install into</span><select className="select-input" value={destination} onChange={(event) => setDestination(event.target.value as FileKind)}>{destinations.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
          {provider === "huggingface" && <label><span>Repository file filters</span><input className="text-input" value={patterns} onChange={(event) => setPatterns(event.target.value)} placeholder="*.safetensors, *.json" /><small>Comma-separated allow patterns. File URLs ignore this field.</small></label>}
          <button className="primary-button" disabled={busy || !sourceUrl} onClick={() => void preview()}>Inspect before download</button>
        </div>
        {civitaiPreview && <div className="download-preview"><strong>{civitaiPreview.model_name} · {civitaiPreview.version_name}</strong><small>{civitaiPreview.model_type ?? "Model"} · {civitaiPreview.base_model ?? "Unknown base"}</small><select className="select-input" value={fileId} onChange={(event) => { setFileId(event.target.value); setAllowUnsafe(false); }}>{civitaiPreview.files.map((file) => <option key={file.id} value={file.id}>{file.filename} · {formatBytes(file.size_bytes ?? 0)}{file.preferred ? " · preferred" : ""}</option>)}</select>{selectedFile?.requires_unsafe_confirmation && <UnsafeConfirmation checked={allowUnsafe} onChange={setAllowUnsafe} />}</div>}
        {huggingFacePreview && <div className="download-preview"><strong>{huggingFacePreview.repo_id}</strong><small>{huggingFacePreview.mirror_repository ? `Repository mirror · ${huggingFacePreview.files.length} files` : huggingFacePreview.filename} · {formatBytes(huggingFacePreview.required_bytes)} required</small>{requiresUnsafe && <UnsafeConfirmation checked={allowUnsafe} onChange={setAllowUnsafe} />}</div>}
        {canStart && !active && transfer?.state !== "completed" && <button className="primary-button" disabled={busy || (requiresUnsafe && !allowUnsafe)} onClick={() => void start(Boolean(transfer))}>{transfer ? "Retry / resume transfer" : "Start provider download"}</button>}
        {transfer && <div className="remote-transfer"><div className="transfer-progress"><div><i style={{ width: `${progress * 100}%` }} /></div><span>{transfer.state} · {formatBytes(transfer.bytes_complete)}{transfer.bytes_total !== null ? ` / ${formatBytes(transfer.bytes_total)}` : ""}{speed > 0 ? ` · ${formatBytes(speed)}/s${eta !== null ? ` · ${Math.ceil(eta)}s remaining` : ""}` : ""}</span></div>{active && <button className="danger-text-button" disabled={busy} onClick={() => void cancel()}>Cancel and keep resumable data</button>}{transfer.state === "completed" && <p className="success-line">Installed {transfer.files.length} verified file{transfer.files.length === 1 ? "" : "s"}.</p>}{transfer.error_message && <div className="error-banner">{transfer.error_message} <small>{transfer.error_code}</small></div>}</div>}
        {error && <div className="error-banner">{error}</div>}
        <p className="field-help">Tokens are encrypted by the control plane and sent to the agent only for one operation. URLs with embedded credentials and redirects to unapproved hosts are rejected.</p>
      </section>
    </div>
  );
}

function UnsafeConfirmation({ checked, onChange }: { checked: boolean; onChange: (value: boolean) => void }) { return <label className="check-row warning-check"><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><span><strong>Allow pickle-based model format</strong><small>Only continue if you trust the publisher. Safetensors is preferred.</small></span></label>; }
function terminal(state: string) { return ["completed", "failed", "cancelled", "paused"].includes(state); }
function patternList(value: string) { return value.split(",").map((item) => item.trim()).filter(Boolean); }
function unsafeName(value: string) { return [".bin", ".ckpt", ".pt", ".pth", ".pkl", ".pickle"].some((suffix) => value.toLowerCase().endsWith(suffix)); }
function message(caught: unknown) { return caught instanceof Error ? caught.message : "Provider transfer failed"; }
function formatBytes(value: number) { if (value < 1024) return `${value} B`; if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KiB`; if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MiB`; return `${(value / 1024 ** 3).toFixed(1)} GiB`; }
