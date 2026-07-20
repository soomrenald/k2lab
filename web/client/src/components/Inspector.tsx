import { useState } from "react";
import type { RegionBox, RegionLayer, StudioMode } from "./RegionCanvas";
import { Icon } from "./Icon";

type InspectorTab = "prompt" | "regions" | "loras" | "advanced";

interface LoraItem {
  id: string;
  name: string;
  active: boolean;
  strength: number;
  scope: "Global" | string;
}

interface Props {
  mode: StudioMode;
  activeLayer: RegionLayer;
  regions: RegionBox[];
  selectedId: string | null;
  globalPrompt: string;
  onGlobalPrompt: (value: string) => void;
  onRegions: (regions: RegionBox[]) => void;
  onSelect: (id: string | null) => void;
}

export function Inspector({
  mode,
  activeLayer,
  regions,
  selectedId,
  globalPrompt,
  onGlobalPrompt,
  onRegions,
  onSelect,
}: Props) {
  const [tab, setTab] = useState<InspectorTab>("prompt");
  const [loras, setLoras] = useState<LoraItem[]>([
    { id: "demo-style", name: "Studio light · example", active: true, strength: 0.75, scope: "Global" },
  ]);
  const visibleRegions = mode === "face"
    ? []
    : regions.filter((region) => region.layer === activeLayer);
  const selected = regions.find((region) => region.id === selectedId) ?? null;

  function updateSelected(patch: Partial<RegionBox>) {
    if (!selected) return;
    onRegions(regions.map((region) => region.id === selected.id ? { ...region, ...patch } : region));
  }

  return (
    <aside className="inspector">
      <div className="inspector-head">
        <div><p className="kicker">Inspector</p><h2>{mode === "edit" ? "Image edit" : mode === "face" ? "Face refinement" : "Generation"}</h2></div>
        <span className="layer-count">{visibleRegions.length} region{visibleRegions.length === 1 ? "" : "s"}</span>
      </div>
      <nav className="inspector-tabs" aria-label="Inspector sections">
        {(["prompt", "regions", "loras", "advanced"] as InspectorTab[]).map((item) => (
          <button key={item} className={tab === item ? "active" : ""} onClick={() => setTab(item)}>
            {item === "prompt" ? "Prompt" : item === "regions" ? "Regions" : item === "loras" ? "LoRAs" : "Advanced"}
          </button>
        ))}
      </nav>

      <div className="inspector-content">
        {tab === "prompt" && (
          <>
            <div className="inspector-section">
              <label className="field-label" htmlFor="global-prompt">
                {mode === "edit" && activeLayer === "targets" ? "Edit instruction" : mode === "edit" ? "Original global prompt · reference" : "Global prompt"}
              </label>
              <textarea id="global-prompt" className="prompt-area global-area"
                placeholder={mode === "edit" ? "Describe the overall edit intent…" : "Describe the complete scene…"}
                value={globalPrompt} onChange={(event) => onGlobalPrompt(event.target.value)} />
              {mode === "edit" && activeLayer === "targets" && (
                <p className="field-help">Combined with each target prompt. Leave blank for box-only instructions.</p>
              )}
            </div>
            <div className="inspector-section">
              <div className="section-inline-title">
                <span>{selected ? selected.name : "Regional prompt"}</span>
                {selected && <span className="active-pill">Selected</span>}
              </div>
              {selected ? (
                <>
                  <input className="text-input compact-input" value={selected.name}
                    onChange={(event) => updateSelected({ name: event.target.value })} />
                  <textarea className="prompt-area" value={selected.prompt}
                    placeholder={mode === "edit" && activeLayer === "targets" ? "Describe the edit inside this box…" : "Describe this region…"}
                    onChange={(event) => updateSelected({ prompt: event.target.value })} />
                </>
              ) : (
                <div className="empty-inspector"><Icon name="layers" /><span>Select a region to edit its prompt.</span></div>
              )}
            </div>
          </>
        )}

        {tab === "regions" && (
          <div className="inspector-section region-panel">
            <div className="section-inline-title"><span>{activeLayer === "reference" ? "Reference regions" : activeLayer === "targets" ? "Edit targets" : "Scene regions"}</span></div>
            <div className="region-list">
              {visibleRegions.map((region, index) => (
                <button className={`region-list-row ${selectedId === region.id ? "selected" : ""}`} key={region.id}
                  onClick={() => onSelect(region.id)}>
                  <span className="region-swatch" style={{ opacity: region.enabled ? 1 : 0.35 }}>{index + 1}</span>
                  <span className="region-list-copy"><strong>{region.name}</strong><small>{Math.round(region.width)} × {Math.round(region.height)}</small></span>
                  <input type="checkbox" aria-label={`Enable ${region.name}`} checked={region.enabled}
                    onClick={(event) => event.stopPropagation()}
                    onChange={(event) => onRegions(regions.map((item) => item.id === region.id ? { ...item, enabled: event.target.checked } : item))} />
                </button>
              ))}
              {visibleRegions.length === 0 && <div className="empty-inspector"><Icon name="plus" /><span>Draw a box on the canvas to add a region.</span></div>}
            </div>
            {selected && selected.layer === activeLayer && (
              <button className="danger-text-button" onClick={() => { onRegions(regions.filter((region) => region.id !== selected.id)); onSelect(null); }}>
                <Icon name="trash" /> Remove selected region
              </button>
            )}
          </div>
        )}

        {tab === "loras" && (
          <div className="inspector-section lora-panel">
            <div className="section-inline-title"><span>LoRA library</span><button className="tiny-button"><Icon name="plus" /> Add</button></div>
            {loras.map((lora) => (
              <div className={`lora-card ${!lora.active ? "inactive" : ""}`} key={lora.id}>
                <div className="lora-title-row">
                  <label className="toggle"><input type="checkbox" checked={lora.active}
                    onChange={(event) => setLoras(loras.map((item) => item.id === lora.id ? { ...item, active: event.target.checked } : item))} /><span /></label>
                  <div><strong>{lora.name}</strong><small>{lora.scope}</small></div>
                  <button className="icon-button danger" aria-label={`Remove ${lora.name}`}
                    onClick={() => setLoras(loras.filter((item) => item.id !== lora.id))}><Icon name="trash" /></button>
                </div>
                <LinkedValue label="Strength" value={lora.strength} min={-4} max={4} step={0.05}
                  onChange={(value) => setLoras(loras.map((item) => item.id === lora.id ? { ...item, strength: value } : item))} />
                <select className="select-input" value={lora.scope}
                  onChange={(event) => setLoras(loras.map((item) => item.id === lora.id ? { ...item, scope: event.target.value } : item))}>
                  <option>Global</option>
                  {visibleRegions.map((region) => <option key={region.id}>{region.name}</option>)}
                </select>
              </div>
            ))}
            <div className="drop-zone"><Icon name="upload" /><span>Drop `.safetensors` here when cloud file transfer is connected.</span></div>
          </div>
        )}

        {tab === "advanced" && (
          <div className="inspector-section advanced-panel">
            <LinkedValue label={mode === "edit" ? "Denoise" : "Steps"} value={mode === "edit" ? 0.15 : 8}
              min={mode === "edit" ? 0.05 : 1} max={mode === "edit" ? 1 : 100} step={mode === "edit" ? 0.05 : 1} onChange={() => undefined} />
            <LinkedValue label="Inside boost" value={1} min={0.1} max={10} step={0.1} onChange={() => undefined} />
            <LinkedValue label="Outside penalty" value={1} min={0} max={10} step={0.1} onChange={() => undefined} />
            {mode === "edit" && <>
              <LinkedValue label="Latent feather" value={64} min={0} max={256} step={1} onChange={() => undefined} />
              <LinkedValue label="Composite feather" value={48} min={0} max={256} step={1} onChange={() => undefined} />
            </>}
            <label className="check-row compact-check"><input type="checkbox" defaultChecked /><span><strong>Separate overlapping subjects</strong></span></label>
            <label className="check-row compact-check"><input type="checkbox" defaultChecked /><span><strong>Preserve reference identity</strong></span></label>
          </div>
        )}
      </div>
    </aside>
  );
}

function LinkedValue({ label, value, min, max, step, onChange }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (value: number) => void;
}) {
  const [local, setLocal] = useState(value);
  function change(next: number) { setLocal(next); onChange(next); }
  return (
    <div className="linked-value">
      <div className="linked-label"><span>{label}</span><input type="number" value={local} min={min} max={max} step={step}
        onChange={(event) => change(Math.max(min, Math.min(max, Number(event.target.value))))} /></div>
      <input className="range-input" type="range" value={local} min={min} max={max} step={step}
        onChange={(event) => change(Number(event.target.value))} />
    </div>
  );
}
