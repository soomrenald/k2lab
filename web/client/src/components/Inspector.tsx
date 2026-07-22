import { useRef, useState } from "react";
import type { DetectedFaceRecord } from "../api";
import type { RegionBox, RegionLayer, StudioMode } from "./RegionCanvas";
import { Icon } from "./Icon";
import {
  COMFYUI_SAMPLERS,
  COMFYUI_SCHEDULERS,
  PROJECTOR_PRESETS,
  type GenerationSettings,
  type LoraLayerBinding,
  type ProjectorSettings,
  type PromptEmphasisState,
  type StudioLora,
  type StudioSettings,
} from "../studioProject";

type InspectorTab = "prompt" | "regions" | "loras" | "advanced";

interface Props {
  mode: StudioMode;
  activeLayer: RegionLayer;
  regions: RegionBox[];
  selectedId: string | null;
  globalPrompt: string;
  settings: StudioSettings;
  loras: StudioLora[];
  onGlobalPrompt: (value: string) => void;
  onSettings: (settings: StudioSettings) => void;
  onLoras: (loras: StudioLora[]) => void;
  onChooseLora: () => void;
  onChooseUpscaleModel: () => void;
  onPreviewUnifiedPrompt: () => void;
  faces: DetectedFaceRecord[];
  selectedFaceIndices: number[];
  manualFacePaths: number[][][];
  lassoMode: boolean;
  onDetectFaces: () => void;
  onToggleFace: (index: number) => void;
  onSelectAllFaces: (selected: boolean) => void;
  onLassoMode: (enabled: boolean) => void;
  onUndoLasso: () => void;
  onClearLassos: () => void;
  onUseLatestFaceSource: () => void;
  onRegions: (regions: RegionBox[]) => void;
  onSelect: (id: string | null) => void;
}

export function Inspector(props: Props) {
  const {
    mode, activeLayer, regions, selectedId, globalPrompt, settings, loras,
    onGlobalPrompt, onSettings, onLoras, onChooseLora, onChooseUpscaleModel,
    onPreviewUnifiedPrompt, faces, selectedFaceIndices, manualFacePaths, lassoMode,
    onDetectFaces, onToggleFace, onSelectAllFaces, onLassoMode, onUndoLasso,
    onClearLassos, onUseLatestFaceSource, onRegions, onSelect,
  } = props;
  const [tab, setTab] = useState<InspectorTab>("prompt");
  const [emphasisStrength, setEmphasisStrength] = useState(0.5);
  const globalPromptRef = useRef<HTMLTextAreaElement>(null);
  const regionPromptRef = useRef<HTMLTextAreaElement>(null);
  const visibleRegions = regions.filter((region) => region.layer === activeLayer);
  const selected = regions.find((region) => region.id === selectedId && region.layer === activeLayer) ?? null;
  const emphasisAvailable = mode === "generation" || (mode === "edit" && activeLayer === "reference");
  const emphases = mode === "generation"
    ? settings.generation.promptEmphases
    : settings.edit.referencePromptEmphases;

  function updateSelected(patch: Partial<RegionBox>) {
    if (!selected) return;
    onRegions(regions.map((region) => region.id === selected.id ? { ...region, ...patch } : region));
  }

  function updateGeneration(patch: Partial<GenerationSettings>) {
    if (patch.width !== undefined || patch.height !== undefined) {
      const width = patch.width ?? settings.generation.width;
      const height = patch.height ?? settings.generation.height;
      onRegions(regions.map((region) => {
        if (region.layer !== "generation") return region;
        const x = Math.min(region.x, width - 16);
        const y = Math.min(region.y, height - 16);
        return {
          ...region,
          x,
          y,
          width: Math.min(region.width, width - x),
          height: Math.min(region.height, height - y),
        };
      }));
    }
    onSettings({ ...settings, generation: { ...settings.generation, ...patch } });
  }

  function updateEdit(patch: Partial<StudioSettings["edit"]>) {
    onSettings({ ...settings, edit: { ...settings.edit, ...patch } });
  }

  function updateFace(patch: Partial<StudioSettings["face"]>) {
    onSettings({ ...settings, face: { ...settings.face, ...patch } });
  }

  function setEmphases(next: PromptEmphasisState[]) {
    if (mode === "generation") updateGeneration({ promptEmphases: next });
    else updateEdit({ referencePromptEmphases: next });
  }

  function addEmphasis(scopeId: "__global__" | string, editor: HTMLTextAreaElement | null) {
    if (!editor || editor.selectionStart === editor.selectionEnd) return;
    const phrase = editor.value.slice(editor.selectionStart, editor.selectionEnd);
    if (!phrase.trim()) return;
    const prefix = editor.value.slice(0, editor.selectionStart);
    const occurrence = prefix.split(phrase).length - 1;
    setEmphases([...emphases, {
      id: crypto.randomUUID(), scopeId, phrase, strength: emphasisStrength, occurrence,
    }]);
  }

  function emphasisMatches(item: PromptEmphasisState) {
    const source = item.scopeId === "__global__"
      ? globalPrompt
      : regions.find((region) => region.id === item.scopeId)?.prompt ?? "";
    let offset = -1;
    let start = 0;
    for (let index = 0; index <= item.occurrence; index += 1) {
      offset = source.indexOf(item.phrase, start);
      if (offset < 0) return false;
      start = offset + item.phrase.length;
    }
    return true;
  }

  function moveSelected(offset: -1 | 1) {
    if (!selected) return;
    const peers = regions.filter((region) => region.layer === selected.layer);
    const peerIndex = peers.findIndex((region) => region.id === selected.id);
    const target = peers[peerIndex + offset];
    if (!target) return;
    const next = [...regions];
    const sourceIndex = next.findIndex((region) => region.id === selected.id);
    const targetIndex = next.findIndex((region) => region.id === target.id);
    [next[sourceIndex], next[targetIndex]] = [next[targetIndex], next[sourceIndex]];
    onRegions(next);
  }

  function removeSelectedRegion() {
    if (!selected) return;
    const regionId = selected.id;
    onRegions(regions.filter((region) => region.id !== regionId));
    onLoras(loras.map((lora) => {
      const generationIds = lora.generation.regionIds.filter((id) => id !== regionId);
      const referenceIds = lora.reference.regionIds.filter((id) => id !== regionId);
      const targetIds = lora.targets.regionIds.filter((id) => id !== regionId);
      return {
        ...lora,
        generation: generationIds.length || lora.generation.global
          ? { ...lora.generation, regionIds: generationIds }
          : { ...lora.generation, global: true, regionIds: [], routingMode: "standard", triggerPhrase: "" },
        reference: { ...lora.reference, enabled: lora.reference.global || referenceIds.length > 0, regionIds: referenceIds },
        targets: { ...lora.targets, enabled: lora.targets.global || targetIds.length > 0, regionIds: targetIds },
      };
    }));
    onSettings({
      ...settings,
      generation: { ...settings.generation, promptEmphases: settings.generation.promptEmphases.filter((item) => item.scopeId !== regionId) },
      edit: { ...settings.edit, referencePromptEmphases: settings.edit.referencePromptEmphases.filter((item) => item.scopeId !== regionId) },
    });
    onSelect(null);
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
        {tab === "prompt" && <>
          <div className="inspector-section">
            <label className="field-label" htmlFor="global-prompt">
              {mode === "edit" && activeLayer === "targets" ? "Edit instruction" : mode === "edit" ? "Original global prompt · reference" : mode === "face" ? "Generation prompt · reference" : "Global prompt"}
            </label>
            <textarea ref={globalPromptRef} id="global-prompt" className="prompt-area global-area"
              placeholder={mode === "edit" ? "Describe the overall edit intent…" : "Describe the complete scene…"}
              value={globalPrompt} onChange={(event) => onGlobalPrompt(event.target.value)} />
            {mode === "edit" && activeLayer === "targets" && <p className="field-help">Combined with each target prompt. Leave blank for box-only instructions.</p>}
          </div>
          {mode !== "face" && <div className="inspector-section">
            <div className="section-inline-title"><span>{selected ? selected.name : "Regional prompt"}</span>{selected && <span className="active-pill">Selected</span>}</div>
            {selected ? <>
              <input className="text-input compact-input" value={selected.name} onChange={(event) => updateSelected({ name: event.target.value })} />
              <label className="field-label">Spatial role</label>
              <select className="select-input compact-select" value={selected.spatialRole} onChange={(event) => updateSelected({ spatialRole: event.target.value as RegionBox["spatialRole"] })}>
                <option value="auto">Auto (based on box width)</option><option value="subject">Subject target</option><option value="background">Background band</option>
              </select>
              <label className="field-label">Region prompt</label>
              <textarea ref={regionPromptRef} className="prompt-area" value={selected.prompt}
                placeholder={mode === "edit" && activeLayer === "targets" ? "Describe the edit inside this box…" : "Describe this region…"}
                onChange={(event) => updateSelected({ prompt: event.target.value })} />
              <label className="field-label">Face identity prompt</label>
              <textarea className="prompt-area identity-area" value={selected.faceIdentityPrompt}
                placeholder="Stable facial identity, person class, face, and hair…"
                onChange={(event) => updateSelected({ faceIdentityPrompt: event.target.value })} />
              <div className="inline-actions"><button className="tiny-button" onClick={() => moveSelected(-1)}>Move forward</button><button className="tiny-button" onClick={() => moveSelected(1)}>Move backward</button></div>
            </> : <div className="empty-inspector"><Icon name="layers" /><span>Select a region to edit its prompt.</span></div>}
          </div>}
          {emphasisAvailable && <div className="inspector-section emphasis-panel">
            <div className="section-inline-title"><span>Phrase emphasis</span></div>
            <LinkedValue label="Selected phrase boost" value={emphasisStrength} min={0} max={2} step={0.1} onChange={setEmphasisStrength} />
            <div className="inline-actions"><button className="tiny-button" onClick={() => addEmphasis("__global__", globalPromptRef.current)}>Emphasize global selection</button>{selected && <button className="tiny-button" onClick={() => addEmphasis(selected.id, regionPromptRef.current)}>Region selection</button>}</div>
            {emphases.map((item) => <div className={`emphasis-row ${emphasisMatches(item) ? "" : "invalid"}`} key={item.id}>
              <span>{item.scopeId === "__global__" ? "Global" : regions.find((region) => region.id === item.scopeId)?.name ?? "Missing region"}: “{item.phrase}”</span>
              <input type="number" min="0" max="2" step="0.1" value={item.strength} onChange={(event) => setEmphases(emphases.map((entry) => entry.id === item.id ? { ...entry, strength: Number(event.target.value) } : entry))} />
              <button className="icon-button danger" onClick={() => setEmphases(emphases.filter((entry) => entry.id !== item.id))}><Icon name="trash" /></button>
            </div>)}
          </div>}
        </>}

        {tab === "regions" && mode === "face" && <FaceSelectionPanel
          faces={faces}
          selectedFaceIndices={selectedFaceIndices}
          manualFacePaths={manualFacePaths}
          lassoMode={lassoMode}
          onDetect={onDetectFaces}
          onToggle={onToggleFace}
          onSelectAll={onSelectAllFaces}
          onLassoMode={onLassoMode}
          onUndoLasso={onUndoLasso}
          onClearLassos={onClearLassos}
          onUseLatest={onUseLatestFaceSource}
        />}

        {tab === "regions" && mode !== "face" && <div className="inspector-section region-panel">
          <div className="section-inline-title"><span>{activeLayer === "reference" ? "Reference regions · front to back" : activeLayer === "targets" ? "Edit targets · front to back" : "Scene regions · front to back"}</span></div>
          <div className="region-list">{visibleRegions.map((region, index) => <button className={`region-list-row ${selectedId === region.id ? "selected" : ""}`} key={region.id} onClick={() => onSelect(region.id)}>
            <span className="region-swatch" style={{ opacity: region.enabled ? 1 : 0.35 }}>{index + 1}</span>
            <span className="region-list-copy"><strong>{region.name}</strong><small>{region.spatialRole} · {Math.round(region.width)} × {Math.round(region.height)}</small></span>
            <input type="checkbox" aria-label={`Enable ${region.name}`} checked={region.enabled} onClick={(event) => event.stopPropagation()} onChange={(event) => onRegions(regions.map((item) => item.id === region.id ? { ...item, enabled: event.target.checked } : item))} />
          </button>)}</div>
          {visibleRegions.length === 0 && <div className="empty-inspector"><Icon name="plus" /><span>Draw a box on the canvas to add a region.</span></div>}
          {selected && <button className="danger-text-button" onClick={removeSelectedRegion}><Icon name="trash" /> Remove selected region</button>}
        </div>}

        {tab === "loras" && <LoraPanel activeLayer={activeLayer} regions={visibleRegions} loras={loras} onLoras={onLoras} onChoose={onChooseLora} />}

        {tab === "advanced" && <AdvancedPanel mode={mode} activeLayer={activeLayer} settings={settings} updateGeneration={updateGeneration} updateEdit={updateEdit} updateFace={updateFace} onChooseUpscaleModel={onChooseUpscaleModel} onPreviewUnifiedPrompt={onPreviewUnifiedPrompt} />}
      </div>
    </aside>
  );
}

function FaceSelectionPanel({ faces, selectedFaceIndices, manualFacePaths, lassoMode, onDetect, onToggle, onSelectAll, onLassoMode, onUndoLasso, onClearLassos, onUseLatest }: {
  faces: DetectedFaceRecord[];
  selectedFaceIndices: number[];
  manualFacePaths: number[][][];
  lassoMode: boolean;
  onDetect: () => void;
  onToggle: (index: number) => void;
  onSelectAll: (selected: boolean) => void;
  onLassoMode: (enabled: boolean) => void;
  onUndoLasso: () => void;
  onClearLassos: () => void;
  onUseLatest: () => void;
}) {
  return <div className="inspector-section face-selection-panel">
    <div className="section-inline-title"><span>Detected faces</span><span className="active-pill">{selectedFaceIndices.length} selected</span></div>
    <div className="inline-actions">
      <button className="tiny-button" onClick={onDetect}>Detect faces</button>
      <button className="tiny-button" onClick={onUseLatest}>Use latest first pass</button>
      <button className="tiny-button" onClick={() => onSelectAll(true)}>Select all</button>
      <button className="tiny-button" onClick={() => onSelectAll(false)}>Select none</button>
    </div>
    <div className="region-list face-list">{faces.map((face) => <button className={`region-list-row ${selectedFaceIndices.includes(face.index) ? "selected" : ""}`} key={face.index} onClick={() => onToggle(face.index)}>
      <span className="region-swatch">{face.index + 1}</span>
      <span className="region-list-copy"><strong>Face {face.index + 1}</strong><small>Confidence {face.score.toFixed(3)} · {Math.round(face.box[2] - face.box[0])} × {Math.round(face.box[3] - face.box[1])}</small></span>
      <input type="checkbox" readOnly checked={selectedFaceIndices.includes(face.index)} />
    </button>)}</div>
    {faces.length === 0 && <div className="empty-inspector"><Icon name="face" /><span>Choose a cloud source, then detect faces.</span></div>}
    <SectionTitle text="Manual face lassos" />
    <p className="field-help">Enable lasso drawing, then drag a closed path around each additional face on the image.</p>
    <div className="inline-actions">
      <button className={`tiny-button ${lassoMode ? "active" : ""}`} onClick={() => onLassoMode(!lassoMode)}>{lassoMode ? "Drawing lasso…" : "Draw lasso"}</button>
      <button className="tiny-button" disabled={manualFacePaths.length === 0} onClick={onUndoLasso}>Undo lasso</button>
      <button className="tiny-button" disabled={manualFacePaths.length === 0} onClick={onClearLassos}>Clear lassos</button>
    </div>
    <p className="field-help">{manualFacePaths.length} manual lasso{manualFacePaths.length === 1 ? "" : "s"} prepared.</p>
  </div>;
}

function LoraPanel({ activeLayer, regions, loras, onLoras, onChoose }: { activeLayer: RegionLayer; regions: RegionBox[]; loras: StudioLora[]; onLoras: (items: StudioLora[]) => void; onChoose: () => void }) {
  const bindingKey = activeLayer;
  function update(id: string, patch: Partial<StudioLora>) { onLoras(loras.map((lora) => lora.id === id ? { ...lora, ...patch } : lora)); }
  function updateBinding(lora: StudioLora, patch: Partial<LoraLayerBinding>) {
    update(lora.id, { [bindingKey]: { ...lora[bindingKey], ...patch } });
  }
  return <div className="inspector-section lora-panel">
    <div className="section-inline-title"><span>LoRA library</span><button className="tiny-button" onClick={onChoose}><Icon name="plus" /> Add cloud LoRA</button></div>
    {loras.map((lora) => {
      const binding = lora[bindingKey];
      return <div className={`lora-card ${!lora.active ? "inactive" : ""}`} key={lora.id}>
        <div className="lora-title-row"><label className="toggle"><input type="checkbox" checked={lora.active} onChange={(event) => update(lora.id, { active: event.target.checked })} /><span /></label><div><strong>{lora.name}</strong><small>{binding.enabled ? binding.global ? "Global" : `${binding.regionIds.length} region(s)` : "Not used on this layer"}</small></div><button className="icon-button danger" onClick={() => onLoras(loras.filter((item) => item.id !== lora.id))}><Icon name="trash" /></button></div>
        <LinkedValue label="Strength" value={lora.strength} min={-4} max={4} step={0.05} onChange={(strength) => update(lora.id, { strength })} />
        {activeLayer !== "generation" && <label className="check-row compact-check"><input type="checkbox" checked={binding.enabled} onChange={(event) => updateBinding(lora, { enabled: event.target.checked, global: event.target.checked ? (binding.global || binding.regionIds.length === 0) : false, routingMode: event.target.checked && binding.regionIds.length === 0 ? "standard" : binding.routingMode })} /><span><strong>Use on this layer</strong></span></label>}
        {binding.enabled && <>
          <label className="check-row compact-check"><input type="checkbox" checked={binding.global} onChange={(event) => updateBinding(lora, { global: event.target.checked, regionIds: event.target.checked ? [] : binding.regionIds, routingMode: event.target.checked ? "standard" : binding.routingMode })} /><span><strong>Global</strong></span></label>
          {!binding.global && <div className="region-assignment-list">{regions.map((region) => <label className="check-row compact-check" key={region.id}><input type="checkbox" checked={binding.regionIds.includes(region.id)} onChange={(event) => updateBinding(lora, { regionIds: event.target.checked ? [...binding.regionIds, region.id] : binding.regionIds.filter((id) => id !== region.id) })} /><span>{region.name}</span></label>)}</div>}
          <label className="field-label">Routing</label><select className="select-input compact-select" value={binding.routingMode} disabled={binding.global} onChange={(event) => updateBinding(lora, { routingMode: event.target.value as LoraLayerBinding["routingMode"] })}><option value="standard">Standard regional</option><option value="character_identity">Character identity (face)</option></select>
          {binding.routingMode === "character_identity" && !binding.global && <><label className="field-label">Training trigger</label><input className="text-input compact-input" value={binding.triggerPhrase} placeholder="For example lface" onChange={(event) => updateBinding(lora, { triggerPhrase: event.target.value })} /><p className="field-help">Inserted automatically into the assigned region identity anchor; do not duplicate it in the visible prompt.</p></>}
        </>}
      </div>;
    })}
    {loras.length === 0 && <div className="drop-zone"><Icon name="upload" /><span>Add an uploaded `.safetensors` file from Cloud files.</span></div>}
  </div>;
}

function AdvancedPanel({ mode, activeLayer, settings, updateGeneration, updateEdit, updateFace, onChooseUpscaleModel, onPreviewUnifiedPrompt }: {
  mode: StudioMode; activeLayer: RegionLayer; settings: StudioSettings;
  updateGeneration: (patch: Partial<GenerationSettings>) => void;
  updateEdit: (patch: Partial<StudioSettings["edit"]>) => void;
  updateFace: (patch: Partial<StudioSettings["face"]>) => void;
  onChooseUpscaleModel: () => void;
  onPreviewUnifiedPrompt: () => void;
}) {
  const generation = settings.generation;
  const edit = settings.edit;
  const face = settings.face;
  if (mode === "face") return <div className="inspector-section advanced-panel">
    <NumberGrid items={[
      ["Steps", face.steps, 1, 100, 1, (steps) => updateFace({ steps })], ["Seed", face.seed, 0, 2147483647, 1, (seed) => updateFace({ seed })],
      ["Denoise", face.denoise, 0.05, 1, 0.05, (denoise) => updateFace({ denoise })], ["Padding", face.padding, 1, 4, 0.1, (padding) => updateFace({ padding })],
      ["Edge feather", face.feather, 0, 0.5, 0.02, (feather) => updateFace({ feather })], ["Blend", face.blend, 0, 1, 0.05, (blend) => updateFace({ blend })],
      ["Regional LoRA scale", face.loraScale, 0, 4, 0.05, (loraScale) => updateFace({ loraScale })], ["Detector threshold", face.detectorThreshold, 0.05, 0.95, 0.05, (detectorThreshold) => updateFace({ detectorThreshold })],
    ]} />
    <Choice label="Crop working resolution" value={face.cropSize} options={[[256, "256 px"], [512, "512 px"], [768, "768 px"], [1024, "1024 px"]]} onChange={(value) => updateFace({ cropSize: value as typeof face.cropSize })} />
    <Choice label="Detector device" value={face.detectorProvider} options={[["auto", "Auto (CUDA when available)"], ["cpu", "CPU"], ["cuda", "NVIDIA CUDA"]]} onChange={(value) => updateFace({ detectorProvider: value as typeof face.detectorProvider })} />
  </div>;
  const values = mode === "generation" ? generation : edit;
  const update = mode === "generation" ? updateGeneration : updateEdit;
  return <div className="inspector-section advanced-panel">
    <div className="settings-grid">
      <Choice label="Sampler" value={values.sampler} options={COMFYUI_SAMPLERS.map((value) => [value, value])} onChange={(sampler) => update({ sampler })} />
      <Choice label="Scheduler" value={values.scheduler} options={COMFYUI_SCHEDULERS.map((value) => [value, value])} onChange={(scheduler) => update({ scheduler })} />
    </div>
    <NumberGrid items={mode === "generation" ? [
      ["Steps", generation.steps, 1, 100, 1, (steps) => updateGeneration({ steps })], ["Seed", generation.seed, 0, 2147483647, 1, (seed) => updateGeneration({ seed })],
      ["Width", generation.width, 256, 4096, 16, (width) => updateGeneration({ width })], ["Height", generation.height, 256, 4096, 16, (height) => updateGeneration({ height })],
      ["Inside boost", generation.insideBoost, 0.1, 10, 0.1, (insideBoost) => updateGeneration({ insideBoost })], ["Outside penalty", generation.outsidePenalty, 0, 10, 0.1, (outsidePenalty) => updateGeneration({ outsidePenalty })],
      ["Spatial falloff", generation.spatialFalloff, 0, 2048, 16, (spatialFalloff) => updateGeneration({ spatialFalloff })], ["Late-step scale", generation.lateStepScale, 0, 1, 0.05, (lateStepScale) => updateGeneration({ lateStepScale })],
    ] : [
      ["Steps", edit.steps, 1, 100, 1, (steps) => updateEdit({ steps })], ["Seed · fixed", edit.seed, 0, 2147483647, 1, (seed) => updateEdit({ seed })],
      ["Denoise", edit.denoise, 0.05, 1, 0.05, (denoise) => updateEdit({ denoise })], ["Reference retention", edit.referenceRetention, 0, 1, 0.05, (referenceRetention) => updateEdit({ referenceRetention })],
      ["Latent feather", edit.latentFeather, 0, 256, 1, (latentFeather) => updateEdit({ latentFeather })], ["Composite feather", edit.compositeFeather, 0, 256, 1, (compositeFeather) => updateEdit({ compositeFeather })],
      ["Inside boost", edit.insideBoost, 0.1, 10, 0.1, (insideBoost) => updateEdit({ insideBoost })], ["Outside penalty", edit.outsidePenalty, 0, 10, 0.1, (outsidePenalty) => updateEdit({ outsidePenalty })],
      ["Spatial falloff", edit.spatialFalloff, 0, 2048, 1, (spatialFalloff) => updateEdit({ spatialFalloff })], ["Late-step scale", edit.lateStepScale, 0, 1, 0.01, (lateStepScale) => updateEdit({ lateStepScale })],
    ]} />
    {mode === "generation" && <Choice label="Seed behavior" value={generation.seedMode} options={generation.batchMode ? [["random", "Random"], ["increment", "Increment"]] : [["fixed", "Fixed"], ["random", "Random"], ["increment", "Increment"]]} onChange={(seedMode) => updateGeneration({ seedMode: seedMode as GenerationSettings["seedMode"] })} />}
    {mode === "generation" && <Check label="Run generation in batch mode" checked={generation.batchMode} onChange={(batchMode) => updateGeneration({ batchMode, seedMode: batchMode && generation.seedMode === "fixed" ? "random" : generation.seedMode })} />}
    {mode === "generation" && generation.batchMode && <LinkedValue label="Batch runs" value={generation.batchCount} min={1} max={100} step={1} onChange={(batchCount) => updateGeneration({ batchCount })} />}
    {mode === "generation" && <Check label="Use unified spatial prompting" checked={generation.regionalPrompting} onChange={(regionalPrompting) => updateGeneration({ regionalPrompting })} />}
    {mode === "generation" && <button className="quiet-button full-button" onClick={onPreviewUnifiedPrompt}>Preview unified prompt…</button>}
    <Check label="Separate overlapping subject targets" checked={values.subjectCompetition} onChange={(subjectCompetition) => update({ subjectCompetition })} />
    <Check label="Make subjects fill their boxes" checked={values.subjectFill} onChange={(subjectFill) => update({ subjectFill })} />
    {mode === "generation" && <Check label="Relax spatial guidance during late steps" checked={generation.relaxation} onChange={(relaxation) => updateGeneration({ relaxation })} />}
    <Check label="Adapt spatial guidance from regional LoRA delta" checked={values.loraAdaptation} onChange={(loraAdaptation) => update({ loraAdaptation })} />
    {values.loraAdaptation && <LinkedValue label="LoRA delta response" value={values.loraResponse} min={0} max={1} step={0.05} onChange={(loraResponse) => update({ loraResponse })} />}
    {mode === "edit" && <><Check label="Preserve reference identity" checked={edit.preserveIdentity} onChange={(preserveIdentity) => updateEdit({ preserveIdentity })} /><Check label="Edit entire image" checked={edit.editEntireImage} onChange={(editEntireImage) => updateEdit({ editEntireImage })} /></>}
    {mode === "generation" && <><SectionTitle text="Post-upscale" /><Check label="Post-upscale after releasing Krea VRAM" checked={generation.postUpscale} onChange={(postUpscale) => updateGeneration({ postUpscale })} />{generation.postUpscale && <><Choice label="Output scale" value={generation.upscaleScale} options={[[2, "2×"], [4, "4×"]]} onChange={(upscaleScale) => updateGeneration({ upscaleScale: upscaleScale as 2 | 4 })} /><Choice label="Upscaler" value={generation.upscaleMethod} options={[["lanczos", "CPU Lanczos"], ["model", "Neural model (tiled GPU)"]]} onChange={(upscaleMethod) => updateGeneration({ upscaleMethod: upscaleMethod as GenerationSettings["upscaleMethod"] })} />{generation.upscaleMethod === "model" && <button className="quiet-button full-button" onClick={onChooseUpscaleModel}>{generation.upscaleModelName || "Choose cloud upscaler model…"}</button>}</>}</>}
    {(mode === "generation" || activeLayer === "reference") && <ProjectorPanel projector={mode === "generation" ? generation.projector : edit.referenceProjector} onChange={(projector) => mode === "generation" ? updateGeneration({ projector }) : updateEdit({ referenceProjector: projector })} />}
  </div>;
}

function ProjectorPanel({ projector, onChange }: { projector: ProjectorSettings; onChange: (value: ProjectorSettings) => void }) {
  return <><SectionTitle text="Projector" /><Check label="Apply global projector vector" checked={projector.enabled} onChange={(enabled) => onChange({ ...projector, enabled })} />{projector.enabled && <>
    <Choice label="Preset" value={projector.preset} options={[["filter_bypass2", "FilterBypass2"], ["filter_bypass3", "FilterBypass3"], ["skc3vo", "skc3vo"], ["z0jglf", "z0jglf"], ["custom", "Custom values"]]} onChange={(preset) => onChange({ ...projector, preset, values: PROJECTOR_PRESETS[preset] ? [...PROJECTOR_PRESETS[preset]] : projector.values })} />
    <div className="projector-grid">{projector.values.map((value, index) => <input key={index} aria-label={`Projector vector ${index + 1}`} type="number" step="0.0001" min="-1000" max="1000" value={value} onChange={(event) => { const values = [...projector.values]; values[index] = Number(event.target.value); onChange({ ...projector, preset: "custom", values }); }} />)}</div>
    <LinkedValue label="Global multiplier" value={projector.multiplier} min={-20} max={20} step={0.1} onChange={(multiplier) => onChange({ ...projector, multiplier })} />
    <LinkedValue label="Face identity protection" value={projector.identityProtection} min={0} max={1} step={0.05} onChange={(identityProtection) => onChange({ ...projector, identityProtection })} />
  </>}</>;
}

type NumberItem = [string, number, number, number, number, (value: number) => void];
function NumberGrid({ items }: { items: NumberItem[] }) { return <div className="settings-grid">{items.map(([label, value, min, max, step, onChange]) => <LinkedValue key={label} label={label} value={value} min={min} max={max} step={step} onChange={onChange} />)}</div>; }
function Choice({ label, value, options, onChange }: { label: string; value: string | number; options: readonly (readonly [string | number, string])[]; onChange: (value: never) => void }) { return <label className="choice-field"><span>{label}</span><select className="select-input" value={value} onChange={(event) => { const match = options.find(([candidate]) => String(candidate) === event.target.value); if (match) onChange(match[0] as never); }}>{options.map(([option, text]) => <option key={option} value={option}>{text}</option>)}</select></label>; }
function Check({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) { return <label className="check-row compact-check"><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><span><strong>{label}</strong></span></label>; }
function SectionTitle({ text }: { text: string }) { return <div className="advanced-section-title">{text}</div>; }
function LinkedValue({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (value: number) => void }) {
  function change(next: number) { if (Number.isFinite(next)) onChange(Math.max(min, Math.min(max, next))); }
  return <div className="linked-value"><div className="linked-label"><span>{label}</span><input type="number" value={value} min={min} max={max} step={step} onChange={(event) => change(Number(event.target.value))} /></div><input className="range-input" type="range" value={value} min={min} max={max} step={step} onChange={(event) => change(Number(event.target.value))} /></div>;
}
