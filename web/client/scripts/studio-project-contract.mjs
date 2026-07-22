import assert from "node:assert/strict";
import {
  buildProjectDocument,
  createStudioSettings,
  loadStudioProjectDocument,
  projectDocumentFromPng,
} from "../src/studioProject.ts";
import { appendBoundedEvents, EVENT_LOG_LIMIT } from "../src/eventLog.ts";

const settings = createStudioSettings();
settings.generation.seed = 8123;
settings.generation.seedMode = "increment";
settings.generation.batchMode = true;
settings.generation.batchCount = 4;
settings.generation.promptEmphases = [{
  id: "not-persisted", scopeId: "person", phrase: "red coat", strength: 1.2, occurrence: 0,
}];
settings.edit.width = 768;
settings.edit.height = 1152;
settings.edit.referencePromptEmphases = [{
  id: "not-persisted", scopeId: "__global__", phrase: "portrait", strength: 0.7, occurrence: 1,
}];
settings.face.cropSize = 768;
settings.runtime.filenamePrefix = "portrait study";
settings.runtime.diffusionModelName = "chosen-transformer.safetensors";
settings.runtime.textEncoderName = "chosen-text.safetensors";
settings.runtime.vaeName = "chosen-vae.safetensors";
settings.runtime.faceDetectorName = "chosen-detector.onnx";
const regions = [
  { id: "person", name: "Person", layer: "generation", x: 80, y: 40, width: 400, height: 900, prompt: "red coat", faceIdentityPrompt: "green eyes", spatialRole: "subject", enabled: true },
  { id: "wall", name: "Wall", layer: "generation", x: 0, y: 0, width: 1024, height: 1024, prompt: "brick wall", faceIdentityPrompt: "", spatialRole: "background", enabled: true },
  { id: "reference", name: "Reference", layer: "reference", x: 20, y: 30, width: 300, height: 700, prompt: "portrait", faceIdentityPrompt: "same person", spatialRole: "subject", enabled: true },
  { id: "target", name: "Target", layer: "targets", x: 350, y: 300, width: 200, height: 250, prompt: "blue jacket", faceIdentityPrompt: "", spatialRole: "subject", enabled: true },
];
const loras = [{
  id: "not-persisted",
  fileId: "opaque-cloud-id",
  name: "character.safetensors",
  active: true,
  strength: 0.85,
  generation: { enabled: true, global: false, regionIds: ["person"], routingMode: "character_identity", triggerPhrase: "lface" },
  reference: { enabled: true, global: false, regionIds: ["reference"], routingMode: "standard", triggerPhrase: "" },
  targets: { enabled: false, global: false, regionIds: [], routingMode: "standard", triggerPhrase: "" },
}];
const prompts = { generation: "studio portrait", reference: "portrait reference", targets: "change clothing" };
const first = buildProjectDocument(regions, prompts, settings, loras, "source.png");
const loaded = loadStudioProjectDocument(first);
const second = buildProjectDocument(loaded.regions, loaded.prompts, loaded.settings, loaded.loras, loaded.sourceName);
assert.deepEqual(second, first);
assert.equal(second.image_edit.width, 768);
assert.deepEqual(second.regions.map((region) => [region.id, region.priority, region.spatial_role]), [
  ["person", 2, "subject"], ["wall", 1, "background"],
]);

const encoded = new TextEncoder().encode(`k2lab_project\0${JSON.stringify(first)}`);
const chunk = new Uint8Array(12 + encoded.length);
new DataView(chunk.buffer).setUint32(0, encoded.length);
chunk.set(new TextEncoder().encode("tEXt"), 4);
chunk.set(encoded, 8);
const png = new Blob([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk]);
assert.deepEqual(await projectDocumentFromPng(png), first);

const events = appendBoundedEvents([], Array.from({ length: EVENT_LOG_LIMIT + 25 }, (_value, index) => index));
assert.equal(events.length, EVENT_LOG_LIMIT);
assert.equal(events[0], 25);
assert.equal(events.at(-1), EVENT_LOG_LIMIT + 24);

console.log("studio project JSON and PNG round-trip contracts passed");
