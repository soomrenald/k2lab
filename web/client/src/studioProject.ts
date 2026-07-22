import type { RegionBox, RegionLayer } from "./components/RegionCanvas";

export type SeedMode = "fixed" | "random" | "increment";
export type LoraRoutingMode = "standard" | "character_identity";

export const COMFYUI_SAMPLERS = [
  "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun",
  "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral",
  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral",
  "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m",
  "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun",
  "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
  "ipndm", "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp",
  "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp", "gradient_estimation",
  "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3", "sa_solver",
  "sa_solver_pece", "ddim", "uni_pc", "uni_pc_bh2",
] as const;

export const COMFYUI_SCHEDULERS = [
  "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal",
  "linear_quadratic", "kl_optimal",
] as const;

export const PROJECTOR_PRESETS: Record<string, number[]> = {
  filter_bypass2: [0, 0, 0, 0, 0, 0, 0, 0, -0.5117, -0.8906, 0, 0],
  filter_bypass3: [0, 0, 0, 0, 0, 0, 0, 0, -0.5117, -0.8906, -0.6094, 0],
  skc3vo: [-5.44, -16.11, -37.11, -50.39, -70.7, -39.45, -39.84, -143.7511, -51.17, -89.06, -60.94, -11.28],
  z0jglf: [-13.6, -40.275, -92.775, -159.75, -176.75, -98.625, -99.6, -359.3778, -127.925, -222.65, -152.35, -28.2],
};

export interface PromptEmphasisState {
  id: string;
  scopeId: "__global__" | string;
  phrase: string;
  strength: number;
  occurrence: number;
}

export interface ProjectorSettings {
  enabled: boolean;
  preset: string;
  values: number[];
  multiplier: number;
  identityProtection: number;
}

export interface GenerationSettings {
  width: number;
  height: number;
  steps: number;
  sampler: string;
  scheduler: string;
  seed: number;
  seedMode: SeedMode;
  batchMode: boolean;
  batchCount: number;
  regionalPrompting: boolean;
  insideBoost: number;
  outsidePenalty: number;
  spatialFalloff: number;
  subjectCompetition: boolean;
  subjectFill: boolean;
  relaxation: boolean;
  lateStepScale: number;
  loraAdaptation: boolean;
  loraResponse: number;
  postUpscale: boolean;
  upscaleScale: 2 | 4;
  upscaleMethod: "lanczos" | "model";
  upscaleModelFileId: string;
  upscaleModelName: string;
  projector: ProjectorSettings;
  promptEmphases: PromptEmphasisState[];
}

export interface EditSettings {
  width: number;
  height: number;
  steps: number;
  sampler: string;
  scheduler: string;
  seed: number;
  denoise: number;
  latentFeather: number;
  compositeFeather: number;
  referenceRetention: number;
  insideBoost: number;
  outsidePenalty: number;
  spatialFalloff: number;
  subjectCompetition: boolean;
  subjectFill: boolean;
  lateStepScale: number;
  loraAdaptation: boolean;
  loraResponse: number;
  preserveIdentity: boolean;
  editEntireImage: boolean;
  referencePromptEmphases: PromptEmphasisState[];
  referenceProjector: ProjectorSettings;
}

export interface FaceSettings {
  steps: number;
  seed: number;
  denoise: number;
  cropSize: 256 | 512 | 768 | 1024;
  padding: number;
  feather: number;
  blend: number;
  loraScale: number;
  detectorThreshold: number;
  detectorProvider: "auto" | "cpu" | "cuda";
}

export interface RuntimeSettings {
  filenamePrefix: string;
  diffusionModelFileId: string;
  diffusionModelName: string;
  textEncoderFileId: string;
  textEncoderName: string;
  vaeFileId: string;
  vaeName: string;
  faceDetectorFileId: string;
  faceDetectorName: string;
}

export interface LoraLayerBinding {
  enabled: boolean;
  global: boolean;
  regionIds: string[];
  routingMode: LoraRoutingMode;
  triggerPhrase: string;
}

export interface StudioLora {
  id: string;
  fileId: string;
  name: string;
  active: boolean;
  strength: number;
  generation: LoraLayerBinding;
  reference: LoraLayerBinding;
  targets: LoraLayerBinding;
}

export interface StudioSettings {
  generation: GenerationSettings;
  edit: EditSettings;
  face: FaceSettings;
  runtime: RuntimeSettings;
}

const defaultProjectorValues = [0, 0, 0, 0, 0, 0, 0, 0, -0.5117, -0.8906, 0, 0];

function defaultProjector(): ProjectorSettings {
  return {
    enabled: false,
    preset: "filter_bypass2",
    values: [...defaultProjectorValues],
    multiplier: 1,
    identityProtection: 1,
  };
}

export function createStudioSettings(): StudioSettings {
  return {
    generation: {
      width: 1024,
      height: 1024,
      steps: 8,
      sampler: "euler",
      scheduler: "simple",
      seed: 0,
      seedMode: "fixed",
      batchMode: false,
      batchCount: 2,
      regionalPrompting: true,
      insideBoost: 1,
      outsidePenalty: 1,
      spatialFalloff: 128,
      subjectCompetition: true,
      subjectFill: true,
      relaxation: true,
      lateStepScale: 0.35,
      loraAdaptation: false,
      loraResponse: 0.35,
      postUpscale: false,
      upscaleScale: 2,
      upscaleMethod: "lanczos",
      upscaleModelFileId: "",
      upscaleModelName: "",
      projector: defaultProjector(),
      promptEmphases: [],
    },
    edit: {
      width: 1024,
      height: 1024,
      steps: 8,
      sampler: "euler",
      scheduler: "simple",
      seed: 0,
      denoise: 0.15,
      latentFeather: 64,
      compositeFeather: 48,
      referenceRetention: 1,
      insideBoost: 1,
      outsidePenalty: 1,
      spatialFalloff: 128,
      subjectCompetition: true,
      subjectFill: true,
      lateStepScale: 0.35,
      loraAdaptation: false,
      loraResponse: 0.35,
      preserveIdentity: true,
      editEntireImage: false,
      referencePromptEmphases: [],
      referenceProjector: defaultProjector(),
    },
    face: {
      steps: 8,
      seed: 0,
      denoise: 0.15,
      cropSize: 512,
      padding: 2,
      feather: 0.12,
      blend: 0.5,
      loraScale: 0.5,
      detectorThreshold: 0.15,
      detectorProvider: "auto",
    },
    runtime: {
      filenamePrefix: "baseline",
      diffusionModelFileId: "",
      diffusionModelName: "",
      textEncoderFileId: "",
      textEncoderName: "",
      vaeFileId: "",
      vaeName: "",
      faceDetectorFileId: "",
      faceDetectorName: "",
    },
  };
}

export function createStudioLora(fileId: string, name: string): StudioLora {
  const inactive = (): LoraLayerBinding => ({
    enabled: false,
    global: false,
    regionIds: [],
    routingMode: "standard",
    triggerPhrase: "",
  });
  return {
    id: crypto.randomUUID(),
    fileId,
    name,
    active: true,
    strength: 1,
    generation: { ...inactive(), enabled: true, global: true },
    reference: inactive(),
    targets: inactive(),
  };
}

export function buildProjectDocument(
  regions: RegionBox[],
  prompts: Record<RegionLayer, string>,
  settings: StudioSettings,
  loras: StudioLora[],
  sourceName: string | null = null,
): Record<string, unknown> {
  const generation = settings.generation;
  const edit = settings.edit;
  const face = settings.face;
  const runtime = settings.runtime;
  return {
    schema: "k2-region-lab-project",
    version: 18,
    canvas: { width: generation.width, height: generation.height },
    generation: {
      global_prompt: prompts.generation,
      steps: generation.steps,
      sampler: generation.sampler,
      scheduler: generation.scheduler,
      seed: generation.seed,
      seed_mode: generation.seedMode,
      batch_mode: generation.batchMode,
      batch_count: generation.batchCount,
      regional_prompting: generation.regionalPrompting,
      regional_prompt_strength: generation.insideBoost,
      regional_outside_penalty: generation.outsidePenalty,
      regional_feather_pixels: generation.spatialFalloff,
      regional_subject_competition: generation.subjectCompetition,
      regional_subject_fill: generation.subjectFill,
      regional_relaxation: generation.relaxation,
      regional_late_step_scale: generation.lateStepScale,
      regional_lora_delta_adaptation: generation.loraAdaptation,
      regional_lora_delta_adaptation_gain: generation.loraResponse,
      prompt_emphases: emphasisDocuments(generation.promptEmphases),
      projector_enabled: generation.projector.enabled,
      projector_preset: generation.projector.preset,
      projector_values: generation.projector.values,
      projector_multiplier: generation.projector.multiplier,
      projector_identity_protection: generation.projector.identityProtection,
      face_detail_seed: face.seed,
      face_detail_steps: face.steps,
      face_detail_denoise: face.denoise,
      face_detail_crop_size: face.cropSize,
      face_detail_padding: face.padding,
      face_detail_feather: face.feather,
      face_detail_blend: face.blend,
      face_detail_lora_scale: face.loraScale,
      face_detail_detector_threshold: face.detectorThreshold,
      face_detail_detector_provider: face.detectorProvider,
      post_upscale: generation.postUpscale,
      upscale_scale: generation.upscaleScale,
      upscale_method: generation.upscaleMethod,
      upscale_model: generation.upscaleModelName || null,
    },
    regions: layerRegions(regions, "generation"),
    loras: loras.map(loraDocument),
    image_edit: {
      source_image: sourceName,
      associated_project: null,
      width: edit.width,
      height: edit.height,
      reference_global_prompt: prompts.reference,
      reference_prompt_emphases: emphasisDocuments(edit.referencePromptEmphases),
      reference_projector_enabled: edit.referenceProjector.enabled,
      reference_projector_preset: edit.referenceProjector.preset,
      reference_projector_values: edit.referenceProjector.values,
      reference_projector_multiplier: edit.referenceProjector.multiplier,
      reference_projector_identity_protection: edit.referenceProjector.identityProtection,
      global_prompt: prompts.targets,
      steps: edit.steps,
      sampler: edit.sampler,
      scheduler: edit.scheduler,
      seed: edit.seed,
      denoise: edit.denoise,
      latent_feather_pixels: edit.latentFeather,
      composite_feather_pixels: edit.compositeFeather,
      edit_entire_image: edit.editEntireImage,
      preserve_identity: edit.preserveIdentity,
      reference_description_retention: edit.referenceRetention,
      regional_prompt_strength: edit.insideBoost,
      regional_outside_penalty: edit.outsidePenalty,
      regional_feather_pixels: edit.spatialFalloff,
      regional_subject_competition: edit.subjectCompetition,
      regional_subject_fill: edit.subjectFill,
      regional_late_step_scale: edit.lateStepScale,
      regional_lora_delta_adaptation: edit.loraAdaptation,
      regional_lora_delta_adaptation_gain: edit.loraResponse,
      regions: layerRegions(regions, "targets"),
      reference_regions: layerRegions(regions, "reference"),
    },
    runtime: {
      filename_prefix: runtime.filenamePrefix,
      diffusion_model_file: runtime.diffusionModelName || null,
      text_encoder_file: runtime.textEncoderName || null,
      vae_file: runtime.vaeName || null,
      face_detector_path: runtime.faceDetectorName || null,
    },
    background_image: sourceName,
  };
}

function emphasisDocuments(items: PromptEmphasisState[]) {
  return items.map(({ scopeId, phrase, strength, occurrence }) => ({
    scope_id: scopeId,
    phrase,
    strength,
    occurrence,
  }));
}

function layerRegions(regions: RegionBox[], layer: RegionLayer) {
  const selected = regions.filter((region) => region.layer === layer);
  return selected.map((region, index) => ({
    id: region.id,
    name: region.name,
    box: {
      x0: region.x,
      y0: region.y,
      x1: region.x + region.width,
      y1: region.y + region.height,
    },
    prompt: region.prompt,
    face_identity_prompt: region.faceIdentityPrompt,
    enabled: region.enabled,
    priority: selected.length - index,
    spatial_role: region.spatialRole,
  }));
}

function loraDocument(lora: StudioLora) {
  return {
    path: lora.name,
    global: lora.generation.global,
    region_ids: lora.generation.regionIds,
    strength: lora.active ? lora.strength : 0,
    routing_mode: lora.generation.routingMode,
    trigger_phrase: lora.generation.triggerPhrase,
    image_edit: {
      enabled: lora.targets.enabled,
      global: lora.targets.global,
      region_ids: lora.targets.regionIds,
      routing_mode: lora.targets.routingMode,
      trigger_phrase: lora.targets.triggerPhrase,
    },
    image_edit_reference: {
      enabled: lora.reference.enabled,
      global: lora.reference.global,
      region_ids: lora.reference.regionIds,
      routing_mode: lora.reference.routingMode,
      trigger_phrase: lora.reference.triggerPhrase,
    },
  };
}

export interface LoadedStudioProject {
  regions: RegionBox[];
  prompts: Record<RegionLayer, string>;
  settings: StudioSettings;
  loras: StudioLora[];
  sourceName: string;
}

type JsonObject = Record<string, unknown>;

export function loadStudioProjectDocument(value: unknown): LoadedStudioProject {
  const document = objectValue(value);
  if (document.schema !== "k2-region-lab-project") throw new Error("Not a K2 Region Lab project");
  if (document.version !== 18) throw new Error(`Unsupported project version: ${String(document.version)}`);
  const canvas = objectValue(document.canvas);
  const generation = objectValue(document.generation);
  const edit = objectValue(document.image_edit);
  const runtime = objectValue(document.runtime);
  const settings = createStudioSettings();
  const width = integerValue(canvas.width, settings.generation.width);
  const height = integerValue(canvas.height, settings.generation.height);
  const projectorValues = numberList(generation.projector_values, settings.generation.projector.values);
  const referenceProjectorValues = numberList(edit.reference_projector_values, settings.edit.referenceProjector.values);
  settings.generation = {
    ...settings.generation,
    width,
    height,
    steps: integerValue(generation.steps, settings.generation.steps),
    sampler: stringValue(generation.sampler, settings.generation.sampler),
    scheduler: stringValue(generation.scheduler, settings.generation.scheduler),
    seed: integerValue(generation.seed, settings.generation.seed),
    seedMode: seedModeValue(generation.seed_mode, settings.generation.seedMode),
    batchMode: booleanValue(generation.batch_mode, settings.generation.batchMode),
    batchCount: integerValue(generation.batch_count, settings.generation.batchCount),
    regionalPrompting: booleanValue(generation.regional_prompting, settings.generation.regionalPrompting),
    insideBoost: numberValue(generation.regional_prompt_strength, settings.generation.insideBoost),
    outsidePenalty: numberValue(generation.regional_outside_penalty, settings.generation.outsidePenalty),
    spatialFalloff: numberValue(generation.regional_feather_pixels, settings.generation.spatialFalloff),
    subjectCompetition: booleanValue(generation.regional_subject_competition, settings.generation.subjectCompetition),
    subjectFill: booleanValue(generation.regional_subject_fill, settings.generation.subjectFill),
    relaxation: booleanValue(generation.regional_relaxation, settings.generation.relaxation),
    lateStepScale: numberValue(generation.regional_late_step_scale, settings.generation.lateStepScale),
    loraAdaptation: booleanValue(generation.regional_lora_delta_adaptation, settings.generation.loraAdaptation),
    loraResponse: numberValue(generation.regional_lora_delta_adaptation_gain, settings.generation.loraResponse),
    promptEmphases: emphasisStates(generation.prompt_emphases),
    postUpscale: booleanValue(generation.post_upscale, settings.generation.postUpscale),
    upscaleScale: generation.upscale_scale === 4 ? 4 : 2,
    upscaleMethod: generation.upscale_method === "model" ? "model" : "lanczos",
    upscaleModelName: basename(stringValue(generation.upscale_model, "")),
    upscaleModelFileId: "",
    projector: {
      enabled: booleanValue(generation.projector_enabled, false),
      preset: stringValue(generation.projector_preset, settings.generation.projector.preset),
      values: projectorValues,
      multiplier: numberValue(generation.projector_multiplier, 1),
      identityProtection: numberValue(generation.projector_identity_protection, 1),
    },
  };
  settings.edit = {
    ...settings.edit,
    width: integerValue(edit.width, width),
    height: integerValue(edit.height, height),
    steps: integerValue(edit.steps, settings.edit.steps),
    sampler: stringValue(edit.sampler, settings.edit.sampler),
    scheduler: stringValue(edit.scheduler, settings.edit.scheduler),
    seed: integerValue(edit.seed, settings.edit.seed),
    denoise: numberValue(edit.denoise, settings.edit.denoise),
    latentFeather: integerValue(edit.latent_feather_pixels, settings.edit.latentFeather),
    compositeFeather: integerValue(edit.composite_feather_pixels, settings.edit.compositeFeather),
    referenceRetention: numberValue(edit.reference_description_retention, settings.edit.referenceRetention),
    insideBoost: numberValue(edit.regional_prompt_strength, settings.edit.insideBoost),
    outsidePenalty: numberValue(edit.regional_outside_penalty, settings.edit.outsidePenalty),
    spatialFalloff: integerValue(edit.regional_feather_pixels, settings.edit.spatialFalloff),
    subjectCompetition: booleanValue(edit.regional_subject_competition, settings.edit.subjectCompetition),
    subjectFill: booleanValue(edit.regional_subject_fill, settings.edit.subjectFill),
    lateStepScale: numberValue(edit.regional_late_step_scale, settings.edit.lateStepScale),
    loraAdaptation: booleanValue(edit.regional_lora_delta_adaptation, settings.edit.loraAdaptation),
    loraResponse: numberValue(edit.regional_lora_delta_adaptation_gain, settings.edit.loraResponse),
    preserveIdentity: booleanValue(edit.preserve_identity, settings.edit.preserveIdentity),
    editEntireImage: booleanValue(edit.edit_entire_image, settings.edit.editEntireImage),
    referencePromptEmphases: emphasisStates(edit.reference_prompt_emphases),
    referenceProjector: {
      enabled: booleanValue(edit.reference_projector_enabled, false),
      preset: stringValue(edit.reference_projector_preset, settings.edit.referenceProjector.preset),
      values: referenceProjectorValues,
      multiplier: numberValue(edit.reference_projector_multiplier, 1),
      identityProtection: numberValue(edit.reference_projector_identity_protection, 1),
    },
  };
  settings.face = {
    steps: integerValue(generation.face_detail_steps, settings.face.steps),
    seed: integerValue(generation.face_detail_seed, settings.face.seed),
    denoise: numberValue(generation.face_detail_denoise, settings.face.denoise),
    cropSize: cropSizeValue(generation.face_detail_crop_size),
    padding: numberValue(generation.face_detail_padding, settings.face.padding),
    feather: numberValue(generation.face_detail_feather, settings.face.feather),
    blend: numberValue(generation.face_detail_blend, settings.face.blend),
    loraScale: numberValue(generation.face_detail_lora_scale, settings.face.loraScale),
    detectorThreshold: numberValue(generation.face_detail_detector_threshold, settings.face.detectorThreshold),
    detectorProvider: detectorProviderValue(generation.face_detail_detector_provider),
  };
  settings.runtime = {
    filenamePrefix: stringValue(runtime.filename_prefix, settings.runtime.filenamePrefix),
    diffusionModelFileId: "",
    diffusionModelName: basename(stringValue(runtime.diffusion_model_file, "")),
    textEncoderFileId: "",
    textEncoderName: basename(stringValue(runtime.text_encoder_file, "")),
    vaeFileId: "",
    vaeName: basename(stringValue(runtime.vae_file, "")),
    faceDetectorFileId: "",
    faceDetectorName: basename(stringValue(runtime.face_detector_path, "")),
  };
  return {
    settings,
    sourceName: basename(stringValue(edit.source_image, stringValue(document.background_image, ""))),
    prompts: {
      generation: stringValue(generation.global_prompt, ""),
      reference: stringValue(edit.reference_global_prompt, ""),
      targets: stringValue(edit.global_prompt, ""),
    },
    regions: [
      ...regionStates(document.regions, "generation"),
      ...regionStates(edit.reference_regions, "reference"),
      ...regionStates(edit.regions, "targets"),
    ],
    loras: arrayValue(document.loras).map((item) => loraState(objectValue(item))),
  };
}

export async function projectDocumentFromPng(file: Blob): Promise<unknown> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const signature = [137, 80, 78, 71, 13, 10, 26, 10];
  if (signature.some((value, index) => bytes[index] !== value)) throw new Error("Project image must be a PNG file");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let offset = 8; offset + 12 <= bytes.length;) {
    const length = view.getUint32(offset);
    const type = new TextDecoder("ascii").decode(bytes.subarray(offset + 4, offset + 8));
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    if (dataEnd + 4 > bytes.length) throw new Error("PNG metadata is truncated");
    if (type === "tEXt") {
      const data = bytes.subarray(dataStart, dataEnd);
      const separator = data.indexOf(0);
      if (separator >= 0) {
        const key = new TextDecoder("latin1").decode(data.subarray(0, separator));
        if (key === "k2lab_project") {
          return JSON.parse(new TextDecoder("latin1").decode(data.subarray(separator + 1)));
        }
      }
    }
    if (type === "iTXt") {
      const data = bytes.subarray(dataStart, dataEnd);
      const keywordEnd = data.indexOf(0);
      if (keywordEnd >= 0 && new TextDecoder("latin1").decode(data.subarray(0, keywordEnd)) === "k2lab_project") {
        const compressionFlag = data[keywordEnd + 1];
        let cursor = keywordEnd + 3;
        for (let field = 0; field < 2; field += 1) {
          const end = data.indexOf(0, cursor);
          if (end < 0) throw new Error("PNG project metadata is malformed");
          cursor = end + 1;
        }
        if (compressionFlag !== 0) throw new Error("Compressed PNG project metadata is not supported");
        return JSON.parse(new TextDecoder("utf-8").decode(data.subarray(cursor)));
      }
    }
    offset = dataEnd + 4;
  }
  throw new Error("PNG does not contain K2 Region Lab project metadata");
}

function regionStates(value: unknown, layer: RegionLayer): RegionBox[] {
  return arrayValue(value).map((item, index) => {
    const region = objectValue(item);
    const box = objectValue(region.box);
    const x = numberValue(box.x0, 0);
    const y = numberValue(box.y0, 0);
    return {
      id: stringValue(region.id, crypto.randomUUID()),
      name: stringValue(region.name, `Region ${index + 1}`),
      layer,
      x,
      y,
      width: numberValue(box.x1, x + 16) - x,
      height: numberValue(box.y1, y + 16) - y,
      prompt: stringValue(region.prompt, ""),
      faceIdentityPrompt: stringValue(region.face_identity_prompt, ""),
      spatialRole: spatialRoleValue(region.spatial_role),
      enabled: booleanValue(region.enabled, true),
      priority: integerValue(region.priority, 0),
    };
  }).sort((left, right) => (right.priority ?? 0) - (left.priority ?? 0)).map(({ priority: _priority, ...region }) => region);
}

function loraState(item: JsonObject): StudioLora {
  const edit = objectValue(item.image_edit);
  const reference = objectValue(item.image_edit_reference);
  const path = stringValue(item.path, "LoRA.safetensors");
  const binding = (value: JsonObject, enabled: boolean): LoraLayerBinding => ({
    enabled,
    global: booleanValue(value.global, false),
    regionIds: stringList(value.region_ids),
    routingMode: routingModeValue(value.routing_mode),
    triggerPhrase: stringValue(value.trigger_phrase, ""),
  });
  const strength = numberValue(item.strength, 1);
  return {
    id: crypto.randomUUID(),
    fileId: "",
    name: basename(path),
    active: strength !== 0,
    strength: strength === 0 ? 1 : strength,
    generation: binding(item, true),
    targets: binding(edit, booleanValue(edit.enabled, false)),
    reference: binding(reference, booleanValue(reference.enabled, false)),
  };
}

function emphasisStates(value: unknown): PromptEmphasisState[] {
  return arrayValue(value).map((entry) => {
    const item = objectValue(entry);
    return {
      id: crypto.randomUUID(),
      scopeId: stringValue(item.scope_id, "__global__"),
      phrase: stringValue(item.phrase, ""),
      strength: numberValue(item.strength, 0.5),
      occurrence: integerValue(item.occurrence, 0),
    };
  });
}

function objectValue(value: unknown): JsonObject { return value !== null && typeof value === "object" && !Array.isArray(value) ? value as JsonObject : {}; }
function arrayValue(value: unknown): unknown[] { return Array.isArray(value) ? value : []; }
function stringValue(value: unknown, fallback: string): string { return typeof value === "string" ? value : fallback; }
function numberValue(value: unknown, fallback: number): number { return typeof value === "number" && Number.isFinite(value) ? value : fallback; }
function integerValue(value: unknown, fallback: number): number { return Math.trunc(numberValue(value, fallback)); }
function booleanValue(value: unknown, fallback: boolean): boolean { return typeof value === "boolean" ? value : fallback; }
function stringList(value: unknown): string[] { return arrayValue(value).filter((item): item is string => typeof item === "string"); }
function numberList(value: unknown, fallback: number[]): number[] { const values = arrayValue(value); return values.length === 12 && values.every((item) => typeof item === "number" && Number.isFinite(item)) ? values as number[] : [...fallback]; }
function basename(path: string): string { return path.split(/[\\/]/).pop() ?? path; }
function seedModeValue(value: unknown, fallback: SeedMode): SeedMode { return value === "random" || value === "increment" || value === "fixed" ? value : fallback; }
function spatialRoleValue(value: unknown): RegionBox["spatialRole"] { return value === "subject" || value === "background" ? value : "auto"; }
function routingModeValue(value: unknown): LoraRoutingMode { return value === "character_identity" ? value : "standard"; }
function cropSizeValue(value: unknown): FaceSettings["cropSize"] { return value === 256 || value === 768 || value === 1024 ? value : 512; }
function detectorProviderValue(value: unknown): FaceSettings["detectorProvider"] { return value === "cpu" || value === "cuda" ? value : "auto"; }
