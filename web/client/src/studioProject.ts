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
): Record<string, unknown> {
  const generation = settings.generation;
  const edit = settings.edit;
  const face = settings.face;
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
      source_image: null,
      associated_project: null,
      width: generation.width,
      height: generation.height,
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
    runtime: {},
    background_image: null,
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
