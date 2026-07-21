# Local GUI feature parity

The Qt Quick workspace is a presentation layer over the same project state and worker
controller as the original Qt Widgets interface. A UI refresh must not remove a project field
or make an existing workflow impossible to edit.

The current Qt Quick interface exposes the original workflow controls for:

- global, regional, edit, reference, and face-identity prompts, including overflowing-text
  scrollbars;
- exact global and regional phrase emphasis, strength editing, removal, saved-match validation,
  and legacy project reload;
- region creation, geometry, deletion, naming, spatial role, enablement, and front/back order;
- LoRA loading, activation, strength, global or multi-region scope, standard or character-
  identity routing, identity triggers, removal, and Krea compatibility diagnostics;
- unified spatial prompting, inside/outside guidance, falloff, subject separation, subject fill,
  late-step relaxation, and LoRA-delta adaptation;
- fixed/random/increment seeds, batch generation, post-upscaling, and unified-prompt preview;
- projector presets, all 12 custom vector values, multiplier, and face-identity protection;
- image-edit sampling, retention, feathering, subject behavior, identity preservation, whole-image
  editing, and LoRA adaptation;
- face detection device/threshold, crop size/padding, denoise, feather, blend, regional LoRA scale,
  latest-first-pass selection, and manual lasso actions;
- runtime/model paths, memory policy, output defaults, worker/model actions, diagnostics, and GPU
  memory release.

`uv run k2lab --legacy-widgets` remains available as an exact compatibility interface. Automated
Qt tests load the new interface offscreen, exercise overflowing prompt scrollbars, round-trip a
saved phrase-emphasis/character-LoRA workflow, and assert controller coverage for every legacy
workflow setting.
