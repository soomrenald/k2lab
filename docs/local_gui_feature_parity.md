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
- fixed/random/increment seeds beside the seed value, the legacy fixed-seed batch guard,
  batch generation, post-upscaling, and unified-prompt preview;
- projector presets, all 12 custom vector values, multiplier, and face-identity protection;
- image-edit sampling, retention, feathering, subject behavior, identity preservation, whole-image
  editing, and LoRA adaptation;
- face detection device/threshold, crop size/padding, denoise, feather, blend, regional LoRA scale,
  latest-first-pass selection, and manual lasso actions;
- runtime/model paths, discovered Krea checkpoint selection, per-artifact status, memory policy,
  output defaults, worker/model actions, diagnostics, and GPU memory release;
- the Events history, current GPU VRAM/RAM/activity telemetry, and the original New/Open/Import/
  Save/Save As/Quit keyboard shortcuts.

## Drift prevention

The Qt Quick numeric fields and choice selectors obtain their minimum, maximum, precision,
step, labels, values, and enabled state from the corresponding legacy widget through
`QmlWorkspaceController.settingSpec()`. Bounds and option lists are not maintained as a second
hard-coded copy in QML.

Prompt editors synchronize on every text change and are flushed again before Run. This matches
the legacy `QTextEdit` behavior and prevents the visible text from getting ahead of the project
state used for phrase emphasis, project saving, or worker submission.

The unified prompt compiler is shared by both interfaces. Its source is byte-for-byte unchanged
from the commit immediately before the Qt Quick migration (`093ec16^`): mixed subject/background
clauses remain in front-to-back region priority order, role-specific wording is preserved, and
the same relationship clause and token spans are produced.

`uv run k2lab --legacy-widgets` remains available as an exact compatibility interface. Automated
Qt tests load the new interface offscreen, exercise overflowing prompt scrollbars, verify live
prompt synchronization and a golden mixed subject/background unified prompt, round-trip a saved
phrase-emphasis/character-LoRA workflow, assert every legacy numeric range and seed/crop choice,
and confirm the seed, role, checkpoint, Events, and telemetry controls exist in QML.
