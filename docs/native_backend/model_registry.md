# K2Lab model registry

Phase 1 adds an inference-independent TOML registry. It is not selected by the worker and
does not load, map, or generate with model tensors.

## Format

```toml
schema_version = "k2lab-model-registry/1"
source = "k2lab"

[[models]]
name = "krea2_turbo_fp8_scaled"
architecture = "krea2"
default_dtype = "bfloat16"

[models.transformer]
path = "/models/krea2/krea2_turbo_fp8_scaled.safetensors"
sha256 = "64-lowercase-hex-characters"

[models.text_encoder]
path = "/models/krea2/qwen3vl_4b_fp8_scaled.safetensors"
sha256 = "64-lowercase-hex-characters"

[models.vae]
path = "/models/krea2/qwen_image_vae.safetensors"
sha256 = "64-lowercase-hex-characters"

[models.tokenizer]
path = "/models/krea2/qwen-tokenizer"
sha256 = "deterministic-directory-sha256"
```

Each model name must be unique without regard to case. Component paths may be anywhere on
the filesystem and may be symlinks. A valid model currently requires the `krea2`
architecture and exact transformer, Qwen3-VL encoder, and Qwen image VAE header
fingerprints.

The tokenizer entry is optional for discovery/loading but required for native prompt
encoding. Its hash covers every file using sorted relative paths and individual file
hashes. The required standalone Qwen tokenizer assets are `merges.txt`,
`tokenizer_config.json`, and `vocab.json`.

## Legacy scanner

The scanner uses the existing configured ComfyUI directories only as an import source. It
streams each component once to calculate SHA-256, reads the safetensors JSON header, and
prints configuration to stdout:

```bash
k2lab --scan-comfyui-models > /tmp/k2lab-models.toml
```

It does not create directories, configure a worker, write a log, copy weights, update
metadata, or change model timestamps or bytes. If any required shared component cannot be
discovered, it exits nonzero without printing a partial registry.

## Standalone validation

```bash
k2lab --validate-model-registry /tmp/k2lab-models.toml
```

Validation:

- resolves every path and symlink;
- requires regular safetensors files;
- reads headers without reading tensor payloads into memory;
- checks required tensor names, shapes, and architecture layer counts;
- streams SHA-256 and compares it with the registry;
- reports every component as `PASS` or `FAIL`;
- exits `0` only if every registered model and component is valid.

This command does not read the desktop configuration, start Qt, construct an inference
backend, or access a GPU.

For native worker loading, configure the emitted registry and one model name:

```toml
[models]
registry = "/absolute/path/to/k2lab-models.toml"
registered_model = "krea2_turbo_fp8_scaled"
```

The equivalent environment variables are `K2LAB_MODEL_REGISTRY` and
`K2LAB_REGISTERED_MODEL`. These settings are ignored by the default ComfyUI backend.
