# Model use and distribution policy

Last reviewed: 2026-07-30

K2Lab is noncommercial open-source software licensed under Apache-2.0. The software
license does not license any model, LoRA, detector, or upscaler weights.

## No model distribution

K2Lab source releases, wheels, installers, and container images must not bundle,
download, cache, mirror, or redistribute model weights. This includes:

- Krea 2 Raw or Turbo transformers and conversions;
- Qwen text encoders, including the reviewed FP8 conversion;
- Qwen Image VAEs;
- LoRAs;
- face detectors; and
- neural upscalers.

Users and deployment operators must obtain these assets from their authorized source
and configure local paths. Registry hashes identify compatible files but do not grant
permission to obtain or use them.

The converted Qwen FP8 text encoder has no separately documented conversion and notice
chain in the reviewed mirror. It must not be redistributed by this project unless that
chain is confirmed and all required Apache-2.0 notices are preserved.

## Krea terms and acceptance

Krea 2 weights are governed by the
[Krea 2 Community License Agreement](https://www.krea.ai/krea-2-licensing) and
[Krea Acceptable Use Policy](https://www.krea.ai/krea-2-use-policy), not Apache-2.0.
Obtaining, configuring, or using Krea weights is the operator's direct interaction with
those terms; K2Lab does not accept them on the operator's behalf. An operator who does
not accept the current terms must not configure or use Krea weights.

The project itself is maintained and released noncommercially. Downstream users remain
responsible for determining whether their own use is commercial and, if so, whether
they qualify for Krea's community terms or require an enterprise license.

## Deployment safeguards

K2Lab does not provide a public hosted inference service. The tested desktop and RunPod
configurations are private, single-operator workspaces in which the operator reviews
prompts and outputs. They must not be exposed as public or shared generation services
without content filtering or an equivalent review process appropriate to the use case
and the current Krea license and Acceptable Use Policy.

Operators must not use Krea weights, derivatives, or outputs for prohibited purposes.
They are responsible for access control, prompt and output review, incident handling,
and compliance with applicable law and all upstream terms.

## Change control

Repeat the model-license review before:

- distributing or automatically downloading any weight;
- changing a reviewed model source or hash;
- enabling public or multi-user inference;
- removing operator review or adding unattended generation; or
- using the project commercially.

`THIRD_PARTY_NOTICES.md` records the reviewed artifacts and upstream terms.
