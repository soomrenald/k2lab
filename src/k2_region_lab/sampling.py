from __future__ import annotations


# Ordered to match ComfyUI's KSampler registries as of 2026-07-18. The worker
# validates the selected values against the installed ComfyUI runtime again
# before generation, so an environment mismatch produces a clear error.
COMFYUI_SAMPLERS = (
    "euler",
    "euler_cfg_pp",
    "euler_ancestral",
    "euler_ancestral_cfg_pp",
    "heun",
    "heunpp2",
    "exp_heun_2_x0",
    "exp_heun_2_x0_sde",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ipndm",
    "ipndm_v",
    "deis",
    "res_multistep",
    "res_multistep_cfg_pp",
    "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp",
    "gradient_estimation",
    "gradient_estimation_cfg_pp",
    "er_sde",
    "seeds_2",
    "seeds_3",
    "sa_solver",
    "sa_solver_pece",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
)

COMFYUI_SCHEDULERS = (
    "simple",
    "sgm_uniform",
    "karras",
    "exponential",
    "ddim_uniform",
    "beta",
    "normal",
    "linear_quadratic",
    "kl_optimal",
)

DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"


def validate_sampler(value: str) -> str:
    sampler = str(value)
    if sampler not in COMFYUI_SAMPLERS:
        raise ValueError(f"unsupported ComfyUI sampler: {sampler!r}")
    return sampler


def validate_scheduler(value: str) -> str:
    scheduler = str(value)
    if scheduler not in COMFYUI_SCHEDULERS:
        raise ValueError(f"unsupported ComfyUI scheduler: {scheduler!r}")
    return scheduler
