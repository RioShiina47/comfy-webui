"""
MCP Tool: get_sampler_scheduler_list
Query all supported Sampler algorithms and Noise Schedulers available for image generation tasks.
"""


def ImageGen_get_sampler_scheduler_list() -> dict:
    """Query all supported Sampler algorithms and Noise Schedulers available for image generation tasks."""
    try:
        from core import node_info_manager
        samplers = node_info_manager.get_node_input_options("KSampler", "sampler_name")
        schedulers = node_info_manager.get_node_input_options("KSampler", "scheduler")
    except Exception:
        samplers = None
        schedulers = None

    if not samplers:
        samplers = [
            "euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
            "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
            "ddpm", "lcm", "ddim", "uni_pc", "uni_pc_bh2", "res_multistep", "er_sde"
        ]
    if not schedulers:
        schedulers = [
            "normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"
        ]

    return {
        "samplers": list(samplers),
        "schedulers": list(schedulers),
    }
