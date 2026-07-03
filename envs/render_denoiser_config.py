import os


ENV_NAME = "ROBOTWIN_DENOISER"
DEFAULT_DENOISER = "oidn"
SUPPORTED_DENOISERS = ("oidn", "optix", "none")


def _format_supported():
    return ", ".join(SUPPORTED_DENOISERS)


def normalize_denoiser_backend(value):
    backend = (value or DEFAULT_DENOISER).strip().lower()
    if backend not in SUPPORTED_DENOISERS:
        raise ValueError(
            f"Unsupported denoiser backend: {value}. "
            f"Expected one of: {_format_supported()}."
        )
    return backend


def resolve_denoiser_backend(cli_backend=None, environ=None):
    environ = os.environ if environ is None else environ
    value = cli_backend if cli_backend is not None else environ.get(ENV_NAME)
    return normalize_denoiser_backend(value)


def configure_sapien_denoiser(backend=None, sapien_module=None):
    backend = resolve_denoiser_backend(backend)
    if sapien_module is None:
        import sapien.core as sapien_module

    sapien_module.render.set_ray_tracing_denoiser(backend)
    print(f"[RoboTwin] Using SAPIEN ray tracing denoiser: {backend}")
    return backend
