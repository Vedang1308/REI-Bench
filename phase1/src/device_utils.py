"""
Device detection and setup utilities.

Auto-detects whether running on:
  - NVIDIA A100 GPU (via nvidia-smi)
  - Intel Gaudi HL-225 HPU (via hl-smi)
  - CPU fallback

Provides a unified interface for model placement.
"""

import subprocess
import sys
import logging

logger = logging.getLogger(__name__)


def _run_cmd(cmd: str) -> bool:
    """Check if a shell command succeeds."""
    try:
        result = subprocess.run(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def detect_device() -> str:
    """
    Detect the available accelerator.

    Returns:
        One of: "cuda", "hpu", "cpu"
    """
    # Check Intel Gaudi HPU first (more specific)
    if _run_cmd("hl-smi"):
        try:
            import habana_frameworks.torch  # noqa: F401
            logger.info("Detected Intel Gaudi HPU via hl-smi")
            return "hpu"
        except ImportError:
            logger.warning(
                "hl-smi found but habana_frameworks not installed. "
                "Install optimum-habana: pip install optimum-habana"
            )

    # Check NVIDIA GPU
    if _run_cmd("nvidia-smi"):
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Detected NVIDIA GPU: {gpu_name}")
                return "cuda"
        except Exception:
            pass

    logger.info("No GPU/HPU detected, falling back to CPU")
    return "cpu"


def get_device_info() -> dict:
    """
    Get detailed device information for logging.

    Returns:
        Dictionary with device type, name, and memory info.
    """
    device_type = detect_device()
    info = {"device_type": device_type, "device_name": "unknown", "memory_gb": 0}

    if device_type == "cuda":
        import torch
        info["device_name"] = torch.cuda.get_device_name(0)
        info["memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
        )
        info["device_count"] = torch.cuda.device_count()

    elif device_type == "hpu":
        try:
            result = subprocess.run(
                ["hl-smi", "-Q", "name,memory.total", "-f", "csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                info["device_name"] = parts[0].strip() if len(parts) > 0 else "Gaudi"
                if len(parts) > 1:
                    mem_str = parts[1].strip().replace("MiB", "").strip()
                    try:
                        info["memory_gb"] = round(float(mem_str) / 1024, 1)
                    except ValueError:
                        pass
        except Exception:
            info["device_name"] = "Intel Gaudi"

    else:
        import platform
        info["device_name"] = f"CPU ({platform.processor() or platform.machine()})"

    return info


def setup_device():
    """
    Set up the compute device and return a torch device object.

    For Gaudi HPU, this also initializes the habana bridge.

    Returns:
        torch.device object ready for model placement.
    """
    import torch

    device_type = detect_device()

    if device_type == "hpu":
        # Initialize Habana bridge
        import habana_frameworks.torch.core as htcore  # noqa: F401
        import habana_frameworks.torch.hpu as hthpu  # noqa: F401
        device = torch.device("hpu")
        logger.info(f"Using Intel Gaudi HPU: {get_device_info()['device_name']}")

    elif device_type == "cuda":
        device = torch.device("cuda:0")
        logger.info(f"Using NVIDIA GPU: {get_device_info()['device_name']}")

    else:
        device = torch.device("cpu")
        logger.warning("Using CPU - inference will be slow!")

    return device


def print_device_banner():
    """Print a formatted banner showing device information."""
    info = get_device_info()
    banner = f"""
╔══════════════════════════════════════════════════╗
║           REI-Bench Device Configuration         ║
╠══════════════════════════════════════════════════╣
║  Device Type : {info['device_type']:>33s} ║
║  Device Name : {info['device_name']:>33s} ║
║  Memory      : {str(info['memory_gb']) + ' GB':>33s} ║
╚══════════════════════════════════════════════════╝
"""
    print(banner)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_device_banner()
