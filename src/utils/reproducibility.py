"""System hardware and library provenance logging."""

import os
import platform
import sys
import torch


def get_system_signature() -> dict:
    """Collect runtime environment metadata for scientific reporting."""
    return {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU Execution)",
        "cpu_count": os.cpu_count(),
    }
