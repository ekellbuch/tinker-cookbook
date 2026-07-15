"""
Code execution backends for sandboxed code evaluation.

The sandbox/ directory provides thin wrappers around different sandbox backends:
- SandboxFusionClient: HTTP-based sandbox using SandboxFusion Docker container
- ModalSandbox: Cloud sandbox using Modal's infrastructure
- DaytonaSandbox: Cloud sandbox using Daytona's infrastructure

ModalSandbox and DaytonaSandbox depend on optional packages and are imported
from their own modules on demand, not re-exported here.
"""

from enum import StrEnum

from tinker_cookbook.sandbox.sandbox_interface import (
    SandboxInterface,
    SandboxResult,
    SandboxTerminatedError,
)
from tinker_cookbook.sandbox.sandboxfusion import SandboxFusionClient


class SandboxBackend(StrEnum):
    SANDBOXFUSION = "sandboxfusion"
    MODAL = "modal"
    DAYTONA = "daytona"


__all__ = [
    "SandboxBackend",
    "SandboxFusionClient",
    "SandboxInterface",
    "SandboxResult",
    "SandboxTerminatedError",
]
