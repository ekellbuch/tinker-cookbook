"""
Code execution backends for sandboxed code evaluation.

The sandbox/ directory provides thin wrappers around different sandbox backends:
- SandboxFusionClient: HTTP-based sandbox using SandboxFusion Docker container
- ModalSandbox: Cloud sandbox using Modal's infrastructure
- DaytonaSandbox: Cloud sandbox using Daytona's infrastructure
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
    # DAYTONA is intentionally omitted until the code_rl grader has a Daytona
    # dispatch path; adding it now would make grading silently fail-closed.


__all__ = [
    "SandboxBackend",
    "SandboxFusionClient",
    "SandboxInterface",
    "SandboxResult",
    "SandboxTerminatedError",
]
