"""
Thin wrapper around the Daytona sandbox API.

Daytona provides cloud-based sandboxed execution environments. Requires a
DAYTONA_API_KEY (and optionally DAYTONA_API_URL / DAYTONA_TARGET for
self-hosted control planes).

See: https://www.daytona.io/docs
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex

try:
    from daytona import (
        AsyncDaytona,
        AsyncSandbox,
        DaytonaNotFoundError,
    )
except ImportError:
    raise ImportError(
        "daytona is required for DaytonaSandbox. Install it with: "
        "uv pip install 'tinker-cookbook[daytona]'"
    ) from None

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult, SandboxTerminatedError

logger = logging.getLogger(__name__)


def _is_sandbox_terminated(e: BaseException) -> bool:
    """Check if an exception indicates the sandbox is gone (deleted/stopped/archived)."""
    if isinstance(e, DaytonaNotFoundError):
        return True
    msg = str(e).lower()
    return any(k in msg for k in ("terminated", "destroyed", "not found", "stopped", "archived"))


class DaytonaSandbox:
    """
    Persistent Daytona sandbox for code execution. Conforms to SandboxInterface.

    Usage:
        sandbox = await DaytonaSandbox.create()

        await sandbox.write_file("/workspace/code.py", "print('hello')")
        result = await sandbox.run_command("python /workspace/code.py")
        print(result.stdout)

        await sandbox.cleanup()
    """

    def __init__(self, client: AsyncDaytona, sandbox: AsyncSandbox) -> None:
        self._client = client
        self._sandbox = sandbox
        self._closed = False

    @classmethod
    async def create(cls, timeout: int = 600) -> DaytonaSandbox:
        """Create a new Daytona sandbox.

        `timeout` bounds how long to wait for the sandbox to start, matching the
        SandboxInterface create() convention. Reads DAYTONA_API_KEY from the
        environment.
        """
        client = AsyncDaytona()
        sandbox = await client.create(timeout=timeout)
        return cls(client, sandbox)

    @property
    def sandbox_id(self) -> str:
        return self._sandbox.id

    async def send_heartbeat(self, timeout: int = 30) -> None:
        try:
            await self._sandbox.process.exec("true", timeout=timeout)
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            raise

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        """Run a shell command in the sandbox.

        Daytona returns combined output in `result` and does not split stderr, so
        stderr is empty on success and carries the client-side error otherwise.
        """
        try:
            resp = await self._sandbox.process.exec(command, cwd=workdir, timeout=timeout)
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

        stdout = resp.result or ""
        if max_output_bytes is not None:
            stdout = stdout[:max_output_bytes]
        return SandboxResult(stdout=stdout, stderr="", exit_code=resp.exit_code)

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        """Read a file from the sandbox via the Daytona filesystem API."""
        try:
            # download_file's real signature is (*args: str); a positional timeout
            # would be misread as a local path, so enforce the timeout here instead.
            data = await asyncio.wait_for(self._sandbox.fs.download_file(path), timeout=timeout)
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)

        if max_bytes is not None:
            data = data[:max_bytes]
        return SandboxResult(stdout=data.decode("utf-8", errors="replace"), stderr="", exit_code=0)

    async def write_file(
        self,
        path: str,
        content: str | bytes = "",
        executable: bool = False,
        timeout: int = 60,
    ) -> SandboxResult:
        """Write content to a file in the sandbox via the Daytona filesystem API."""
        if isinstance(content, str):
            content = content.encode()

        try:
            parent = os.path.dirname(path)
            if parent:
                await self._sandbox.process.exec(f"mkdir -p {shlex.quote(parent)}", timeout=timeout)
            await self._sandbox.fs.upload_file(content, path, timeout=timeout)
            if executable:
                await self._sandbox.process.exec(f"chmod +x {shlex.quote(path)}", timeout=timeout)
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)

        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def cleanup(self) -> None:
        """Delete the sandbox and close the client. Idempotent and safe to call twice."""
        if self._closed:
            return
        self._closed = True

        try:
            await self._sandbox.delete()
        except DaytonaNotFoundError:
            pass  # already gone (e.g. terminated out-of-band)
        except Exception as e:
            logger.warning("DaytonaSandbox delete raised during cleanup: %s", e)

        try:
            await self._client.close()
        except Exception as e:
            logger.warning("DaytonaSandbox client close raised during cleanup: %s", e)
