"""
Thin wrapper around the Daytona sandbox API.

Daytona provides cloud-based sandboxed execution environments. Requires a
DAYTONA_API_KEY (and optionally DAYTONA_API_URL / DAYTONA_TARGET for
self-hosted control planes).

See: https://www.daytona.io/docs
"""

from __future__ import annotations

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
    """Check if an exception indicates the sandbox itself is gone (deleted/stopped/archived).

    Keyed on DaytonaNotFoundError from a sandbox-level call. Callers must not pass
    file-level 404s here (e.g. download_file on a missing file), which would
    otherwise be misread as sandbox death.
    """
    if isinstance(e, DaytonaNotFoundError):
        return True
    msg = str(e).lower()
    return any(k in msg for k in ("sandbox not found", "destroyed", "stopped", "archived"))


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

    def __init__(
        self,
        client: AsyncDaytona,
        sandbox: AsyncSandbox,
        max_output_bytes: int = 128 * 1024,
    ) -> None:
        self._client = client
        self._sandbox = sandbox
        self._max_output_bytes = max_output_bytes
        self._closed = False

    @classmethod
    async def create(cls, timeout: int = 600) -> DaytonaSandbox:
        """Create a new Daytona sandbox.

        `timeout` bounds how long to wait for the sandbox to start, matching the
        SandboxInterface create() convention. Reads DAYTONA_API_KEY from the
        environment.
        """
        client = AsyncDaytona()
        try:
            sandbox = await client.create(timeout=timeout)
        except BaseException:
            await client.close()  # don't leak the aiohttp session if start fails
            raise
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
        cap = max_output_bytes if max_output_bytes is not None else self._max_output_bytes
        try:
            resp = await self._sandbox.process.exec(command, cwd=workdir, timeout=timeout)
        except Exception as e:
            if _is_sandbox_terminated(e):
                raise SandboxTerminatedError(str(e)) from e
            return SandboxResult(stdout="", stderr=f"{type(e).__name__}: {e}", exit_code=-1)

        raw = (resp.result or "").encode()
        stdout = (
            raw[:cap].decode("utf-8", errors="replace") if len(raw) > cap else (resp.result or "")
        )
        exit_code = resp.exit_code if resp.exit_code is not None else -1
        return SandboxResult(stdout=stdout, stderr="", exit_code=exit_code)

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        """Read a file from the sandbox.

        Implemented via a shell read (like ModalSandbox) rather than the Daytona
        filesystem API, so a missing file returns a nonzero exit code instead of a
        DaytonaNotFoundError that would be misread as sandbox death.
        """
        quoted = shlex.quote(path)
        cmd = f"head -c {max_bytes} {quoted}" if max_bytes is not None else f"cat {quoted}"
        return await self.run_command(cmd, timeout=timeout)

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
            return SandboxResult(stdout="", stderr=f"{type(e).__name__}: {e}", exit_code=1)

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
