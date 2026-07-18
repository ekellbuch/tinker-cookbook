"""
Thin wrapper around the Daytona Sandbox API.

Daytona provides cloud-based sandboxed execution environments. This module is
the provider-specific equivalent of ``modal_sandbox.py``: it exposes the same
public capabilities, lifecycle semantics, pooling behavior, and Harbor
integration as the Modal backend. Differences exist only where the Daytona SDK
requires a different implementation.

Requires Daytona authentication: ``export DAYTONA_API_KEY=...``
(or ``DAYTONA_JWT_TOKEN`` + ``DAYTONA_ORGANIZATION_ID``).

Configuration via environment variables:
    DAYTONA_POOL_SIZE: Number of sandboxes in the pool (default: 32)
    DAYTONA_CREATION_RATE_LIMIT: Max sandboxes created per maintenance step (default: 4)
    DAYTONA_SNAPSHOT: Optional pre-created snapshot name. Not required — image
        builds are content-hashed and cached across sandboxes automatically.

See: https://www.daytona.io
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import uuid
from pathlib import Path
from typing import Any

try:
    from daytona import (
        AsyncDaytona,
        CreateSandboxFromImageParams,
        CreateSandboxFromSnapshotParams,
        DaytonaConfig,
        DaytonaNotFoundError,
        FileUpload,
        Image,
    )
except ImportError:
    raise ImportError(
        "daytona is required for DaytonaSandbox. "
        "Install it with: uv pip install 'tinker-cookbook[daytona] @ "
        "git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'"
    ) from None

from tinker_cookbook.exceptions import SandboxError
from tinker_cookbook.sandbox.sandbox_interface import (
    SandboxInterface,
    SandboxResult,
    SandboxTerminatedError,
)

logger = logging.getLogger(__name__)


def _cap_output(text: str, max_bytes: int) -> str:
    """Cap a string to *max_bytes* bytes of UTF-8, decoding safely at the boundary."""
    candidate = text[:max_bytes]
    encoded = candidate.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return candidate
    return encoded[:max_bytes].decode("utf-8", errors="replace")


# 15-minute auto-stop matches the Daytona SDK default; 30-minute auto-delete
# ensures a crashed rollout or forgotten ``cleanup()`` call does not leak the
# sandbox indefinitely. Both are overridable via ``DaytonaSandbox.create``.
_DEFAULT_AUTO_STOP_MINUTES = 15
_DEFAULT_AUTO_DELETE_MINUTES = 30
_DEFAULT_MAX_OUTPUT_BYTES = 128 * 1024


class DaytonaSandbox(SandboxInterface):
    """
    Persistent Daytona sandbox for code execution. Conforms to SandboxInterface.

    Matches :class:`ModalSandbox`: each ``run_command`` executes independently in
    a fresh shell (via ``process.exec``), so shell state (cwd, exported env,
    shell variables) does not persist across calls. Filesystem changes persist
    because they are changes to the sandbox filesystem.

    Usage:
        sandbox = await DaytonaSandbox.create()
        await sandbox.write_file("/workspace/code.py", "print('hello')")
        result = await sandbox.run_command("python /workspace/code.py")
        print(result.stdout)
        await sandbox.cleanup()
    """

    def __init__(
        self,
        *,
        client: AsyncDaytona,
        sandbox: Any,  # daytona.AsyncSandbox — Any to avoid a hard dep at module import
        max_stream_output_bytes: int,
        owns_client: bool,
    ) -> None:
        self._client = client
        self._sandbox = sandbox
        self._max_stream_output_bytes = max_stream_output_bytes
        self._owns_client = owns_client
        self._cleaned_up = False
        self._cleanup_lock = asyncio.Lock()

    @classmethod
    async def create(
        cls,
        *,
        image: Image | str | None = None,
        snapshot: str | None = None,
        timeout: int = 600,
        auto_stop_minutes: int | None = None,
        auto_delete_minutes: int | None = None,
        max_stream_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
        client: AsyncDaytona | None = None,
    ) -> DaytonaSandbox:
        """Create a new Daytona sandbox.

        Args:
            image: Image to use. Mutually exclusive with *snapshot*. If both are
                ``None`` (and ``DAYTONA_SNAPSHOT`` is unset), defaults to
                ``Image.debian_slim()``. Image builds are cached by Daytona
                across sandboxes, so passing the same image repeatedly does not
                re-build.
            snapshot: Name of a pre-created Daytona snapshot. Mutually exclusive
                with *image*. Falls back to ``DAYTONA_SNAPSHOT`` if unset.
            timeout: Max wait time in seconds for sandbox creation.
            auto_stop_minutes: Minutes of inactivity before auto-stop. ``0``
                disables. Defaults to 15.
            auto_delete_minutes: Minutes after stopping before auto-delete. ``0``
                means delete immediately, negative disables. Defaults to 30.
                Leak protection if ``cleanup()`` is skipped.
            max_stream_output_bytes: Cap per-stream output at this many bytes.
                Defaults to 128 KB.
            client: Existing ``AsyncDaytona`` client to reuse (e.g. from a pool).
                If ``None``, a new client is created and owned by this sandbox.
        """
        if image is not None and snapshot is not None:
            raise ValueError("Provide either image or snapshot, not both.")

        resolved_auto_stop = (
            auto_stop_minutes if auto_stop_minutes is not None else _DEFAULT_AUTO_STOP_MINUTES
        )
        resolved_auto_delete = (
            auto_delete_minutes if auto_delete_minutes is not None else _DEFAULT_AUTO_DELETE_MINUTES
        )

        owns_client = client is None
        if owns_client:
            client = AsyncDaytona(DaytonaConfig())
        assert client is not None  # for type checker

        try:
            resolved_snapshot = snapshot if snapshot is not None else os.getenv("DAYTONA_SNAPSHOT")
            if resolved_snapshot:
                params: CreateSandboxFromImageParams | CreateSandboxFromSnapshotParams = (
                    CreateSandboxFromSnapshotParams(
                        snapshot=resolved_snapshot,
                        auto_stop_interval=resolved_auto_stop,
                        auto_delete_interval=resolved_auto_delete,
                    )
                )
            else:
                resolved_image: Image | str = image if image is not None else Image.debian_slim()
                params = CreateSandboxFromImageParams(
                    image=resolved_image,
                    auto_stop_interval=resolved_auto_stop,
                    auto_delete_interval=resolved_auto_delete,
                )
            sandbox = await client.create(params, timeout=float(timeout))
        except Exception:
            if owns_client:
                # Best-effort close of the HTTP session we just opened.
                with contextlib.suppress(Exception):
                    await client.close()
            raise

        return cls(
            client=client,
            sandbox=sandbox,
            max_stream_output_bytes=max_stream_output_bytes,
            owns_client=owns_client,
        )

    @property
    def sandbox_id(self) -> str:
        return self._sandbox.id

    def _check_live(self) -> None:
        """Raise ``SandboxTerminatedError`` if the sandbox was already cleaned up.

        After ``cleanup()`` the sandbox is deleted (and an owned client closed);
        calling into the SDK afterwards would either error or reopen a fresh,
        unclosed aiohttp session. This mirrors Modal, where a command against a
        terminated sandbox surfaces as ``SandboxTerminatedError``.
        """
        if self._cleaned_up:
            raise SandboxTerminatedError("sandbox has been cleaned up")

    async def send_heartbeat(self, timeout: int = 30) -> None:
        self._check_live()
        try:
            await asyncio.wait_for(self._sandbox.refresh_activity(), timeout=timeout)
        except DaytonaNotFoundError as e:
            raise SandboxTerminatedError(str(e)) from e

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        """Run a shell command in the sandbox.

        Each call executes independently in a fresh shell via ``process.exec``,
        so cwd changes and exported environment variables do not persist across
        calls. ``workdir`` applies only to this command; ``workdir=None`` uses
        the image's default working directory.
        """
        self._check_live()
        cap = max_output_bytes if max_output_bytes is not None else self._max_stream_output_bytes
        try:
            response = await self._sandbox.process.exec(command, cwd=workdir, timeout=timeout)
        except DaytonaNotFoundError as e:
            raise SandboxTerminatedError(str(e)) from e
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

        exit_code = response.exit_code if response.exit_code is not None else -1
        # Daytona's exec returns combined output as a single ``result`` stream;
        # unlike Modal there is no separate stderr channel for a normal exit.
        stdout = response.result or ""
        return SandboxResult(
            stdout=_cap_output(stdout, cap),
            stderr="",
            exit_code=exit_code,
        )

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        """Read a file from the sandbox via a shell read.

        Uses ``cat``/``head -c`` through ``run_command`` rather than
        ``fs.download_file`` because the filesystem API raises
        ``DaytonaNotFoundError`` for a *missing file*, which is indistinguishable
        from a missing *sandbox* and would be misreported as
        ``SandboxTerminatedError`` on a healthy sandbox. Going through
        ``run_command`` yields a nonzero exit code for a missing file while still
        raising ``SandboxTerminatedError`` when the sandbox itself is gone.
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
        """Write content to a file in the sandbox.

        Uses Daytona's native ``fs.upload_files`` (a multipart HTTP POST) rather
        than piping through stdin. Binary content is written exactly as-is.
        """
        self._check_live()
        if isinstance(content, str):
            content = content.encode()

        try:
            await asyncio.wait_for(
                self._sandbox.fs.upload_files([FileUpload(source=content, destination=path)]),
                timeout=timeout,
            )
        except DaytonaNotFoundError as e:
            raise SandboxTerminatedError(str(e)) from e
        except Exception as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=-1)

        if executable:
            chmod = await self.run_command(f"chmod +x {shlex.quote(path)}", timeout=timeout)
            if chmod.exit_code != 0:
                return chmod
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def cleanup(self) -> None:
        """Delete the sandbox and close the owned Daytona client, idempotently.

        Safe to call twice and safe to call after the sandbox has already been
        auto-reaped: provider "not found"/"already deleted" errors are
        suppressed. If the client is borrowed from a pool it is left open.
        """
        async with self._cleanup_lock:
            if self._cleaned_up:
                return
            self._cleaned_up = True

            with contextlib.suppress(Exception):
                await self._client.delete(self._sandbox)

            if self._owns_client:
                with contextlib.suppress(Exception):
                    await self._client.close()


class DaytonaSandboxPool:
    """
    Pool of Daytona sandboxes for concurrent execution.

    Provider-specific equivalent of :class:`ModalSandboxPool`. Each sandbox
    handles one request at a time; a used sandbox is terminated and replaced
    rather than returned to the warm pool. A single ``AsyncDaytona`` client is
    shared across the pool as an internal implementation detail.

    Configuration via environment variables:
        DAYTONA_POOL_SIZE: Number of sandboxes in the pool (default: 32)
        DAYTONA_CREATION_RATE_LIMIT: Max sandboxes created per maintenance step
            (default: 4)
    """

    def __init__(
        self,
        *,
        pool_size: int | None = None,  # Number of warm sandboxes to maintain during the job run.
        sandbox_timeout_secs: int = 1200,  # Lifetime after which a sandbox is auto-stopped.
        image: Image | str | None = None,
        snapshot: str | None = None,
    ):
        self._pool_size = pool_size or int(os.getenv("DAYTONA_POOL_SIZE", "32"))
        self._creation_rate_limit = int(os.getenv("DAYTONA_CREATION_RATE_LIMIT", "4"))
        self._sandbox_timeout_secs = sandbox_timeout_secs
        self._image = image
        self._snapshot = snapshot
        self._terminated = False
        self._client = AsyncDaytona(DaytonaConfig())

        self._warm_pool: asyncio.Queue[DaytonaSandbox] = asyncio.Queue()  # Warm pool of sandboxes.
        self._to_terminate: list[DaytonaSandbox] = []  # Sandboxes pending termination.
        self._active_count = 0  # Number of in-use sandboxes.

        asyncio.create_task(self._maintain_pool())

    async def _create(self) -> DaytonaSandbox:
        return await DaytonaSandbox.create(
            image=self._image,
            snapshot=self._snapshot,
            timeout=self._sandbox_timeout_secs,
            auto_stop_minutes=max(1, self._sandbox_timeout_secs // 60),
            client=self._client,
        )

    async def _maintain_pool(self) -> None:
        """Background task to handle all sandbox creation and termination."""
        while not self._terminated:
            try:
                await self._maintain_pool_step()
            except Exception as e:
                logger.error(f"Error maintaining DaytonaSandboxPool: {e}")
            await asyncio.sleep(1.0)

    async def _maintain_pool_step(self) -> None:
        """Single iteration of pool maintenance: terminate used sandboxes, create new ones."""
        # Batch terminate used sandboxes
        if self._to_terminate:
            to_terminate, self._to_terminate = self._to_terminate, []
            await asyncio.gather(*(sb.cleanup() for sb in to_terminate))

        # Create new sandboxes in parallel (respecting rate limit)
        total = self._warm_pool.qsize() + self._active_count
        need = min(self._creation_rate_limit, self._pool_size - total)
        if need > 0:
            new_sandboxes = await asyncio.gather(
                *(self._create() for _ in range(need)),
                return_exceptions=True,
            )
            for sb in new_sandboxes:
                if isinstance(sb, BaseException):
                    logger.error(f"Error creating Daytona sandbox: {sb}")
                else:
                    await self._warm_pool.put(sb)

    async def run_in_workdir(
        self,
        files: dict[str, str],
        command: list[str],
        timeout: int | None = None,
    ) -> SandboxResult:
        """
        Execute command with files using an available sandbox from the pool.
        If all sandboxes are busy, waits until one becomes available.

        Creates an isolated workdir, writes files, and runs the command.

        Args:
            files: Files to write {filename: content}
            command: Command and arguments (e.g., ["python", "run.py"])
            timeout: Execution timeout in seconds
        """
        if self._terminated:
            raise SandboxError("DaytonaSandboxPool has been terminated.")

        sandbox = await self._warm_pool.get()
        self._active_count += 1

        try:
            workdir = f"/workspace/{uuid.uuid4().hex[:12]}"
            result = await sandbox.run_command(
                f"mkdir -p {shlex.quote(workdir)}", timeout=timeout or 60
            )
            if result.exit_code != 0:
                return SandboxResult(
                    stdout="",
                    stderr=f"Failed to create workdir: {workdir}",
                    exit_code=result.exit_code,
                )

            if files:
                write_results = await asyncio.gather(
                    *(
                        sandbox.write_file(f"{workdir}/{filename}", content)
                        for filename, content in files.items()
                    )
                )
                for filename, write_result in zip(files, write_results, strict=True):
                    if write_result.exit_code != 0:
                        return SandboxResult(
                            stdout="",
                            stderr=f"Failed to write {filename}: {write_result.stderr}",
                            exit_code=write_result.exit_code,
                        )

            return await sandbox.run_command(
                shlex.join(command), workdir=workdir, timeout=timeout or self._sandbox_timeout_secs
            )
        finally:
            self._active_count -= 1
            self._to_terminate.append(sandbox)

    async def terminate(self) -> None:
        """Exit the pool, terminate all sandboxes, and close the shared client."""
        self._terminated = True

        # Wait for active sandboxes to finish and be added to _to_terminate
        while self._active_count > 0:
            await asyncio.sleep(0.5)

        # Collect and terminate all sandboxes
        all_sandboxes = list(self._to_terminate)
        while not self._warm_pool.empty():
            try:
                all_sandboxes.append(self._warm_pool.get_nowait())
            except asyncio.QueueEmpty:
                break
        await asyncio.gather(*(sb.cleanup() for sb in all_sandboxes))

        # Close the shared client once, after all sandboxes are gone.
        with contextlib.suppress(Exception):
            await self._client.close()


# ---------------------------------------------------------------------------
# Harbor RL factory
# ---------------------------------------------------------------------------


async def daytona_sandbox_factory(env_dir: Path, timeout: int) -> DaytonaSandbox:
    """Create a Daytona sandbox from a Harbor task environment directory.

    Provider-specific equivalent of Modal's Harbor factory. ``env_dir`` is used
    as the Docker build context: ``Image.from_dockerfile`` archives the
    Dockerfile's ``COPY``/``ADD`` sources relative to ``env_dir``, so those
    instructions resolve against the task environment just as they do on Modal.

    Signature matches ``harbor_env.SandboxFactory``.

    Args:
        env_dir: Path to the task's ``environment/`` directory (must contain a
            ``Dockerfile``).
        timeout: Max wait time in seconds for sandbox creation.
    """
    dockerfile_path = env_dir / "Dockerfile"
    image = Image.from_dockerfile(str(dockerfile_path))
    return await DaytonaSandbox.create(image=image, timeout=timeout)


__all__ = [
    "DaytonaSandbox",
    "DaytonaSandboxPool",
    "daytona_sandbox_factory",
]
