"""Smoke tests for DaytonaSandbox.

Daytona sibling of test_modal_sandbox.py. Require a Daytona API key and network
access; skipped when DAYTONA_API_KEY is not set, and skipped entirely unless the
optional `daytona` dependency is installed.

These exercise the SandboxInterface contract against a real backend
(write_file / read_file / run_command / cleanup), guard against write_file
latency regressions, and mirror the Modal cleanup-resilience coverage
(cleanup after the sandbox is terminated out-of-band, and after a command
outlasts its timeout) in Daytona terms.
"""

import os
import time

import pytest
import pytest_asyncio

from tinker_cookbook.sandbox.sandbox_interface import SandboxTerminatedError

# Skip the whole module unless the optional daytona dependency is installed.
daytona_sandbox = pytest.importorskip("tinker_cookbook.sandbox.daytona_sandbox")
DaytonaSandbox = daytona_sandbox.DaytonaSandbox
AsyncDaytona = pytest.importorskip("daytona").AsyncDaytona

_has_daytona_auth = bool(os.environ.get("DAYTONA_API_KEY"))

requires_daytona = pytest.mark.skipif(
    not _has_daytona_auth, reason="Daytona not configured locally (no DAYTONA_API_KEY)"
)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sandbox():
    """Shared Daytona sandbox for all tests in this module."""
    sb = await DaytonaSandbox.create(timeout=180)
    yield sb
    await sb.cleanup()


async def _timed(coro):
    """Await a coroutine and return (result, elapsed_seconds)."""
    start = time.monotonic()
    result = await coro
    return result, time.monotonic() - start


@requires_daytona
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(60)
async def test_write_file_latency(sandbox):
    """write_file should complete in seconds, and round-trip through run_command."""
    content = "#!/bin/bash\necho hello world\n"

    result, elapsed = await _timed(
        sandbox.write_file("/tmp/test.sh", content, executable=True, timeout=30)
    )
    assert result.exit_code == 0, f"write_file failed: {result.stderr}"
    assert elapsed < 15, f"write_file took {elapsed:.1f}s (expected <15s)"

    # Content round-trips.
    read_result = await sandbox.run_command("cat /tmp/test.sh")
    assert read_result.exit_code == 0
    assert read_result.stdout == content

    # Executable bit was applied.
    stat_result = await sandbox.run_command("test -x /tmp/test.sh && echo yes")
    assert stat_result.stdout.strip() == "yes"


@requires_daytona
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(60)
async def test_write_file_binary(sandbox):
    """write_file should handle binary content correctly."""
    content = bytes(range(256))

    result, elapsed = await _timed(sandbox.write_file("/tmp/binary.bin", content, timeout=30))
    assert result.exit_code == 0, f"write_file failed: {result.stderr}"
    assert elapsed < 15, f"write_file took {elapsed:.1f}s (expected <15s)"

    size_result = await sandbox.run_command("wc -c < /tmp/binary.bin")
    assert size_result.exit_code == 0
    assert int(size_result.stdout.strip()) == 256


@requires_daytona
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(60)
async def test_read_file_roundtrip(sandbox):
    """read_file should return exactly what write_file wrote."""
    content = "line one\nline two\n"
    write_result = await sandbox.write_file("/tmp/roundtrip.txt", content, timeout=30)
    assert write_result.exit_code == 0, f"write_file failed: {write_result.stderr}"

    read_result = await sandbox.read_file("/tmp/roundtrip.txt", timeout=30)
    assert read_result.exit_code == 0, f"read_file failed: {read_result.stderr}"
    assert read_result.stdout == content


@requires_daytona
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(60)
async def test_read_missing_file_is_not_termination(sandbox):
    """Reading a missing file returns a nonzero exit code, not SandboxTerminatedError."""
    result = await sandbox.read_file("/tmp/does_not_exist.txt")
    assert result.exit_code != 0
    # The sandbox is still alive after a missing-file read.
    assert (await sandbox.run_command("echo alive")).stdout.strip() == "alive"


@requires_daytona
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(60)
async def test_run_command_exit_code(sandbox):
    """run_command should surface both success and nonzero exit codes."""
    ok = await sandbox.run_command("true")
    assert ok.exit_code == 0

    fail = await sandbox.run_command("exit 7")
    assert fail.exit_code == 7


@requires_daytona
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(180)
async def test_cleanup_is_idempotent():
    """cleanup() should not raise if called twice (sandbox already terminated)."""
    sb = await DaytonaSandbox.create(timeout=180)

    # First cleanup terminates normally.
    await sb.cleanup()

    # Second cleanup should be a no-op, not an error.
    await sb.cleanup()


@requires_daytona
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(180)
async def test_cleanup_after_external_termination():
    """cleanup() should stay graceful if the sandbox was terminated out-of-band."""
    sb = await DaytonaSandbox.create(timeout=180)

    # Delete the sandbox through a separate client, simulating external death.
    client = AsyncDaytona()
    try:
        target = await client.get(sb.sandbox_id)
        await client.delete(target)
    finally:
        await client.close()

    # send_heartbeat must report the death per the SandboxInterface contract.
    with pytest.raises(SandboxTerminatedError):
        await sb.send_heartbeat()

    # The wrapper's own cleanup must not raise even though the sandbox is gone.
    await sb.cleanup()


@requires_daytona
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(180)
async def test_cleanup_after_command_timeout():
    """A command that outlasts its timeout should fail without wedging cleanup()."""
    sb = await DaytonaSandbox.create(timeout=180)
    try:
        result = await sb.run_command("sleep 30", timeout=3)
        assert result.exit_code != 0, "expected the timed-out command to report failure"
    finally:
        await sb.cleanup()
