"""Smoke tests for DaytonaSandbox.

Daytona sibling of test_modal_sandbox.py. Require a Daytona API key and network
access; skipped when DAYTONA_API_KEY is not set. The module is also skipped
until the DaytonaSandbox backend exists, so this file can land test-first ahead
of the implementation.

These exercise the SandboxInterface contract against a real backend
(write_file / read_file / run_command / cleanup) and, like the Modal tests,
guard against write_file latency regressions.
"""

import os
import time

import pytest
import pytest_asyncio

# Skip the whole module until the Daytona backend is implemented, so this test
# can be committed test-first without breaking collection.
daytona_sandbox = pytest.importorskip("tinker_cookbook.sandbox.daytona_sandbox")
DaytonaSandbox = daytona_sandbox.DaytonaSandbox

_has_daytona_auth = bool(os.environ.get("DAYTONA_API_KEY"))

requires_daytona = pytest.mark.skipif(
    not _has_daytona_auth, reason="Daytona not configured locally (no DAYTONA_API_KEY)"
)


@pytest_asyncio.fixture(scope="module")
async def sandbox():
    """Shared Daytona sandbox for all tests in this module."""
    sb = await DaytonaSandbox.create(timeout=120)
    yield sb
    await sb.cleanup()


async def _timed(coro):
    """Await a coroutine and return (result, elapsed_seconds)."""
    start = time.monotonic()
    result = await coro
    return result, time.monotonic() - start


@requires_daytona
@pytest.mark.asyncio
@pytest.mark.timeout(30)
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
@pytest.mark.asyncio
@pytest.mark.timeout(30)
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
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_read_file_roundtrip(sandbox):
    """read_file should return exactly what write_file wrote."""
    content = "line one\nline two\n"
    write_result = await sandbox.write_file("/tmp/roundtrip.txt", content, timeout=30)
    assert write_result.exit_code == 0, f"write_file failed: {write_result.stderr}"

    read_result = await sandbox.read_file("/tmp/roundtrip.txt", timeout=30)
    assert read_result.exit_code == 0, f"read_file failed: {read_result.stderr}"
    assert read_result.stdout == content


@requires_daytona
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_run_command_exit_code(sandbox):
    """run_command should surface both success and nonzero exit codes."""
    ok = await sandbox.run_command("true")
    assert ok.exit_code == 0

    fail = await sandbox.run_command("exit 7")
    assert fail.exit_code == 7


@requires_daytona
@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_cleanup_is_idempotent():
    """cleanup() should not raise if called twice (sandbox already terminated)."""
    sb = await DaytonaSandbox.create(timeout=60)

    # First cleanup terminates normally.
    await sb.cleanup()

    # Second cleanup should be a no-op, not an error.
    await sb.cleanup()
