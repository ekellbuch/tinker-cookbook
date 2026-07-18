"""Smoke tests for DaytonaSandbox.

Mirror ``tests/test_modal_sandbox.py``: the same five behaviors and no others.
Require Daytona authentication and network access; skipped when no
``DAYTONA_API_KEY`` or ``DAYTONA_JWT_TOKEN`` is set in the environment.

The primary goal is to catch latency regressions in write_file and to confirm
cleanup is idempotent and resilient to a sandbox that has already timed out.
"""

import asyncio
import os
import time

import pytest
import pytest_asyncio

from tinker_cookbook.sandbox.daytona_sandbox import DaytonaSandbox

_has_daytona_auth = bool(os.environ.get("DAYTONA_API_KEY") or os.environ.get("DAYTONA_JWT_TOKEN"))

requires_daytona = pytest.mark.skipif(
    not _has_daytona_auth,
    reason="Daytona not configured (set DAYTONA_API_KEY or DAYTONA_JWT_TOKEN)",
)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
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
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(30)
async def test_write_file_latency(sandbox):
    """write_file should complete in seconds, not minutes."""
    content = "#!/bin/bash\necho hello world\n"

    result, elapsed = await _timed(
        sandbox.write_file("/tmp/test.sh", content, executable=True, timeout=30)
    )
    assert result.exit_code == 0, f"write_file failed: {result.stderr}"
    assert elapsed < 15, f"write_file took {elapsed:.1f}s (expected <15s)"

    # Verify content was written correctly
    read_result = await sandbox.run_command("cat /tmp/test.sh")
    assert read_result.exit_code == 0
    assert read_result.stdout == content

    # Verify executable bit
    stat_result = await sandbox.run_command("test -x /tmp/test.sh && echo yes")
    assert stat_result.stdout.strip() == "yes"

    print(f"\nwrite_file latency: {elapsed:.2f}s")


@requires_daytona
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(30)
async def test_write_file_binary(sandbox):
    """write_file should handle binary content correctly."""
    content = bytes(range(256))

    result, elapsed = await _timed(sandbox.write_file("/tmp/binary.bin", content, timeout=30))
    assert result.exit_code == 0, f"write_file failed: {result.stderr}"
    assert elapsed < 15, f"write_file took {elapsed:.1f}s (expected <15s)"

    # Verify size
    size_result = await sandbox.run_command("wc -c < /tmp/binary.bin")
    assert size_result.exit_code == 0
    assert int(size_result.stdout.strip()) == 256

    print(f"\nwrite_file (binary) latency: {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# cleanup() resilience tests
# ---------------------------------------------------------------------------


@requires_daytona
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(180)
async def test_cleanup_after_timeout():
    """cleanup() should not raise even if the sandbox has already timed out.

    Daytona's shortest practical lifetime is a 1-minute auto-stop; auto-delete
    at 0 removes the sandbox as soon as it stops.
    """
    sb = await DaytonaSandbox.create(timeout=120, auto_stop_minutes=1, auto_delete_minutes=0)

    # Wait for the sandbox to auto-stop and be deleted.
    await asyncio.sleep(75)

    # cleanup() should succeed without raising even though the sandbox is gone.
    await sb.cleanup()


@requires_daytona
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(120)
async def test_cleanup_after_terminate():
    """cleanup() should not raise if called twice (sandbox already terminated)."""
    sb = await DaytonaSandbox.create(timeout=60)

    # First cleanup terminates normally
    await sb.cleanup()

    # Second cleanup should not raise even though sandbox is already dead
    await sb.cleanup()


@requires_daytona
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(180)
async def test_cleanup_after_command_timeout():
    """cleanup() should work after a command hits the command/sandbox timeout."""
    sb = await DaytonaSandbox.create(timeout=120, auto_stop_minutes=1, auto_delete_minutes=0)

    # Run a command that will outlast its own timeout.
    await sb.run_command("sleep 30", timeout=5)

    # cleanup() should not raise
    await sb.cleanup()
