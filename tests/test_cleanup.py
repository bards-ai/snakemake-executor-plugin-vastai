"""Tests for the emergency-cleanup safety net (atexit / SIGTERM).

The Executor is instantiated without snakemake plumbing (bypassing
RemoteExecutor.__init__) so the cleanup logic can be exercised directly.
"""

import logging
import os
import signal
import subprocess
import sys
import textwrap

import pytest

from snakemake_executor_plugin_vastai import Executor, ExecutorSettings


class FakeVast:
    def __init__(self):
        self.destroyed = []

    def destroy_instance(self, id):
        self.destroyed.append(id)


@pytest.fixture
def executor():
    ex = Executor.__new__(Executor)
    ex.settings = ExecutorSettings()
    ex.logger = logging.getLogger("test")
    ex.vast = FakeVast()
    ex._destroyed_instances = set()
    ex._rented_instances = set()
    return ex


def test_emergency_cleanup_destroys_leaked_instances(executor):
    executor._rented_instances = {111, 222}
    executor._emergency_cleanup()
    assert sorted(executor.vast.destroyed) == [111, 222]


def test_emergency_cleanup_skips_already_destroyed(executor):
    executor._rented_instances = {111, 222}
    executor._destroyed_instances = {111}
    executor._emergency_cleanup()
    assert executor.vast.destroyed == [222]


def test_emergency_cleanup_is_idempotent(executor):
    executor._rented_instances = {111}
    executor._emergency_cleanup()
    executor._emergency_cleanup()
    assert executor.vast.destroyed == [111]


def test_emergency_cleanup_noop_on_clean_run(executor):
    executor._emergency_cleanup()
    assert executor.vast.destroyed == []


def test_emergency_cleanup_respects_keep_instances(executor):
    executor.settings.keep_instances = True
    executor._rented_instances = {111}
    executor._emergency_cleanup()
    assert executor.vast.destroyed == []


def test_signal_cleanup_chains_previous_handler(executor):
    calls = []
    executor._emergency_cleanup = lambda: calls.append("cleanup")

    previous = {
        sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        def prev(signum, frame):
            calls.append("prev")

        signal.signal(signal.SIGTERM, prev)
        executor._install_signal_cleanup()
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert calls == ["cleanup", "prev"]
    finally:
        for sig, prev_handler in previous.items():
            signal.signal(sig, prev_handler)


def test_sigterm_destroys_rented_instance_in_subprocess(tmp_path):
    """End-to-end check that a plain `kill` (SIGTERM) triggers destruction."""
    script = textwrap.dedent(
        """
        import logging, os, signal, sys, time

        from snakemake_executor_plugin_vastai import Executor, ExecutorSettings

        class FakeVast:
            def destroy_instance(self, id):
                print(f"DESTROYED {id}", flush=True)

        ex = Executor.__new__(Executor)
        ex.settings = ExecutorSettings()
        ex.logger = logging.getLogger("test")
        ex.vast = FakeVast()
        ex._destroyed_instances = set()
        ex._rented_instances = {424242}
        ex._install_signal_cleanup()
        print("READY", flush=True)
        time.sleep(30)
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout.readline().strip() == "READY"
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=10)
    assert "DESTROYED 424242" in out
    assert proc.returncode != 0  # default SIGTERM action still applies
