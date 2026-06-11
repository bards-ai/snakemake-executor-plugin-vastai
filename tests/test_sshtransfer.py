"""Tests for the SSH transfer pipeline with stubbed SSH/SCP plumbing."""

import logging
import subprocess
from pathlib import Path

import pytest

from snakemake_executor_plugin_vastai import ExecutorSettings, SubmittedJobInfo
from snakemake_executor_plugin_vastai.sshtransfer import SSHJobRunner


class FakeJob:
    jobid = 1
    name = "rule_a"
    input = []
    output = []
    log = []


class FakeExecutor:
    def __init__(self, tmp_path):
        self.vast = None
        self.logger = logging.getLogger("test")
        self.settings = ExecutorSettings()
        self.ssh_keyfile = tmp_path / "key"
        self.log_dir = tmp_path
        self.source_archive_path = tmp_path / "sources.tar.xz"
        self.destroyed = []

    def _destroy_instance(self, instance_id):
        self.destroyed.append(instance_id)

    def job_setup_script(self):
        return ""


def make_runner(tmp_path, exec_job="run-the-job"):
    executor = FakeExecutor(tmp_path)
    job_info = SubmittedJobInfo(
        job=FakeJob(), external_jobid="123", aux={"label": "test-label"}
    )
    runner = SSHJobRunner(executor, job_info, exec_job)
    runner.host = "host"
    runner.port = 22
    return runner


def completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_successful_pipeline(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    ssh_commands = []

    monkeypatch.setattr(runner, "_wait_for_boot", lambda: None)
    monkeypatch.setattr(runner, "_wait_for_ssh", lambda: None)
    monkeypatch.setattr(runner, "_scp", lambda src, dst: None)
    monkeypatch.setattr(
        SSHJobRunner, "_remote_extract", lambda self, name: None
    )

    def fake_ssh(command, check=True, input_text=None, timeout=3600):
        ssh_commands.append((command, input_text))
        if "cat /snakemake-workdir/exit_code" in command:
            return completed(stdout="0\n")
        if "job.log" in command:
            return completed(stdout="job ran fine")
        return completed()

    monkeypatch.setattr(runner, "_ssh", fake_ssh)
    runner._run()

    assert runner.job_info.aux["outcome"] == ("success", None)
    assert runner.executor.destroyed == [123]
    # The job script was uploaded and contains the exec command + exit_code.
    script = next(t for c, t in ssh_commands if c.endswith("job.sh"))
    assert "run-the-job" in script
    assert "echo $? > exit_code" in script
    assert (tmp_path / "test-label.log").read_text() == "job ran fine"


def test_failing_job_reports_exit_code(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    monkeypatch.setattr(runner, "_wait_for_boot", lambda: None)
    monkeypatch.setattr(runner, "_wait_for_ssh", lambda: None)
    monkeypatch.setattr(runner, "_upload", lambda: None)

    def fake_ssh(command, check=True, input_text=None, timeout=3600):
        if "cat /snakemake-workdir/exit_code" in command:
            return completed(stdout="2\n")
        return completed(stdout="boom")

    monkeypatch.setattr(runner, "_ssh", fake_ssh)
    runner._run()

    kind, msg = runner.job_info.aux["outcome"]
    assert kind == "error"
    assert "exit code 2" in msg
    assert runner.executor.destroyed == [123]


def test_boot_failure_destroys_and_reports(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)

    class FakeVast:
        def show_instance(self, id):
            return {"actual_status": "exited", "status_msg": "host gone"}

    runner.vast = FakeVast()
    runner.settings.boot_timeout = 1
    monkeypatch.setattr(runner, "_download_log", lambda: None)
    runner._run()

    kind, msg = runner.job_info.aux["outcome"]
    assert kind == "error"
    assert "terminated while booting" in msg
    assert runner.executor.destroyed == [123]


def test_cancellation_sets_no_outcome(tmp_path):
    runner = make_runner(tmp_path)
    runner.cancel_event.set()
    runner._run()
    assert "outcome" not in runner.job_info.aux
    # cancel_jobs() destroys the instance, not the runner
    assert runner.executor.destroyed == []
