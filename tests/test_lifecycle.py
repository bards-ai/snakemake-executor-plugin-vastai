"""Tests for the instance status state machine, using a stubbed Vast.ai SDK.

The Executor is instantiated without snakemake plumbing (bypassing
RemoteExecutor.__init__) so that _check_job/_finalize_job can be exercised
directly.
"""

import logging

import pytest

from snakemake_executor_plugin_vastai import (
    Executor,
    ExecutorSettings,
    SubmittedJobInfo,
)


class FakeVast:
    def __init__(self, instance=None, log_text=""):
        self.instance = instance
        self.log_text = log_text
        self.destroyed = []

    def show_instance(self, id):
        if isinstance(self.instance, Exception):
            raise self.instance
        return self.instance

    def logs(self, id, tail=None):
        if isinstance(self.log_text, Exception):
            raise self.log_text
        return self.log_text

    def destroy_instance(self, id):
        self.destroyed.append(id)


class FakeJob:
    jobid = 1
    name = "rule_a"


@pytest.fixture
def executor(tmp_path):
    ex = Executor.__new__(Executor)
    ex.settings = ExecutorSettings()
    ex.logger = logging.getLogger("test")
    ex.log_dir = tmp_path
    ex.storage_mode = True
    ex._destroyed_instances = set()
    ex.reported = []
    ex.report_job_success = lambda ji: ex.reported.append(("success", ji))
    ex.report_job_error = lambda ji, msg=None, **kw: ex.reported.append(
        ("error", ji, msg)
    )
    return ex


def make_job_info(submitted=None):
    import time

    return SubmittedJobInfo(
        job=FakeJob(),
        external_jobid="123",
        aux={
            "submitted": submitted if submitted is not None else time.time(),
            "label": "snakemake-test-1-1",
            "seen_running": False,
            "unreachable_since": None,
            "api_misses": 0,
        },
    )


def test_running_yields(executor):
    executor.vast = FakeVast(instance={"actual_status": "running"})
    job_info = make_job_info()
    assert executor._check_job(job_info) is True
    assert job_info.aux["seen_running"] is True
    assert executor.reported == []


def test_loading_yields_until_boot_timeout(executor):
    executor.vast = FakeVast(instance={"actual_status": "loading"})
    assert executor._check_job(make_job_info()) is True

    expired = make_job_info(submitted=0)  # submitted long ago
    assert executor._check_job(expired) is False
    assert executor.reported[0][0] == "error"
    assert executor.vast.destroyed == [123]


def test_exited_with_zero_exit_code_is_success(executor):
    executor.vast = FakeVast(
        instance={"actual_status": "exited"},
        log_text="job output\nsnakemake_vastai_exit_code=0\n",
    )
    assert executor._check_job(make_job_info()) is False
    assert executor.reported[0][0] == "success"
    assert executor.vast.destroyed == [123]
    assert (executor.log_dir / "snakemake-test-1-1.log").exists()


def test_exited_with_nonzero_exit_code_is_error(executor):
    executor.vast = FakeVast(
        instance={"actual_status": "exited"},
        log_text="boom\nsnakemake_vastai_exit_code=1\n",
    )
    assert executor._check_job(make_job_info()) is False
    kind, _, msg = executor.reported[0]
    assert kind == "error"
    assert "exit code 1" in msg


def test_exited_without_sentinel_is_error(executor):
    executor.vast = FakeVast(
        instance={"actual_status": "exited", "status_msg": "host died"},
        log_text="partial output, container killed",
    )
    assert executor._check_job(make_job_info()) is False
    kind, _, msg = executor.reported[0]
    assert kind == "error"
    assert "without reporting a job exit code" in msg


def test_unreachable_gets_grace_then_fails(executor, monkeypatch):
    executor.vast = FakeVast(instance={"actual_status": "offline"}, log_text="")
    job_info = make_job_info()
    assert executor._check_job(job_info) is True
    assert job_info.aux["unreachable_since"] is not None

    # Simulate the grace period having expired.
    job_info.aux["unreachable_since"] -= 10_000
    assert executor._check_job(job_info) is False
    assert executor.reported[0][0] == "error"


def test_api_errors_tolerated_then_fatal(executor):
    executor.vast = FakeVast(instance=RuntimeError("api down"), log_text="")
    job_info = make_job_info()
    for _ in range(5):
        assert executor._check_job(job_info) is True
    assert executor._check_job(job_info) is False
    assert executor.reported[0][0] == "error"


def test_keep_instances_skips_destroy(executor):
    executor.settings = ExecutorSettings(keep_instances=True)
    executor.vast = FakeVast(
        instance={"actual_status": "exited"},
        log_text="snakemake_vastai_exit_code=0",
    )
    assert executor._check_job(make_job_info()) is False
    assert executor.vast.destroyed == []
    assert executor.reported[0][0] == "success"


def test_cancel_jobs_destroys_instances(executor):
    executor.vast = FakeVast()
    executor.cancel_jobs([make_job_info()])
    assert executor.vast.destroyed == [123]
