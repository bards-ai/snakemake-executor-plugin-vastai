"""Regression tests for behaviors discovered during live Vast.ai testing."""

from snakemake_executor_plugin_vastai import (
    Executor,
    ExecutorSettings,
)
from snakemake_executor_plugin_vastai._common import (
    PYTHON_PATH_SETUP,
    fatal_boot_error,
)


def make_executor(storage_mode=False):
    ex = Executor.__new__(Executor)
    ex.settings = ExecutorSettings()
    ex.storage_mode = storage_mode
    ex.ssh_pubkey = "ssh-ed25519 AAAATESTKEY snakemake-test"
    return ex


def test_onstart_provisions_key_without_attach_api():
    # The attach-ssh API on a live instance is racy; the key must land in
    # authorized_keys via onstart.
    script = make_executor()._ssh_onstart_script()
    assert "ssh-ed25519 AAAATESTKEY" in script
    assert "authorized_keys" in script
    # Best-effort OpenSSH install for Debian-family images lacking it.
    assert "apt-get install -y openssh-server" in script


def test_job_setup_discovers_python_before_pip():
    # Non-interactive SSH sessions don't get the image ENV PATH; python
    # discovery (preferring the /venv/main torch env) must come first.
    script = make_executor().job_setup_script()
    assert "/venv/main/bin/python3" in script
    assert script.index("/venv/main") < script.index("pip install")


def test_ssh_mode_installs_storage_plugins_remotely():
    # Locally installed storage plugins put --storage-* args into every job
    # command; the remote snakemake can only parse them with the plugins.
    assert "snakemake-storage-plugin-s3==" in make_executor().job_setup_script()
    # Storage mode handles plugins via the auto-deploy precommand instead.
    assert (
        "snakemake-storage-plugin-s3"
        not in make_executor(storage_mode=True).job_setup_script()
    )


def test_fatal_boot_errors_fail_fast():
    # Hosts hitting Docker Hub pull limits never recover; waiting out the
    # boot timeout would just burn money.
    assert fatal_boot_error(
        "Error response from daemon: toomanyrequests: You have reached "
        "your unauthenticated pull rate limit."
    )
    assert fatal_boot_error("manifest unknown") is not None
    assert fatal_boot_error("downloading layer 3/12") is None
    assert fatal_boot_error(None) is None


def test_python_path_setup_symlinks_python():
    # Images with only python3 still need a `python` for spawned commands.
    assert "ln -sf" in PYTHON_PATH_SETUP
    assert "/usr/local/bin/python" in PYTHON_PATH_SETUP
