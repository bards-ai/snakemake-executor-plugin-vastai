"""SSH-based file transfer mode, used when no default storage provider is
configured.

Each job gets a worker thread that waits for its instance to boot, uploads
the workflow sources and the job's input files over SSH, runs the job
command, downloads the output files back into the local working directory,
and destroys the instance. The executor's status loop only inspects the
outcome recorded by the thread.
"""

import base64
import json
import subprocess
import tarfile
import tempfile
import threading
import time
from pathlib import Path

from snakemake_executor_plugin_vastai._common import (
    PYTHON_PATH_SETUP,
    fatal_boot_error,
)

REMOTE_WORKDIR = "/snakemake-workdir"
SSH_RETRY_SECONDS = 300
POLL_SECONDS = 15
# How long to tolerate SSH connection failures before checking for a stuck
# proxy tunnel and rebooting the instance once.
TUNNEL_REBOOT_AFTER_SECONDS = 75


class JobCancelled(Exception):
    pass


class TransferError(Exception):
    pass


class SSHJobRunner:
    """Runs a single job on a Vast.ai instance via SSH, in a thread.

    The outcome is stored in job_info.aux["outcome"] as
    ("success", None) or ("error", message).
    """

    def __init__(self, executor, job_info, exec_job: str):
        self.executor = executor
        self.vast = executor.vast
        self.logger = executor.logger
        self.settings = executor.settings
        self.job_info = job_info
        self.job = job_info.job
        self.instance_id = int(job_info.external_jobid)
        self.exec_job = exec_job
        self.keyfile = executor.ssh_keyfile
        self.cancel_event = threading.Event()
        job_info.aux["cancel_event"] = self.cancel_event
        self.host = None
        self.port = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    # ------------------------------------------------------------------
    # pipeline
    # ------------------------------------------------------------------

    def _run(self):
        try:
            self._wait_for_boot()
            self._wait_for_ssh()
            self._upload()
            exit_code = self._execute()
            self._download_log()
            if exit_code == 0:
                self._download_outputs()
                self._finish(("success", None))
            else:
                self._finish(
                    ("error", f"Job finished with exit code {exit_code}.")
                )
        except JobCancelled:
            pass
        except Exception as e:
            try:
                self._download_log()
            except Exception:
                pass
            self._finish(("error", str(e)))

    def _finish(self, outcome):
        self.executor._destroy_instance(self.instance_id)
        self.job_info.aux["outcome"] = outcome

    def _check_cancelled(self):
        if self.cancel_event.is_set():
            raise JobCancelled()

    def _wait_for_boot(self):
        deadline = time.time() + self.settings.boot_timeout
        last_status = None
        while time.time() < deadline:
            self._check_cancelled()
            try:
                instance = self.vast.show_instance(self.instance_id)
                if isinstance(instance, list):
                    instance = instance[0] if instance else {}
            except Exception:
                instance = {}
            last_status = instance.get("actual_status")
            if last_status == "running":
                self.host = instance.get("ssh_host") or instance.get(
                    "public_ipaddr"
                )
                self.port = instance.get("ssh_port")
                if self.host and self.port:
                    return
            elif last_status in ("exited", "stopped"):
                raise TransferError(
                    f"Instance terminated while booting (status: "
                    f"{last_status!r}, status_msg: "
                    f"{instance.get('status_msg')!r})."
                )
            elif fatal_boot_error(instance.get("status_msg")):
                raise TransferError(
                    f"Instance failed to start (status_msg: "
                    f"{instance.get('status_msg')!r}). This host will not "
                    "recover; rerun with --retries to resubmit on a "
                    "different machine."
                )
            time.sleep(POLL_SECONDS)
        raise TransferError(
            f"Instance did not become reachable within "
            f"{self.settings.boot_timeout}s (last status: {last_status!r})."
        )

    def _wait_for_ssh(self):
        deadline = time.time() + SSH_RETRY_SECONDS
        reboot_at = time.time() + TUNNEL_REBOOT_AFTER_SECONDS
        rebooted = False
        while True:
            self._check_cancelled()
            result = self._ssh("true", check=False, timeout=30)
            if result.returncode == 0:
                return
            if not rebooted and time.time() > reboot_at:
                # A stuck reverse tunnel on Vast's proxy ("remote port
                # forwarding failed") is a known flake; a reboot
                # re-establishes it with the image already cached.
                if self._tunnel_is_stuck():
                    self.logger.warning(
                        f"Vast.ai instance {self.instance_id}: SSH proxy "
                        "tunnel failed to establish; rebooting the "
                        "instance once to retry."
                    )
                    try:
                        self.vast.reboot_instance(self.instance_id)
                        deadline = time.time() + SSH_RETRY_SECONDS
                        self._wait_for_boot()
                    except Exception as e:
                        self.logger.warning(f"Reboot failed: {e}")
                rebooted = True
            if time.time() > deadline:
                raise TransferError(
                    f"Could not establish an SSH connection to "
                    f"root@{self.host}:{self.port} within "
                    f"{SSH_RETRY_SECONDS}s: {result.stderr.strip()}. "
                    "If you use a custom image, make sure it has OpenSSH "
                    "installed (or installable via apt), or use the "
                    "default image."
                )
            time.sleep(10)

    def _tunnel_is_stuck(self) -> bool:
        try:
            logs = self.vast.logs(
                self.instance_id, tail="50", filter="port forwarding failed"
            )
        except Exception:
            return False
        return "port forwarding failed" in (logs or "")

    def _upload(self):
        self._check_cancelled()
        self._ssh(f"mkdir -p {REMOTE_WORKDIR}")
        # Python discovery, sourced by the remote helper commands below
        # (non-interactive SSH sessions don't get the image ENV's PATH).
        self._ssh(f"cat > {REMOTE_WORKDIR}/env.sh", input_text=PYTHON_PATH_SETUP)

        # Workflow sources (created once per run by the executor).
        self._scp_to(self.executor.source_archive_path, "sources.tar.xz")
        self._remote_extract("sources.tar.xz")

        # Input files, packed preserving relative paths.
        inputs = [f for f in self.job.input if Path(f).exists()]
        if inputs:
            with tempfile.NamedTemporaryFile(suffix=".tar") as tf:
                with tarfile.open(tf.name, "w") as tar:
                    for f in inputs:
                        tar.add(f)
                self._scp_to(Path(tf.name), "inputs.tar")
            self._remote_extract("inputs.tar")

    def _execute(self) -> int:
        self._check_cancelled()
        # Detached so that transient SSH disconnects don't kill the job;
        # completion is signalled through the exit_code file.
        script = (
            f"cd {REMOTE_WORKDIR}\n"
            f"{self.executor.job_setup_script()}"
            f"{self.exec_job}\n"
            "echo $? > exit_code\n"
        )
        self._ssh(f"cat > {REMOTE_WORKDIR}/job.sh", input_text=script)
        self._ssh(
            f"cd {REMOTE_WORKDIR} && "
            "nohup sh job.sh > job.log 2>&1 < /dev/null &"
        )
        while True:
            self._check_cancelled()
            result = self._ssh(
                f"cat {REMOTE_WORKDIR}/exit_code 2>/dev/null",
                check=False,
                timeout=60,
            )
            output = result.stdout.strip()
            if result.returncode == 0 and output:
                return int(output)
            time.sleep(POLL_SECONDS)

    def _download_log(self):
        result = self._ssh(
            f"cat {REMOTE_WORKDIR}/job.log 2>/dev/null", check=False
        )
        if result.returncode == 0 and result.stdout:
            log_file = self.executor.log_dir / f"{self.job_info.aux['label']}.log"
            log_file.write_text(result.stdout)
            self.job_info.aux["log_file"] = str(log_file)

    def _download_outputs(self):
        files = [str(f) for f in self.job.output]
        files.extend(str(f) for f in getattr(self.job, "log", []) or [])
        if not files:
            return
        # Pack remotely with python (always present) instead of tar (feature
        # set varies between images); the file list goes in base64-encoded
        # JSON to dodge shell quoting.
        payload = base64.b64encode(json.dumps(files).encode()).decode()
        pack = (
            "import tarfile, os, json, base64; "
            f"files = json.loads(base64.b64decode('{payload}')); "
            "t = tarfile.open('outputs.tar', 'w'); "
            "[t.add(f) for f in files if os.path.exists(f)]; "
            "t.close()"
        )
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tf:
            local_tar = Path(tf.name)
        try:
            self._ssh(f'cd {REMOTE_WORKDIR} && . ./env.sh && python -c "{pack}"')
            self._scp_from("outputs.tar", local_tar)
            with tarfile.open(local_tar) as tar:
                tar.extractall(".", filter="data")
        finally:
            local_tar.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # ssh plumbing
    # ------------------------------------------------------------------

    def _ssh_options(self):
        return [
            "-i",
            str(self.keyfile),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
        ]

    def _ssh(self, command, check=True, input_text=None, timeout=3600):
        argv = (
            ["ssh", "-p", str(self.port)]
            + self._ssh_options()
            + [f"root@{self.host}", command]
        )
        try:
            result = subprocess.run(
                argv,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            result = subprocess.CompletedProcess(
                argv, 255, "", f"ssh timed out after {timeout}s"
            )
        if check and result.returncode != 0:
            raise TransferError(
                f"Remote command failed ({command[:80]}...): "
                f"{result.stderr.strip()}"
            )
        return result

    def _scp(self, source, destination):
        argv = (
            ["scp", "-q", "-P", str(self.port)]
            + self._ssh_options()
            + [str(source), str(destination)]
        )
        result = subprocess.run(argv, capture_output=True, text=True)
        if result.returncode != 0:
            raise TransferError(
                f"File transfer failed ({source} -> {destination}): "
                f"{result.stderr.strip()}"
            )

    def _scp_to(self, local_path, remote_name):
        self._scp(local_path, f"root@{self.host}:{REMOTE_WORKDIR}/{remote_name}")

    def _scp_from(self, remote_name, local_path):
        self._scp(f"root@{self.host}:{REMOTE_WORKDIR}/{remote_name}", local_path)

    def _remote_extract(self, archive_name):
        # python is guaranteed in the image (snakemake needs it); tar with xz
        # support is not.
        self._ssh(
            f"cd {REMOTE_WORKDIR} && . ./env.sh && python -c "
            f"\"import tarfile; tarfile.open('{archive_name}').extractall()\" "
            f"&& rm {archive_name}"
        )
