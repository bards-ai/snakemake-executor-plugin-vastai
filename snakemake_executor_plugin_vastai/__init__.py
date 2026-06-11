__author__ = "Michał Pogoda"
__copyright__ = "Copyright 2026, bards.ai"
__email__ = "michal.pogoda@bards.ai"
__license__ = "MIT"

import base64
import importlib.metadata
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import AsyncGenerator, List, Mapping, Optional

from snakemake_interface_common.exceptions import WorkflowError
from snakemake_interface_executor_plugins.executors.base import SubmittedJobInfo
from snakemake_interface_executor_plugins.executors.remote import RemoteExecutor
from snakemake_interface_executor_plugins.jobs import JobExecutorInterface
from snakemake_interface_executor_plugins.settings import (
    CommonSettings,
    ExecutorSettingsBase,
)

from snakemake_executor_plugin_vastai._common import (
    PYTHON_PATH_SETUP,
    fatal_boot_error,
)
from snakemake_executor_plugin_vastai.sshtransfer import SSHJobRunner

# Printed by the job wrapper script as the very last line of container output.
# check_active_jobs() parses it from the instance logs to obtain the exit code.
EXIT_CODE_PATTERN = re.compile(r"snakemake_vastai_exit_code=(\d+)")

# Number of cheapest matching offers to attempt when creating an instance.
# Offers are single-use and can be taken by other users between search and
# create, so we fall through to the next candidate on failure.
OFFER_ATTEMPTS = 5

# Instances reporting stopped/offline/unknown may recover (host heartbeat
# loss is often transient); only give up after this many seconds.
UNREACHABLE_GRACE_SECONDS = 300

# Tolerated consecutive failures of the status API before declaring the
# instance lost.
MAX_STATUS_API_MISSES = 5


# Local environment variables that are forwarded to job containers when
# forward_credentials is enabled, so that the default storage provider (e.g.
# S3, GCS, Azure, or any S3-compatible service) works remotely without any
# --envvars ceremony.
CREDENTIAL_ENVVARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AWS_ENDPOINT_URL",
    "GOOGLE_CLOUD_PROJECT",
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_STORAGE_ACCOUNT",
    "AZURE_STORAGE_KEY",
    "AZURE_STORAGE_SAS_TOKEN",
)

# Google credentials are a *file* (GOOGLE_APPLICATION_CREDENTIALS points to
# it), so the content is shipped base64-encoded in this variable and
# materialized inside the container by the credential setup script.
GCP_CREDENTIALS_CONTENT_VAR = "SNAKEMAKE_VASTAI_GCP_CREDENTIALS_B64"

GCP_ADC_DEFAULT_PATH = "~/.config/gcloud/application_default_credentials.json"

# Region shortcuts for the geolocation setting, matching the vastai CLI's
# georegion expansion.
GEO_REGIONS = {
    "EU": (
        "AL,AD,AT,BY,BE,BA,BG,HR,CY,CZ,DK,EE,FI,FR,GE,DE,GR,HU,IS,IT,KZ,LV,"
        "LI,LT,LU,MT,MD,MC,ME,NL,NO,PL,PT,RO,RU,RS,SK,SI,ES,SE,CH,UA,GB,VA,MK"
    ),
    "NA": "CA,US",
    "AS": (
        "AE,AM,AR,AU,AZ,BD,BH,BN,BT,MM,KH,KP,IN,ID,IR,IQ,IL,JP,JO,KZ,LV,LI,"
        "MY,MV,MN,NP,KR,PK,PH,QA,SA,SG,LK,SY,TW,TJ,TH,TR,TM,VN,YE,HK,CN,OM"
    ),
    "AF": (
        "DZ,AO,BJ,BW,BF,BI,CM,CV,CF,TD,KM,CG,CD,DJ,EG,GQ,ER,ET,GA,GM,GH,GN,"
        "GW,KE,LS,LR,LY,MW,MA,ML,MR,MU,MZ,NA,NE,NG,RW,SH,ST,SN,SC,SL,SO,ZA,"
        "SS,SD,SZ,TZ,TG,TN,UG,YE,ZM,ZW"
    ),
    "LC": (
        "AG,AR,BS,BB,BZ,BO,BR,CL,CO,CR,CU,DO,EC,SV,GY,HT,HN,JM,MX,NI,PA,PY,"
        "PE,PR,RD,SUR,TT,UR,VZ"
    ),
    "OC": "AU,FJ,GU,KI,MH,FM,NR,NZ,PG,PW,SL,TO,TV,VU",
}

# Vast.ai's curated PyTorch image: CUDA matched to the machine's driver via
# the server-side @vastai-automatic-tag, OpenSSH preinstalled (required for
# SSH transfer mode), python on PATH. Snakemake itself is pip-installed by
# the job bootstrap below.
DEFAULT_IMAGE = "vastai/pytorch:@vastai-automatic-tag"

CREDENTIAL_SETUP_SCRIPT = (
    f'if [ -n "${{{GCP_CREDENTIALS_CONTENT_VAR}:-}}" ]; then\n'
    f'  echo "${{{GCP_CREDENTIALS_CONTENT_VAR}}}" | base64 -d '
    "> /tmp/gcp-credentials.json\n"
    "  export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-credentials.json\n"
    f"  unset {GCP_CREDENTIALS_CONTENT_VAR}\n"
    "fi\n"
)


@dataclass
class ExecutorSettings(ExecutorSettingsBase):
    api_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "Vast.ai API key. If not set, it is resolved like the vastai "
            "CLI does: from the VAST_API_KEY environment variable or from "
            "~/.config/vastai/vast_api_key.",
            "env_var": True,
            "required": False,
        },
    )
    gpu_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Default GPU model to request, in Vast.ai naming with "
            "underscores for spaces (e.g. RTX_4090, H100_SXM, A100_SXM4). "
            "Can be overridden per job with the gpu_model resource. If unset, "
            "any GPU matching the other constraints is used.",
        },
    )
    image: Optional[str] = field(
        default=None,
        metadata={
            "help": "Docker image to run jobs in. Defaults to "
            f"{DEFAULT_IMAGE} (CUDA+PyTorch matched to the machine, OpenSSH "
            "included); an explicitly set --container-image takes "
            "precedence over that default. Snakemake is pip-installed into "
            "the container automatically if the image does not provide it; "
            "bake `pip install snakemake` into your own image to skip that "
            "startup cost.",
        },
    )
    disk: float = field(
        default=40.0,
        metadata={
            "help": "Disk space to allocate per instance in GB. Can be raised "
            "per job with the disk_mb resource.",
        },
    )
    max_price: Optional[float] = field(
        default=None,
        metadata={
            "help": "Maximum on-demand price per instance in $/h (dph_total, "
            "including storage). Offers above this price are never rented.",
        },
    )
    reliability: float = field(
        default=0.98,
        metadata={
            "help": "Minimum host reliability score (0-1) required for offers. "
            "Set to 0 to disable the filter.",
        },
    )
    order: str = field(
        default="dph_total",
        metadata={
            "help": "Sort order for choosing among matching offers (vastai "
            "search order syntax; append '-' for descending). The default "
            "rents the cheapest matching offer. Use e.g. 'dlperf_usd-' for "
            "best DL performance per dollar.",
        },
    )
    geolocation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Restrict offers to a geographic area: a region "
            "shortcut (EU, NA, AS, AF, LC, OC) or comma-separated ISO "
            "country codes (e.g. 'PL,DE,CZ').",
        },
    )
    no_datacenter: bool = field(
        default=False,
        metadata={
            "help": "Also rent from non-datacenter (hobbyist) hosts. By "
            "default only verified datacenter hosts are used — they are "
            "slightly pricier but avoid the most common marketplace "
            "flakiness (Docker Hub pull rate limits, slow residential "
            "uplinks). Disabling the filter gives access to the cheapest "
            "offers.",
        },
    )
    search_query: Optional[str] = field(
        default=None,
        metadata={
            "help": "Additional filters appended to every offer search, in "
            "vastai query syntax (e.g. 'cuda_vers>=12.4 geolocation=EU "
            "inet_down>=500'). Can be extended per job with the vastai_query "
            "resource.",
        },
    )
    boot_timeout: int = field(
        default=1800,
        metadata={
            "help": "Maximum seconds to wait for an instance to reach the "
            "running state (includes docker image pull) before the job is "
            "failed and the instance destroyed.",
        },
    )
    no_forward_credentials: bool = field(
        default=False,
        metadata={
            "help": "Do not forward cloud storage credentials (AWS_*, "
            "AZURE_STORAGE_*, GOOGLE_* including the application "
            "credentials file content) from the local environment into the "
            "job containers. By default they are forwarded so that the "
            "default storage provider works remotely without declaring "
            "--envvars. Credentials configured via the storage plugin's own "
            "settings (e.g. --storage-s3-access-key or "
            "SNAKEMAKE_STORAGE_S3_ACCESS_KEY) are always forwarded by "
            "Snakemake itself, independent of this option.",
        },
    )
    keep_instances: bool = field(
        default=False,
        metadata={
            "help": "Do not destroy instances after job completion or failure. "
            "Useful for debugging, but instances keep accruing charges until "
            "destroyed manually!",
        },
    )


common_settings = CommonSettings(
    non_local_exec=True,
    implies_no_shared_fs=True,
    job_deploy_sources=True,
    pass_default_storage_provider_args=True,
    pass_default_resources_args=True,
    # Envvars are injected through the container environment instead of being
    # inlined into the command (see run_job).
    pass_envvar_declarations_to_cmd=False,
    # Lifts Snakemake's requirement for a default storage provider: with a
    # storage provider the executor restores the storage-based behavior
    # itself (see Executor.common_settings), without one it transfers files
    # over SSH (see sshtransfer.py).
    can_transfer_local_files=True,
    auto_deploy_default_storage_provider=False,
    # Instances need to be scheduled and pull the image first; no point in
    # checking immediately.
    init_seconds_before_status_checks=30,
)


def build_offer_query(
    settings: ExecutorSettings,
    resources: Mapping,
    threads: int,
) -> str:
    """Build a vastai offer search query for a job.

    Resources understood: gpu / nvidia_gpu (count), gpu_model, mem_mb,
    disk_mb, vastai_query (extra filters in vastai query syntax).
    """
    parts = []

    num_gpus = resources.get("gpu", resources.get("nvidia_gpu", 1))
    try:
        num_gpus = max(1, int(num_gpus))
    except (TypeError, ValueError):
        raise WorkflowError(
            f"Resource gpu/nvidia_gpu must be an integer, got {num_gpus!r}."
        )
    parts.append(f"num_gpus={num_gpus}")

    gpu_model = resources.get("gpu_model") or settings.gpu_name
    if gpu_model:
        parts.append(f"gpu_name={str(gpu_model).replace(' ', '_')}")

    if threads:
        parts.append(f"cpu_cores_effective>={int(threads)}")

    mem_mb = resources.get("mem_mb")
    if mem_mb:
        # cpu_ram is specified in GB in vastai query syntax.
        parts.append(f"cpu_ram>={float(mem_mb) / 1000:g}")

    parts.append(f"disk_space>={required_disk_gb(settings, resources):g}")

    if settings.reliability > 0:
        parts.append(f"reliability>{settings.reliability:g}")

    if settings.geolocation:
        geo = settings.geolocation.strip()
        codes = GEO_REGIONS.get(geo.upper(), geo)
        parts.append(f"geolocation in [{codes}]")

    if not settings.no_datacenter:
        parts.append("datacenter=true")

    if settings.max_price is not None:
        parts.append(f"dph_total<={settings.max_price:g}")

    if settings.search_query:
        parts.append(settings.search_query)

    job_query = resources.get("vastai_query")
    if job_query:
        parts.append(str(job_query))

    return " ".join(parts)


def required_disk_gb(settings: ExecutorSettings, resources: Mapping) -> float:
    disk_mb = resources.get("disk_mb")
    disk_gb = float(disk_mb) / 1000 if disk_mb else 0.0
    return max(float(settings.disk), disk_gb)


def resolve_container_image(
    settings_image: Optional[str], snakemake_image: Optional[str]
) -> str:
    """Pick the job image: --vastai-image > explicit --container-image >
    the Vast.ai PyTorch default.

    Snakemake's own default for --container-image is snakemake/snakemake,
    which lacks both CUDA and OpenSSH, so it is replaced unless the user
    explicitly asked for something.
    """
    if settings_image:
        return settings_image
    if snakemake_image and not snakemake_image.startswith("snakemake/snakemake"):
        return snakemake_image
    return DEFAULT_IMAGE


def _pip_pin(package: str) -> str:
    try:
        return f"{package}=={importlib.metadata.version(package)}"
    except importlib.metadata.PackageNotFoundError:
        return package


def snakemake_bootstrap_script(with_storage_plugins: bool = False) -> str:
    """Shell snippet installing snakemake into the container, pinned to the
    local version so spawned-job CLI args stay compatible.

    With with_storage_plugins (SSH mode), the locally installed storage
    plugins are installed too: their settings are serialized into every job
    command (e.g. --storage-s3-retries), which the remote snakemake can only
    parse with the plugins present. Storage mode covers this via the
    auto-deploy precommand instead.
    """
    pins = [_pip_pin("snakemake")]
    if with_storage_plugins:
        from snakemake_interface_storage_plugins.registry import (
            StoragePluginRegistry,
        )

        registry = StoragePluginRegistry()
        pins.extend(
            sorted(
                _pip_pin(registry.get_plugin_package_name(name))
                for name in registry.get_registered_plugins()
            )
        )
    quoted = " ".join(f'"{p}"' for p in pins)
    return (
        f"echo '[snakemake-vastai] ensuring job dependencies: {len(pins)} "
        "package(s)'\n"
        f"python -m pip install --quiet {quoted}\n"
    )


def credential_envvars(environ: Mapping[str, str] = os.environ) -> dict:
    env = {var: environ[var] for var in CREDENTIAL_ENVVARS if environ.get(var)}
    gcp_path = environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gcp_path is None and environ is os.environ:
        # Fall back to gcloud's application default credentials.
        gcp_path = os.path.expanduser(GCP_ADC_DEFAULT_PATH)
    if gcp_path:
        try:
            with open(gcp_path, "rb") as f:
                env[GCP_CREDENTIALS_CONTENT_VAR] = base64.b64encode(
                    f.read()
                ).decode()
        except OSError:
            pass
    return env


class Executor(RemoteExecutor):
    def __post_init__(self):
        # Imported lazily so that merely having the plugin installed does not
        # require the vastai package at snakemake startup of other executors.
        from vastai import VastAI

        self.settings: ExecutorSettings = self.workflow.executor_settings
        self.vast = VastAI(api_key=self.settings.api_key, raw=True, quiet=True)
        self.container_image = resolve_container_image(
            self.settings.image,
            self.workflow.remote_execution_settings.container_image,
        )
        self.run_id = uuid.uuid4().hex[:8]
        self.log_dir = Path(self.workflow.persistence.aux_path) / "vastai-logs"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info(
            f"Using container image {self.container_image} for Vast.ai jobs."
        )
        if not self.settings.no_forward_credentials:
            forwarded = sorted(credential_envvars())
            if forwarded:
                self.logger.info(
                    "Forwarding credentials to job containers: "
                    f"{', '.join(forwarded)} (disable with "
                    "--vastai-no-forward-credentials)."
                )

        self._destroyed_instances = set()
        self.storage_mode = (
            self.workflow.storage_registry.default_storage_provider is not None
        )
        if self.storage_mode:
            # Snakemake core skips the source upload when
            # can_transfer_local_files is set, so it is done here instead.
            self.workflow.upload_sources()
        else:
            self.logger.info(
                "No default storage provider configured; transferring files "
                "between this machine and the instances over SSH. For large "
                "data or many jobs, an S3-compatible bucket "
                "(--default-storage-provider s3) is faster and more robust."
            )
            self._init_ssh_transfer()

    def _init_ssh_transfer(self):
        for tool in ("ssh", "scp", "ssh-keygen"):
            if shutil.which(tool) is None:
                raise WorkflowError(
                    f"SSH transfer mode requires '{tool}' on PATH. Install "
                    "OpenSSH, or configure a default storage provider "
                    "instead (--default-storage-provider)."
                )
        self.ssh_keyfile = Path(self.tmpdir) / "vastai_ssh_key"
        subprocess.run(
            [
                "ssh-keygen",
                "-t",
                "ed25519",
                "-N",
                "",
                "-q",
                "-C",
                f"snakemake-vastai-{self.run_id}",
                "-f",
                str(self.ssh_keyfile),
            ],
            check=True,
        )
        self.ssh_pubkey = (
            self.ssh_keyfile.with_suffix(".pub").read_text().strip()
        )
        self.source_archive_path = Path(self.tmpdir) / "sources.tar.xz"
        self.workflow.write_source_archive(self.source_archive_path)

    @property
    def common_settings(self):
        # The module-level can_transfer_local_files=True makes Snakemake core
        # skip both the storage requirement and the storage-based source
        # deployment. In storage mode the latter is wanted after all, so the
        # spawned-job machinery is presented with settings that re-enable it
        # (pip-install of storage plugins + `snakemake --deploy-sources` in
        # the job's precommand).
        if getattr(self, "storage_mode", False):
            return replace(
                common_settings,
                can_transfer_local_files=False,
                auto_deploy_default_storage_provider=True,
            )
        return common_settings

    def job_setup_script(self) -> str:
        """Shell lines run inside the container before the job command: put
        python on PATH, materialize shipped credentials, install snakemake
        if absent."""
        return (
            PYTHON_PATH_SETUP
            + CREDENTIAL_SETUP_SCRIPT
            + snakemake_bootstrap_script(
                with_storage_plugins=not self.storage_mode
            )
        )

    def _ssh_onstart_script(self) -> str:
        """Instance startup script for SSH transfer mode.

        Provisions the per-run SSH key by writing authorized_keys directly —
        the attach-key API on a live instance is known to be racy, while
        onstart runs as the container user before sshd accepts connections.
        Also best-effort installs OpenSSH for images that lack it (Vast's
        launch script keeps retrying the proxy tunnel until `ssh` appears).
        """
        log = "/tmp/snakemake-vastai-openssh.log"
        return (
            "#!/bin/sh\n"
            "if ! command -v ssh >/dev/null 2>&1 "
            "|| ! command -v sshd >/dev/null 2>&1; then\n"
            "  echo '[snakemake-vastai] image lacks OpenSSH, installing it'\n"
            "  { apt-get update && apt-get install -y openssh-server "
            f"openssh-client; }} >{log} 2>&1 \\\n"
            "    || echo '[snakemake-vastai] could not install OpenSSH; "
            "SSH transfer will fail (use an image with OpenSSH, e.g. the "
            "default vastai/pytorch)'\n"
            "fi\n"
            "for d in \"$HOME\" /root; do\n"
            "  mkdir -p \"$d/.ssh\" 2>/dev/null || continue\n"
            f"  echo '{self.ssh_pubkey}' >> \"$d/.ssh/authorized_keys\"\n"
            "  chmod 700 \"$d/.ssh\"\n"
            "  chmod 600 \"$d/.ssh/authorized_keys\"\n"
            "done\n"
        )

    def run_job(self, job: JobExecutorInterface):
        exec_job = self.format_job_exec(job)
        query = build_offer_query(self.settings, dict(job.resources), job.threads)
        disk_gb = required_disk_gb(self.settings, job.resources)
        self.logger.debug(f"Searching Vast.ai offers with query: {query}")

        try:
            offers = self.vast.search_offers(
                query=query,
                type="on-demand",
                order=self.settings.order,
                limit=OFFER_ATTEMPTS,
                storage=disk_gb,
            )
        except Exception as e:
            raise WorkflowError(f"Failed to search Vast.ai offers: {e}")
        if not offers:
            raise WorkflowError(
                f"No Vast.ai offers match the requirements of job {job.name} "
                f"(query: {query}). Relax the filters (e.g. --vastai-max-price, "
                "--vastai-gpu-name, --vastai-search-query) or free up budget."
            )

        label = f"snakemake-{self.run_id}-{job.jobid}-{job.attempt}"
        if self.storage_mode:
            # The wrapper runs the job command and emits the exit code as the
            # last log line, where check_active_jobs() picks it up. The
            # container exits afterwards (entrypoint mode), flipping the
            # instance to 'exited'.
            script = (
                f"echo '[snakemake-vastai] starting job {job.name} "
                f"(jobid={job.jobid}, attempt={job.attempt})'\n"
                "mkdir -p /snakemake-workdir && cd /snakemake-workdir\n"
                f"{self.job_setup_script()}"
                f"{exec_job}\n"
                "ec=$?\n"
                'echo "snakemake_vastai_exit_code=${ec}"\n'
                "exit ${ec}\n"
            )
            create_kwargs = dict(
                runtype="args",
                onstart_cmd="/bin/sh",
                args=["-c", script],
            )
        else:
            # SSH transfer mode: the container only has to be alive and
            # reachable; an SSHJobRunner thread drives the job. Note that
            # only "ssh_proxy" gets the proxy tunnel wired up (plain "ssh"
            # leaves the ssh_host:ssh_port unreachable).
            create_kwargs = dict(
                runtype="ssh_proxy",
                onstart_cmd=self._ssh_onstart_script(),
            )

        instance_id = None
        last_error = None
        for offer in offers:
            try:
                response = self.vast.create_instance(
                    id=offer["id"],
                    image=self.container_image,
                    disk=disk_gb,
                    label=label,
                    env=self._container_env(),
                    cancel_unavail=True,
                    **create_kwargs,
                )
            except Exception as e:
                # Offers are single-use; this one was likely taken in the
                # meantime. Try the next candidate.
                last_error = e
                continue
            if response.get("success"):
                instance_id = response["new_contract"]
                self.logger.info(
                    f"Job {job.jobid} ({job.name}): rented Vast.ai instance "
                    f"{instance_id} ({offer.get('num_gpus')}x "
                    f"{offer.get('gpu_name')} @ ${offer.get('dph_total', 0):.3f}/h, "
                    f"machine {offer.get('machine_id')})"
                )
                break
            last_error = WorkflowError(f"Unexpected response: {response}")

        if instance_id is None:
            raise WorkflowError(
                f"Could not rent any of the {len(offers)} matching Vast.ai "
                f"offers for job {job.name}. Last error: {last_error}"
            )

        job_info = SubmittedJobInfo(
            job=job,
            external_jobid=str(instance_id),
            aux={
                "submitted": time.time(),
                "label": label,
                "seen_running": False,
                "unreachable_since": None,
                "api_misses": 0,
            },
        )

        if not self.storage_mode:
            try:
                self.vast.attach_ssh(instance_id, self.ssh_pubkey)
            except Exception as e:
                self._destroy_instance(instance_id)
                raise WorkflowError(
                    f"Failed to attach SSH key to Vast.ai instance "
                    f"{instance_id}: {e}"
                )
            runner = SSHJobRunner(self, job_info, exec_job)
            job_info.aux["runner"] = runner
            runner.start()

        self.report_job_submission(job_info)

    def _container_env(self) -> dict:
        env = {}
        if not self.settings.no_forward_credentials:
            env.update(credential_envvars())
        # Declared envvars and storage plugin secrets (forwarded by Snakemake
        # itself) take precedence over ambient credentials.
        env.update(self.envvars())
        return env

    async def check_active_jobs(
        self, active_jobs: List[SubmittedJobInfo]
    ) -> AsyncGenerator[SubmittedJobInfo, None]:
        for active_job in active_jobs:
            async with self.status_rate_limiter:
                still_active = self._check_job(active_job)
            if still_active:
                yield active_job

    def _check_job(self, job_info: SubmittedJobInfo) -> bool:
        """Check one job; returns True if it is still running."""
        if not self.storage_mode:
            return self._check_ssh_job(job_info)

        aux = job_info.aux
        instance_id = int(job_info.external_jobid)

        try:
            instance = self.vast.show_instance(instance_id)
            if isinstance(instance, list):
                instance = instance[0] if instance else {}
        except Exception as e:
            aux["api_misses"] += 1
            if aux["api_misses"] > MAX_STATUS_API_MISSES:
                self._finalize_job(job_info, instance={})
                return False
            self.logger.warning(
                f"Failed to query status of Vast.ai instance {instance_id} "
                f"({aux['api_misses']}/{MAX_STATUS_API_MISSES}): {e}"
            )
            return True
        aux["api_misses"] = 0

        status = instance.get("actual_status") if instance else None

        if not instance:
            # Instance vanished (e.g. destroyed externally).
            self._finalize_job(job_info, instance={})
            return False

        if status == "running":
            aux["seen_running"] = True
            aux["unreachable_since"] = None
            return True

        if status == "exited":
            # Container entrypoint finished; outcome is in the logs.
            self._finalize_job(job_info, instance)
            return False

        if status in (None, "", "created", "loading", "rebooting"):
            if not aux["seen_running"] and fatal_boot_error(
                instance.get("status_msg")
            ):
                self._fail_job(
                    job_info,
                    f"Instance failed to start "
                    f"(status_msg: {instance.get('status_msg')!r}). This "
                    "host will not recover; rerun with --retries to "
                    "resubmit on a different machine.",
                )
                return False
            elapsed = time.time() - aux["submitted"]
            if not aux["seen_running"] and elapsed > self.settings.boot_timeout:
                self._fail_job(
                    job_info,
                    f"Instance did not reach running state within "
                    f"{self.settings.boot_timeout}s (status: {status!r}, "
                    f"status_msg: {instance.get('status_msg')!r}).",
                )
                return False
            return True

        # stopped / offline / unknown / anything unexpected: possibly a
        # transient host problem, give it a grace period.
        now = time.time()
        if aux["unreachable_since"] is None:
            aux["unreachable_since"] = now
            self.logger.warning(
                f"Vast.ai instance {instance_id} reports status {status!r} "
                f"(status_msg: {instance.get('status_msg')!r}); waiting up to "
                f"{UNREACHABLE_GRACE_SECONDS}s for it to recover."
            )
        if now - aux["unreachable_since"] > UNREACHABLE_GRACE_SECONDS:
            self._finalize_job(job_info, instance)
            return False
        return True

    def _check_ssh_job(self, job_info: SubmittedJobInfo) -> bool:
        """Inspect the outcome recorded by the job's SSHJobRunner thread."""
        runner = job_info.aux["runner"]
        outcome = job_info.aux.get("outcome")
        if outcome is None:
            if not runner.thread.is_alive():
                self._destroy_instance(int(job_info.external_jobid))
                self.report_job_error(
                    job_info,
                    msg="The transfer worker for this job died unexpectedly. ",
                )
                return False
            return True
        kind, msg = outcome
        log_file = job_info.aux.get("log_file")
        if kind == "success":
            self.report_job_success(job_info)
        else:
            self.report_job_error(
                job_info,
                msg=f"Vast.ai instance {job_info.external_jobid}: {msg} ",
                aux_logs=[log_file] if log_file else None,
            )
        return False

    def _finalize_job(self, job_info: SubmittedJobInfo, instance: dict):
        """Fetch logs, parse the exit code, destroy the instance and report."""
        instance_id = int(job_info.external_jobid)

        log_text = ""
        try:
            log_text = self.vast.logs(instance_id, tail="10000") or ""
        except Exception as e:
            self.logger.warning(
                f"Could not retrieve logs of Vast.ai instance {instance_id}: {e}"
            )

        log_file = None
        if log_text:
            log_file = self.log_dir / f"{job_info.aux['label']}.log"
            log_file.write_text(log_text)

        self._destroy_instance(instance_id)

        matches = EXIT_CODE_PATTERN.findall(log_text)
        exit_code = int(matches[-1]) if matches else None

        if exit_code == 0:
            self.report_job_success(job_info)
        elif exit_code is not None:
            self.report_job_error(
                job_info,
                msg=f"Job finished with exit code {exit_code} on Vast.ai "
                f"instance {instance_id}. ",
                aux_logs=[str(log_file)] if log_file else None,
            )
        else:
            status = instance.get("actual_status") if instance else "gone"
            status_msg = instance.get("status_msg") if instance else None
            self.report_job_error(
                job_info,
                msg=f"Vast.ai instance {instance_id} terminated without "
                f"reporting a job exit code (instance status: {status!r}, "
                f"status_msg: {status_msg!r}). The host may have failed or "
                "the container crashed. ",
                aux_logs=[str(log_file)] if log_file else None,
            )

    def _fail_job(self, job_info: SubmittedJobInfo, msg: str):
        instance_id = int(job_info.external_jobid)
        self._destroy_instance(instance_id)
        self.report_job_error(
            job_info, msg=f"Vast.ai instance {instance_id}: {msg} "
        )

    def _destroy_instance(self, instance_id: int):
        if instance_id in self._destroyed_instances:
            return
        if self.settings.keep_instances:
            self.logger.info(
                f"Keeping Vast.ai instance {instance_id} as requested "
                "(--vastai-keep-instances). Remember to destroy it manually, "
                "it accrues charges until then!"
            )
            return
        try:
            self.vast.destroy_instance(instance_id)
            self._destroyed_instances.add(instance_id)
            self.logger.debug(f"Destroyed Vast.ai instance {instance_id}.")
        except Exception as e:
            self.logger.error(
                f"Failed to destroy Vast.ai instance {instance_id}: {e}. "
                "Please destroy it manually at https://console.vast.ai/instances/ "
                "to stop billing."
            )

    def cancel_jobs(self, active_jobs: List[SubmittedJobInfo]):
        for job_info in active_jobs:
            self.logger.info(
                f"Cancelling job {job_info.job.jobid} on Vast.ai instance "
                f"{job_info.external_jobid}."
            )
            cancel_event = job_info.aux.get("cancel_event")
            if cancel_event is not None:
                cancel_event.set()
            self._destroy_instance(int(job_info.external_jobid))
