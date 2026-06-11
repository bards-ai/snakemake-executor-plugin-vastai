---
name: vastai-executor
description: Testing the snakemake-executor-plugin-vastai (unit + live e2e on real GPUs), debugging live instances, and the design decisions behind the plugin. Use when working on this repo, running its tests, or debugging Vast.ai job failures.
---

# Testing

- Unit (free, fast): `uv run pytest` — covers query building, status state
  machine, scripts, credentials.
- Live e2e (rents a real GPU, ~$0.02–0.05, needs `VAST_API_KEY` with credit):
  `SNAKEMAKE_VASTAI_E2E=1 pytest tests/test_e2e.py -s`
- After ANY live run, verify nothing is still billing:
  `vastai show instances --raw` must show no `snakemake-*` labels. Destroy
  stragglers immediately (`vastai destroy instance <id> -y`).

## Debugging a live run

- Run snakemake from a git-initialized scratch dir (source archiving needs git).
- Status: `VastAI().show_instance(id)` → `actual_status` + `status_msg`.
  Container logs: `VastAI().logs(id, tail='200')`.
- SSH-mode instances: connect with the per-run key
  `.snakemake/tmp.*/vastai_ssh_key`; remote workdir is `/snakemake-workdir`
  (`job.sh`, `job.log`, `exit_code`, `env.sh`).
- Watch costs: prefer `--vastai-max-price`, small `--vastai-disk`, EU
  datacenter offers (~$0.06–0.11/h for RTX 3060-class).

# Design decisions (all verified on real instances, 2026-06)

- **Two modes**, chosen by whether `--default-storage-provider` is set:
  - *Storage mode*: entrypoint containers (`runtype="args"`); completion =
    instance `exited` + `snakemake_vastai_exit_code=N` log sentinel; logs
    fetched once at finalization.
  - *SSH mode* (zero config): `runtype="ssh_proxy"` + per-job thread in
    `sshtransfer.py` (scp sources/inputs, detached run, poll `exit_code`,
    scp outputs). Enabled by `can_transfer_local_files=True`; in storage
    mode the `Executor.common_settings` property presents a doctored copy
    so core's source-deploy precommand is regenerated, and the executor
    calls `workflow.upload_sources()` itself.
- Plain `runtype="ssh"` never gets the proxy tunnel wired — must be `ssh_proxy`.
- **Default image** `vastai/pytorch:@vastai-automatic-tag`: driver-matched
  CUDA + OpenSSH (`snakemake/snakemake` has neither). Snakemake and the
  locally installed storage plugins are pip-installed in-job, version-pinned
  (their settings leak into spawned CLI args, e.g. `--storage-s3-retries`).
- **SSH keys via onstart** writing `authorized_keys` — the attach-key API on
  a live instance is racy; account keys would leak across runs. apt-based
  OpenSSH bootstrap for custom Debian images (conda installs are unusable).
- Non-interactive SSH doesn't get the image ENV PATH → `PYTHON_PATH_SETUP`
  prelude discovers python, preferring `/venv/main` (where torch lives).
- **`datacenter=true` is the search default**: hobbyist hosts routinely hit
  Docker Hub pull rate limits. Unrecoverable boot errors
  (`FATAL_BOOT_ERRORS`) fail fast so `--retries` resubmits elsewhere; a
  stuck proxy tunnel triggers one instance reboot.
- Interface bug: bool settings with `default=True` break CLI parsing
  (argparse_dataclass renames the flag) → only default-False bools, hence
  `--vastai-no-datacenter`, `--vastai-no-forward-credentials`.
- Credentials: snakemake core forwards storage-plugin settings itself; the
  plugin additionally forwards ambient `AWS_*`/`AZURE_STORAGE_*` vars and
  ships the GCP credentials *file content* base64-encoded (path-based
  GOOGLE_APPLICATION_CREDENTIALS is useless remotely).
