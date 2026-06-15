# snakemake-executor-plugin-vastai

Run [Snakemake](https://snakemake.github.io) jobs on cheap
[Vast.ai](https://vast.ai) GPUs. For every job the plugin rents the cheapest
marketplace offer matching the job's resources, runs it in a Docker
container, ships the files back, and destroys the instance.

## Quickstart

```bash
pip install snakemake-executor-plugin-vastai
export VAST_API_KEY=...   # https://console.vast.ai/manage-keys/
```

Write a Snakefile as usual — resources decide what gets rented:

```python
rule train:
    input: "data/train.parquet"
    output: "models/model.pt"
    threads: 8
    resources:
        gpu=1,
        gpu_model="RTX_4090",
        mem_mb=32000,
    shell:
        "python train.py {input} {output}"
```

Run it:

```bash
snakemake --executor vastai --jobs 4
```

That's all — no bucket, no shared filesystem, no cluster setup. Jobs run in
Vast.ai's PyTorch CUDA image, and every instance is destroyed as soon as
its job finishes, fails, or you hit Ctrl-C.

### What ends up on the machine

Each instance receives two uploads before the job starts:

1. **Your code**, as Snakemake's source archive: the Snakefile (plus
   anything you `include:`), config files, `script:`/`notebook:` files, and
   **every git-tracked file** in the workflow directory. In the example
   above, `train.py` gets there this way — which is why the workflow must
   be a git repository and why an uncommitted-and-untracked script would be
   missing remotely (`git add` is enough, no commit needed). Files over
   10 MB are skipped — declare those as `input:` instead.
2. **The job's declared `input:` files** (`data/train.parquet` above),
   preserving their relative paths.

The job then runs in the same relative layout, and only its declared
`output:` (and `log:`) files are copied back to your machine. Anything else
written on the instance is discarded when the instance is destroyed.

## Configuration

Pin your defaults in `profiles/default/config.yaml` next to the Snakefile —
Snakemake loads it automatically, so the command stays plain `snakemake`:

```yaml
executor: vastai
jobs: 4
vastai-max-price: 1.0     # $/h cap per instance
vastai-geolocation: EU    # region shortcut or country codes (PL,DE,CZ)
```

| Option | Default | Description |
|---|---|---|
| `--vastai-api-key` | – | API key (or `VAST_API_KEY` / `SNAKEMAKE_VASTAI_API_KEY`) |
| `--vastai-gpu-name` | any | Default GPU model, e.g. `RTX_4090`, `H100_SXM` |
| `--vastai-max-price` | – | Max price per instance in $/h (`dph_total`) |
| `--vastai-geolocation` | any | Region shortcut (EU, NA, AS, AF, LC, OC) or country codes |
| `--vastai-disk` | 40 | Disk allocation per instance (GB) |
| `--vastai-image` | `vastai/pytorch:@vastai-automatic-tag` | Docker image for jobs |
| `--vastai-reliability` | 0.98 | Minimum host reliability (0 disables) |
| `--vastai-no-datacenter` | off | Also allow non-datacenter (hobbyist) hosts |
| `--vastai-order` | `dph_total` | Offer ranking (e.g. `dlperf_usd-` for perf/$) |
| `--vastai-search-query` | – | Extra offer filters (vastai query syntax) |
| `--vastai-boot-timeout` | 1800 | Max seconds for an instance to start running |
| `--vastai-no-forward-credentials` | off | Don't forward local cloud credentials to jobs |
| `--vastai-keep-instances` | off | Don't destroy instances (debugging; **keeps billing!**) |
| `--vastai-deploy-paths` | – | Ship only these comma-separated paths/globs (plus the Snakefile/`.smk`/config) instead of the whole `git ls-files` tree. Unioned with every rule's `deploy` resource — prefer declaring files per-rule (`resources: deploy="train.py,scripts/**"`) |
| `--vastai-max-runtime` | 0 | Hard cap (s) on job runtime after the instance is `running`; force-finalize + destroy past it (0 = no cap) |

By default only verified datacenter hosts are rented — slightly pricier,
but they avoid the most common marketplace flakiness (Docker Hub pull rate
limits, slow residential uplinks). `--vastai-no-datacenter` unlocks the
cheapest hobbyist offers.

### Per-job resources

| Resource | Effect on the offer search |
|---|---|
| `gpu` / `nvidia_gpu` | `num_gpus=N` (minimum 1 — Vast.ai only rents GPU machines) |
| `gpu_model` | GPU model (Vast.ai naming, underscores for spaces) |
| `threads` | `cpu_cores_effective>=N` |
| `mem_mb` | minimum system RAM |
| `disk_mb` | minimum disk and rented allocation |
| `vastai_query` | extra filters, appended verbatim (e.g. `"cuda_vers>=12.4"`) |

### Container image

Jobs run in `vastai/pytorch:@vastai-automatic-tag` by default — Vast.ai's
curated PyTorch image with CUDA matched to each machine's driver, usually
cached on hosts. Snakemake is pip-installed into the container automatically
(pinned to your local version, ~30 s per instance).

For other stacks set `--vastai-image`. Requirements: `python` on `PATH`;
for SSH mode also OpenSSH (auto-installed via apt on Debian/Ubuntu images
if missing). Bake `pip install snakemake` into the image to skip the
bootstrap cost.

## File transfer

**SSH mode** (the default, used when no storage is configured): the source
archive and input files are scp'd from your machine to each instance, and
outputs are scp'd back (see "What ends up on the machine" above). Zero
setup, ideal for getting started and small/medium data. Caveats: everything
flows through your uplink, intermediate files between dependent jobs
round-trip through your machine, and the Snakemake process must stay
online.

**Storage mode** (recommended for real pipelines): the same things move,
but through a bucket instead of your machine — Snakemake uploads the source
archive once per run, and each job downloads the archive plus its inputs
from the bucket and uploads its outputs there (your laptop only orchestrates).
Faster, resumable:

```yaml
executor: vastai
jobs: 4
default-storage-provider: s3
default-storage-prefix: s3://my-bucket/my-workflow
```

Install the matching storage plugin (`pip install
snakemake-storage-plugin-s3`, or `-gcs` / `-azure`). Any S3-compatible
service works: AWS S3, MinIO, Cloudflare R2, Backblaze B2, …

### Credentials

Storage credentials reach the jobs automatically — no `--envvars` needed:
local `AWS_*` and `AZURE_STORAGE_*` variables are forwarded into the job
containers, and the Google credentials file
(`GOOGLE_APPLICATION_CREDENTIALS` or gcloud ADC) is shipped and materialized
inside the container. Settings configured on the storage plugin itself
(`SNAKEMAKE_STORAGE_S3_ACCESS_KEY` etc.) are forwarded by Snakemake core.

Notes: credentials living only in `~/.aws/credentials` are not forwarded —
export them as `AWS_*` variables. Other secrets your rules need (e.g.
`HF_TOKEN`) go through the standard `envvars:` directive or `--envvars`.
Forwarded credentials are visible inside containers on third-party hosts —
use scoped, revocable keys.

## Debugging & costs

- Remote job logs land in `.snakemake/auxiliary/vastai-logs/`; `--verbose`
  prints the generated offer queries.
- `--vastai-keep-instances` keeps instances alive for inspection
  (`vastai logs <id>`) — destroy them manually, they bill until then.
- Flaky hosts happen on a marketplace: unrecoverable boot errors fail fast,
  so run with `--retries 2` to resubmit on a different machine.
- The plugin destroys every instance it rents, even on failure, Ctrl-C,
  plain `kill` (SIGTERM/SIGHUP), or interpreter exit. Only SIGKILL
  (`kill -9`) cannot be intercepted — after one, check
  https://console.vast.ai/instances/ for leftovers.
- Each job rents its own instance, so prefer fewer, larger jobs (or job
  grouping) over many tiny ones — boot overhead is paid per job.

## Development

```bash
uv sync
uv run pytest                                  # unit tests, free
SNAKEMAKE_VASTAI_E2E=1 uv run pytest tests/test_e2e.py -s   # rents a real GPU (~$0.05)
```
