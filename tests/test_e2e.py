"""Live end-to-end test against real Vast.ai — rents a GPU, costs a few cents.

Skipped unless explicitly enabled:

    SNAKEMAKE_VASTAI_E2E=1 VAST_API_KEY=... pytest tests/test_e2e.py -s

Covers what was verified manually on 2026-06-11: SSH transfer mode round
trip (input upload, GPU job, output download, instance teardown).
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SNAKEMAKE_VASTAI_E2E") != "1",
    reason="live Vast.ai test; set SNAKEMAKE_VASTAI_E2E=1 to rent a real GPU",
)

SNAKEFILE = '''rule all:
    input: "results/out.txt"

rule probe:
    input: "data/message.txt"
    output: "results/out.txt"
    shell:
        """
        cat {input} > {output}
        echo "host: $(hostname)" >> {output}
        nvidia-smi -L >> {output} 2>&1 || echo "nvidia-smi unavailable" >> {output}
        """
'''


def test_ssh_mode_round_trip(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "message.txt").write_text("hello from pytest\n")
    (tmp_path / "Snakefile").write_text(SNAKEFILE)
    # Source archiving requires a git repo.
    for cmd in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=e2e@test", "-c", "user.name=e2e",
         "commit", "-qm", "e2e"],
    ):
        subprocess.run(cmd, cwd=tmp_path, check=True)

    result = subprocess.run(
        [
            sys.executable, "-m", "snakemake",
            "--executor", "vastai",
            "--jobs", "1",
            "--vastai-max-price", "0.3",
            "--vastai-disk", "25",
            "--vastai-geolocation", "EU",
            "--vastai-search-query", "inet_down>=400",
            "--retries", "2",
            "--verbose",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=2400,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    out = (tmp_path / "results" / "out.txt").read_text()
    assert "hello from pytest" in out  # input made it to the instance
    assert "host:" in out  # job actually ran remotely

    # Every rented instance must be destroyed again (they bill until then).
    from vastai import VastAI

    instances = VastAI().show_instances_v1({"limit": 25}).get("instances", [])
    leftovers = [
        i["id"] for i in instances
        if (i.get("label") or "").startswith("snakemake-")
    ]
    assert leftovers == [], f"leftover instances still billing: {leftovers}"
