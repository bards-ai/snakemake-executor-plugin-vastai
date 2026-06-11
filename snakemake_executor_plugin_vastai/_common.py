from typing import Optional

# status_msg fragments during boot that will not resolve by waiting (e.g.
# the host hit Docker Hub's pull rate limit) — fail fast instead of burning
# the whole boot timeout. Snakemake --retries then resubmits, usually on a
# different machine.
FATAL_BOOT_ERRORS = (
    "pull rate limit",
    "manifest unknown",
    "unauthorized",
    "no space left on device",
)


def fatal_boot_error(status_msg: Optional[str]) -> Optional[str]:
    msg = (status_msg or "").lower()
    for pattern in FATAL_BOOT_ERRORS:
        if pattern in msg:
            return pattern
    return None


# Locates a usable python and puts it on PATH. Non-interactive SSH sessions
# don't get the PATH that the docker image's ENV sets up, so e.g. on
# vastai/* images the /venv/main virtualenv (where torch lives) must be
# found explicitly. Preference order: Vast's main venv, conda base, system.
PYTHON_PATH_SETUP = """PYBIN=""
for cand in /venv/main/bin/python3 /opt/conda/bin/python3 \
    "$(command -v python3 2>/dev/null || true)" \
    "$(command -v python 2>/dev/null || true)"; do
  if [ -n "$cand" ] && [ -x "$cand" ]; then PYBIN="$cand"; break; fi
done
if [ -n "$PYBIN" ]; then
  PATH="$(dirname "$PYBIN"):$PATH"; export PATH
  command -v python >/dev/null 2>&1 \
    || ln -sf "$PYBIN" /usr/local/bin/python 2>/dev/null || true
fi
"""
