from snakemake_executor_plugin_vastai import (
    DEFAULT_IMAGE,
    resolve_container_image,
    snakemake_bootstrap_script,
)


def test_plugin_setting_wins():
    assert (
        resolve_container_image("my/image:1", "other/image:2") == "my/image:1"
    )


def test_explicit_container_image_respected():
    assert (
        resolve_container_image(None, "pytorch/pytorch:2.4.0")
        == "pytorch/pytorch:2.4.0"
    )


def test_snakemake_default_image_is_replaced():
    assert (
        resolve_container_image(None, "snakemake/snakemake:v9.22.0")
        == DEFAULT_IMAGE
    )
    assert resolve_container_image(None, None) == DEFAULT_IMAGE


def test_bootstrap_pins_local_snakemake_version():
    import snakemake

    script = snakemake_bootstrap_script()
    assert f"snakemake=={snakemake.__version__}" in script
    assert "pip install" in script


def test_bootstrap_includes_storage_plugins_for_ssh_mode():
    script = snakemake_bootstrap_script(with_storage_plugins=True)
    # The test environment has the s3 storage plugin installed; its settings
    # appear in spawned job commands, so it must be installed remotely too.
    assert "snakemake-storage-plugin-s3==" in script
