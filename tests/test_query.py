from snakemake_executor_plugin_vastai import (
    ExecutorSettings,
    build_offer_query,
    required_disk_gb,
)


def test_defaults():
    query = build_offer_query(ExecutorSettings(), {}, threads=1)
    assert query == (
        "num_gpus=1 cpu_cores_effective>=1 disk_space>=40 reliability>0.98 "
        "datacenter=true"
    )


def test_no_datacenter_lifts_filter():
    query = build_offer_query(
        ExecutorSettings(no_datacenter=True), {}, threads=1
    )
    assert "datacenter" not in query


def test_gpu_resources_override_settings():
    settings = ExecutorSettings(gpu_name="RTX_4090")
    query = build_offer_query(
        settings,
        {"gpu": 2, "gpu_model": "H100 SXM", "mem_mb": 64000, "disk_mb": 100000},
        threads=8,
    )
    assert "num_gpus=2" in query
    assert "gpu_name=H100_SXM" in query
    assert "cpu_cores_effective>=8" in query
    assert "cpu_ram>=64" in query
    assert "disk_space>=100" in query


def test_price_cap_and_extra_queries():
    settings = ExecutorSettings(
        max_price=1.5, search_query="cuda_vers>=12.4 geolocation=EU"
    )
    query = build_offer_query(
        settings, {"vastai_query": "inet_down>=500"}, threads=1
    )
    assert "dph_total<=1.5" in query
    assert query.index("cuda_vers>=12.4 geolocation=EU") < query.index(
        "inet_down>=500"
    )


def test_reliability_filter_can_be_disabled():
    query = build_offer_query(ExecutorSettings(reliability=0), {}, threads=1)
    assert "reliability" not in query


def test_required_disk_gb_takes_maximum():
    settings = ExecutorSettings(disk=40.0)
    assert required_disk_gb(settings, {}) == 40.0
    assert required_disk_gb(settings, {"disk_mb": 10000}) == 40.0
    assert required_disk_gb(settings, {"disk_mb": 120000}) == 120.0


def test_geolocation_region_expansion():
    settings = ExecutorSettings(geolocation="EU")
    query = build_offer_query(settings, {}, threads=1)
    assert "geolocation in [AL,AD,AT" in query

    settings = ExecutorSettings(geolocation="PL,DE")
    query = build_offer_query(settings, {}, threads=1)
    assert "geolocation in [PL,DE]" in query
