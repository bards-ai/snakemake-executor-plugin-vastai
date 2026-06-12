import base64

from snakemake_executor_plugin_vastai import (
    GCP_CREDENTIALS_CONTENT_VAR,
    credential_envvars,
)


def test_forwards_only_known_credential_vars():
    environ = {
        "AWS_ACCESS_KEY_ID": "AKIA123",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_ENDPOINT_URL": "https://r2.example.com",
        "AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=https;...",
        "HOME": "/home/user",
        "VAST_API_KEY": "must-never-be-forwarded",
        "SNAKEMAKE_VASTAI_API_KEY": "must-never-be-forwarded",
    }
    env = credential_envvars(environ)
    assert env == {
        "AWS_ACCESS_KEY_ID": "AKIA123",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_ENDPOINT_URL": "https://r2.example.com",
        "AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=https;...",
    }


def test_empty_values_are_skipped():
    assert credential_envvars({"AWS_ACCESS_KEY_ID": ""}) == {}


def test_gcp_credentials_file_content_is_shipped(tmp_path):
    creds = tmp_path / "sa.json"
    creds.write_text('{"type": "service_account"}')
    env = credential_envvars({"GOOGLE_APPLICATION_CREDENTIALS": str(creds)})
    decoded = base64.b64decode(env[GCP_CREDENTIALS_CONTENT_VAR])
    assert decoded == b'{"type": "service_account"}'


def test_missing_gcp_credentials_file_is_ignored():
    env = credential_envvars(
        {"GOOGLE_APPLICATION_CREDENTIALS": "/nonexistent/sa.json"}
    )
    assert GCP_CREDENTIALS_CONTENT_VAR not in env


def test_gcp_project_derived_from_adc_quota_project(tmp_path):
    # authorized_user ADC carries no project usable by google.auth; the
    # plugin must export quota_project_id so remote clients can resolve it.
    creds = tmp_path / "adc.json"
    creds.write_text(
        '{"type": "authorized_user", "quota_project_id": "my-project"}'
    )
    env = credential_envvars({"GOOGLE_APPLICATION_CREDENTIALS": str(creds)})
    assert env["GOOGLE_CLOUD_PROJECT"] == "my-project"


def test_gcp_project_falls_back_to_service_account_project_id(tmp_path):
    creds = tmp_path / "sa.json"
    creds.write_text('{"type": "service_account", "project_id": "sa-project"}')
    env = credential_envvars({"GOOGLE_APPLICATION_CREDENTIALS": str(creds)})
    assert env["GOOGLE_CLOUD_PROJECT"] == "sa-project"


def test_explicit_gcp_project_is_not_overridden(tmp_path):
    creds = tmp_path / "adc.json"
    creds.write_text(
        '{"type": "authorized_user", "quota_project_id": "from-adc"}'
    )
    env = credential_envvars(
        {
            "GOOGLE_APPLICATION_CREDENTIALS": str(creds),
            "GOOGLE_CLOUD_PROJECT": "explicit",
        }
    )
    assert env["GOOGLE_CLOUD_PROJECT"] == "explicit"
