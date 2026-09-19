"""Global pytest fixtures for SharK test suite."""
import os
import pytest


@pytest.fixture(autouse=True)
def isolate_test_job_directories(tmp_path_factory, monkeypatch):
    """Ensure tests write temporary job outputs to a sandbox, not the repository."""
    sandbox = tmp_path_factory.mktemp("shark_test_jobs")
    monkeypatch.setenv("SHARK_OUTPUT_DIR", str(sandbox))
