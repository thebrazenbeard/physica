"""Source-level guardrails for the seven-job GitHub Actions qualification."""

import re
from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
JOBS = ("baseline", "scipy-reference", "torch", "torch-minimum", "wheel-smoke")


def _job_body(content: str, job: str) -> str:
    match = re.search(rf"(?ms)^  {re.escape(job)}:\n(.*?)(?=^  [\w-]+:\n|\Z)", content)
    assert match is not None, f"expected CI job {job}"
    return match.group(1)


def test_ci_token_is_read_only_for_all_jobs():
    content = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"(?m)^permissions:\s*\n  contents: read\s*$", content)


def test_every_qualification_job_uses_node24_actions_without_checkout_credentials():
    content = WORKFLOW.read_text(encoding="utf-8")
    for job in JOBS:
        block = _job_body(content, job)
        assert re.search(
            r"(?m)^      - uses: actions/checkout@v5\n"
            r"        with:\n          persist-credentials: false\s*$",
            block,
        ), job
        assert re.search(r"(?m)^      - uses: actions/setup-python@v6\s*$", block), job

    assert len(re.findall(r"(?m)^      - uses: actions/checkout@", content)) == len(JOBS)
