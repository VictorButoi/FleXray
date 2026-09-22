"""Keep the software repository independent of website deployment."""

from pathlib import Path

import yaml


def test_package_workflows_do_not_deploy_the_website() -> None:
    """Website publication and Pages permissions belong to FleXray-website."""
    for path in Path(".github/workflows").glob("*.yml"):
        text = path.read_text()
        workflow = yaml.load(text, Loader=yaml.BaseLoader)
        assert "pages" not in workflow.get("permissions", {})
        assert "actions/deploy-pages@" not in text
        assert "actions/upload-pages-artifact@" not in text
        assert "WEBSITE_READ_KEY" not in text
        for job in workflow["jobs"].values():
            assert "pages" not in job.get("permissions", {})


def test_normal_ci_needs_no_website_credentials() -> None:
    """Package checks remain independent of website access."""
    ci = Path(".github/workflows/ci.yml").read_text()
    assert "WEBSITE_READ_KEY" not in ci
    assert "FLEXRAY_READ_KEY" not in ci
