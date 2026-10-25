"""Regression guards for the concealed workspace-open loader removed in #8.

These are source-tree checks, not a replacement for code review or malware
scanning. In particular, they must never execute an alleged font asset.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_vscode_cannot_run_repository_tasks_on_folder_open():
    settings = json.loads((ROOT / ".vscode/settings.json").read_text(encoding="utf-8"))
    tasks = json.loads((ROOT / ".vscode/tasks.json").read_text(encoding="utf-8"))

    assert settings["task.allowAutomaticTasks"] == "off"
    # This research package has no documented need for editor tasks. Keeping
    # the inventory empty rejects the exact task reintroduced by stale PR #7.
    assert tasks.get("tasks", []) == []


def test_font_directory_contains_only_expected_asset_formats():
    fonts = ROOT / "public/fonts"
    allowed_suffixes = {".eot", ".svg", ".ttf", ".woff", ".woff2"}
    for asset in fonts.iterdir():
        if asset.name == "README.md":
            continue
        assert asset.is_file() and not asset.is_symlink(), asset.name
        assert asset.suffix in allowed_suffixes, asset.name
        if asset.suffix in {".woff", ".woff2"}:
            with asset.open("rb") as stream:
                magic = stream.read(4)
            expected = b"wOF2" if asset.suffix == ".woff2" else b"wOFF"
            assert magic == expected, asset.name
