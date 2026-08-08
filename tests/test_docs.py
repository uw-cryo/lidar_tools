"""The README is the only documentation shipped in the built package
(docs/ is not included), so a command that is registered but absent from
it is invisible to anyone who installs lidar_tools."""

from pathlib import Path

from lidar_tools import cli

README = Path(__file__).resolve().parents[1] / "README.md"


def _registered_command_names() -> set[str]:
    return {name for name in cli.app if isinstance(name, str)} - {
        "--help",
        "-h",
        "--version",
    }


def test_readme_documents_every_registered_command():
    readme = README.read_text()
    missing = sorted(n for n in _registered_command_names() if f"`{n}`" not in readme)
    assert not missing, (
        f"commands registered in cli.py but absent from README.md: {missing}. "
        "Add a row to the CLI Commands table."
    )
