"""Project setup: the submodule pin, the uv settings shared with upstream and task discovery (CPU)."""

from __future__ import annotations

import subprocess
import tomllib
from importlib.metadata import entry_points
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "third_party" / "microduck-playground"
UPSTREAM_COMMIT = "f010ef1cefcad49c05c20a4a78f840af83a643b9"
DANCE_TASKS = (
    "Mjlab-Dance-Flat-MicroDuck",
    "Mjlab-DanceLib-Flat-MicroDuck",
    "Mjlab-DanceV3-Flat-MicroDuck",
    "Mjlab-DanceV3-Rough-MicroDuck",
    "Mjlab-DanceV3-RoughMix-MicroDuck",
    "Mjlab-BallDance-MicroDuck",
    "Mjlab-RollerDance-MicroDuck",
)


def _toml(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def _git(*args, cwd=ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def _in_git() -> bool:
    try:
        return _git("rev-parse", "--is-inside-work-tree").strip() == "true"
    except (OSError, subprocess.CalledProcessError):
        return False


needs_git = pytest.mark.skipif(not _in_git(), reason="not a git checkout")


def test_uv_settings_match_upstream():
    ours, up = _toml(ROOT / "pyproject.toml"), _toml(SUB / "pyproject.toml")
    assert ours["tool"]["uv"]["override-dependencies"] == up["tool"]["uv"]["override-dependencies"]
    assert ours["tool"]["uv"]["sources"]["torch"] == up["tool"]["uv"]["sources"]["torch"]
    assert ours["tool"]["uv"]["index"] == up["tool"]["uv"]["index"]
    assert ours["project"]["requires-python"] == up["project"]["requires-python"]
    torch_pin = [d for d in up["project"]["dependencies"] if d.startswith("torch")]
    assert torch_pin and torch_pin[0] in ours["project"]["dependencies"]
    src = ours["tool"]["uv"]["sources"]["mjlab-microduck"]
    assert src == {"path": "third_party/microduck-playground", "editable": True}


def test_dance_tasks_are_an_mjlab_entry_point():
    eps = {e.name: e.value for e in entry_points().select(group="mjlab.tasks")}
    assert eps.get("microduck_dance") == "microduck_dance.tasks"
    assert eps.get("mjlab_microduck") == "mjlab_microduck.tasks"


def test_importing_mjlab_registers_the_dance_tasks():
    import mjlab  # noqa: F401
    from mjlab.tasks.registry import list_tasks

    tasks = set(list_tasks())
    assert set(DANCE_TASKS) <= tasks
    assert "Mjlab-Velocity-Flat-MicroDuck" in tasks


def test_upstream_package_is_the_submodule():
    import mjlab_microduck

    assert Path(mjlab_microduck.__file__).resolve().is_relative_to(SUB.resolve())


@needs_git
def test_submodule_is_pinned_and_unchanged():
    line = _git("ls-tree", "HEAD", "third_party/microduck-playground").split()
    assert line[:3] == ["160000", "commit", UPSTREAM_COMMIT]
    assert _git("rev-parse", "HEAD", cwd=SUB).strip() == UPSTREAM_COMMIT
    # No local change in the upstream tree (the dance code must not patch it).
    assert _git("status", "--porcelain", "--untracked-files=no", cwd=SUB) == ""
    url = _git("config", "-f", ".gitmodules", "submodule.third_party/microduck-playground.url").strip()
    assert url == "https://github.com/Vottivott/microduck-playground"
