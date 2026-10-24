"""The report figure script imports on CPU and builds the two diagram figures."""
import importlib.util
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "dance/scripts/video/report_figures.py"


def _load():
    spec = importlib.util.spec_from_file_location("report_figures", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_builds_workflow_and_architecture(tmp_path):
    rf = _load()
    rf.setup_fonts()
    for name in ("workflow", "architecture"):
        out = rf.FIGURES[name](tmp_path)
        assert out == tmp_path / f"{name}.png"
        with Image.open(out) as im:
            w, h = im.size
            dpi = im.info.get("dpi", (0, 0))[0]
        assert abs(dpi - 300) < 1
        assert abs(w - round(16 / 2.54 * 300)) <= 2  # 16 cm wide at 300 dpi
        assert h > 0


def test_cli_rejects_unknown_figure(tmp_path):
    rf = _load()
    import pytest

    with pytest.raises(SystemExit):
        rf.main(["nope", "--out-dir", str(tmp_path)])
