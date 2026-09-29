"""Run the real notebook end to end on fake, Tiingo-shaped data (quick mode) so library changes
cannot silently break it. Checks the wiring only; the numbers mean nothing."""
import re
import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from stock_lstm import report
from tests.conftest import write_synthetic_tiingo_cache

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ["raw_vs_adjusted_close", "walk_forward_predictions", "rmse_ratio_vs_persistence",
           "forecast_vs_actual", "rolling_origin_backtest"]


@pytest.mark.slow
def test_notebook_runs_top_to_bottom_in_quick_mode_and_saves_consistent_results(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "stock_lstm", tmp_path / "stock_lstm", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "README.md", tmp_path / "README.md")
    (tmp_path / "notebooks").mkdir()
    nb_src = ROOT / "notebooks" / "Apple_Stock_Prediction.ipynb"
    shutil.copy(nb_src, tmp_path / "notebooks" / nb_src.name)

    nb = nbformat.read(tmp_path / "notebooks" / nb_src.name, as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    start = re.search(r'^START = "([^"]+)"', code, re.M).group(1)     # follow whatever dates the notebook pins
    end = re.search(r'^END = "([^"]+)"', code, re.M).group(1)
    write_synthetic_tiingo_cache(tmp_path / "data" / f"AAPL_{start}_{end}.csv", start=start, end=end)

    monkeypatch.setenv("STOCK_LSTM_QUICK", "1")
    monkeypatch.setenv("STOCK_LSTM_SAVE_RESULTS", "1")
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)              # the cache must make the key unnecessary
    NotebookClient(nb, timeout=900, resources={"metadata": {"path": str(tmp_path / "notebooks")}}).execute()

    text = "\n".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.outputs)
    assert "Split events in the raw data" in text     # data section ran and found the split
    assert "'tomorrow = today'" in text               # the computed one-step verdict printed
    assert "forecasts of 30 days" in text             # the rolling-origin section ran
    assert any("image/png" in o.get("data", {}) for c in nb.cells if c.cell_type == "code" for o in c.outputs)

    # the save-results cell wrote the JSON and figures, and the README section is rendered from that JSON
    results = report.load_results(tmp_path / "docs" / "results.json")
    assert results["one_step"]["models"]["LSTM"]["ratio"] > 0 and results["rolling"]["n_origins"] >= 2
    assert all((tmp_path / "docs" / "images" / f"{name}.png").is_file() for name in FIGURES)
    assert report.update_readme(tmp_path / "README.md", results, check=True)
