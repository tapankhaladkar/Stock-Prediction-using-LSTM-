"""Run the real notebook end to end on fake, Tiingo-shaped data (quick mode) so library changes
cannot silently break it. Checks the wiring only; the numbers mean nothing."""
import os
import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from tests.conftest import write_synthetic_tiingo_cache

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_notebook_runs_top_to_bottom_in_quick_mode(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "stock_lstm", tmp_path / "stock_lstm", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "notebooks").mkdir()
    nb_src = ROOT / "notebooks" / "Apple_Stock_Prediction.ipynb"
    shutil.copy(nb_src, tmp_path / "notebooks" / nb_src.name)
    write_synthetic_tiingo_cache(tmp_path / "data" / "AAPL_2015-01-01_2025-09-30.csv")
    monkeypatch.setenv("STOCK_LSTM_QUICK", "1")
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)     # the cache must make the key unnecessary

    nb = nbformat.read(tmp_path / "notebooks" / nb_src.name, as_version=4)
    NotebookClient(nb, timeout=600, resources={"metadata": {"path": str(tmp_path / "notebooks")}}).execute()

    text = "\n".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.outputs)
    assert "Split events in the raw data" in text     # data section ran and found the split
    assert "'tomorrow = today'" in text               # the computed verdict printed (whichever way it went)
    assert any("image/png" in o.get("data", {}) for c in nb.cells if c.cell_type == "code" for o in c.outputs)
