import copy
import json
import re

import numpy as np
import pandas as pd
import pytest

from stock_lstm import report


def fake_results():
    """Made-up numbers with the real schema; only used to test rendering."""
    def m(rmse, mae, ratio, lo, hi, dirn):
        return {"RMSE": rmse, "MAE": mae, "DirAcc_%": dirn, "ratio": ratio, "lo": lo, "hi": hi}
    return {
        "run_date": "2030-01-02",
        "data": {"ticker": "AAPL", "first_date": "2015-01-02", "last_date": "2029-12-31", "origin": "2029-11-15",
                 "modelling_days": 3700, "held_out_days": 32},
        "setup": {"window": 60, "units": [32, 32], "n_folds": 5, "test_size": 126, "val_size": 126,
                  "n_seeds": 3, "horizon": 30, "step": 21},
        "one_step": {
            "n_days": 630, "test_start": "2027-01-04", "test_end": "2029-11-15", "up_day_rate": 55.3,
            "models": {
                "Persistence": m(3.217, 2.177, 1.0, 1.0, 1.0, None),
                "Drift": m(3.219, 2.178, 1.001, 0.999, 1.002, 51.0),
                "LSTM": m(3.223, 2.172, 1.002, 0.997, 1.007, 53.8),
                "ARIMA(1, 1, 1)": m(3.237, 2.197, 1.006, 0.999, 1.014, 49.4),
                "MA(5)": m(5.003, 3.611, 1.555, 1.443, 1.669, 47.1),
                "MA(20)": m(8.156, 6.386, 2.535, 2.203, 2.959, 47.1),
            },
            "per_fold": {"LSTM": [2.0, 2.3, 2.9, 2.9, 5.1], "Persistence": [2.01, 2.25, 2.91, 2.92, 5.07]},
            "seed_rmse": [3.221, 3.225, 3.226],
        },
        "forecast": {
            "origin_date": "2029-11-15", "origin_price": 210.16, "forecast_end": 217.96, "actual_end": 231.28,
            "band80": [192.28, 245.16],
            "summary": {"days_scored": 30, "RMSE_forecast": 10.42, "RMSE_flat": 14.19, "RMSE_drift": 11.59,
                        "forecast_move_%": 3.7, "actual_move_%": 10.1, "drift_move_%": 2.5,
                        "coverage_80_%": 100.0, "coverage_95_%": 100.0},
        },
        "rolling": {
            "n_origins": 28, "horizon": 30, "step": 21, "first_origin": "2027-01-01", "last_origin": "2029-09-03",
            "wins_vs_flat": 50.0, "wins_vs_drift": 46.4,
            "table": {
                "Forecast": {"RMSE": 12.0, "MAE": 9.0, "vs_Flat": 1.01, "vs_Flat_lo": 0.97, "vs_Flat_hi": 1.05,
                             "vs_Drift": 1.02, "vs_Drift_lo": 0.98, "vs_Drift_hi": 1.06},
                "Drift": {"RMSE": 11.8, "MAE": 8.8, "vs_Flat": 0.99, "vs_Flat_lo": 0.96, "vs_Flat_hi": 1.02,
                          "vs_Drift": 1.0, "vs_Drift_lo": 1.0, "vs_Drift_hi": 1.0},
                "Flat": {"RMSE": 11.9, "MAE": 8.9, "vs_Flat": 1.0, "vs_Flat_lo": 1.0, "vs_Flat_hi": 1.0,
                         "vs_Drift": 1.01, "vs_Drift_lo": 0.98, "vs_Drift_hi": 1.04},
            },
            "coverage": {"80": {"mean_%": 82.0, "lo_%": 76.0, "hi_%": 88.0},
                         "95": {"mean_%": 96.0, "lo_%": 93.0, "hi_%": 99.0}},
        },
    }


# ---- json -----------------------------------------------------------------------------------------

def test_to_jsonable_converts_numpy_pandas_and_nan():
    obj = {"a": np.float64(1.5), "b": np.int64(3), "c": float("nan"), "d": float("inf"),
           "e": pd.Timestamp("2024-03-05"), "f": np.array([1.0, np.nan]), "g": (np.bool_(True),),
           "h": pd.DataFrame({"x": [1.0, np.nan]}, index=["r1", "r2"]), 7: "int key"}
    out = report.to_jsonable(obj)
    assert out == {"a": 1.5, "b": 3, "c": None, "d": None, "e": "2024-03-05", "f": [1.0, None], "g": [True],
                   "h": {"r1": {"x": 1.0}, "r2": {"x": None}}, "7": "int key"}
    json.dumps(out, allow_nan=False)                                    # strict JSON, no NaN literals


def test_save_and_load_roundtrip_with_strict_json(tmp_path):
    res = fake_results()
    res["one_step"]["models"]["Persistence"]["DirAcc_%"] = float("nan")
    p = tmp_path / "sub" / "results.json"
    report.save_results(p, res)
    text = p.read_text()
    assert "NaN" not in text and text.endswith("\n")
    assert report.load_results(p)["one_step"]["models"]["Persistence"]["DirAcc_%"] is None


# ---- wording ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("lo,hi,expected", [
    (0.90, 0.99, "better than"), (0.997, 1.007, "statistically indistinguishable from"),
    (1.02, 1.10, "worse than"), (0.90, 1.0, "statistically indistinguishable from"),   # touching 1 is not better
    (1.0, 1.10, "statistically indistinguishable from"),
])
def test_verdict(lo, hi, expected):
    assert report.verdict(lo, hi) == expected


def test_coverage_status():
    assert "as often as intended" in report.coverage_status(80, {"mean_%": 82, "lo_%": 76, "hi_%": 88})
    assert "wider than needed" in report.coverage_status(80, {"mean_%": 95, "lo_%": 92, "hi_%": 98})
    assert "too narrow" in report.coverage_status(95, {"mean_%": 80, "lo_%": 75, "hi_%": 85})


# ---- rendering ------------------------------------------------------------------------------------------

def test_render_contains_every_headline_number_from_the_json():
    text = report.render_markdown(fake_results())
    for token in ["2030-01-02", "2015-01-02", "2029-12-31", "3,700", "630 out-of-sample days",
                  "$3.217", "$3.223", "1.002 (0.997 to 1.007)", "53.8%", "55.3%", "$210.16", "$217.96", "+3.7%",
                  "$192 to $245", "$231.28", "+10.1%", "$10.42", "$14.19", "$11.59", "+2.5%",
                  "28 forecasts of 30 days", "1.010 (0.970 to 1.050)", "82%", "96%"]:
        assert token in text, token
    assert text.count("![") == 5                                        # the five figures
    assert "None" not in text and "nan" not in text.lower().replace("finance", "")


def test_render_lists_models_in_a_fixed_order_and_marks_persistence_direction_na():
    text = report.render_markdown(fake_results())
    order = [text.index(k) for k in ("Persistence (", "Drift (", "LSTM (", "ARIMA(1,1,1)", "| MA(5)", "| MA(20)")]
    assert order == sorted(order)
    assert re.search(r'Persistence \("tomorrow = today"\) \| \$3\.217 \| \$2\.177 \| 1\.000 \| n/a', text)


def test_reading_follows_the_numbers_indistinguishable_better_worse():
    res = fake_results()
    assert "no measurable edge" in report.render_markdown(res)
    better = copy.deepcopy(res)
    better["one_step"]["models"]["LSTM"].update(ratio=0.9, lo=0.85, hi=0.95)
    text = report.render_markdown(better)
    assert "is better than \"tomorrow = today\"" in text and "does measurably better" in text
    assert "no measurable edge" not in text
    worse = copy.deepcopy(res)
    worse["one_step"]["models"]["LSTM"].update(ratio=1.2, lo=1.1, hi=1.3)
    assert "does measurably worse" in report.render_markdown(worse)


def test_live_forecast_sentence_follows_the_benchmarks():
    res = fake_results()                                                # forecast beats flat, not drift? 10.42 < 11.59
    assert "beat both benchmarks" in report.render_markdown(res)
    only_flat = copy.deepcopy(res)
    only_flat["forecast"]["summary"]["RMSE_drift"] = 9.0
    assert "advantage over \"flat\" is drift, not skill" in report.render_markdown(only_flat)
    neither = copy.deepcopy(res)
    neither["forecast"]["summary"]["RMSE_flat"] = 5.0
    assert "did not beat \"flat\"" in report.render_markdown(neither)


def test_rolling_paragraph_reports_verdicts_and_coverage_status():
    text = report.render_markdown(fake_results())
    assert "statistically indistinguishable from \"flat\"" in text
    assert "statistically indistinguishable from drift" in text
    assert "both bands covered about as often as intended" in text
    bad = fake_results()
    bad["rolling"]["coverage"]["95"] = {"mean_%": 80.0, "lo_%": 75.0, "hi_%": 85.0}
    assert ("the 80% band covered about as often as intended, the 95% band covered less often than intended "
            "(bands too narrow)") in report.render_markdown(bad)
    win = fake_results()
    win["rolling"]["table"]["Forecast"].update(vs_Flat=0.8, vs_Flat_lo=0.7, vs_Flat_hi=0.9)
    assert "better than \"flat\"" in report.render_markdown(win)


def test_a_winner_by_a_hair_is_flagged_as_within_noise_only_when_the_verdict_says_so():
    res = fake_results()
    assert "has the lowest RMSE of the models compared (by a margin within noise)" in report.render_markdown(res)
    better = copy.deepcopy(res)
    better["one_step"]["models"]["LSTM"].update(RMSE=2.5, ratio=0.78, lo=0.7, hi=0.85)
    text = report.render_markdown(better)
    assert "LSTM has the lowest RMSE of the models compared." in text and "within noise)" not in text


def test_fold_gap_is_not_rounded_down_into_a_smaller_claim():
    """A largest fold gap of $0.011 must not be reported as 'within $0.01'."""
    res = fake_results()
    res["one_step"]["per_fold"] = {"LSTM": [2.0, 3.011], "Persistence": [2.0, 3.0]}
    text = report.render_markdown(res)
    assert "within $0.011 of persistence's" in text and "within $0.01 of" not in text


def test_render_is_deterministic():
    assert report.render_markdown(fake_results()) == report.render_markdown(fake_results())


# ---- README update ------------------------------------------------------------------------------------------

def _readme(tmp_path, body="OLD"):
    p = tmp_path / "README.md"
    p.write_text(f"# Title\n\nintro\n\n## Results snapshot\n\n{report.START_MARK}\n\n{body}\n\n{report.END_MARK}\n\n## After\n\nkeep me\n")
    return p


def test_update_replaces_only_the_marked_block_and_is_idempotent(tmp_path):
    p = _readme(tmp_path)
    assert report.update_readme(p, fake_results()) is True
    text = p.read_text()
    assert "OLD" not in text and "$3.223" in text
    assert text.startswith("# Title\n\nintro\n\n## Results snapshot") and text.endswith("## After\n\nkeep me\n")
    before = text
    assert report.update_readme(p, fake_results()) is False               # second run changes nothing
    assert p.read_text() == before


def test_check_mode_reports_sync_without_writing(tmp_path):
    p = _readme(tmp_path)
    assert report.update_readme(p, fake_results(), check=True) is False   # out of sync ...
    assert "OLD" in p.read_text()                                         # ... and untouched
    report.update_readme(p, fake_results())
    assert report.update_readme(p, fake_results(), check=True) is True


@pytest.mark.parametrize("bad", ["no markers", "<!-- results:end -->\n<!-- results:start -->",
                                 "<!-- results:start -->\nonly start"])
def test_missing_or_malformed_markers_are_an_error(tmp_path, bad):
    p = tmp_path / "README.md"
    p.write_text(bad)
    with pytest.raises(ValueError, match="results:start"):
        report.update_readme(p, fake_results())


def test_cli_check_exit_codes(tmp_path):
    results = tmp_path / "results.json"
    report.save_results(results, fake_results())
    readme = _readme(tmp_path)
    assert report.main(["--check", str(results), str(readme)]) == 1
    assert report.main([str(results), str(readme)]) == 0
    assert report.main(["--check", str(results), str(readme)]) == 0
