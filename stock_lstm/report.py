"""Turn the notebook's results into ``docs/results.json`` and the README's results section.

The README section is *rendered* from the JSON, never typed by hand: the numbers cannot be
transcribed wrongly, and the prose that interprets them (better than / worse than / cannot be
told apart) is chosen by the numbers, so it cannot claim more than the run shows.

    python -m stock_lstm.report            # rewrite the README section from docs/results.json
    python -m stock_lstm.report --check    # exit 1 if the README section is out of sync
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

START_MARK = "<!-- results:start -->"
END_MARK = "<!-- results:end -->"
MODEL_ORDER = ["Persistence", "Drift", "LSTM", "ARIMA(1, 1, 1)", "MA(5)", "MA(20)"]


# ---- JSON ------------------------------------------------------------------------------------

def to_jsonable(obj):
    """Plain-Python copy of ``obj``: numpy scalars/arrays, pandas timestamps and frames converted,
    and NaN / infinity replaced by null (JSON has no NaN)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="index"))
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_results(path: str | Path, results: dict) -> None:
    text = json.dumps(to_jsonable(results), indent=2, allow_nan=False) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def load_results(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---- wording chosen by the numbers ---------------------------------------------------------------

def verdict(lo: float, hi: float) -> str:
    """How a ratio (below 1 = better) with 95% interval [lo, hi] compares with its reference."""
    return "better than" if hi < 1 else "worse than" if lo > 1 else "statistically indistinguishable from"


def coverage_status(level: int, cov: dict) -> str:
    nominal = float(level)
    if cov["lo_%"] <= nominal <= cov["hi_%"]:
        return "covered about as often as intended"
    if cov["mean_%"] > nominal:
        return "covered more often than intended (bands wider than needed)"
    return "covered less often than intended (bands too narrow)"


def coverage_sentence(c80: dict, c95: dict) -> str:
    s80, s95 = coverage_status(80, c80), coverage_status(95, c95)
    return f"both bands {s80}" if s80 == s95 else f"the 80% band {s80}, the 95% band {s95}"


def _usd(x, nd: int = 2) -> str:
    return f"${x:,.{nd}f}"


def _interval(lo, hi, nd: int = 3) -> str:
    return f"{lo:.{nd}f} to {hi:.{nd}f}"


# ---- markdown ----------------------------------------------------------------------------------------

def render_markdown(results: dict, image_dir: str = "docs/images") -> str:
    """The body of the README's "Results snapshot" section (everything between the markers)."""
    d, s = results["data"], results["setup"]
    o, f, r = results["one_step"], results["forecast"], results["rolling"]
    models = o["models"]
    H = s["horizon"]

    def img(name: str, alt: str) -> str:
        return f"![{alt}]({image_dir}/{name})"

    out = []
    out.append(
        "> One dated run, not a guarantee. This section is rendered from "
        "[`docs/results.json`](docs/results.json) by `python -m stock_lstm.report`, so no number in it is "
        "typed by hand; run the notebook with `STOCK_LSTM_SAVE_RESULTS=1` to regenerate it. The data provider "
        "rewrites adjusted history whenever a dividend is paid, and seeds or hardware differ, so a later run "
        "will differ slightly."
    )
    out.append(
        f"**Setup (run on {results['run_date']}):** {d['ticker']} adjusted daily closes, {d['first_date']} to "
        f"{d['last_date']}. Models saw only data up to {d['origin']} ({d['modelling_days']:,} trading days); "
        f"the {d['held_out_days']} later days were held out to score the live forecast. One-step evaluation: "
        f"{s['n_folds']} walk-forward folds x {s['test_size']} days = **{o['n_days']} out-of-sample days** "
        f"({o['test_start']} to {o['test_end']}), {s['n_seeds']} seeds, {s['window']}-day window, "
        f"LSTM {' x '.join(str(u) for u in s['units'])} units. The data step matters: raw `close` contains the "
        "4:1 split as a fake crash, `adjClose` does not."
    )
    out.append(img("raw_vs_adjusted_close.png",
                   "Raw vs adjusted close: the 2020-08-31 split is a cliff in raw close and invisible in adjusted close"))

    # -- one-step table
    labels = {"Persistence": 'Persistence ("tomorrow = today")',
              "Drift": "Drift (persistence + average past return)",
              "LSTM": f"LSTM ({s['n_seeds']}-seed mean)",
              "ARIMA(1, 1, 1)": "ARIMA(1,1,1)"}
    rows = ["| Model | RMSE | MAE | RMSE vs persistence (95% interval) | Direction correct |",
            "|---|---|---|---|---|"]
    for name in MODEL_ORDER:
        if name not in models:
            continue
        m = models[name]
        rel = "1.000" if name == "Persistence" else f"{m['ratio']:.3f} ({_interval(m['lo'], m['hi'])})"
        direction = "n/a" if m.get("DirAcc_%") is None else f"{m['DirAcc_%']:.1f}%"
        rows.append(f"| {labels.get(name, name)} | {_usd(m['RMSE'], 3)} | {_usd(m['MAE'], 3)} | {rel} | {direction} |")
    out.append(f"**Next-day accuracy** (dollars; every model scored on the same {o['n_days']} days):\n\n"
               + "\n".join(rows))
    out.append(f"{o['up_day_rate']:.1f}% of these days closed up, so always guessing \"up\" would score "
               f"{o['up_day_rate']:.1f}% on direction.")
    out.append(img("walk_forward_predictions.png",
                   "One-step-ahead predictions on the walk-forward test days: the LSTM line lies on top of the persistence line"))
    out.append(img("rmse_ratio_vs_persistence.png",
                   "RMSE relative to persistence with 95% intervals for each model"))

    # -- reading the one-step result
    lstm, pers, drift = models["LSTM"], models["Persistence"], models.get("Drift")
    v = verdict(lstm["lo"], lstm["hi"])
    best = min(models, key=lambda k: models[k]["RMSE"])
    pf = o["per_fold"]
    fold_gap = max(abs(a - b) for a, b in zip(pf["LSTM"], pf["Persistence"]))
    seed_spread = max(o["seed_rmse"]) - min(o["seed_rmse"])
    reading = [f"The LSTM is {v} \"tomorrow = today\" (RMSE ratio {lstm['ratio']:.3f}, 95% interval "
               f"{_interval(lstm['lo'], lstm['hi'])}). "]
    if drift is not None:
        noise = " (by a margin within noise)" if v == "statistically indistinguishable from" else ""
        reading.append(f"Its RMSE is {_usd(lstm['RMSE'], 3)}, against {_usd(drift['RMSE'], 3)} for drift and "
                       f"{_usd(pers['RMSE'], 3)} for persistence; {labels.get(best, best).split(' (')[0]} has the lowest "
                       f"RMSE of the models compared{noise}. ")
    reading.append(f"In every fold the LSTM's RMSE is within {_usd(fold_gap, 2)} of persistence's, and individual "
                   f"seeds differ by {_usd(seed_spread, 3)}. ")
    if v == "statistically indistinguishable from":
        reading.append("On this data the LSTM shows no measurable edge over the naive baseline. ")
    elif v == "better than":
        reading.append("On this data the LSTM does measurably better than the naive baseline; check the per-fold "
                       "numbers in the notebook before relying on that. ")
    else:
        reading.append("On this data the LSTM does measurably worse than the naive baseline. ")
    reading.append("This is a finding about this setup (a univariate LSTM on past returns, one asset, this period). "
                   "The test suite's positive control shows the same evaluation does detect skill when it exists, "
                   "so a null result is a result, not a tooling failure; it is not proof that no model could work.")
    out.append("**Reading it.** " + "".join(reading))

    # -- live forecast
    fs = f["summary"]
    beat_flat = fs["RMSE_forecast"] < fs["RMSE_flat"]
    beat_drift = fs["RMSE_forecast"] < fs["RMSE_drift"]
    out.append(
        f"**Live {H}-day forecast vs what happened.** From the {_usd(f['origin_price'])} adjusted close on "
        f"{f['origin_date']} the forecast reached {_usd(f['forecast_end'])} on day {fs['days_scored']} "
        f"({fs['forecast_move_%']:+.1f}%, 80% band {_usd(f['band80'][0], 0)} to {_usd(f['band80'][1], 0)}); "
        f"the stock actually closed at {_usd(f['actual_end'])} ({fs['actual_move_%']:+.1f}%)."
    )
    out.append(img("forecast_vs_actual.png", f"{H}-day forecast fan from {f['origin_date']} with the realised closes overlaid"))
    if beat_flat and beat_drift:
        sentence = "The forecast beat both benchmarks on this one path."
    elif beat_flat:
        sentence = "It beat \"flat\" but not drift, so its advantage over \"flat\" is drift, not skill."
    else:
        sentence = "It did not beat \"flat\" on this path."
    out.append(
        f"Over those {fs['days_scored']} days the forecast's RMSE was {_usd(fs['RMSE_forecast'])}, against "
        f"{_usd(fs['RMSE_flat'])} for \"flat at the last close\" and {_usd(fs['RMSE_drift'])} for constant drift at the "
        f"average past return ({fs['drift_move_%']:+.1f}% over the horizon). {sentence} Its bands covered "
        f"{fs['coverage_80_%']:.0f}% (80% band) and {fs['coverage_95_%']:.0f}% (95% band) of the realised closes; "
        "on one strongly autocorrelated path that says little about calibration, which is what the rolling-origin "
        "backtest below is for."
    )

    # -- rolling origin
    t, cov = r["table"], r["coverage"]
    rrows = ["| | RMSE | vs flat (95% interval) | vs drift (95% interval) |", "|---|---|---|---|"]
    for name, label in (("Forecast", "Model forecast"), ("Drift", "Constant drift"), ("Flat", "Flat at last close")):
        x = t[name]
        vf = "1.000" if name == "Flat" else f"{x['vs_Flat']:.3f} ({_interval(x['vs_Flat_lo'], x['vs_Flat_hi'])})"
        vd = "1.000" if name == "Drift" else f"{x['vs_Drift']:.3f} ({_interval(x['vs_Drift_lo'], x['vs_Drift_hi'])})"
        rrows.append(f"| {label} | {_usd(x['RMSE'])} | {vf} | {vd} |")
    fc_t = t["Forecast"]
    out.append(
        f"**Rolling-origin backtest.** {r['n_origins']} forecasts of {r['horizon']} days, one every {r['step']} trading "
        f"days from {r['first_origin']} to {r['last_origin']}, each made by the model of its own fold (trained "
        "before the days it forecasts) using only information available at its origin, and scored on what "
        "actually followed:\n\n" + "\n".join(rrows)
    )
    c80, c95 = cov["80"], cov["95"]
    out.append(
        f"The model forecast's RMSE is {fc_t['vs_Flat']:.3f}x \"flat\" "
        f"({_interval(fc_t['vs_Flat_lo'], fc_t['vs_Flat_hi'])}): {verdict(fc_t['vs_Flat_lo'], fc_t['vs_Flat_hi'])} "
        f"\"flat\"; and {fc_t['vs_Drift']:.3f}x drift ({_interval(fc_t['vs_Drift_lo'], fc_t['vs_Drift_hi'])}): "
        f"{verdict(fc_t['vs_Drift_lo'], fc_t['vs_Drift_hi'])} drift. It had the lower error on "
        f"{r['wins_vs_flat']:.0f}% of origins against \"flat\" and {r['wins_vs_drift']:.0f}% against drift. "
        f"The 80% band contained {c80['mean_%']:.0f}% of realised closes (95% interval {c80['lo_%']:.0f} to "
        f"{c80['hi_%']:.0f}) and the 95% band {c95['mean_%']:.0f}% ({c95['lo_%']:.0f} to {c95['hi_%']:.0f}): "
        f"{coverage_sentence(c80, c95)}. Windows overlap, so intervals are block bootstraps over origins, not "
        "over days."
    )
    out.append(img("rolling_origin_backtest.png",
                   f"Per-origin {r['horizon']}-day error relative to the flat and drift benchmarks, and band coverage vs nominal"))
    return "\n\n".join(out)


# ---- README --------------------------------------------------------------------------------------------

def update_readme(readme_path: str | Path, results: dict, check: bool = False) -> bool:
    """Replace the text between the markers with the rendered section.

    Returns True if the file changed (or, with ``check=True``, if it is already in sync).
    """
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8")
    if text.count(START_MARK) != 1 or text.count(END_MARK) != 1 or text.index(START_MARK) > text.index(END_MARK):
        raise ValueError(f"{path} must contain exactly one {START_MARK} ... {END_MARK} pair")
    head, rest = text.split(START_MARK)
    _, tail = rest.split(END_MARK)
    new = f"{head}{START_MARK}\n\n{render_markdown(results)}\n\n{END_MARK}{tail}"
    if check:
        return new == text
    if new != text:
        path.write_text(new, encoding="utf-8")
    return new != text


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    check = "--check" in args
    paths = [a for a in args if not a.startswith("--")]
    results_path = paths[0] if paths else "docs/results.json"
    readme_path = paths[1] if len(paths) > 1 else "README.md"
    results = load_results(results_path)
    if check:
        ok = update_readme(readme_path, results, check=True)
        print("README results section is in sync" if ok else f"README results section is OUT OF SYNC with {results_path}")
        return 0 if ok else 1
    changed = update_readme(readme_path, results)
    print(f"updated {readme_path}" if changed else f"{readme_path} already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
