"""Run the current strategy and export aggregate results for the README."""

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
from io import StringIO
import json
from pathlib import Path
import platform
import re
import runpy
import sys
import warnings
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = {"Ridge": "result", "Momentum": "benchmark_result"}
COLORS = {"Ridge": "#314b7c", "Momentum": "#098c80"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(bt):
    returns = bt["portfolio_return"].astype(float)
    std = returns.std(ddof=1)
    return {
        "CAGR": float(bt["nav"].iloc[-1] ** (12 / len(bt)) - 1),
        "Sharpe": float(returns.mean() / std * np.sqrt(12)) if std > 0 else None,
        "max_drawdown": float(bt["drawdown"].min()),
        "avg_monthly_turnover": float(bt["turnover"].mean()),
    }


def export_results(output_dir, as_of):
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    stdout = StringIO()
    with redirect_stdout(stdout), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive.*")
        ns = runpy.run_path(str(ROOT / "scripts/ridge.py"), run_name="__main__")
    # Do not publish captured stdout: existing diagnostics contain stock-level data.
    plt.close("all")

    results = {name: ns[key] for name, key in STRATEGIES.items()}
    weights = {"Ridge": ns["weights"], "Momentum": ns["benchmark_weights"]}
    months = ns["months"]
    raw_returns = ns["realized"].pivot(
        index="month", columns="permno", values="observed_return"
    )
    monthly_frames, annual_rows, summaries = [], [], {}
    costs_bps = []
    for name, bt in results.items():
        w = weights[name]
        assert bt.index.equals(months), f"{name}: incomplete evaluation calendar"
        assert w.index.equals(months), f"{name}: incomplete weight calendar"
        assert np.isfinite(w.to_numpy(dtype=float)).all(), f"{name}: nonfinite weights"
        assert np.isfinite(bt.to_numpy(dtype=float)).all(), f"{name}: nonfinite results"
        assert np.allclose(w.clip(lower=0).sum(axis=1), 0.5)
        assert np.allclose(w.clip(upper=0).sum(axis=1), -0.5)
        raw = raw_returns.reindex(index=w.index, columns=w.columns)
        filled = raw.fillna(0.0)
        assert np.allclose(bt["gross_return"], (w * filled).sum(axis=1))
        assert np.allclose(bt["nav"], (1 + bt["portfolio_return"]).cumprod())
        costs = bt.loc[bt["turnover"].gt(0), "transaction_cost"] / bt.loc[
            bt["turnover"].gt(0), "turnover"
        ] * 10000
        assert np.allclose(costs, costs.iloc[0])
        costs_bps.append(float(costs.iloc[0]))
        missing = (w.ne(0) & raw.isna()).sum(axis=1)
        monthly = bt.copy()
        monthly["strategy"] = name
        monthly["long_holdings"] = w.gt(0).sum(axis=1)
        monthly["short_holdings"] = w.lt(0).sum(axis=1)
        monthly["missing_held_returns_zero_filled"] = missing
        monthly.index.name = "holding_month"
        monthly_frames.append(monthly.reset_index())
        summaries[name] = summarize(bt)
        for year, period in bt.groupby(bt.index.year):
            r = period["portfolio_return"].astype(float)
            nav = (1 + r).cumprod()
            annual_rows.append({
                "year": int(year), "strategy": name, "months": len(period),
                "net_return": float(nav.iloc[-1] - 1),
                "max_drawdown": float((nav / nav.cummax().clip(lower=1) - 1).min()),
                "avg_monthly_turnover": float(period["turnover"].mean()),
                "missing_held_returns_zero_filled": int(missing.loc[period.index].sum()),
            })
    assert np.allclose(costs_bps, costs_bps[0]), "Strategy cost assumptions differ"
    cost_bps = costs_bps[0]

    summary = pd.DataFrame.from_dict(summaries, orient="index")
    summary.index.name = "strategy"
    assert np.allclose(summary.to_numpy(), ns["comparison"].loc[summary.index].to_numpy())
    summary.to_csv(output_dir / "summary.csv", float_format="%.10f")
    pd.concat(monthly_frames, ignore_index=True).to_csv(
        output_dir / "monthly_results.csv", index=False, float_format="%.10f"
    )
    annual = pd.DataFrame(annual_rows).sort_values(["year", "strategy"])
    annual.to_csv(output_dir / "annual_results.csv", index=False, float_format="%.10f")

    alpha_rows = []
    for year_text, alpha_text in re.findall(
        r"预测年份：(\d+)，选择的 alpha：([\d.eE+\-]+)", stdout.getvalue()
    ):
        year = int(year_text)
        train_start = pd.Timestamp(year=year - ns["TRAIN_YEARS"], month=1, day=1)
        test_start = pd.Timestamp(year=year, month=1, day=1)
        p = ns["panel"]
        train_mask = p["holding_month"].ge(train_start) & p["holding_month"].lt(test_start)
        train_mask &= np.isfinite(p["target_ret_1m"])
        alpha_rows.append({
            "test_year": year, "train_start_year": train_start.year,
            "train_end_year": year - 1, "alpha": float(alpha_text),
            "train_rows": int(train_mask.sum()),
            "forecast_rows": int(ns["test"]["holding_month"].dt.year.eq(year).sum()),
        })
    assert len(alpha_rows) == len(months.year.unique()), "Missing annual alpha diagnostics"
    pd.DataFrame(alpha_rows).to_csv(output_dir / "selected_alphas.csv", index=False)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name, bt in results.items():
        axes[0].plot(bt.index, bt["nav"], label=name, color=COLORS[name], linewidth=2)
        axes[1].plot(bt.index, bt["drawdown"], color=COLORS[name], linewidth=1.7)
    axes[0].axhline(1, color="#7a8592", linewidth=0.7, linestyle="--")
    axes[0].set_ylabel("Net asset value (start = 1)")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_xlabel("Holding month")
    axes[1].set_xlim(months[0], months[-1])
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Rolling RidgeCV vs momentum", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.932, f"{months[0]:%Y-%m} to {months[-1]:%Y-%m} | Long 50% / short 50% | {cost_bps:g} bps trading costs", ha="center", color="#465569")
    fig.text(0.5, 0.02, "Monthly top/bottom deciles; missing realized returns set to zero; borrow and financing costs excluded.", ha="center", fontsize=9, color="#465569")
    fig.tight_layout(rect=[0, 0.04, 1, 0.91])
    fig.savefig(output_dir / "nav_drawdown.png", dpi=170, facecolor="white")
    plt.close(fig)

    yearly = annual.pivot(index="year", columns="strategy", values="net_return")
    ax = yearly[["Ridge", "Momentum"]].plot.bar(
        figsize=(12, 5), color=[COLORS["Ridge"], COLORS["Momentum"]], width=0.78
    )
    ax.axhline(0, color="#687789", linewidth=0.8)
    ax.set_title(f"Calendar-year returns after {cost_bps:g} bps trading costs", pad=15)
    ax.set_xlabel("Holding year")
    ax.set_ylabel("Net return")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "annual_returns.png", dpi=170, facecolor="white")
    plt.close(ax.figure)

    code_paths = ["scripts/ridge.py", "scripts/build_features.py", "scripts/build_panel.py",
                  "backtest/engine.py", "scripts/export_results.py"]
    input_paths = ["data/silver/features.parquet", "data/silver/universe_2010_2024.parquet"]
    manifest = {
        "as_of": as_of, "evaluation_start": str(months[0].date()),
        "evaluation_end": str(months[-1].date()), "months": len(months),
        "forecast_rows": len(ns["test"]), "training_window_years": ns["TRAIN_YEARS"],
        "features": ns["FEATURE_COLUMNS"],
        "alpha_candidates": list(ns["model"].named_steps["ridgecv"].alphas),
        "cv": "RidgeCV default leave-one-out; preprocessing fitted on each outer training window",
        "long_weight": 0.5, "short_weight": -0.5, "selection_fraction_each_side": 0.1,
        "transaction_cost_bps": cost_bps, "risk_free_rate_for_sharpe": 0,
        "missing_realized_return": "fill zero, including held positions",
        "code_sha256": {p: sha256(ROOT / p) for p in code_paths},
        "input_sha256": {p: sha256(ROOT / p) for p in input_paths},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__},
        "artifact_sha256": {p.name: sha256(p) for p in sorted(output_dir.iterdir())
                            if p.is_file() and p.name != "manifest.json"},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))
    print("Missing held returns assigned zero:", {
        name: int(frame["missing_held_returns_zero_filled"].sum())
        for name, frame in zip(STRATEGIES, monthly_frames)
    })
    print(f"Exported aggregate results to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/results")
    parser.add_argument("--as-of", default=datetime.now(ZoneInfo("America/New_York")).date().isoformat())
    args = parser.parse_args()
    export_results(args.output_dir, args.as_of)
