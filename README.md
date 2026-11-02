# buyside1

A quantitative research project studying how predictive information translates into tradable returns under changing market conditions, execution frictions, and information delays.

The research agenda has three parts:

- **Adaptive equity alpha:** test whether adapting forecasts and shrinking uncertain signals improves performance after trading costs.
- **Queue-aware execution:** measure how much queue position adds to fill probability and adverse-selection estimates.
- **Disclosure changes:** test whether changes in SEC filings add predictive information beyond fundamentals and price signals.

The equity research prototype now runs from CRSP daily data through a monthly liquidity universe, monthly features, Ridge forecasts, and an equal-weight long-only backtest. The current experiment trains on 2010–2014 and evaluates monthly holdings in 2015. It selects the top 20% of positive forecasts each month, uses zero transaction costs, and temporarily assigns zero returns to holdings with missing realized returns.

This run verifies the research workflow; controlled comparisons and empirical conclusions remain to be established. Adaptive models, queue-aware execution, and SEC disclosure analysis remain planned research. The approaches below describe the intended studies rather than a list of completed experiments.

## 1. Adaptive Equity Alpha under Distribution Shift and Trading Costs

**Research question:** Can models that adapt to changes in predictive relationships outperform fixed-window models when risk, turnover, and trading costs are held comparable?

**Target dataset:** Approximately 1,000 U.S. equities per month over 2010–2024, with earlier observations used for feature and liquidity lookbacks.

### Approach

1. **Build a point-in-time universe.** At each month-end, rank eligible stocks by their previous 60 trading days of average dollar volume. Select up to 1,000 stocks with a price of at least $5 and sufficient valid history for the following month. Preserve historical membership as stocks enter, leave, or delist. Finalize issuer eligibility before freezing the sample; the current downloader filters U.S. listing venues but does not require U.S. incorporation.
2. **Construct features and labels.** Start with lagged returns, momentum, reversal, volatility, and liquidity features to forecast next-month returns. Fit imputation, scaling, and any feature selection within each training window using only information available at the forecast date.
3. **Establish forecasting baselines.** Compare Ridge and LightGBM using fixed rolling windows and an expanding-window benchmark, with the same features and prediction horizon.
4. **Introduce adaptation and uncertainty shrinkage.** Test recent-performance weighting and adaptive training windows. Use historical validation errors and model disagreement as uncertainty proxies, shrinking less reliable forecasts toward zero. Select adaptation and shrinkage settings using past validation periods.
5. **Translate forecasts into portfolios.** Apply the same exposure limits, risk target, rebalance schedule, and turnover budget across models. Report whether realized risk and turnover remain comparable. Begin with proportional trading costs, then test higher costs and liquidity-dependent assumptions.
6. **Evaluate chronologically.** Use walk-forward training and validation within 2010–2021, then freeze the research design for a three-year holdout covering 2022–2024. Keep overlapping forward-return labels from crossing training and validation boundaries.

**Evaluation:** Forecast error and rank correlation; gross and net returns; Sharpe ratio; maximum drawdown; realized risk; turnover; and break-even trading costs. Ablate adaptation and uncertainty shrinkage separately to identify their contributions.

## 2. Queue-Aware Fill Probability and Adverse Selection

**Research question:** Does knowing an order's place in the queue improve execution forecasts beyond price-level order-book and order-flow information?

**Target dataset:** Order-level messages for one equity index futures contract across approximately 30 trading sessions. Contract selection and data access remain to be established.

### Approach

1. **Reconstruct the order book.** Process additions, modifications, cancellations, and executions in exchange sequence. Validate reconstruction against available snapshots and the selected contract's matching rules. Track order identifiers, queue priority, and quantity ahead.
2. **Define order outcomes.** Start with time to first fill versus cancellation as competing outcomes. Treat orders still resting at the observation boundary as censored, and distinguish partial fills from complete fills in subsequent analysis.
3. **Build features and benchmarks.** Use queue position, quantity ahead, spread, depth imbalance, recent trades, and cancellation intensity. Compare a model using only price-level and flow features with one that also includes queue information.
4. **Test chronologically and calibrate probabilities.** Separate training, calibration, and final testing by session. Compare probability calibration, Brier score, and fill-versus-cancellation performance at prespecified horizons.
5. **Measure execution quality.** Calculate signed post-fill midprice markouts at fixed horizons to assess adverse selection. Repeat the analysis under delayed order arrival and different hypothetical queue placements, documenting the replay assumptions.

**Evaluation:** The incremental forecasting value of queue features, probability reliability, post-fill markouts, and sensitivity to latency. Historical-order prediction and hypothetical-order replay will be reported separately.

## 3. Incremental Information in SEC Disclosure Changes

**Research question:** Do changes in annual disclosures improve return forecasts after accounting for numerical fundamentals and price-based signals?

**Target dataset:** Approximately 5,000 annual filings from 500 U.S. issuers over 2015–2024, subject to filing availability. Earlier filings may be needed to establish the first year-over-year comparisons.

### Approach

1. **Build the filing panel.** Record SEC accession identifiers, issuer identifiers, reporting periods, and public acceptance timestamps. Link issuers to historically valid securities and map each disclosure to its first eligible trading session after a processing delay.
2. **Extract and align sections.** Identify MD&A and Risk Factors, align them with the previous annual filing, and filter repeated boilerplate. Audit extraction quality on a manually reviewed sample.
3. **Measure disclosure changes.** Begin with additions, deletions, section growth, textual novelty, and changes in risk-related language. Compare these with features based on the full section text.
4. **Test incremental information.** Compare a baseline using point-in-time numerical fundamentals and price signals with the same model augmented by disclosure-change features. Keep the stock sample, return horizon, and portfolio rules consistent.
5. **Validate robustness.** Develop models on earlier years and reserve later years for chronological testing. Run feature ablations and delayed-entry experiments, such as one, three, and five trading sessions after eligibility. Use historical fundamentals as available at the time rather than subsequently revised values.

**Evaluation:** Out-of-sample forecast improvement, rank correlation, and cost-adjusted portfolio performance. Test whether any benefit persists after removing boilerplate, changing entry delays, and controlling for numerical information.

## Shared Research Design

- **Information timing:** Record when an observation became available and when it could first affect a trade. Lag signals and portfolio weights accordingly.
- **Chronological validation:** Tune on earlier periods, respect label horizons at split boundaries, and keep final holdouts untouched during development.
- **Controlled comparisons:** Change one component at a time while keeping samples, targets, portfolio constraints, and cost assumptions comparable.
- **Reproducibility:** Save data definitions, feature versions, model settings, split dates, and experiment outputs. Report sensitivity and failed hypotheses alongside successful results.

## Current Implementation

| Component | Current scope |
| --- | --- |
| `scripts/pull_crsp_history.py` | Downloads yearly CRSP daily data; defaults to 2000–2025. Stores security identifiers, dates, returns, prices, volume, exchange, and trading status. |
| `scripts/pull_crsp_sample.py` | Downloads a 10-stock sample for the first half of 2024. |
| `scripts/build_universe.py` | Builds up to 1,000-stock monthly universes using 60-day average dollar volume, an absolute price floor of $5, 60 valid dollar-volume observations, and active status at formation. The default entry point exports holding months in 2010–2024. |
| `scripts/build_features.py` | Aggregates daily data for 2009–2024, constructs four predictors and calendar-aligned next-month labels, and saves `data/silver/features.parquet`. |
| `scripts/build_panel.py` | Left-joins the monthly universe to features on `permno` and `holding_month`, enforcing one-to-one keys and retaining unmatched universe rows. The function returns a DataFrame; the script prints diagnostics without saving a separate panel. |
| `scripts/ridge.py` | Fits training-only median imputation, scaling, and Ridge with `alpha=0.1`; predicts 2015; selects and weights positive signals; aligns full-market realized returns; and calls the backtester. |
| `backtest/engine.py` | Computes portfolio returns, weight drift, turnover, proportional transaction costs, NAV, and drawdown; includes plotting. Risk and turnover constraints remain to be implemented. |
| `scripts/demo_scripts/run_equal_weight.py` | Runs an equal-weight backtest on the 2024 sample with 10 bps trading costs. |
| `scripts/demo_scripts/ridge_demo.py` | Compares Ridge and Lasso on synthetic data. Its test-error-based parameter selection is exploratory; final research will use separate tuning and untouched test samples. |
| `scripts/demo_scripts/fused_lasso.py`, `scripts/demo_scripts/non_linear_model` | Synthetic structural-break, KNN, and kernel-smoothing examples. |
| `scripts/demo_scripts/histGBM.py` | Unfinished synthetic regime example; currently contains an `np.rando` typo and does not fit a forecasting model. |
| `data/pit.py`, `data/linkage.py`, `data/silver/create_tables.sql` | Initial scaffolding for information timing, identifier linkage, and a DuckDB schema. Integration and validation remain in progress. |

### Feature and Label Timing

Each feature row is keyed by a stock and its information month `month`. Dates such as `2015-01-01` identify a month; they do not mean that its full-month return was known on the first day.

| Field | Current definition for information month `m` |
| --- | --- |
| `formation_date` | Last observed market date in month `m`, inferred from the input daily panel. |
| `observed_return`, `ret_1m` | Compounded available daily returns in month `m`. `ret_1m` currently copies `observed_return`. |
| `reversal_1m` | Negative of `ret_1m`. |
| `mom_12m` | Compounded returns from `m-11` through `m-1`: eleven months excluding the information month. The calculation reindexes the full monthly calendar before shifting and rolling. |
| `vol_1m` | Sample standard deviation of available daily returns in month `m`, without annualization. |
| `adv_1m`, `log_adv_1m` | Average valid daily absolute price times volume in month `m`, and `log1p` of that average. |
| `holding_month` | Month `m+1`, whose return the model forecasts. |
| `target_ret_1m` | The same stock's `ret_1m` in month `m+1`, matched by calendar month. |
| `n_valid_returns` | Count of nonmissing daily returns in information month `m`; it does not certify complete coverage or valid numerical values. |

For example, the December 2014 feature row has `holding_month=2015-01-01` and a target equal to January 2015's realized return. For portfolio evaluation, the script separately reads `observed_return` from the full-market row with `month=2015-01-01` and aligns it to January's weights. It does not require the stock to remain in a later month's universe.

### Current Portfolio Experiment

1. Join universe membership to features using `permno` and `holding_month`.
2. Train on holding months January 2010–December 2014 with finite target returns. Fit median imputation and standardization within this training sample, followed by Ridge with `alpha=0.1`.
3. Use the fixed fitted model to forecast all universe rows for January–December 2015. Prediction eligibility is not filtered by whether the future target is available.
4. Each month, retain strictly positive forecasts, sort by forecast descending and `permno` ascending to resolve ties, and select `ceil(0.20 * number_of_positive_forecasts)` stocks.
5. Give selected stocks equal target weights summing to one. Reindex weights to all twelve evaluation months so a month without selections is represented by zero stock weights and implicit cash with zero return.
6. Obtain realized monthly returns from the full-market feature file using `month` and `permno`. Mark observations with no valid daily returns as missing, align returns to the weights, and fill missing values with zero under the current temporary policy. Set nonheld assets' returns to zero.
7. Pass the start-of-month weights and the same month's realized returns to `backtest.engine.backtest`, with `initial_nav=1.0`, `transaction_cost_bps=0`, and plotting enabled. The engine does not shift weights internally.

The script produces NAV and drawdown plots. Predictions, weights, and the result DataFrame remain in memory; experiment reports and model artifacts are not automatically saved.

### Current Assumptions and Limitations

- **Missing realized returns:** Zero-filling applies to held stocks as well as nonheld positions. It assumes a zero monthly return, not a verified failed order, cash settlement, or delisting outcome. The script computes `invalid_held` after filling missing values but does not enforce it with an exception. Delisting and corporate-action reconciliation remain pending.
- **Monthly data quality:** The feature builder records coverage counts but does not enforce complete months. Its default product aggregation can turn entirely missing daily returns into a zero monthly return. The realized-return path masks zero-valid-return observations before its explicit zero-fill; this does not repair training features or labels, nor partial-month observations.
- **Execution timing:** Features use information through the formation close, while realized returns are calendar-month daily-return compounds. The prototype assumes monthly beginning weights; next-open fills, overnight gaps, execution delays, and order availability are not modeled. Calendar alignment alone does not establish executable, leakage-free returns.
- **Evaluation and costs:** The current experiment uses a single 2015 development period and zero transaction costs, without rolling refits, parameter validation, benchmark comparisons, or exposure constraints. The planned 2022–2024 holdout is not evaluated by this script.
- **Research scaffolding:** LightGBM forecasting, adaptive signals, constrained portfolio construction, futures book reconstruction, and SEC processing are not implemented. Some exploratory modules are unfinished, and `data/pit.py` currently has a `pd.Dataframe` annotation typo.

## Getting Started

Use Python 3.11 or later and run commands from the repository root. Install the current dependencies with `uv`:

```bash
uv sync
```

For the real-data workflow, download full calendar years including the 2009 feature and liquidity lookback. Supply your WRDS username:

```bash
export WRDS_USERNAME="your_wrds_username"
uv run python scripts/pull_crsp_history.py \
  --username "$WRDS_USERNAME" \
  --start-date 2009-01-01 \
  --end-date 2024-12-31
```

Then build the universe and features, and run the monthly strategy:

```bash
uv run python scripts/build_universe.py
uv run python scripts/build_features.py
uv run python scripts/ridge.py
```

The universe and feature scripts currently use defaults defined in their Python entry points; they do not expose date arguments through a CLI. The universe builder reads yearly files from 2009 through 2024 and writes `data/silver/universe_2010_2024.parquet`. The feature builder writes `data/silver/features.parquet`. `ridge.py` imports `build_panel.py`, which reads and prints local data during import. Run `ridge.py` as the script shown above; its root-path setup makes the sibling `backtest` package importable.

With generated Parquet files already available, only the last command is needed. To inspect the merged panel separately:

```bash
uv run python scripts/build_panel.py
```

Run the synthetic Ridge/Lasso demonstration without market data, or the separate ten-stock equal-weight sample:

```bash
uv run python -m scripts.demo_scripts.ridge_demo

# Requires WRDS_USERNAME and WRDS access for the sample download.
uv run python scripts/pull_crsp_sample.py
uv run python -m scripts.demo_scripts.run_equal_weight
```

Raw market data, generated Parquet datasets, and local DuckDB files are excluded from Git. WRDS credentials and licensed datasets must be supplied locally. The history downloader filters common equity and selected U.S. listing venues but does not require U.S. incorporation; these filters still need an audit for event and terminal-return coverage. Futures data and SEC ingestion require separate setup; LightGBM is not yet a dependency.

## Local Validation Snapshot

On November 2, 2026, the current strategy was run against the existing local Parquet files with a noninteractive plotting backend. Source compilation and TOML parsing also passed.

| Check | Observed local result |
| --- | --- |
| Training stock-months, 2010–2014 | 59,970 |
| Forecast stock-months, 2015 | 12,000 |
| Selected stock-months | 2,348 |
| Backtest periods | 12 months |
| Missing held monthly returns assigned zero | 7 |
| Weight/return axes and finite backtest outputs | Passed |
| Monthly selection counts, descending ranks, and equal weights | Passed |

These checks validate the current calculation path and temporary assumptions. They do not establish predictive performance, completeness of market data, or execution feasibility. Counts depend on the local dataset; licensed data and output artifacts are not included in the repository.

## Next Steps

1. Reconcile missing held returns and terminal events, audit the downloader's filters, and validate daily inputs and partial-month coverage. Compare the current zero-return assumption with verified outcomes.
2. Save forecasts, weights, data-quality diagnostics, model settings, split dates, and backtest reports so experiments can be reproduced and compared. Add equal-weight and momentum benchmarks using the same eligible sample and return conventions.
3. Establish chronological rolling and expanding-window Ridge and LightGBM baselines, select parameters on past validation periods, and leave the planned 2022–2024 holdout untouched until the design is frozen.
4. Align signals with executable trade times, evaluate positive and liquidity-dependent trading costs, and introduce common exposure, risk, and turnover constraints before testing adaptation and uncertainty shrinkage.
5. Secure order-level futures data and implement a validated book replay before fitting queue models.
6. Build the filing panel and section-change pipeline, then test incremental information using the same chronological evaluation framework.
