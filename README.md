# buyside1

A quantitative research project studying how predictive information translates into tradable returns under changing market conditions, execution frictions, and information delays.

The research agenda has three parts:

- **Adaptive equity alpha:** test whether adapting forecasts and shrinking uncertain signals improves performance after trading costs.
- **Queue-aware execution:** measure how much queue position adds to fill probability and adverse-selection estimates.
- **Disclosure changes:** test whether changes in SEC filings add predictive information beyond fundamentals and price signals.

The repository is in an early research stage. Current code provides CRSP data collection, a monthly liquidity universe, a basic portfolio backtester, and a synthetic Ridge regression demonstration. The studies below describe planned experiments; empirical findings are not yet available.

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
| `scripts/build_universe.py` | Builds monthly universes using 60-day average dollar volume; currently exports holding months in 2010 and retains local path assumptions. |
| `backtest/engine.py` | Computes portfolio returns, weight drift, turnover, proportional transaction costs, NAV, and drawdown; includes plotting. Risk and turnover constraints remain to be implemented. |
| `scripts/demo_scripts/run_equal_weight.py` | Runs an equal-weight backtest on the 2024 sample with 10 bps trading costs. |
| `scripts/demo_scripts/ridge_demo.py` | Illustrates Ridge regularization with synthetic data. Its test-error curve is exploratory; final research will use separate tuning and untouched test samples. |
| `data/pit.py`, `data/linkage.py`, `data/silver/create_tables.sql` | Initial scaffolding for information timing, identifier linkage, and a DuckDB schema. Integration and validation remain in progress. |

LightGBM forecasting, adaptive signals, constrained portfolio construction, futures order-book reconstruction, and SEC disclosure processing are planned work.

## Getting Started

Use Python 3.11 or later and run commands from the repository root. Install the current dependencies with `uv`:

```bash
uv sync
```

Run the synthetic Ridge demonstration without downloading market data:

```bash
uv run python -m scripts.demo_scripts.ridge_demo
```

With WRDS access, download the small CRSP sample and run the equal-weight demonstration:

```bash
export WRDS_USERNAME="your_wrds_username"
uv run python scripts/pull_crsp_sample.py
uv run python -m scripts.demo_scripts.run_equal_weight
```

Download historical CRSP data using explicit date bounds:

```bash
uv run python scripts/pull_crsp_history.py \
  --username "$WRDS_USERNAME" \
  --start-date 2009-01-01 \
  --end-date 2024-12-31
```

The current universe-building entry point still reads 2009–2025 files and an extra local 2000 file; it needs parameterization before it can run directly on the narrower download above. The backtester expects weights for the **start** of each return interval and does not shift them internally.

Raw market data and generated Parquet datasets are excluded from Git. WRDS credentials and licensed datasets must be supplied locally. Futures order-level data and SEC filing ingestion require separate setup; LightGBM is not yet a project dependency.

## Next Steps

1. Parameterize the universe builder for the full 2010–2024 sample and validate missing observations, corporate actions, and delistings.
2. Build leakage-checked features, forward-return labels, and walk-forward experiment tracking.
3. Establish Ridge and LightGBM baselines, then test adaptation and uncertainty shrinkage under matched portfolio constraints.
4. Secure order-level futures data and implement a validated book replay before fitting queue models.
5. Build the filing panel and section-change pipeline, then test incremental information using the same chronological evaluation framework.
