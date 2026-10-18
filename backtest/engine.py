import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def backtest(
        returns:pd.DataFrame,
        weights:pd.DataFrame,
        *,
        plot:bool = True,
        transaction_cost_bps: float=0.0,
        initial_nav:float=1.0
)->pd.DataFrame:
    """
    returns: simple return for each asset in certain time interval
    weights: the holding weight at the starting of each intervel(not shift)
    plot:whether to display the nav and drawdown or not
    transaction_cost_bps:10bps = 0.1%, calculate as turnover*bps
    """
    records=[]

    #error check
    if returns.empty:
        raise ValueError("returns should not be empty")
    if not np.isfinite(initial_nav) or initial_nav<=0:
        raise ValueError("initial nav has to be postive")


    #main logic
    previous_end_weights = np.zeros(len(returns.columns))
    for i in range(len(returns)):
        asset_returns = returns.iloc[i]
        target_weights = weights.iloc[i]

        turnover = np.abs(target_weights-previous_end_weights).sum()
        transaction_cost = turnover * transaction_cost_bps/10000

        gross_return = (asset_returns*target_weights).sum()
        portfolio_return = gross_return-transaction_cost

        records.append({
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "portfolio_return": portfolio_return,
        })
        previous_end_weights = (
            target_weights
            * (1 + asset_returns)
            / (1 + portfolio_return)
        )

    result = pd.DataFrame(records, index=returns.index)

    result["nav"] = initial_nav *(1+result["portfolio_return"]).cumprod()
    peak = result["nav"].cummax().clip(lower = initial_nav)

    result["drawdown"] = result["nav"]/peak-1

    if plot:
        fig,axes = plt.subplots(
            2,1,
            figsize=(12,7)
        )
        result["nav"].plot(
            ax=axes[0]
        )
        axes[0].set_title("Portfolio NAV After Transaction Costs")
        axes[0].set_ylabel("NAV")

        result["drawdown"].plot.area(
            ax=axes[1],
        )
        axes[1].set_title("Drawdown")
        axes[1].set_ylabel("Drawdown")
        axes[1].set_xlabel(returns.index.name or "Period")

        fig.tight_layout()
        plt.show()
    return result
