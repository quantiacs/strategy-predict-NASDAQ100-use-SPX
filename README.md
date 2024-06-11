# Predicting stocks using the SPX index

This trading strategy is designed for the [Quantiacs](https://quantiacs.com/contest) platform, which hosts competitions
for trading algorithms. Detailed information about the competitions is available on
the [official Quantiacs website](https://quantiacs.com/contest).

## How to Run the Strategy

### In an Online Environment

The strategy can be executed in an online environment using Jupiter or JupiterLab on
the [Quantiacs personal dashboard](https://quantiacs.com/personalpage/homepage). To do this, clone the template in your
personal account.

### In a Local Environment

To run the strategy locally, you need to install the [Quantiacs Toolbox](https://github.com/quantiacs/toolbox).

## Strategy Overview

This file is a **Jupyter notebook** that contains a detailed implementation for **predicting stocks** within the
**NASDAQ 100 index** using the **SPX index** and various **technical analysis indicators**. The notebook is structured
into several **markdown and code cells** that describe the approach and implement multiple **trading strategies**. Each
strategy utilizes different technical indicators such as the **Trix**, **RSI**, and **ROC** to generate trading signals.
The notebook also includes functions to **load market data**, **calculate the indicators**, **apply trading logic**,
and **evaluate the performance** of the strategies by displaying statistics and **equity curves**. Additionally, there
are examples demonstrating the use of **moving averages**, **exponential moving averages**, and the **Advance–Decline
line** in trading strategies.

```python

from IPython.display import display
import xarray as xr
import qnt.data as qndata
import qnt.output as qnout
import qnt.ta as qnta
import qnt.stats as qns


def get_SPX(market_data):
    index_name = 'SPX'
    index_data = qndata.index.load_data(assets=[index_name], min_date='2005-01-01', forward_order=True)
    spx_data = index_data.sel(asset=index_name)
    spx_data = xr.align(spx_data, market_data.isel(field=0), join='right')[0]
    return spx_data


def get_strategy_1(data, spx, params):
    def get_trix(prices, index, periods):
        result = prices.copy(True)
        t = qnta.trix(index, periods)
        for a in prices.asset.values:
            result.loc[{"asset": a}] = t
        return result

    trix = get_trix(data.sel(field='close'), spx, 40)

    strategy_1 = trix.shift(time=params[0]) < trix.shift(time=params[1])
    strategy_2 = trix.shift(time=params[2]) > trix.shift(time=params[3])

    weights = strategy_1 * strategy_2 * data.sel(field="is_liquid")
    return weights.fillna(0)


def get_strategy_2(data, spx, params):
    def get_rsi(prices, index, periods):
        result = prices.copy(True)
        r = qnta.rsi(index, periods)
        for a in prices.asset.values:
            result.loc[{"asset": a}] = r
        return result

    rsi = get_rsi(data.sel(field='close'), spx, 40)

    strategy_1 = rsi.shift(time=params[0]) < rsi.shift(time=params[1])
    strategy_2 = rsi.shift(time=params[2]) > rsi.shift(time=params[3])

    weights = strategy_1 * strategy_2 * data.sel(field="is_liquid")
    return weights.fillna(0)


def get_strategy_3(data, spx, params):
    def get_roc(prices, index, periods):
        result = prices.copy(True)
        r = qnta.roc(index, periods)
        for a in prices.asset.values:
            result.loc[{"asset": a}] = r
        return result

    roc = get_roc(data.sel(field='close'), spx, 15)

    strategy_1 = roc.shift(time=params[0]) < roc.shift(time=params[1])
    strategy_2 = roc.shift(time=params[2]) > roc.shift(time=params[3])

    weights = strategy_1 * strategy_2 * data.sel(field="is_liquid")
    return weights.fillna(0)


data = qndata.stocks.load_ndx_data(min_date="2005-01-01")
spx = get_SPX(data)

weights_1_1 = get_strategy_1(data, spx, [142, 54, 132, 63])
weights_1_2 = get_strategy_1(data, spx, [166, 75, 46, 24])
weights_2 = get_strategy_2(data, spx, [159, 78, 77, 167])
weights_3 = get_strategy_3(data, spx, [10, 27, 29, 41])

weights_all = weights_1_1 + weights_1_2 + weights_2 + weights_3
weights = qnout.clean(output=weights_all, data=data, kind="stocks_nasdaq100")


def print_statistic(data, weights_all):
    import plotly.graph_objs as go
    import qnt.stats as qnstats

    stats = qnstats.calc_stat(data, weights_all)
    display(stats.to_pandas().tail(5))

    equity_curve = stats.loc[:, "equity"]
    fig = go.Figure(data=[
        go.Scatter(
            x=equity_curve.time.to_pandas(),
            y=equity_curve,
            hovertext="Equity curve",
        )
    ])
    fig.update_yaxes(fixedrange=False)
    fig.show()


print_statistic(data, weights)
qnout.check(weights, data, "stocks_nasdaq100")
qnout.write(weights)  # To participate in the competition, save this code in a separate cell.

```
