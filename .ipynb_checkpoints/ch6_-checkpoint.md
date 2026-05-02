---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Part II Capstone — Seeing Patterns and Trading Signals

In Part II, we studied how to see patterns in time series data.

We introduced:

- visualization,
- trends,
- cycles,
- noise,
- smoothing,
- moving averages,
- exponential smoothing,
- and trading indicators.

This capstone integrates these ideas through a practical applied project using financial price data.

```{admonition} Central Question
How can we extract useful signals from noisy financial time series?
```

The goal is not to prove that a trading rule works.

The goal is to understand how smoothing and filtering tools transform raw price data into interpretable signals.

---

# Learning Goals

By completing this capstone, you should be able to:

- visualize financial time series
- compute rolling averages
- compare short-run and long-run trends
- construct basic trading indicators
- interpret MACD, RSI, and Bollinger Bands
- understand backtesting at a basic level
- recognize overfitting and false signals

---

# Dataset

We will use the Thai SET Index.

```{admonition} Practical Note
You may replace the SET Index with another financial series, such as:
- SET50,
- SPY,
- AAPL,
- gold prices,
- Bitcoin,
- or an exchange rate.
```

---

# Exercise 1 — Downloading and Plotting the SET Index

```{code-cell} python
import yfinance as yf
import matplotlib.pyplot as plt

set_index = yf.download(
    "^SET.BK",
    start="2018-01-01",
    auto_adjust=False
)

prices = set_index["Adj Close"].squeeze()

prices.plot(figsize=(10,4))

plt.title("Thai SET Index Adjusted Closing Prices")
plt.xlabel("Date")
plt.ylabel("Index Level")

plt.savefig("figs/ch6_/set.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![SET](figs/ch6_/set.png)

```{code-cell} python
# Save CSV file
set_index.to_csv(
    "figs/ch6_/set.csv")
```

````{admonition} Get the Data
To download the data run:

``` python
from IPython.display import FileLink
FileLink("figs/ch6_/set.csv")
```
````

---

## Questions

1. Does the SET Index display a clear trend?
2. Can you identify periods of sharp decline?
3. Are there periods where the series appears especially volatile?

---

# Exercise 2 — Rolling Averages

We now compute short-run and long-run moving averages.

```{code-cell} python
ma30 = prices.rolling(30).mean()

ma100 = prices.rolling(100).mean()

plt.figure(figsize=(10,5))

plt.plot(prices, label="SET Index")
plt.plot(ma30, label="30-Day Moving Average")
plt.plot(ma100, label="100-Day Moving Average")

plt.legend()

plt.title("SET Index with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Index Level")

plt.savefig("figs/ch6_/30-100.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![MAC](figs/ch6_/30-100.png)

---

```{admonition} Observation
Moving averages smooth short-run fluctuations and help reveal broader market trends.
```

---

## Questions

1. Which moving average responds faster to price changes?
2. Which moving average is smoother?
3. What happens to the moving averages during market downturns?

---

# Exercise 3 — Moving Average Crossover Signals

A simple trading rule is:

- buy when the short moving average is above the long moving average,
- stay out of the market otherwise.

```{code-cell} python
import numpy as np
import pandas as pd

signals = pd.DataFrame(index=prices.index)

signals["price"] = prices

signals["ma30"] = ma30

signals["ma100"] = ma100

signals["signal"] = np.where(
    signals["ma30"] > signals["ma100"],
    1,
    0
)

signals["crossover"] = signals["signal"].diff()

signals.tail()
```

---

## Plotting Buy and Sell Signals

```{code-cell} python
buy = signals[signals["crossover"] == 1]

sell = signals[signals["crossover"] == -1]

plt.figure(figsize=(10,5))

plt.plot(signals["price"], label="SET Index")
plt.plot(signals["ma30"], label="30-Day MA")
plt.plot(signals["ma100"], label="100-Day MA")

plt.scatter(
    buy.index,
    buy["price"],
    marker="^",
    s=100,
    label="Buy Signal"
)

plt.scatter(
    sell.index,
    sell["price"],
    marker="v",
    s=100,
    label="Sell Signal"
)

plt.legend()

plt.title("Moving Average Crossover Signals")
plt.xlabel("Date")
plt.ylabel("Index Level")

plt.savefig("figs/ch6_/MAC.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Crossover Signal](figs/ch6_/MAC.png)

---

```{admonition} Key Idea
A crossover rule converts smoothed price information into a trading signal.
```

---

## Questions

1. Do the buy signals appear early or late?
2. Do the sell signals appear early or late?
3. What are the advantages and disadvantages of using moving average crossovers?

---

# Exercise 4 — MACD

MACD compares short-run and long-run exponential moving averages.

```{code-cell} python
ema12 = prices.ewm(
    span=12,
    adjust=False
).mean()

ema26 = prices.ewm(
    span=26,
    adjust=False
).mean()

macd = ema12 - ema26

signal_line = macd.ewm(
    span=9,
    adjust=False
).mean()

plt.figure(figsize=(10,5))

plt.plot(macd, label="MACD")
plt.plot(signal_line, label="Signal Line")

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.legend()

plt.title("MACD Indicator")
plt.xlabel("Date")

plt.savefig("figs/ch6_/MACD.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![MACD](figs/ch6_/MACD.png)

---

```{admonition} Interpretation
When MACD rises above the signal line, upward momentum may be strengthening.

When MACD falls below the signal line, momentum may be weakening.
```

---

## Questions

1. How does MACD differ from the simple moving average crossover?
2. Does MACD react faster or slower than the 50/200 moving average rule?
3. Can you identify false signals?

---

# Exercise 5 — RSI

RSI attempts to identify overbought and oversold conditions.

```{code-cell} python
delta = prices.diff()

up = delta.clip(lower=0)

down = -delta.clip(upper=0)

roll_up = up.rolling(14).mean()

roll_down = down.rolling(14).mean()

rs = roll_up / roll_down

rsi = 100 - (100 / (1 + rs))

plt.figure(figsize=(10,4))

plt.plot(rsi, label="RSI")

plt.axhline(
    70,
    linestyle="--",
    linewidth=1,
    label="Overbought"
)

plt.axhline(
    30,
    linestyle="--",
    linewidth=1,
    label="Oversold"
)

plt.legend()

plt.title("Relative Strength Index (RSI)")
plt.xlabel("Date")

plt.savefig("figs/ch6_/rsi.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![RSI](figs/ch6_/rsi.png)

---

```{admonition} Intuition
RSI summarizes recent gains and losses into a bounded momentum indicator.
```

---

## Questions

1. When does RSI move above 70?
2. When does RSI move below 30?
3. Does an overbought signal always imply that prices will fall immediately?

---

# Exercise 6 — Bollinger Bands

Bollinger Bands combine smoothing and volatility.

```{code-cell} python
ma50 = prices.rolling(50).mean()

std50 = prices.rolling(50).std()

upper = ma50 + 2 * std50

lower = ma50 - 2 * std50

plt.figure(figsize=(10,5))

plt.plot(prices, label="SET Index")
plt.plot(ma50, label="50-Day MA")
plt.plot(upper, label="Upper Band")
plt.plot(lower, label="Lower Band")

plt.legend()

plt.title("Bollinger Bands")
plt.xlabel("Date")
plt.ylabel("Index Level")

plt.savefig("figs/ch6_/BB.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Bollinger Bands](figs/ch6_/BB.png)


---

```{admonition} Observation
Bollinger Bands widen during volatile periods and narrow during calm periods.
```

---

## Questions

1. When do the bands become wider?
2. What does it mean when prices move outside the bands?
3. Why are Bollinger Bands related to volatility?

---

# Exercise 7 — A Very Simple Backtest

We now evaluate a simple moving average strategy.

The strategy is:

- hold the market when the 30-day moving average is above the 100-day moving average,
- otherwise hold cash.

```{code-cell} python
returns = prices.pct_change()

strategy_position = signals["signal"].shift(1)

strategy_returns = strategy_position * returns

comparison = pd.DataFrame(index=prices.index)

comparison["Buy and Hold"] = (1 + returns).cumprod()

comparison["MA Strategy"] = (1 + strategy_returns).cumprod()

comparison.plot(figsize=(10,5))

plt.title("Buy-and-Hold vs Moving Average Strategy")
plt.ylabel("Cumulative Growth")
plt.xlabel("Date")

plt.savefig("figs/ch6_/backtest.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Backtest](figs/ch6_/backtest.png)

---

```{admonition} Important
A backtest is not proof that a strategy will work in the future.
```

---

## Questions

1. Which strategy performed better over this sample?
2. Did the moving average strategy reduce losses during downturns?
3. Did the strategy miss some recoveries?
4. How might transaction costs affect the result?

---

# Exercise 8 — Comparing Indicators

Create a short table summarizing the indicators used in this capstone.

| Indicator | Main Idea | Best Suited For |
|---|---|---|
| Moving average crossover | trend following | persistent trends |
| MACD | momentum | trend acceleration |
| RSI | overbought/oversold | reversal analysis |
| Bollinger Bands | deviation from average | volatility regimes |

---

## Questions

1. Which indicators are trend-following?
2. Which indicators are closer to mean-reversion logic?
3. Why might different indicators give conflicting signals?

---

# Mini Project — Your Own Trading Signal Study

Choose one asset.

Examples:

- SET50,
- SPY,
- AAPL,
- Bitcoin,
- Thai baht per U.S. dollar,
- gold prices.

Then complete the following tasks:

1. Download daily price data.
2. Plot adjusted prices.
3. Compute 50-day and 200-day moving averages.
4. Identify crossover signals.
5. Compute MA cross over.
6. Compute RSI.
7. Plot Bollinger Bands.
8. Conduct a simple backtest using all 3 strategies.
9. Discuss whether the indicators were useful.

---

```{admonition} Suggested Extension
Compare two different periods:

- a calm period,
- and a crisis period.

Do the indicators behave differently?
```

---

# Gretl Version

The same ideas can also be explored in Gretl.

---

## Moving Averages

Menu:

```text
Add → Moving average
```

Choose the window length.

---

## Plotting Indicators

Menu:

```text
Variable → Time series plot
```

Then add smoothed series or generated variables.

---

## Creating Returns

Menu:

```text
Add → Define new variable
```

Example:

```text
return = (P - P(-1))/P(-1)
```

---

# Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Treating indicators as magic signals**  
Indicators are transformations of past prices, not guarantees about the future.

**2. Ignoring transaction costs**  
Frequent trading may reduce or eliminate profits.

**3. Using raw prices instead of adjusted prices**  
Stock splits and dividends can distort signals.

**4. Overfitting parameters**  
Choosing parameters that work only in one sample may fail later.

**5. Ignoring market regimes**  
Indicators may perform differently in trending and sideways markets.
```

---

# Looking Ahead

Part III turns from visual pattern recognition to formal time series concepts.

We will study:

- randomness,
- dependence,
- stationarity,
- autocorrelation,
- unit roots,
- and differencing.

```{admonition} Looking Ahead
The next part gives us the formal tools needed to understand whether visual patterns are meaningful or merely noise.
```

# Key Takeaways

```{admonition} Summary
- Smoothing tools can help reveal patterns in noisy financial data.
- Moving averages are basic filters used in both analysis and trading.
- MACD measures momentum using exponential averages.
- RSI identifies overbought and oversold conditions.
- Bollinger Bands combine smoothing with volatility.
- Backtesting evaluates historical strategy performance.
- Trading indicators are useful learning tools but imperfect forecasting devices.
- Practical analysis requires attention to adjusted prices, costs, and overfitting.
