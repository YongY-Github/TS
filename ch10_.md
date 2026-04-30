---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Part III Capstone — Dependence, Stationarity, and Unit Roots

In Part III, we studied the core ideas that make time series analysis different from ordinary statistics.

We introduced:

- randomness,
- dependence,
- autocorrelation,
- stationarity,
- ACF and PACF,
- unit roots,
- and differencing.

This capstone integrates these ideas using a practical workflow.

```{admonition} Central Question
How can we diagnose whether a time series is stable, persistent, or nonstationary?
```

The goal is to move from visual inspection to formal diagnosis.

---

# Learning Goals

By completing this capstone, you should be able to:

- plot time series data
- distinguish levels from differences
- inspect persistence visually
- compute and interpret ACF and PACF plots
- perform a unit root test
- transform nonstationary data using differencing
- explain why stationarity matters before modeling

---

# Dataset

We use the Thai SET Index as the main example.

```{admonition} Practical Note
You may replace the SET Index with another time series, such as:
- CPI,
- GDP,
- exchange rates,
- stock prices,
- cryptocurrency prices,
- or interest rates.
```

---

# Exercise 1 — Download and Plot the Series

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

plt.title("Thai SET Index")
plt.xlabel("Date")
plt.ylabel("Index Level")

plt.show()
```

---

## Questions

1. Does the series appear to fluctuate around a stable mean?
2. Does it appear persistent?
3. Are there obvious periods of sharp decline or recovery?

---

```{admonition} Observation
Asset prices often display strong persistence in levels.
```

---

# Exercise 2 — Compute Returns

```{code-cell} python
import numpy as np

returns = 100 * np.log(
    prices / prices.shift(1)
)

returns = returns.dropna()

returns.plot(figsize=(10,4))

plt.title("Thai SET Index Log Returns")
plt.xlabel("Date")
plt.ylabel("Log Return (%)")

plt.show()
```

---

## Questions

1. How does the return series differ from the price series?
2. Does the return series appear more stable?
3. Are there periods of high volatility?

---

```{admonition} Key Idea
Prices often appear nonstationary, while returns are usually closer to stationary.
```

---

# Exercise 3 — Autocorrelation of Prices

We now examine the autocorrelation function of the price level.

```{code-cell} python
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(
    prices.dropna(),
    lags=40
)

plt.title("ACF of SET Index Level")

plt.show()
```

---

## Questions

1. Do autocorrelations decline quickly or slowly?
2. What does slow decay suggest?
3. Why might persistent autocorrelation be a warning sign?

---

```{admonition} Interpretation
Slowly decaying autocorrelations often suggest nonstationarity or near-nonstationarity.
```

---

# Exercise 4 — Autocorrelation of Returns

```{code-cell} python
plot_acf(
    returns,
    lags=40
)

plt.title("ACF of SET Index Returns")

plt.show()
```

---

## Questions

1. Are return autocorrelations smaller than price autocorrelations?
2. Do returns appear closer to white noise?
3. Are there any significant autocorrelations?

---

```{admonition} Observation
Financial returns often show weak autocorrelation in the mean, even when volatility is strongly persistent.
```

---

# Exercise 5 — PACF of Returns

The partial autocorrelation function helps identify direct lag relationships.

```{code-cell} python
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(
    returns,
    lags=40,
    method="ywm"
)

plt.title("PACF of SET Index Returns")

plt.show()
```

---

## Questions

1. Are there strong partial autocorrelations?
2. Would an AR model likely be useful for the mean of returns?
3. Why might returns be difficult to forecast?

---

# Exercise 6 — Unit Root Test on Prices

We now perform the Augmented Dickey-Fuller test.

```{code-cell} python
from statsmodels.tsa.stattools import adfuller

adf_price = adfuller(
    prices.dropna()
)

print("ADF Statistic:", adf_price[0])
print("p-value:", adf_price[1])
```

---

## Questions

1. What is the null hypothesis of the ADF test?
2. Is the p-value small or large?
3. Do we reject the unit root null?

---

```{admonition} Important
The ADF test has the null hypothesis that the series has a unit root.
```

A large p-value means we fail to reject the possibility of a unit root.

---

# Exercise 7 — Unit Root Test on Returns

```{code-cell} python
adf_returns = adfuller(
    returns.dropna()
)

print("ADF Statistic:", adf_returns[0])
print("p-value:", adf_returns[1])
```

---

## Questions

1. Is the return series more stationary than the price level?
2. How does the p-value compare with the price-level test?
3. Why does differencing often help with nonstationarity?

---

```{admonition} Key Insight
Taking returns is closely related to differencing log prices.
```

---

# Exercise 8 — First Difference of Prices

Instead of log returns, we can also examine first differences.

```{code-cell} python
price_diff = prices.diff().dropna()

price_diff.plot(figsize=(10,4))

plt.title("First Difference of SET Index")
plt.xlabel("Date")
plt.ylabel("Change in Index Level")

plt.show()
```

---

## Questions

1. How does the first difference compare with log returns?
2. Does differencing remove the long-run trend?
3. Which transformation is easier to interpret financially?

---

# Exercise 9 — Comparing Levels, Differences, and Returns

```{code-cell} python
import pandas as pd

comparison = pd.DataFrame({
    "Level": prices,
    "First Difference": price_diff,
    "Log Return": returns
})

comparison.describe()
```

---

## Questions

1. Which series has the largest scale?
2. Which series appears most stable?
3. Why should we avoid comparing standard deviations across variables with different units?

---

# Exercise 10 — Rolling Mean and Rolling Variance

Stationarity requires more than a stable mean.

It also involves stable variance.

```{code-cell} python
rolling_mean = returns.rolling(60).mean()
rolling_std = returns.rolling(60).std()

plt.figure(figsize=(10,4))

plt.plot(
    rolling_mean,
    label="Rolling Mean"
)

plt.plot(
    rolling_std,
    label="Rolling Standard Deviation"
)

plt.legend()

plt.title("Rolling Mean and Volatility of SET Returns")

plt.show()
```

---

## Questions

1. Is the rolling mean relatively stable?
2. Is the rolling standard deviation stable?
3. What does changing volatility suggest?

---

```{admonition} Looking Ahead
Changing volatility motivates ARCH and GARCH models later in the book.
```

---

# Exercise 11 — Simulating a Stationary AR(1)

Now compare the financial data with a simulated stationary process.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

T = 500
phi = 0.6

e = np.random.normal(size=T)

x = np.zeros(T)

for t in range(1, T):
    x[t] = phi * x[t-1] + e[t]

plt.figure(figsize=(10,4))

plt.plot(x)

plt.title("Simulated Stationary AR(1) Process")

plt.show()
```

---

## Questions

1. Does this series fluctuate around a stable mean?
2. Does it look different from the SET price level?
3. How does it compare with returns?

---

# Exercise 12 — Simulating a Random Walk

```{code-cell} python
np.random.seed(123)

T = 500

e = np.random.normal(size=T)

rw = np.cumsum(e)

plt.figure(figsize=(10,4))

plt.plot(rw)

plt.title("Simulated Random Walk")

plt.show()
```

---

## Questions

1. Does the random walk return to a stable mean?
2. Does it look more like prices or returns?
3. Why is a random walk difficult to forecast?

---

```{admonition} Key Idea
A random walk is highly persistent and contains a unit root.
```

---

# Mini Project — Diagnosing Stationarity

Choose one time series.

Examples:

- SET Index,
- Thai baht exchange rate,
- CPI,
- GDP,
- Bitcoin,
- gold prices,
- oil prices.

Complete the following tasks:

1. Plot the level series.
2. Compute first differences or log returns.
3. Plot the transformed series.
4. Plot ACF and PACF.
5. Perform an ADF test on the level.
6. Perform an ADF test on the transformed series.
7. Compare the results.
8. Explain whether the original series appears stationary.

---

```{admonition} Suggested Extension
Repeat the analysis using two different time periods.

Does the behavior of the series change across periods?
```

---

# GRETL Version

The same workflow can be performed in GRETL.

---

## Plotting the Series

Menu:

```text
Variable → Time series plot
```

---

## First Difference

Menu:

```text
Add → First differences of selected variables
```

or command:

```text
diff x
```

---

## ACF and PACF

Menu:

```text
Variable → Correlogram
```

---

## Unit Root Test

Menu:

```text
Variable → Unit root tests → Augmented Dickey-Fuller test
```

---

```markdown
[GRETL Screenshot Placeholder: ADF test output]
```

---

# Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Judging stationarity only by looking at the graph**  
Visual inspection is important but not enough.

**2. Forgetting the ADF null hypothesis**  
The null is that the series has a unit root.

**3. Treating prices and returns as statistically similar**  
Prices and returns usually have very different time series properties.

**4. Ignoring volatility changes**  
A series may have stable mean but changing variance.

**5. Over-differencing**  
Differencing too much can remove useful structure.
```

---

# Looking Ahead

Part IV introduces linear time series models.

We will use the concepts developed in Part III to build:

- AR models,
- MA models,
- ARMA models,
- and ARIMA models.

```{admonition} Looking Ahead
Stationarity, autocorrelation, and differencing are the foundations for ARIMA modeling.
```

---

# Key Takeaways

```{admonition} Summary
- Time series data often display persistence.
- ACF and PACF help diagnose dependence.
- Price levels are often nonstationary.
- Returns are usually closer to stationary.
- The ADF test helps detect unit roots.
- Differencing can transform nonstationary series.
- Stationarity is essential for many time series models.
- Changing volatility remains important even after differencing.
