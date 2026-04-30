---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Part I Capstone — Working with Financial Data

In Part I, we introduced:

- time series data,
- financial returns,
- probability and uncertainty,
- and the statistical foundations needed for later chapters.

This capstone integrates these ideas through a small applied project using real financial data.

```{admonition} Central Question
How can we transform raw financial prices into useful statistical and economic information?
```

The exercises below combine:

- data collection,
- visualization,
- return calculations,
- probability concepts,
- and interpretation.

The goal is not only to compute statistics, but also to think economically about what the data represent.

---

# Learning Goals

By completing this capstone, you should be able to:

- download and organize financial data
- compute simple and log returns
- visualize prices and returns
- interpret volatility
- understand stylized facts of financial data
- connect statistical concepts with financial interpretation

---

# Background: Financial Time Series

Financial markets generate enormous quantities of time series data.

Examples include:

- stock prices,
- exchange rates,
- interest rates,
- commodity prices,
- cryptocurrency prices.

Raw price data alone are often difficult to interpret statistically.

For this reason, analysts usually transform prices into:

```{admonition} Definition
Returns.
```

Returns provide a more meaningful measure of financial performance and risk.

---

# Dataset

We will use:

```text
SET Index
```

the ETF tracking the Thai SET index.

You may later replace this with:

- S&P 500 index,
- Korean KOSPI,
- exchange rates,
- or other assets.

---

# Exercise 1 — Downloading and Visualizing Prices

We begin by downloading adjusted price data.

```{code-cell} python
import yfinance as yf
import matplotlib.pyplot as plt

# Download SET Index data
set_index = yf.download(
    "^SET.BK",
    start="2018-01-01",
    auto_adjust=False
)

# Adjusted closing prices
prices = set_index["Adj Close"]

# Plot
prices.plot(figsize=(10,4))

plt.title("Thai SET Index Adjusted Closing Prices")

plt.ylabel("Index Level")

plt.xlabel("Date")

plt.savefig("figs/ch4_/set.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![SET](figs/ch4_/set.png)

```{code-cell} python
# Save CSV file
set_index.to_csv(
    "figs/ch4_/set.csv")
```

````{admonition} Get the Data
To download the data run:

``` python
from IPython.display import FileLink
FileLink("figs/ch4_/set.csv")
```
````

---

## Questions

1. Does the series appear stationary?

2. Can you identify periods of:
   - rapid growth,
   - sharp decline,
   - unusual volatility?

3. Why might adjusted prices be preferable to raw prices?

---

```{admonition} Hint
Recall that adjusted prices account for:
- stock splits,
- dividends,
- and other corporate actions.
```

---

# Exercise 2 — Computing Returns

We now compute simple returns.

```{code-cell} python
returns = prices.pct_change().dropna()

returns.head()
```

---

## Plotting Returns

```{code-cell} python
returns.plot(figsize=(10,4))

plt.title("SET Daily Returns")

plt.ylabel("Return")

plt.savefig("figs/ch4_/returns.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![SET](figs/ch4_/returns.png)

---

## Questions

1. How does the return series differ visually from the price series?

2. Does the return series appear more stationary?

3. Can you identify periods of elevated volatility?

---

```{admonition} Observation
Financial returns often fluctuate around relatively stable means even when prices trend strongly upward.
```

---

# Exercise 3 — Simple vs Log Returns

We now compute log returns.

```{code-cell} python
import numpy as np

log_returns = np.log(
    prices / prices.shift(1)
).dropna()

log_returns.head()
```

---

## Comparing Returns

```{code-cell} python
comparison = plt.figure(figsize=(10,4))

plt.plot(
    returns.index,
    returns,
    label="Simple Returns",
    alpha=0.7
)

plt.plot(
    log_returns.index,
    log_returns,
    label="Log Returns",
    alpha=0.7
)

plt.legend()

plt.title("Simple vs Log Returns")

plt.savefig("figs/ch4_/returns2.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Simple vs Log Returns](figs/ch4_/returns2.png)

---

## Questions

1. Are simple and log returns very different for daily data?

2. Why do economists and finance researchers often prefer log returns?

3. Under what circumstances might differences become larger?

---

# Exercise 4 — Measuring Volatility

Volatility measures variability in returns.

One simple measure is the standard deviation.

---

## Daily Volatility

```{code-cell} python
returns.std()
```

---

## Rolling Volatility

```{code-cell} python
rolling_vol = returns.rolling(30).std()

rolling_vol.plot(figsize=(10,4))

plt.title("30-Day Rolling Volatility")

plt.ylabel("Volatility")

plt.savefig("figs/ch4_/rol_vol.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Rolling Volatility](figs/ch4_/rol_vol.png)


---

## Questions

1. Does volatility appear constant over time?

2. Can you identify periods of volatility clustering?

3. Why might volatility matter for investors?

---

```{admonition} Key Idea
Volatility itself often changes through time.
```

This becomes central later when we study:

- ARCH models,
- and GARCH models.

---

# Exercise 5 — Distribution of Returns

We now examine the distribution of returns.

```{code-cell} python
returns.hist(
    bins=50,
    figsize=(8,4)
)

plt.title("Distribution of Daily Returns")

plt.savefig("figs/ch4_/ret_dist.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Returns Distribution](figs/ch4_/ret_dist.png)

---

## Questions

1. Does the distribution appear perfectly normal?

2. Are extreme observations present?

3. Why might extreme returns matter in finance?

---

```{admonition} Observation
Financial returns often display:
- fat tails,
- volatility clustering,
- and occasional extreme observations.
```

---

# Exercise 6 — Comparing Assets

We now compare multiple assets.

```{code-cell} python
aapl = yf.download(
    "AAPL",
    start="2018-01-01",
    auto_adjust=False
)["Adj Close"]

nflx = yf.download(
    "NFLX",
    start="2018-01-01",
    auto_adjust=False
)["Adj Close"]

comparison = plt.figure(figsize=(10,4))

plt.plot(
    aapl / aapl.iloc[0],
    label="Apple"
)

plt.plot(
    nflx / nflx.iloc[0],
    label="NetFlix"
)

plt.legend()

plt.title("Normalized Stock Prices")

plt.show()
```

---

## Questions

1. Why do we normalize prices before comparison?

2. Which asset performed better over the sample?

3. Which asset appears more volatile?

---

# Exercise 7 — Short Selling and Negative Returns

Suppose an investor expects prices to fall.

One possible strategy is:

```{admonition} Definition
Short selling.
```

A short seller:

- borrows an asset,
- sells it today,
- and hopes to repurchase it later at a lower price.

---

## Example

Suppose:

| Day | Price |
|---|---|
| 1 | 100 |
| 2 | 90 |

---

### Long Position Return

```{math}
:enumerated: false
\frac{90-100}{100}
=
-0.10
```

The investor loses 10%.

---

### Short Position Return

```{math}
:enumerated: false
\frac{100-90}{100}
=
0.10
```

The short seller gains 10%.

---

## Questions

1. Why are short positions risky?

2. Why can short-selling losses become very large?

3. Why might short-selling be economically useful?

---

# Exercise 8 — Interpreting Stylized Facts

Using the exercises above, identify examples of:

- volatility clustering,
- fat tails,
- trends,
- noise,
- nonstationarity.

---

```{admonition} Important
Stylized facts are recurring empirical patterns observed across many financial markets.
```

These patterns strongly influence modern time series modeling.

---

# Mini Project — Exploring Thai Financial Data

Choose one Thai financial asset, such as:

- SET Index,
- SET50,
- Thai baht exchange rate,
- a major Thai stock.

Then:

1. Download the data.
2. Compute returns.
3. Plot prices and returns.
4. Measure volatility.
5. Compare simple and log returns.
6. Discuss stylized facts.

---

```{admonition} Suggested Extension
Compare:
- calm periods,
- and crisis periods.

Examples include:
- COVID-19,
- global financial crises,
- political shocks.
```

---

# Gretl Version

The same exercises can also be performed in Gretl and/or Excel.

---

## Downloading Data

Import CSV or Excel financial data.

````{admonition} Get the Data
To download the data from Python run:

``` python
from IPython.display import FileLink
FileLink("figs/ch4_/set.csv")
```
````

---

## Plotting Data

Menu:

```text
Variable → Time series plot
```

---

## Computing Returns

Menu:

```text
Add → Define new variable
```

Example:

```text
return = (P - P(-1))/P(-1)
```

Or.

Right click on `AdjClose` and `Add percent change...`

---

## Histogram

Menu:

```text
Variable → Frequency distribution
```

![Return distribution](figs/ch4_/SET_rnt.png)

---

# Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Using raw instead of adjusted prices**  
Corporate actions may distort returns.

**2. Forgetting to remove missing observations**  
Return calculations generate missing first observations.

**3. Confusing prices and returns**  
Returns usually have different statistical properties.

**4. Ignoring volatility changes**  
Financial risk is rarely constant through time.

**5. Assuming returns are perfectly normal**  
Financial returns often contain fat tails and extreme events.
```

---

# Looking Ahead

Part II begins studying:

- trends,
- smoothing,
- filtering,
- and trading indicators.

We move from:

```{admonition} Observation
Describing financial data.
```

toward:

```{admonition} Observation
Extracting patterns and signals from noisy time series.
```

---

# Key Takeaways

```{admonition} Summary
- Financial prices are transformed into returns for analysis.
- Returns often behave differently from prices.
- Volatility is a central concept in finance.
- Financial returns frequently display stylized facts such as fat tails and volatility clustering.
- Visualization is essential for understanding time series data.
- Adjusted prices are important for meaningful financial analysis.
- Statistical concepts become more meaningful when connected to real financial data.
```