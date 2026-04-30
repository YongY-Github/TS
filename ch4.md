---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 4 — Visualizing Time Series

Before estimating models or producing forecasts, we should first look carefully at the data.

Visualization is one of the most important steps in time series analysis.

A graph often reveals features that summary statistics alone may hide.

```{admonition} Central Question
What patterns can we detect visually in time series data?
```

This chapter introduces:

- plotting time series,
- trends,
- cycles,
- seasonality,
- noise,
- rolling averages,
- and visual interpretation.

The emphasis is practical and intuition-first.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- plot and interpret time series data
- identify trends and cycles visually
- distinguish signal from noise
- recognize seasonality
- understand rolling averages
- compare multiple time series visually
- use Python and Gretl to visualize data

---

# 4.1 Why Visualization Matters

Time series data often contain rich structure.

Plots may reveal:

- trends,
- volatility,
- structural breaks,
- outliers,
- cycles,
- persistence,
- seasonality.

```{admonition} Key Idea
Visualization is often the first step in understanding a time series.
```

A model estimated without visual inspection may miss important features of the data.

---

# 4.2 A First Example

We begin with a simple example using stock market data.

```{code-cell} python
import yfinance as yf
import matplotlib.pyplot as plt

sp500 = yf.download("^GSPC", start="2018-01-01", auto_adjust=False)

sp500["Adj Close"].plot(figsize=(10,4))

plt.title("S&P 500 Adjusted Closing Price")
plt.xlabel("Date")
plt.ylabel("Index Level")

plt.savefig("figs/ch4/sp500.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![sp500](figs/ch4/sp500.png)

```{admonition} Observation
The stock market index does not fluctuate randomly around a constant level.

Instead, we observe long-run movements and periods of turbulence.
```

---

# 4.3 Components of a Time Series

Many time series contain several components.

A useful decomposition is:

```{math}
:enumerated: false
\text{Series}
=
\text{Trend}
+
\text{Cycle}
+
\text{Seasonality}
+
\text{Noise}
```

## Trend

A trend represents long-run movement.

Examples include:

- long-run GDP growth,
- rising stock market indices,
- long-run inflation trends.

## Cycles

Cycles are medium-run fluctuations around the trend.

Examples include:

- business cycles,
- housing cycles,
- commodity cycles.

## Seasonality

Seasonality refers to regular repeating patterns.

Examples include:

- tourism seasons,
- holiday shopping,
- electricity demand,
- agricultural harvest cycles.

## Noise

Noise represents unpredictable random fluctuations.

```{admonition} Important
One of the central goals of time series analysis is separating meaningful structure from noise.
```

---

# 4.4 Trends in Economic Data

Many macroeconomic variables display strong trends.

Examples include:

- GDP,
- population,
- prices,
- productivity.

## Example: GDP

```{code-cell} python
# !pip install pandas_datareader

import pandas_datareader.data as web
import matplotlib.pyplot as plt

thai_gdp = web.DataReader(
    "MKTGDPTHA646NWDB",
    "fred",
    start="2000-01-01"
)

thai_gdp.plot(figsize=(10,4))

plt.title("Thailand GDP")
plt.ylabel("Current U.S. Dollars")

plt.savefig("figs/ch4/thai_gdp.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Thai GDP](figs/ch4/thai_gdp.png)

```{admonition} Observation
GDP displays a strong upward trend over long periods.
```

This has important implications for modeling and forecasting later in the book.

---

# 4.5 Trends vs Stationarity

Some time series fluctuate around relatively stable levels.

Others drift persistently upward or downward.

```{admonition} Looking Ahead
Later chapters will study stationarity formally.

For now, the key point is simple:

some time series have stable statistical properties, while others evolve systematically over time.
```

---

# 4.6 Seasonality

Seasonality refers to repeating patterns linked to the calendar.

Examples include:

- higher retail sales during holidays,
- tourism peaks,
- electricity demand during summer,
- agricultural cycles.

## Example: Monthly Airline Passengers

```{code-cell} python
# import matplotlib.pyplot as plt
import pandas as pd

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

df = pd.read_csv(url)

df["Passengers"].plot(figsize=(10,4))

plt.title("Monthly Airline Passengers")

plt.savefig("figs/ch4/airline.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Airline](figs/ch4/airline.png)

````{admonition} Package Note
The `pandas_datareader` package may require installation.

If needed, run:

```python
%pip install pandas_datareader
```
````

```{admonition} Observation
The series displays both trend and seasonality.
```

The fluctuations repeat systematically through time.

---

# 4.7 Cycles

Economic cycles differ from seasonality.

Seasonality repeats regularly.

Business cycles are more irregular.

Examples include:

- recessions,
- expansions,
- commodity booms,
- housing booms.

```{admonition} Definition
A cycle is a medium-run fluctuation around a longer-run trend.
```

---

# 4.8 Noise and Random Fluctuations

Not all movements are meaningful.

Time series often contain substantial randomness.

```{admonition} Important
Random fluctuations can sometimes appear patterned even when no real structure exists.
```

This is especially important in finance.

Short-run market movements may contain substantial noise.

---

# 4.9 Visualizing Financial Returns

Prices and returns often behave very differently.

```{code-cell} python
# import yfinance as yf
# import matplotlib.pyplot as plt
# import pandas as pd

aapl = yf.download("AAPL", start="2020-01-01", auto_adjust=False)

prices = aapl["Adj Close"]

returns = prices.pct_change()

fig, ax = plt.subplots(2,1, figsize=(10,7))

ax[0].plot(prices)
ax[0].set_title("Apple Adjusted Prices")

ax[1].plot(returns)
ax[1].set_title("Apple Daily Returns")

plt.tight_layout()

plt.savefig("figs/ch4/aapl.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![APPLE](figs/ch4/aapl.png)

```{admonition} Observation
Prices often display trends, while returns fluctuate around relatively stable levels.
```

---

# 4.10 Volatility Clustering

Financial returns often display changing volatility.

Periods of calm are followed by periods of turbulence.

```{admonition} Key Idea
Volatility itself often changes through time.
```

This phenomenon becomes central later in ARCH and GARCH models.

---

# 4.11 Structural Breaks

Some time series experience sudden changes.

These are called structural breaks.

Examples include:

- financial crises,
- policy regime changes,
- pandemics,
- wars.

## Example

The COVID-19 pandemic caused dramatic movements in:

- stock prices,
- unemployment,
- GDP,
- exchange rates.

```{admonition} Important
A model that works well in one period may fail after a structural break.
```

---

# 4.12 Rolling Averages

A rolling average smooths short-run fluctuations.

This helps reveal longer-run patterns.

## Simple Rolling Average

A rolling average over $k$ periods is:

```{math}
:enumerated: false
MA_t
=
\frac{1}{k}
\sum_{i=0}^{k-1}x_{t-i}
```

## Example

```{code-cell} python
# import yfinance as yf
# import matplotlib.pyplot as plt

aapl = yf.download("AAPL", start="2020-01-01", auto_adjust=False)

prices = aapl["Adj Close"]

ma50 = prices.rolling(50).mean()

plt.figure(figsize=(10,4))

plt.plot(prices, label="Price")

plt.plot(ma50, label="50-Day MA")

plt.legend()

plt.title("Apple Stock Price and Moving Average")

plt.savefig("figs/ch4/rolling.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Rolling MA](figs/ch4/rolling.png)

```{admonition} Observation
Rolling averages smooth short-run fluctuations and reveal broader trends.
```

---

# 4.13 Comparing Multiple Series

Visualization also helps compare multiple variables.

```{code-cell} python
# import yfinance as yf
# import matplotlib.pyplot as plt

aapl = yf.download("AAPL", start="2020-01-01", auto_adjust=False)["Adj Close"]

msft = yf.download("MSFT", start="2020-01-01", auto_adjust=False)["Adj Close"]

plt.figure(figsize=(10,4))

plt.plot(aapl / aapl.iloc[0], label="Apple")

plt.plot(msft / msft.iloc[0], label="Microsoft")

plt.legend()

plt.title("Normalized Stock Prices")

plt.savefig("figs/ch4/twostocks.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Two Stocks](figs/ch4/twostocks.png)

```{admonition} Practical Advice
Normalizing series allows easier visual comparison across assets with different price levels.
```

---

# 4.14 Log Scales

Some time series grow exponentially over long periods.

In such cases, log scales can improve interpretation.

## Example

```{code-cell} python
# import yfinance as yf
# import matplotlib.pyplot as plt

sp500 = yf.download("^GSPC", start="1990-01-01", auto_adjust=False)

plt.figure(figsize=(10,4))

plt.plot(sp500["Adj Close"])

plt.yscale("log")

plt.title("S&P 500 (Log Scale)")

plt.savefig("figs/ch4/logscale.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Log Scale](figs/ch4/logscale.png)

```{admonition} Intuition
Log scales emphasize proportional rather than absolute changes.
```

---

# 4.15 Visualization and Forecasting

Good forecasting begins with understanding the data visually.

Plots may reveal:

- trends,
- persistence,
- seasonality,
- changing volatility,
- structural breaks.

These features influence model selection later.

```{admonition} Key Idea
Visualization guides model building.
```

---

# 4.16 Gretl Example: Plotting a Time Series

Gretl provides simple visualization tools.

---

## Step 1 — Open Data

Menu:

```text
File → Open data
```

---

## Step 2 — Plot a Variable

Select a variable and choose:

```text
Variable → Time series plot
```

---

```markdown
[GRETL Screenshot Placeholder: Time series plot]
```

---

## Step 3 — Add Moving Average

Menu:

```text
Add → Moving average
```

---

```markdown
[GRETL Screenshot Placeholder: Moving average plot]
```

---

# 4.17 Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Ignoring trends**  
Trending data may require transformation before modeling.

**2. Confusing cycles with noise**  
Not every fluctuation is meaningful.

**3. Overinterpreting visual patterns**  
Some apparent patterns occur by chance.

**4. Comparing variables with different scales improperly**  
Normalization may be necessary.

**5. Ignoring structural breaks**  
Relationships may change abruptly over time.
```

---

# 4.18 Looking Ahead

This chapter introduced visual exploration of time series data.

The next chapter studies smoothing and trend estimation more formally.

We will examine:

- moving averages,
- exponential smoothing,
- Holt models,
- HP filters,
- and trend extraction.

# Key Takeaways

```{admonition} Summary
- Visualization is a critical first step in time series analysis.
- Time series often contain trends, cycles, seasonality, and noise.
- Financial returns behave differently from prices.
- Rolling averages help smooth noisy fluctuations.
- Structural breaks can dramatically alter time series behavior.
- Visualization helps guide model selection and forecasting.
