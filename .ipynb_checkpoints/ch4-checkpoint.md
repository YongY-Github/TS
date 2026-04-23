# Chapter 4 — Visualizing Time Series

## 4.1 Introduction

In the previous chapter, we learned how to compute **returns** and saw that financial data behave very differently from prices.

Before building models, we must first **look at the data**.

Time series analysis always begins with a simple but powerful step:

```{admonition} Key Idea
Plot the data.
````

Visualization helps us:

* detect trends
* identify patterns
* spot unusual behavior
* distinguish signal from noise

---

## 4.2 A First Time Series Plot

Let us begin with a simple example.

```markdown id="v1plh8"
[Figure Placeholder: Time series plot (e.g. GDP, stock index, or exchange rate)]
```

Even without formal tools, we can already ask:

* Is there an upward or downward trend?
* Are fluctuations large or small?
* Are there sudden jumps or breaks?

```{admonition} Insight
Many important features of time series can be seen directly from a plot.
```

---

## 4.3 What Should We Look For?

When examining a time series, there are several key features to look for.

---

### 1. Trend

A **trend** is a long-run movement in the series.

```markdown
[Figure Placeholder: Series with upward trend]
```

Examples:

* GDP growth
* stock market indices

---

### 2. Cycles and Fluctuations

Time series often exhibit **short-run fluctuations** around a trend.

```markdown
[Figure Placeholder: Business cycle example]
```

These may reflect:

* economic cycles
* policy changes
* external shocks

---

### 3. Volatility

Volatility refers to how much the series fluctuates.

```markdown
[Figure Placeholder: High vs low volatility comparison]
```

In financial data:

* calm periods → small changes
* turbulent periods → large swings

```{admonition} Key Insight
Volatility is often **time-varying**, especially in financial markets.
```

---

### 4. Structural Breaks

A **structural break** is a sudden change in the behavior of the series.

```markdown
[Figure Placeholder: Structural break example]
```

Examples:

* financial crises
* policy regime changes
* pandemics

---

### 5. Seasonality

Some series show regular patterns over time.

```markdown
[Figure Placeholder: Seasonal pattern (monthly or quarterly)]
```

Examples:

* tourism
* retail sales
* electricity demand

---

## 4.4 Prices vs Returns

A crucial distinction introduced earlier is between **prices** and **returns**.

---

### Price Series

```markdown
[Figure Placeholder: Stock price series]
```

Prices often show:

* trends
* persistence
* smooth movements

---

### Return Series

```markdown
[Figure Placeholder: Return series]
```

Returns typically show:

* no clear trend
* rapid fluctuations
* volatility clustering

```{admonition} Key Insight
Returns are much more suitable for statistical modeling than prices.
```

---

## 4.5 Rolling Statistics

To better understand how a series evolves, we often compute **rolling (moving) statistics**.

---

### Rolling Mean

The rolling mean shows how the average changes over time.

```markdown
[Excel Screenshot Placeholder: Rolling average calculation]
```

Steps in Excel:

1. Choose a window size (e.g. 5 or 10 periods)
2. Compute average over that window
3. Slide the window forward

---

### Rolling Variance

Similarly, we can track how variability changes.

```markdown
[Excel Screenshot Placeholder: Rolling variance calculation]
```

```{admonition} Insight
Rolling statistics help reveal **time variation in mean and volatility**.
```

---

````{dropdown} Python (optional) id="c0h9k2"
```python
import pandas as pd

df["rolling_mean"] = df["price"].rolling(window=5).mean()
df["rolling_std"] = df["price"].rolling(window=5).std()
df
````

````id="f0z5bx"

---

## 4.6 Smoothing the Series

Sometimes the raw data are too noisy to interpret clearly.

```markdown
[Figure Placeholder: Noisy vs smoothed series]
````

Smoothing helps:

* reduce noise
* reveal underlying structure

Common methods include:

* moving averages
* exponential smoothing
* LOESS

```{admonition} Key Idea
Smoothing helps us focus on the **signal**, not the noise.
```

We will study these methods in detail in the next chapter.

---

## 4.7 Comparing Multiple Series

Visualization is especially powerful when comparing multiple time series.

```markdown
[Figure Placeholder: Multiple series comparison]
```

Examples:

* GDP across countries
* stock indices across markets
* inflation across regions

```{admonition} Practical Tip
Always put series on comparable scales (e.g. index = 100) when comparing.
```

---

## 4.8 Log Scale Plots

Sometimes it is useful to plot data on a **log scale**.

```markdown
[Figure Placeholder: Linear vs log scale comparison]
```

Why?

* growth rates become easier to interpret
* exponential growth becomes linear

```{admonition} Insight
Log scales are especially useful for long-run economic data.
```

---

## 4.9 Common Mistakes in Visualization

```{admonition} Common Pitfalls
:class: warning

**1. Not plotting the data first**  
Always start with visualization.

**2. Ignoring scale**  
Different scales can distort interpretation.

**3. Over-interpreting noise**  
Not every fluctuation is meaningful.

**4. Mixing price and return interpretations**  
These behave very differently.

**5. Using too many transformations too early**  
Start simple.
```

---

## 4.10 Why Visualization Comes First

Before any modeling:

* we must understand the data
* we must identify patterns
* we must form hypotheses

```{admonition} Key Principle
Good modeling begins with good visualization.
```

---

## 4.11 Looking Ahead

In the next chapter, we build on these ideas by introducing:

* smoothing techniques
* trend estimation
* separating signal from noise

---

## Key Takeaways

* Visualization is the first step in time series analysis
* Important features include trend, volatility, seasonality, and breaks
* Returns behave differently from prices
* Rolling statistics help track changes over time
* Smoothing helps reveal underlying patterns

---

```

---

# 👍 Why this chapter works

- Very **intuitive and visual**
- No math overload
- Prepares students perfectly for:
  - smoothing (next)
  - stationarity (later)
- Strong connection to:
  - trading
  - real data

---

# 🚀 Next step (very important moment)

You now have:

- Chapter 1 ✅  
- Chapter 2 ✅  
- Chapter 4 ✅  
- Chapter 5 (smoothing) already drafted  

👉 The next logical step is:

# **Chapter 6 — Trading Indicators (MACD, RSI, Bollinger Bands)**

This will:
- hook students strongly  
- connect intuition → practice  
- prepare for ACF/PACF later  

---

If you want, I can write:

👉 **Chapter 6 — Trading Indicators (CORE-style, Excel-first, very engaging)**  

This is where your book becomes *really unique*.

Just say 👍
```
