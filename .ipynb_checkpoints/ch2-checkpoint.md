# Chapter 2 — Returns and Financial Data

## 2.1 Introduction

In the previous chapter, we introduced time series and saw examples from economics and finance.  

In financial applications, however, we rarely analyze **prices directly**. Instead, we focus on **returns**.

Why?

Because returns:

- measure gains and losses  
- allow comparisons across assets  
- are easier to model statistically  

```{admonition} Key Idea
In finance, we model **returns**, not prices.
````

---

## 2.2 A Simple Example

Let us begin with a very simple example.

```markdown
[Figure Placeholder: Price series 100 → 80 → 100 (simple line chart)]
```

Suppose the price of an asset evolves as follows:

| Year | Price |
| ---- | ----- |
| 2020 | 100   |
| 2021 | 80    |
| 2022 | 100   |

We observe:

* a drop from 100 to 80 (−20%)
* a rise from 80 to 100 (+25%)

At first glance, it may seem that:

> −20% + 25% = +5%

But this is misleading.

```{admonition} Important
A −20% return followed by +25% does **not** imply a net gain of 5%.  
In fact, the price returns exactly to its starting point.
```

This simple example motivates the need to think carefully about how returns are defined and aggregated.

---

## 2.3 Simple (Arithmetic) Returns

For a price $P_t$, the **simple return** is defined as:

$$
R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1.
$$

This is the standard percentage change.

---

### Example

```markdown
[Excel Screenshot Placeholder: Calculating simple returns step-by-step]
```

Steps in Excel:

1. Enter prices in a column
2. Compute:

   ```
   = (P_t / P_{t-1}) - 1
   ```
3. Format as percentage

---

````{dropdown} Python (optional)
```python
import numpy as np
import pandas as pd

price = [100, 80, 100]
year = [2020, 2021, 2022]

df = pd.DataFrame({"price": price}, index=year)
df["rtn"] = df["price"].pct_change()
df
````

````

We obtain:

- $R_{2021} = -0.20$  
- $R_{2022} = +0.25$  

---

## 2.4 Why Simple Returns Don’t “Add Up”

Suppose we try to combine returns across time.

For two periods:

$$
(1+R_1)(1+R_2) - 1 \neq R_1 + R_2
$$

In general, simple returns are **not additive over time**.

---

### Example

```markdown
[Excel Screenshot Placeholder: Compounding returns example]
````

Compute:

$$
(1 - 0.20)(1 + 0.25) - 1 = 0
$$

So the true total return is **0%**, not +5%.

```{admonition} Key Insight
Returns combine **multiplicatively**, not additively.
```

---

## 2.5 Log Returns

To solve this issue, we define **log returns**:

$$
r_t = \log(1 + R_t) = \log\left(\frac{P_t}{P_{t-1}}\right)
$$

---

### Example

```markdown
[Excel Screenshot Placeholder: Computing log returns using LN()]
```

Steps in Excel:

```
= LN(P_t / P_{t-1})
```

---

````{dropdown} Python (optional)
```python
df["log_rtn"] = np.log(df["price"] / df["price"].shift(1))
df[["rtn", "log_rtn"]]
````

````

---

```{admonition} Why Logs?
Logs convert multiplication into addition.

This makes returns **time-additive**, which is extremely useful.
````

---

## 2.6 Time Additivity (The Key Advantage)

Over multiple periods:

$$
\prod_{t=1}^T (1 + R_t) = \frac{P_T}{P_0}
$$

Taking logs:

$$
\sum_{t=1}^T r_t = \log\left(\frac{P_T}{P_0}\right)
$$

```{admonition} Key Idea
Log returns **add over time**.
```

This is one of the most important properties in financial modeling.

---

## 2.7 A Longer Example

Let us extend the data:

| Year | Price |
| ---- | ----- |
| 2020 | 100   |
| 2021 | 80    |
| 2022 | 100   |
| 2023 | 115   |
| 2024 | 125   |

---

````{dropdown} Python (optional)
```python
price = [100, 80, 100, 115, 125]
year = [2020, 2021, 2022, 2023, 2024]

df = pd.DataFrame({"price": price}, index=year)

df["rtn"] = df["price"].pct_change()
df["log_rtn"] = np.log(df["price"] / df["price"].shift(1))
df
````

````

---

## 2.8 Cumulative Returns

### Simple Returns (Geometric Compounding)

```markdown
[Excel Screenshot Placeholder: Cumulative return using product formula]
````

Formula:

$$
\prod_{t=1}^T (1 + R_t)
$$

---

````{dropdown} Python (optional)
```python
cum_gross = (df["rtn"] + 1).cumprod()
cum_gross
````

````

---

### Log Returns

```markdown
[Excel Screenshot Placeholder: Cumulative log return (sum)]
````

Formula:

$$
\sum_{t=1}^T r_t
$$

---

````{dropdown} Python (optional)
```python
cum_log = np.log(df["rtn"] + 1).cumsum()
cum_log
````

````

---

### Consistency Check

```{dropdown} Python (optional)
```python
np.log(df["price"].iloc[-1]) - np.log(df["price"].iloc[0])
np.exp(0.22314)
````

````

```{admonition} Insight
Exponentiating cumulative log returns gives the total compounded return.
````

---

## 2.9 Small Return Approximation

For small returns:

$$
\log(1+R) \approx R
$$

---

````{dropdown} Python (optional)
```python
for r in [0.001, 0.01, 0.05, 0.1]:
    print(r, np.log(1+r))
````

````

```{admonition} Rule of Thumb
For returns below about 5–10%, simple and log returns are very similar.
````

---

## 2.10 Practical Considerations

```{admonition} Practical Notes
- **Modeling:** log returns are often preferred  
- **Reporting:** simple returns are easier to interpret  
- **Compounding:** use geometric returns or log sums  
- **Volatility:** arithmetic averages can be misleading  
```

---

## 2.11 Why This Matters for Time Series

Returns behave very differently from prices:

```markdown
[Figure Placeholder: Price vs return comparison]
```

* prices → trending, persistent
* returns → noisy, fluctuating

```{admonition} Key Insight
Prices often look non-stationary, while returns are closer to stationary.
```

This distinction will be central in later chapters.

---

## 2.12 Looking Ahead

In the next chapter, we begin exploring:

* how to visualize time series
* how to separate signal from noise
* how to identify patterns

---

## Key Takeaways

* Returns measure percentage change in prices
* Simple returns do not add over time
* Log returns are time-additive
* Compounding is multiplicative
* Returns are more suitable than prices for modeling

---

```

---

# 👍 What we achieved

- Preserved your **core logic and examples**
- Made it **more readable + structured**
- Added:
  - Excel pedagogy
  - optional Python
  - CORE-style flow
- Strengthened transitions to:
  - smoothing
  - stationarity
  - ARIMA

---

# 🚀 Next step (important)

Now your book is flowing very nicely:

1. Chapter 1 ✅  
2. Chapter 2 ✅  
3. Next: **Chapter 3 — Probability & Statistics (light, intuitive)**  

OR (my recommendation):

👉 Skip ahead to **Chapter 4: Visualization & Patterns**  
(to keep momentum and avoid math fatigue early)

---

If you want, I can:

✅ Write Chapter 3 (light, intuitive, non-mathy)  
✅ OR jump to Chapter 4 (very engaging, visual)

Just tell me 👍
```
