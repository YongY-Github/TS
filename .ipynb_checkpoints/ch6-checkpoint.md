# Chapter 6 — Trading Indicators as Filters

## 6.1 Introduction

In the previous chapter, we introduced **smoothing methods** as tools for revealing the underlying movement of a time series.

In this chapter, we explore a different—but closely related—set of tools:

```{admonition} Key Idea
Many popular trading indicators are simply **filters applied to time series data**.
````

These indicators are widely used in financial markets to:

* detect trends
* identify momentum
* signal potential turning points

Rather than thinking of them as “black-box trading rules,” we will interpret them using the tools we already know:

👉 **moving averages, smoothing, and transformations**

---

## 6.2 Prices vs Signals

A raw price series contains:

* long-run movement
* short-run noise
* temporary fluctuations

Trading indicators attempt to extract **signals** from this noisy data.

```markdown
[Figure Placeholder: Price series with overlaid indicator]
```

```{admonition} Insight
Trading indicators are designed to separate **signal from noise**, just like smoothing methods.
```

---

## 6.3 Moving Average (MA) Indicators

The simplest trading indicator is the **moving average**.

### Simple Moving Average (SMA)

The $k$-period moving average is:

$$
SMA_t = \frac{1}{k} \sum_{j=0}^{k-1} P_{t-j}
$$

This is exactly the smoothing method we studied earlier.

---

### Short vs Long Moving Averages

Traders often use:

* a **short window** (e.g. 10 days) → responsive
* a **long window** (e.g. 50 days) → smooth

```markdown
[Figure Placeholder: Short vs long moving averages]
```

---

### Moving Average Crossover

A common strategy is:

* **Buy signal**: short MA crosses above long MA
* **Sell signal**: short MA crosses below long MA

```markdown
[Figure Placeholder: MA crossover example]
```

```{admonition} Interpretation
A crossover suggests a **change in trend direction**.
```

---

### Excel Implementation

```markdown
[Excel Screenshot Placeholder: Computing moving averages step-by-step]
```

Steps:

1. Choose window size (e.g. 10, 50)
2. Use AVERAGE() over rolling window
3. Plot together with price

---

````{dropdown} Python (optional)
```python
df["ma10"] = df["price"].rolling(window=10).mean()
df["ma50"] = df["price"].rolling(window=50).mean()
````

````

---

## 6.4 MACD (Moving Average Convergence Divergence)

The **MACD** is a more refined indicator based on exponential smoothing.

It is defined as:

$$
MACD_t = EMA_{short} - EMA_{long}
$$

where EMA is an **exponential moving average**.

---

### Interpretation

- Positive MACD → upward momentum  
- Negative MACD → downward momentum  

```markdown
[Figure Placeholder: MACD with signal line]
````

---

### Signal Line

A further smoothing is applied:

$$
Signal_t = EMA(MACD_t)
$$

Trading signals:

* MACD crosses above signal → buy
* MACD crosses below signal → sell

---

```{admonition} Key Insight
MACD is essentially a **difference between two smoothers**.
```

---

### Excel Implementation

```markdown
[Excel Screenshot Placeholder: MACD calculation]
```

---

````{dropdown} Python (optional)
```python
ema12 = df["price"].ewm(span=12).mean()
ema26 = df["price"].ewm(span=26).mean()

df["macd"] = ema12 - ema26
df["signal"] = df["macd"].ewm(span=9).mean()
````

````

---

## 6.5 RSI (Relative Strength Index)

The **RSI** measures the strength of recent gains relative to losses.

It is defined as:

$$
RSI = 100 - \frac{100}{1 + RS}
$$

where:

$$
RS = \frac{\text{average gain}}{\text{average loss}}
$$

---

### Interpretation

- RSI > 70 → overbought  
- RSI < 30 → oversold  

```markdown
[Figure Placeholder: RSI indicator]
````

---

```{admonition} Intuition
RSI compares **upward vs downward movements** over a recent window.
```

---

### Excel Implementation

```markdown
[Excel Screenshot Placeholder: RSI calculation steps]
```

---

````{dropdown} Python (optional)
```python
delta = df["price"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df["rsi"] = 100 - (100 / (1 + rs))
````

````

---

## 6.6 Bollinger Bands

Bollinger Bands combine:

- a moving average  
- a measure of volatility  

They are defined as:

$$
Upper_t = MA_t + k \cdot \sigma_t
$$

$$
Lower_t = MA_t - k \cdot \sigma_t
$$

---

### Interpretation

- Price near upper band → relatively high  
- Price near lower band → relatively low  

```markdown
[Figure Placeholder: Bollinger Bands]
````

---

```{admonition} Insight
Bollinger Bands adapt to **changing volatility**.
```

---

### Excel Implementation

```markdown
[Excel Screenshot Placeholder: Bollinger Bands steps]
```

---

````{dropdown} Python (optional)
```python
ma20 = df["price"].rolling(20).mean()
std20 = df["price"].rolling(20).std()

df["upper"] = ma20 + 2 * std20
df["lower"] = ma20 - 2 * std20
````

````

---

## 6.7 Indicators as Filters

We can now reinterpret all these indicators.

```{admonition} Big Insight
Trading indicators are **filters applied to time series data**.
````

| Indicator      | Interpretation                   |
| -------------- | -------------------------------- |
| Moving Average | Low-pass filter (removes noise)  |
| MACD           | Difference of smoothers          |
| RSI            | Transformation of recent changes |
| Bollinger      | Mean + volatility bands          |

---

## 6.8 Limitations of Trading Indicators

```{admonition} Important
:class: warning

Trading indicators do not guarantee profits.
```

Common issues:

* lagging signals
* false positives
* sensitivity to parameter choice

---

```{admonition} Deeper Insight
Many indicators are based on **past data only**, so they cannot perfectly predict future movements.
```

---

## 6.9 Connection to Time Series Models

These indicators may seem very different from econometric models, but they are closely related.

```{admonition} Key Connection
- Moving averages → smoothing filters  
- ARMA models → stochastic filters  
```

Later, we will formalize these ideas using:

* autocorrelation
* AR models
* MA models

---

## 6.10 Looking Ahead

In the next chapters, we move from:

* descriptive tools
* heuristic indicators

to:

👉 **formal time series models**

---

## Key Takeaways

* Trading indicators are based on smoothing and transformations
* Moving averages are the foundation of many indicators
* MACD, RSI, and Bollinger Bands extract different types of signals
* These tools help separate signal from noise
* They are closely related to formal time series models

---

```

---

# 👍 Why this chapter works

- builds directly on smoothing  
- highly engaging (students recognize indicators)  
- intuitive but conceptually deep  
- sets up:
  - ACF/PACF  
  - ARMA  
  - filtering interpretation  

---

# 🚀 Suggested next move

Now you have a *very strong arc*:

- Ch 4 → visualization  
- Ch 5 → smoothing  
- Ch 6 → trading indicators  

👉 Next best chapter:

# **Chapter 7 — White Noise and Random Walks**

This is where the theory *starts*.

---

If you want, I can write:

👉 Chapter 7 (CORE-style, with strong intuition and minimal math first)

This is a critical bridge chapter.
```
