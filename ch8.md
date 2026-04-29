---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 8 — Stationarity

In the previous chapter, we introduced stochastic processes, white noise, persistence, and random walks.

We now turn to one of the most important concepts in time series analysis:

```{admonition} Central Question
Does the probabilistic behavior of a time series remain stable over time?
```

This idea is captured by the concept of **stationarity**.

Stationarity plays a central role in:

- forecasting
- statistical inference
- ARIMA modeling
- autocorrelation analysis
- regression with time series data

Many classical time series methods assume stationarity in one form or another.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- explain what stationarity means
- distinguish strict and weak stationarity
- understand why stationarity matters
- identify nonstationary behavior visually
- distinguish white noise from random walks
- understand the role of mean, variance, and autocovariance

---

## 8.1 Why Does Stationarity Matter?

Suppose we use historical data to forecast the future.

This only makes sense if the underlying probabilistic structure remains reasonably stable over time.

```{admonition} Intuition
If the statistical behavior of a series changes continuously over time, then past data may provide little guidance about the future.
```

### Examples

A stationary series might exhibit:

- stable fluctuations around a constant mean
- roughly constant variability
- dependence patterns that remain stable

A nonstationary series might exhibit:

- trending behavior
- changing variance
- structural shifts
- persistent drift

```{admonition} Important
Many statistical methods rely on the assumption that the underlying process is stable over time.
```

---

## 8.2 Strict Stationarity

We begin with the most general definition.

```{admonition} Definition
A time series $\{X_t\}$ is **strictly stationary** if its joint distribution does not change over time.
```

More formally:

$$
F_{t_1+h,\dots,t_n+h}(x_1,\dots,x_n)
=
F_{t_1,\dots,t_n}(x_1,\dots,x_n)
$$

for all:

- time points $t_1,\dots,t_n$
- shifts $h$
- sample sizes $n$

```{admonition} Intuition
Strict stationarity means that the probabilistic behavior of the process is unchanged by shifts in time.
```

### Example

If we look at:

- $(X_1,X_2)$
- $(X_{101},X_{102})$

their joint distributions should be identical under strict stationarity.

---

## 8.3 Weak Stationarity

In practice, strict stationarity is often stronger than necessary.

Most time series methods instead rely on **weak stationarity**.

```{admonition} Definition
A process $\{X_t\}$ is **weakly stationary** if:

1. $E[X_t] = \mu$ is constant

2. $Var(X_t) = \sigma^2$ is constant

3. $Cov(X_t, X_{t-h})$ depends only on the lag $h$, not on time $t$
```

```{admonition} Important
Unless otherwise stated, “stationary” usually means weakly stationary.
```

---

## 8.4 Mean and Variance Stability

A stationary process has stable first and second moments.

### Constant Mean

$$
E[X_t] = \mu
$$

does not depend on time.

### Constant Variance

$$
Var(X_t) = \sigma^2
$$

does not change over time.

### Stable Covariance Structure

The covariance:

$$
Cov(X_t, X_{t-h})
$$

depends only on lag $h$.

```{admonition} Key Insight
In stationary processes, dependence depends on “distance through time,” not on the specific calendar date.
```

---

## 8.5 White Noise Revisited

Recall white noise from Chapter 7.

White noise satisfies:

- constant mean
- constant variance
- zero covariance across time

Thus white noise is stationary.

### Simulating White Noise

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10101)

wn = np.random.normal(0, 2, 500)

plt.figure(figsize=(10,4))
plt.plot(wn, lw=1)
plt.title("Simulated White Noise")
plt.xlabel("Time")
plt.ylabel("$w_t$")

plt.savefig("figs/ch7/wn.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![White Noise](figs/ch7/wn.png)


```{admonition} Observation
The series fluctuates around a stable mean with roughly constant variability.
```

---

## 8.7 Random Walks Revisited

Now consider the random walk:

$$
x_t = x_{t-1} + w_t
$$

where $w_t$ is white noise.

### Simulating a Random Walk

```{code-cell} python
:tags: [hide-input]

np.random.seed(123)

w = np.random.normal(0, 1, 500)
x = np.cumsum(w)

plt.figure(figsize=(10,4))
plt.plot(x, lw=1)
plt.title("Simulated Random Walk")
plt.xlabel("Time")
plt.ylabel("$x_t$")

plt.savefig("figs/ch7/rw.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Random Walk](figs/ch7/rw.png)

```{admonition} Observation
The random walk drifts over time and does not fluctuate around a stable mean.
```

---

## 8.9 Why Random Walks Are Nonstationary

A random walk violates stationarity because:

- shocks accumulate over time
- variance grows continuously
- no stable long-run mean exists

### Variance of a Random Walk

Recall:

```{math}
:enumerated: false
x_t = \sum_{s=1}^t w_s
```

Therefore:

```{math}
:enumerated: false
Var(x_t) = t \sigma_w^2
```

```{admonition} Key Observation
The variance of a random walk increases over time.
```

This violates weak stationarity.

---

## 8.10 Stationary vs Nonstationary Series

| Feature | Stationary | Nonstationary |
|---|---|---|
| Mean stable | Yes | Not necessarily |
| Variance stable | Yes | Often no |
| Dependence stable | Yes | Often no |
| Shocks temporary | Usually | Often persistent |

---

## 8.11 Trend Stationary vs Difference Stationary

Not all nonstationary series behave in the same way.

### Trend-Stationary Processes

Some series fluctuate around a deterministic trend:

$$
x_t = \alpha + \beta t + u_t
$$

where $u_t$ is stationary.

Removing the trend leaves a stationary process.

### Difference-Stationary Processes

Other series become stationary only after differencing.

Random walks are the classic example.

```{admonition} Important
Understanding whether a series is trend-stationary or difference-stationary is crucial in applied work.
```

---

## 8.12 Why Stationarity Matters for Forecasting

Forecasting relies heavily on stable patterns.

For stationary processes:

- dependence structure is stable
- past relationships remain informative

For nonstationary processes:

- statistical properties evolve over time
- prediction becomes more difficult

```{admonition} Key Insight
Many forecasting methods assume that the future behaves statistically like the past.
```

---

## 8.13 Why Stationarity Matters for Regression

Nonstationary variables can create misleading statistical relationships.

Two unrelated trending variables may appear strongly related.

This problem is called:

```{admonition} Preview
**Spurious regression**
```

We study this in detail later in latter chapters.

---

## 8.14 Looking Ahead

In this chapter, we introduced the idea of stationarity and saw how random walks violate it.

In the next chapter, we study:

- autocorrelation
- autocovariance
- partial autocorrelation

which help us understand dependence structures in stationary time series.

## Key Takeaways

```{admonition} Summary
- Stationarity means statistical stability over time
- Weak stationarity requires stable mean, variance, and covariance structure
- White noise is stationary
- Random walks are nonstationary
- Stationarity is fundamental for forecasting and inference
```