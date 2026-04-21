---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Non-Stationarity and ARIMA Models

In the previous chapter, we studied AR, MA, and ARMA models under the assumption that the series is **stationary**. In practice, however, many economic and financial time series are **not stationary**.

For example:

- GDP often grows over time
- consumer prices usually trend upward
- population tends to increase
- stock prices may drift and wander

In such cases, it is often inappropriate to fit an ARMA model directly to the series in levels. Instead, we first transform the series into one that is approximately stationary, and then model the transformed series.

The most common transformation is **differencing**.

This leads to the **ARIMA** model.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- explain what it means for a time series to be non-stationary
- distinguish deterministic trend from stochastic trend
- understand the random walk as a canonical non-stationary process
- explain the role of differencing
- understand the idea of a unit root
- define an ARIMA($p,d,q$) model
- recognize the dangers of under-differencing and over-differencing

---

## Why Stationarity Matters

Many of the tools developed in the previous chapter assume stationarity. In particular, for a weakly stationary process:

- the mean is constant over time
- the variance is constant over time
- the autocovariance depends only on lag, not on time

When a series is non-stationary, these properties fail, and our usual tools may become misleading.

```{admonition} Key Idea
Before fitting an ARMA-type model, we should first ask:

**Does the series appear stationary?**
````

---

## A First Visual Comparison

The contrast between a stationary series and a non-stationary one is easiest to see graphically.

### White Noise vs Random Walk

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
n = 300

w = np.random.normal(size=n)
rw = np.cumsum(w)

fig, ax = plt.subplots(figsize=(8,3))
ax.plot(w, lw=1)
ax.set_title("White Noise")
ax.set_xlabel("Time")
ax.set_ylabel("$x_t$")
plt.show()

fig, ax = plt.subplots(figsize=(8,3))
ax.plot(rw, lw=1)
ax.set_title("Random Walk")
ax.set_xlabel("Time")
ax.set_ylabel("$x_t$")
plt.show()
```

The white noise series fluctuates around a roughly constant mean and variance. By contrast, the random walk wanders persistently and does not return to any stable level.

```{admonition} Intuition
A stationary series tends to fluctuate around a stable center.

A random walk has no such stable center: shocks accumulate over time.
```

---

## Different Sources of Non-Stationarity

A series may be non-stationary for different reasons. Two especially important cases are:

1. **deterministic trend**
2. **stochastic trend**

These are not the same thing, and they require different responses.

---

## Deterministic Trend

A simple trend-stationary model is

$$
x_t = \alpha + \beta t + y_t,
$$

where $y_t$ is stationary.

Here the non-stationarity comes from the deterministic function $\alpha + \beta t$. If we remove that trend, the remaining series may be stationary.

### Simulated Deterministic Trend

```{code-cell} python
np.random.seed(123)
n = 300
t = np.arange(n)
y = np.random.normal(scale=1.0, size=n)
x_trend = 2 + 0.05*t + y

fig, ax = plt.subplots(figsize=(8,3))
ax.plot(x_trend, lw=1)
ax.set_title("Series with Deterministic Linear Trend")
ax.set_xlabel("Time")
ax.set_ylabel("$x_t$")
plt.show()
```

```{admonition} Interpretation
This series trends upward, but the trend is smooth and predictable. The fluctuations around the trend are stationary.
```

---

## Stochastic Trend

A series has a **stochastic trend** when shocks have permanent effects. The most important example is the **random walk**:

$$
x_t = x_{t-1} + w_t.
$$

By repeated substitution,

```{math}
:enumerated: false
x_t = x_0 + \sum_{j=1}^t w_j.
```

So the current level is the cumulative sum of all past shocks.

```{admonition} Key Insight
In a random walk, every shock has a permanent effect.

A positive shock today raises the expected path of the series in all future periods.
```

---

## Random Walk with Drift

A random walk with drift is

$$
x_t = \delta + x_{t-1} + w_t.
$$

This can be written as

$$
x_t = x_0 + \delta t + \sum_{j=1}^t w_j.
$$

So it combines:

* a deterministic upward or downward drift, and
* a stochastic wandering component

### Simulated Random Walk with Drift

```{code-cell} python
np.random.seed(123)
n = 300
drift = 0.1
w = np.random.normal(size=n)
rw_drift = np.zeros(n)

for i in range(1, n):
    rw_drift[i] = drift + rw_drift[i-1] + w[i]

fig, ax = plt.subplots(figsize=(8,3))
ax.plot(rw_drift, lw=1)
ax.set_title("Random Walk with Drift")
ax.set_xlabel("Time")
ax.set_ylabel("$x_t$")
plt.show()
```

This series tends to move upward over time, but not smoothly. Its path is irregular because shocks accumulate permanently.

---

## Why the Random Walk is Non-Stationary

If

```{math}
:enumerated: false
x_t = x_{t-1} + w_t,
```

then the variance grows over time:

```{math}
:enumerated: false
\operatorname{Var}(x_t) = t \sigma_w^2,
```

assuming the starting value is fixed.

So the variance is not constant. Hence the random walk is not stationary.

```{admonition} Why This Matters
A non-stationary series may show high persistence even when there is no stable underlying mean to revert to.
```

---

## Differencing

The most common way to transform a non-stationary series is to take **differences**.

The first difference is

$$
\Delta x_t = x_t - x_{t-1}.
$$

Using the backshift operator $B$:

```{math}
:enumerated: false
\Delta x_t = (1-B)x_t.
```

The second difference is

```{math}
:enumerated: false
\Delta^2 x_t = (1-B)^2 x_t.
```

More generally,

```{math}
:enumerated: false
\Delta^d x_t = (1-B)^d x_t.
```

---

## Why Differencing Works

For a random walk,

```{math}
:enumerated: false
x_t = x_{t-1} + w_t.
```

Subtracting $x_{t-1}$ from both sides gives

```{math}
:enumerated: false
\Delta x_t = w_t.
```

So the first-differenced series is white noise, which is stationary.

```{admonition} Key Idea
A random walk is non-stationary in levels but stationary in first differences.
```

---

## A Visual Example of Differencing

### Original Series and First Difference

```{code-cell} python
np.random.seed(123)
n = 300
w = np.random.normal(size=n)
rw = np.cumsum(w)
drw = np.diff(rw)

fig, ax = plt.subplots(figsize=(8,3))
ax.plot(rw, lw=1)
ax.set_title("Random Walk")
ax.set_xlabel("Time")
ax.set_ylabel("$x_t$")
plt.show()

fig, ax = plt.subplots(figsize=(8,3))
ax.plot(drw, lw=1)
ax.set_title("First Difference of the Random Walk")
ax.set_xlabel("Time")
ax.set_ylabel("$\\Delta x_t$")
plt.show()
```

The differenced series looks much more stable than the original series.

---

## ACF of the Level Series and the Differenced Series

```{code-cell} python
from statsmodels.graphics.tsaplots import plot_acf

fig, ax = plt.subplots(figsize=(8,3))
plot_acf(rw, lags=30, ax=ax)
ax.set_title("ACF: Random Walk")
plt.show()

fig, ax = plt.subplots(figsize=(8,3))
plot_acf(drw, lags=30, ax=ax)
ax.set_title("ACF: First Difference of Random Walk")
plt.show()
```

The ACF of the random walk decays very slowly, reflecting strong persistence. After differencing, the autocorrelation structure is much weaker.

```{admonition} Interpretation
A slowly decaying ACF in levels is often a warning sign of non-stationarity.
```

---

## Trend-Stationary vs Difference-Stationary

We now distinguish two important ideas.

### Trend-stationary

A series is **trend-stationary** if removing a deterministic trend leaves a stationary residual.

Example:

```{math}
:enumerated: false
x_t = \alpha + \beta t + y_t,
```

where $y_t$ is stationary.

### Difference-stationary

A series is **difference-stationary** if it becomes stationary only after differencing.

Example:

```{math}
:enumerated: false
x_t = x_{t-1} + w_t.
```

```{admonition} Important Distinction
- **Trend-stationary** series are made stationary by detrending  
- **Difference-stationary** series are made stationary by differencing
```

---

## Unit Roots

Consider the AR(1) model

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t.
```

We know this process is stationary only if $|\phi| < 1$.

The case $\phi = 1$ gives

```{math}
:enumerated: false
x_t = x_{t-1} + w_t,
```

which is the random walk.

This is called a **unit root** process.

```{admonition} Definition: Unit Root
A series has a unit root when the autoregressive polynomial has a root equal to 1.

In the AR(1) case, this corresponds to $\phi = 1$.
```

---

## Rewriting the AR(1) Model

Start from

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t.
```

Subtract $x_{t-1}$ from both sides:

```{math}
:enumerated: false
x_t - x_{t-1} = (\phi - 1)x_{t-1} + w_t.
```

So

```{math}
:enumerated: false
\Delta x_t = \gamma x_{t-1} + w_t,
```

where

```{math}
:enumerated: false
\gamma = \phi - 1.
```

Testing for a unit root is therefore equivalent to testing

```{math}
:enumerated: false
H_0: \gamma = 0.
```

---

## The Dickey–Fuller Idea

The basic Dickey–Fuller test is built around the regression

$$
\Delta x_t = \gamma x_{t-1} + w_t.
$$

The hypotheses are:

* $H_0: \gamma = 0$  (unit root)
* $H_1: \gamma < 0$  (stationary)

If we reject the null, we have evidence against a unit root.

```{admonition} Important
The Dickey–Fuller test does not use the usual $t$ critical values. Specialized critical values are required.
```

---

## The Augmented Dickey–Fuller (ADF) Test

In practice, we usually allow more complicated short-run dynamics by estimating

```{math}
:enumerated: falselta x_t = \alpha + \beta t + \gamma x_{t-1}

* \delta_1 \Delta x_{t-1}
* \delta_2 \Delta x_{t-2}
* \cdots
* \delta_k \Delta x_{t-k}
* u_t.
```

The additional lagged differences help absorb serial correlation in the errors.

```{admonition} Intuition
The ADF test asks whether the series behaves like a wandering process with no tendency to return, or whether it shows evidence of mean reversion.
```

---

## ARIMA Models

We are now ready to define the ARIMA model.

```{admonition} Definition: ARIMA(p,d,q)
A time series follows an **ARIMA($p,d,q$)** model if the $d$-th differenced series follows an ARMA($p,q$) model.
```

In compact form:

```{math}
:enumerated: false
\phi(B)(1-B)^d x_t = \theta(B) w_t.
```

Here:

* $p$ = AR order
* $d$ = number of differences
* $q$ = MA order

---

## Interpreting the Parameters

```{admonition} Intuition
ARIMA modeling has two steps:

1. Difference the series until it is approximately stationary  
2. Model the differenced series using ARMA tools
```

So:

* ARIMA(0,1,0) is a random walk
* ARIMA(1,1,0) means the first difference follows AR(1)
* ARIMA(0,1,1) means the first difference follows MA(1)
* ARIMA(1,1,1) means the first difference follows ARMA(1,1)

---

## Example Forms

### ARIMA(0,1,0)

```{math}
:enumerated: false
(1-B)x_t = w_t
```

This is the random walk.

### ARIMA(1,1,0)

```{math}
:enumerated: false
(1-\phi B)(1-B)x_t = w_t
```

Equivalently,

```{math}
:enumerated: false
\Delta x_t = \phi \Delta x_{t-1} + w_t.
```

### ARIMA(0,1,1)

```{math}
:enumerated: false
(1-B)x_t = (1+\theta B)w_t.
```

### ARIMA(1,1,1)

```{math}
:enumerated: false
(1-\phi B)(1-B)x_t = (1+\theta B)w_t.
````
---

## A Practical ARIMA Workflow

A beginner-friendly workflow is:

1. plot the series in levels
2. ask whether it appears stationary
3. difference the series if needed
4. inspect the differenced series
5. examine ACF/PACF of the differenced series
6. fit small candidate ARIMA models
7. check residual diagnostics
8. compare models using AIC/BIC

```{admonition} Practical Advice
Start with simple candidates such as:

- ARIMA(0,1,1)
- ARIMA(1,1,0)
- ARIMA(1,1,1)
```

---

## Under-Differencing and Over-Differencing

Differencing is useful, but it should not be done mechanically.

### Under-differencing

If we difference too little, the series may remain non-stationary.

### Over-differencing

If we difference too much, we may remove useful structure and introduce extra noise.

```{admonition} Common Pitfall
Difference **only as much as needed** to achieve approximate stationarity.
```

---

## A Visual Hint of Over-Differencing

A common warning sign of over-differencing is a strong negative autocorrelation at lag 1.

To illustrate this, let us difference white noise, which is already stationary.

```{code-cell} python
np.random.seed(123)
n = 300
wn = np.random.normal(size=n)
dwn = np.diff(wn)

fig, ax = plt.subplots(figsize=(8,3))
ax.plot(wn, lw=1)
ax.set_title("White Noise")
plt.show()

fig, ax = plt.subplots(figsize=(8,3))
ax.plot(dwn, lw=1)
ax.set_title("Difference of White Noise")
plt.show()
```

```{code-cell} python
fig, ax = plt.subplots(figsize=(8,3))
plot_acf(dwn, lags=30, ax=ax)
ax.set_title("ACF: Difference of White Noise")
plt.show()
```

The strong negative spike at lag 1 is a classic signal that differencing may have gone too far.

```{admonition} Interpretation
If a series is already stationary, differencing it can create artificial autocorrelation and make modeling harder rather than easier.
```

---

## Detrending or Differencing?

Not every trending series should be differenced.

* If the series fluctuates around a smooth deterministic trend, detrending may be better.
* If shocks have permanent effects, differencing may be better.

This is one reason why plots and economic reasoning remain important.

```{admonition} Important
A trending series is not automatically a unit-root series.
```

---

## A Small Demonstration: Trend and Detrending

```{code-cell} python
import statsmodels.api as sm

np.random.seed(123)
n = 300
t = np.arange(n)
y = np.random.normal(size=n)
x_det = 1 + 0.03*t + y

X = sm.add_constant(t)
fit = sm.OLS(x_det, X).fit()
trend_hat = fit.fittedvalues
resid_det = fit.resid

fig, ax = plt.subplots(figsize=(8,3))
ax.plot(x_det, lw=1, label="Series")
ax.plot(trend_hat, lw=2, label="Fitted trend")
ax.set_title("Deterministic Trend and Fitted Trend")
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(8,3))
ax.plot(resid_det, lw=1)
ax.set_title("Detrended Residuals")
plt.show()
```

This example illustrates the idea of a trend-stationary series: after removing the deterministic trend, the residual series may look approximately stationary.

---

## Common Mistakes in ARIMA Modeling

```{admonition} Common Mistakes
:class: warning

**1. Treating every trend as a unit root**  
Some series are trend-stationary rather than difference-stationary.

**2. Differencing too quickly**  
Always inspect the series first.

**3. Identifying AR and MA terms from the wrong series**  
ACF and PACF for ARIMA should be examined on the differenced series, not the original non-stationary levels.

**4. Over-differencing**  
Too much differencing may remove genuine structure.

**5. Blind reliance on formal tests**  
Unit-root tests are useful, but plots and economic reasoning remain important.
```

---

## Summary

* Many real-world series are non-stationary
* Deterministic trend and stochastic trend are different ideas
* The random walk is the canonical unit-root process
* Differencing can convert some non-stationary series into stationary ones
* A series that becomes stationary after $d$ differences is called $I(d)$
* ARIMA($p,d,q$) means the $d$-th differenced series follows ARMA($p,q$)

```{admonition} Looking Ahead
In later chapters, these ideas can be extended to:

- seasonal differencing
- seasonal ARIMA models
- cointegration
- error-correction models
```

