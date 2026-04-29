---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 10 — Unit Roots and Differencing

In previous chapters, we introduced:

- random walks
- persistence
- stationarity
- autocorrelation

We now study one of the most important sources of nonstationarity in time series analysis:

```{admonition} Central Idea
Some time series contain a **unit root**, meaning that shocks have permanent effects.
```

Understanding unit roots is fundamental because:

- many economic and financial time series are highly persistent
- standard regression methods may fail under nonstationarity
- forecasting behavior changes dramatically
- differencing often becomes necessary

This chapter introduces:

- unit root intuition
- random walks revisited
- differencing
- detrending vs differencing
- the Augmented Dickey–Fuller (ADF) test

---

## Learning Objectives

By the end of this chapter, you should be able to:

- understand what a unit root is
- explain why random walks are nonstationary
- distinguish stationary and unit root processes
- difference a time series
- distinguish detrending from differencing
- interpret the ADF test intuitively

---

## 10.1 Persistence Revisited

Recall the random walk:

```{math}
:enumerated: false
x_t = x_{t-1} + w_t
```

where:

```{math}
:enumerated: false
w_t \sim wn(0,\sigma_w^2)
```

Each new observation equals:

- the previous value
- plus a random shock

```{admonition} Key Insight
In a random walk, shocks accumulate permanently over time.
```

---

## 10.2 Simulating a Random Walk

````{dropdown} Python Code
```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

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
````

![Random Walk](figs/ch7/rw.png)

```{admonition} Observation
The series drifts over time and does not fluctuate around a stable mean.
```

---

## 10.3 Why Random Walks Matter

Random walks are central in economics and finance.

Examples include:

- stock prices
- exchange rates
- some macroeconomic aggregates

The random walk model implies:

- strong persistence
- unpredictable long-run movement
- permanent effects of shocks

---

## 10.4 The Unit Root Idea

The random walk can be rewritten as:

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t
```

with:

```{math}
:enumerated: false
\phi = 1
```

```{admonition} Definition
A process has a **unit root** when the coefficient on the lagged variable equals one.
```

---

## 10.5 Why the Unit Root Matters

Consider:

$$
x_t = \phi x_{t-1} + w_t
$$

### Case 1: $|\phi| < 1$

- shocks gradually disappear
- process reverts toward its mean
- process is stationary

### Case 2: $\phi = 1$

- shocks never disappear
- effects accumulate permanently
- process becomes nonstationary

```{admonition} Intuition
The closer $\phi$ is to 1, the more persistent the series becomes.
```

---

## 10.6 Stationary vs Unit Root Processes

### Stationary Process

- stable mean
- stable variance
- temporary shocks

### Unit Root Process

- drifting behavior
- growing variance
- permanent shocks

```{admonition} Important
Unit root processes exhibit a fundamentally different type of persistence from stationary processes.
```

---

## 10.7 Simulating Different Levels of Persistence

```{code-cell} python
:tags: [hide-input]
np.random.seed(123)

n = 500
w = np.random.normal(0,1,n)

x1 = np.zeros(n)
x2 = np.zeros(n)
x3 = np.zeros(n)

phi1 = 0.4
phi2 = 0.9
phi3 = 1.0

for t in range(1,n):
    x1[t] = phi1*x1[t-1] + w[t]
    x2[t] = phi2*x2[t-1] + w[t]
    x3[t] = phi3*x3[t-1] + w[t]

fig, ax = plt.subplots(3,1, figsize=(10,8))

ax[0].plot(x1)
ax[0].set_title(r"$\phi = 0.4$")

ax[1].plot(x2)
ax[1].set_title(r"$\phi = 0.9$")

ax[2].plot(x3)
ax[2].set_title(r"$\phi = 1.0$ (Unit Root)")

plt.tight_layout()

plt.savefig("figs/ch10/ar-sim.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Persistence](figs/ch10/ar-sim.png)


```{admonition} Observation
As $\phi$ approaches 1, persistence becomes increasingly strong.
```

---

## 10.8 Differencing

A common way to remove unit roots is differencing.

```{admonition} Definition
The first difference of a series is:

$$
\Delta x_t = x_t - x_{t-1}
$$
```

---

## 10.9 Differencing a Random Walk

For a random walk:

$$
x_t = x_{t-1} + w_t
$$

taking first differences gives:

$$
\Delta x_t = w_t
$$

```{admonition} Key Insight
Differencing converts a random walk into white noise.
```

---

## 10.10 Simulating Differencing

```{code-cell} python
dx = np.diff(x)

plt.figure(figsize=(10,4))
plt.plot(dx, lw=1)
plt.title("First Difference of Random Walk")
plt.xlabel("Time")
plt.ylabel(r"$\Delta x_t$")

plt.savefig("figs/ch10/rw-diff.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![RW Differencing](figs/ch10/rw-diff.png)

```{admonition} Observation
The differenced series fluctuates around a stable mean and appears stationary.
```

---

## 10.11 Integrated Processes

A unit root process is said to be integrated of order one.

```{admonition} Definition
A series is:

- $I(0)$ if stationary
- $I(1)$ if first differences are stationary
```

### Example

| Series | Order |
|---|---|
| White noise | $I(0)$ |
| Random walk | $I(1)$ |

---

## 10.12 Detrending vs Differencing

Nonstationarity may arise for different reasons.

### Deterministic Trend

Suppose:

$$
x_t = \alpha + \beta t + u_t
$$

where $u_t$ is stationary.

Removing the trend may produce stationarity.

### Stochastic Trend

For a random walk:

$$
x_t = x_{t-1} + w_t
$$

detrending alone is insufficient.

Differencing is required.

```{admonition} Important
Trend-stationary and difference-stationary processes require different treatments.
```

---

## 10.13 Visual Comparison

```{code-cell} python
:tags: [hide-input]

np.random.seed(123)

t = np.arange(500)

trend_stationary = 0.03*t + np.random.normal(0,1,500)

plt.figure(figsize=(10,4))
plt.plot(trend_stationary)
plt.title("Trend-Stationary Series")
plt.xlabel("Time")

plt.savefig("figs/ch10/trend.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Deterministic Trend](figs/ch10/trend.png)


```{admonition} Observation
Trend-stationary series fluctuate around a deterministic trend rather than drifting permanently.
```

---

## 10.14 Why Unit Roots Matter

Unit roots affect:

- forecasting
- inference
- regression
- model selection

```{admonition} Key Problem
Regressions involving nonstationary variables may produce misleading statistical relationships.
```

This problem becomes central later in:

- spurious regression
- cointegration
- ECM models

---

## 10.15 The Dickey–Fuller Test

We now need a way to test for unit roots.

Consider:

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t
```

Subtract $x_{t-1}$ from both sides:

```{math}
:enumerated: false
x_t - x_{t-1}
=
(\phi - 1)x_{t-1} + w_t
```

or:

```{math}
:enumerated: false
\Delta x_t
=
\theta x_{t-1} + w_t
```

where:

```{math}
:enumerated: false
\theta = \phi - 1
```

### Hypotheses

```{math}
:enumerated: false
H_0: \theta = 0
\quad \text{(unit root)}
```

```{math}
:enumerated: false
H_1: \theta < 0
\quad \text{(stationary)}
```

```{admonition} Intuition
The Dickey–Fuller test checks whether the series exhibits mean reversion.
```

---

## 10.16 Augmented Dickey–Fuller (ADF) Test

Real-world data often exhibit serial correlation.

The Augmented Dickey–Fuller (ADF) test extends the Dickey–Fuller test by including lagged differences:

$$
\Delta x_t
=
\alpha
+
\theta x_{t-1}
+
\sum_{i=1}^p \gamma_i \Delta x_{t-i}
+
u_t
$$

```{admonition} Important
The additional lagged differences help remove serial correlation from the residuals.
```

---

## 10.17 Interpreting the ADF Test

### Reject $H_0$

- evidence against unit root
- series likely stationary


### Fail to Reject $H_0$

- insufficient evidence against unit root
- series may be nonstationary

```{admonition} Caution
Failing to reject a unit root does not prove the process is exactly a random walk.
```

---

## 10.18 ADF Testing in Gretl

### Menu

```text
Variable → Unit root tests → Augmented Dickey-Fuller
```

### Typical Steps

1. Choose variable
2. Include constant and/or trend if appropriate
3. Select lag length
4. Interpret test statistic and p-value

```markdown
[GRETL Screenshot Placeholder: ADF test dialog]
```

```markdown
[GRETL Screenshot Placeholder: ADF test output]
```

---

## 10.19 KPSS Test (Optional)

The KPSS test reverses the hypotheses.

### KPSS Hypotheses

```{math}
:enumerated: false
H_0: \text{stationary}
```

```{math}
:enumerated: false
H_1: \text{unit root}
```

```{admonition} Practical Advice
Many analysts use both:

- ADF test
- KPSS test

to obtain complementary evidence.
```

---

## 10.20 Looking Ahead

In this chapter, we introduced:

- unit roots
- differencing
- ADF testing

We are now ready to build formal stochastic models for stationary time series.

In the next chapters, we study:

- autoregressive (AR) models
- moving average (MA) models
- ARMA and ARIMA models

## Key Takeaways

```{admonition} Summary
- Unit roots imply permanent effects of shocks
- Random walks are classic unit root processes
- Differencing often removes nonstationarity
- Trend-stationary and difference-stationary processes differ fundamentally
- The ADF test helps detect unit roots
```