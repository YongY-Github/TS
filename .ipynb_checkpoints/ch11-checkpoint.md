---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 11 — Autoregressive Models

In previous chapters, we studied:

- stationarity
- autocorrelation
- persistence
- unit roots and differencing

We are now ready to build formal stochastic models for stationary time series.

One of the most important and widely used classes of models is the **autoregressive (AR)** model.

AR models capture a simple but powerful idea:

```{admonition} Central Idea
The present is often related to the recent past.
```

This idea may sound simple, but it is extremely powerful.

Many economic and financial variables adjust gradually rather than instantaneously. Inflation, unemployment, interest rates, exchange rates, and GDP growth all tend to exhibit inertia or persistence.

Autoregressive models capture this gradual adjustment process mathematically.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- understand the logic of autoregressive models
- define AR($p$) processes
- derive the mean and variance of AR(1)
- understand stationarity conditions
- interpret persistence
- derive the autocovariance structure of AR models
- understand the Yule–Walker equations
- interpret the ACF and PACF of AR models
- estimate simple AR models

---

# 11.1 The Basic Idea

Suppose today's value depends partly on yesterday's value.

A simple model is:

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t
```

where:

- $x_t$ is the current value
- $\phi$ measures persistence
- $w_t$ is white noise

```{admonition} Intuition
The AR(1) model says that the current value partly “inherits” information from the previous period.

- If $\phi$ is close to zero, the past matters little.
- If $\phi$ is large and positive, shocks fade away slowly.
- If $\phi$ is negative, the series tends to oscillate.
```

---

# 11.2 The AR(1) Model

```{admonition} Definition
An autoregressive process of order 1, denoted AR(1), is:

$$
x_t = \phi x_{t-1} + \mu + w_t
$$

where:

- $w_t \sim wn(0,\sigma_w^2)$
- $\mu$ is a constant
- $\phi$ measures persistence
```

## Mean of AR(1)

Assume stationarity and let:

```{math}
:enumerated: false
E[x_t] = m
```

Taking expectations:

```{math}
:enumerated: false
m = \phi m + \mu
```

Hence:

```{math}
:enumerated: false
m = \frac{\mu}{1-\phi}
```

provided:

```{math}
:enumerated: false
\phi \neq 1
```

```{admonition} Key Result
For a stationary AR(1):

$$
E[x_t]
=
\frac{\mu}{1-\phi}
$$
```

---

# 11.3 Mean-Centered Form

Define:

```{math}
:enumerated: false
y_t = x_t - m
```

Then:

```{math}
:enumerated: false
y_t = \phi y_{t-1} + w_t
```

So we can usually work with the simpler mean-zero form:

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t
```

---

# 11.4 Recursive Representation

To understand the dynamics of the AR(1) model more deeply, it is useful to repeatedly substitute lagged values into the equation itself.

This reveals how current observations depend on the entire history of past shocks.

Substitute repeatedly:

```{math}
:enumerated: false
x_t = \phi(\phi x_{t-2} + w_{t-1}) + w_t
```

giving:

```{math}
:enumerated: false
x_t
=
\phi^2 x_{t-2}
+
\phi w_{t-1}
+
w_t
```

Continuing recursively:

```{math}
:enumerated: false
x_t
=
\phi^k x_{t-k}
+
\sum_{j=0}^{k-1} \phi^j w_{t-j}
```

If:

```{math}
:enumerated: false
|\phi| < 1
```

then:

```{math}
:enumerated: false
\phi^k x_{t-k} \to 0
```

and therefore:

```{math}
:enumerated: false
x_t
=
\sum_{j=0}^{\infty} \phi^j w_{t-j}
```

```{admonition} Key Insight
A stationary AR(1) process can be written as an infinite weighted sum of past shocks.
```

This representation is extremely important.

It shows that an AR(1) process is built from:

- current shocks,
- recent shocks,
- and increasingly distant past shocks,

with weights that decline geometrically over time.

---

# 11.5 Stationarity Condition

```{admonition} Stationarity Condition
An AR(1) process is stationary if:

$$
|\phi| < 1
$$
```

## Why?

If:

```{math}
:enumerated: false
|\phi| < 1
```

then:

- effects of shocks decay over time
- variance remains finite
- the process fluctuates around a stable mean

```{admonition} Intuition
Stationarity requires that shocks eventually lose their influence.

If shocks never fade away, the series “remembers” disturbances forever and the variance grows without bound.
```

## What Happens if $\phi = 1$?

Then:

```{math}
:enumerated: false
x_t = x_{t-1} + w_t
```

which is a random walk.

```{admonition} Important
The random walk is NOT stationary.
```

---

# 11.6 Simulating AR(1) Processes

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

n = 300
w = np.random.normal(size=n)

phis = [0.2, 0.7, 0.95]

fig, ax = plt.subplots(3,1, figsize=(10,8))

for i, phi in enumerate(phis):

    x = np.zeros(n)

    for t in range(1,n):
        x[t] = phi*x[t-1] + w[t]

    ax[i].plot(x, lw=1)
    ax[i].set_title(rf"AR(1): $\phi={phi}$")

plt.tight_layout()

plt.savefig("figs/ch11/AR1.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![AR 1](figs/ch11/AR1.png)


```{admonition} Observation
As $\phi$ approaches 1, persistence becomes increasingly strong.
```

---

# 11.7 Variance of AR(1)

Starting from:

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t
```

take variances:

```{math}
:enumerated: false
Var(x_t)
=
\phi^2 Var(x_{t-1})
+
\sigma_w^2
```

Under stationarity:

```{math}
:enumerated: false
Var(x_t)
=
Var(x_{t-1})
=
\gamma(0)
```

Hence:

```{math}
:enumerated: false
\gamma(0)
=
\phi^2 \gamma(0)
+
\sigma_w^2
```

Therefore:

```{math}
:enumerated: false
\gamma(0)
=
\frac{\sigma_w^2}{1-\phi^2}
```

```{admonition} Result
For stationary AR(1):

$$
Var(x_t)
=
\frac{\sigma_w^2}{1-\phi^2}
$$
```

Notice that as $\phi \to 1$, the denominator approaches zero and the variance becomes very large.

This reflects the increasing persistence of the process.

---

# 11.8 Autocovariance Function

Recall:

```{math}
:enumerated: false
\gamma(h)
=
Cov(x_t,x_{t-h})
```

Using the AR(1) recursion:

```{math}
:enumerated: false
\gamma(h)
=
\phi \gamma(h-1)
```

Repeated substitution gives:

```{math}
:enumerated: false
\gamma(h)
=
\phi^h \gamma(0)
```

---

# 11.9 Yule–Walker Equations

The recursive structure of autoregressive models implies a set of important relationships between autocovariances and model parameters.

These are called the **Yule–Walker equations**.

For the AR(1) model:

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t
```

multiply both sides by $x_{t-h}$ and take expectations:

```{math}
:enumerated: false
E[x_t x_{t-h}]
=
\phi E[x_{t-1}x_{t-h}]
+
E[w_t x_{t-h}]
```

Since white noise is uncorrelated with past values:

```{math}
:enumerated: false
E[w_t x_{t-h}] = 0
\quad \text{for } h \geq 1
```

Therefore:

```{math}
:enumerated: false
\gamma(h)
=
\phi \gamma(h-1)
```

Dividing by $\gamma(0)$ gives:

```{math}
:enumerated: false
\rho(h)
=
\phi \rho(h-1)
```

```{admonition} Key Insight
The Yule–Walker equations connect the autocorrelation structure directly to the model parameters.
```

---

# 11.10 Autocorrelation Function (ACF)

Since:

```{math}
:enumerated: false
\rho(h)
=
\frac{\gamma(h)}{\gamma(0)}
```

we obtain:

```{math}
:enumerated: false
\rho(h)
=
\phi^h
```

```{admonition} Key Result
The ACF of AR(1) decays geometrically.
```

## Interpretation

### If $0<\phi<1$

- smooth positive decay
- persistent behavior

### If $\phi$ close to 1

- slow decay
- strong persistence

### If $-1<\phi<0$

- oscillating autocorrelation
- alternating signs

---

# 11.11 Simulated ACF

```{code-cell} python
from statsmodels.graphics.tsaplots import plot_acf

np.random.seed(123)

phi = 0.8
n = 500

w = np.random.normal(size=n)

x = np.zeros(n)

for t in range(1,n):
    x[t] = phi*x[t-1] + w[t]

plot_acf(x, lags=30)

plt.savefig("figs/ch11/AR1-acf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![AR 1 ACF](figs/ch11/AR1-acf.png)

```{admonition} Observation
The ACF tails off gradually rather than cutting off sharply.
```

---

# 11.12 The Partial Autocorrelation Function (PACF)

Recall:

- ACF measures total correlation
- PACF measures direct correlation after controlling for intermediate lags

The ACF alone is sometimes insufficient because correlations at long lags may arise indirectly through intermediate lags.

The PACF helps isolate the “direct” dependence at each lag.

```{admonition} Fundamental Result
For an AR($p$) process:

- ACF tails off
- PACF cuts off after lag $p$
```

## AR(1) PACF

For AR(1):

- PACF large at lag 1
- PACF approximately zero afterward

## Simulated PACF

```{code-cell} python
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(x, lags=30, method='ywm')

plt.savefig("figs/ch11/AR1-pacf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![AR 1 PACF](figs/ch11/AR1-pacf.png)

```{admonition} Identification Rule
AR(1):

- ACF tails off
- PACF cuts off after lag 1
```

This “cutoff” property makes the PACF extremely useful for identifying autoregressive order in practice.

---

# 11.13 The AR(2) Model

We now allow dependence on two lags:

$$
x_t
=
\phi_1 x_{t-1}
+
\phi_2 x_{t-2}
+
w_t
$$

Allowing two lags greatly enriches the behavior of the model.

Unlike AR(1), AR(2) processes can generate:

- cyclical movements,
- oscillations,
- and damped business-cycle-like behavior.

## Yule–Walker Equations for AR(2)

For AR(2):

```{math}
:enumerated: false
\gamma(h)
=
\phi_1 \gamma(h-1)
+
\phi_2 \gamma(h-2)
```

for:

```{math}
:enumerated: false
h \geq 1
```

These equations determine the autocorrelation structure of the process.

---

# 11.14 Simulating AR(2)

```{code-cell} python
np.random.seed(123)

n = 400
w = np.random.normal(size=n)

phi1 = 1.0
phi2 = -0.6

x = np.zeros(n)

for t in range(2,n):
    x[t] = phi1*x[t-1] + phi2*x[t-2] + w[t]

plt.figure(figsize=(10,4))
plt.plot(x)
plt.title("AR(2) Process")

plt.savefig("figs/ch11/AR2.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![AR 2 PACF](figs/ch11/AR2.png)


```{admonition} Observation
AR(2) models may generate damped cyclical behavior.
```

---

# 11.15 ACF and PACF of AR(2)

```{code-cell} python
plot_acf(x, lags=30)
plt.savefig("figs/ch11/AR2-acf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()

plot_pacf(x, lags=30, method='ywm')

plt.savefig("figs/ch11/AR2-pacf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![AR 2 ACF](figs/ch11/AR2-acf.png)

![AR 2 PACF](figs/ch11/AR2-pacf.png)

```{admonition} Identification Rule
AR(2):

- ACF tails off
- PACF cuts off after lag 2
```

---

# 11.16 General AR(p) Models

```{admonition} Definition
An AR($p$) process is:

$$
x_t
=
\phi_1 x_{t-1}
+
\phi_2 x_{t-2}
+
\cdots
+
\phi_p x_{t-p}
+
w_t
$$
```

## Backshift Representation

As autoregressive models become more complicated, compact notation becomes useful.

Time series analysis therefore often uses the **backshift operator** (or lag operator).

Define:

```{math}
:enumerated: false
Bx_t = x_{t-1}
```

Then:

```{math}
:enumerated: false
B^2x_t = x_{t-2}
```

and more generally:

```{math}
:enumerated: false
B^k x_t = x_{t-k}
```

Using the lag operator:

```{math}
:enumerated: false
\phi(B)x_t = w_t
```

where:

```{math}
:enumerated: false
\phi(B)
=
1
-
\phi_1 B
-
\phi_2 B^2
-
\cdots
-
\phi_p B^p
```

---

# 11.17 General Yule–Walker Equations

For an AR($p$) process:

```{math}
:enumerated: false
\gamma(h)
=
\phi_1 \gamma(h-1)
+
\phi_2 \gamma(h-2)
+
\cdots
+
\phi_p \gamma(h-p)
```

These equations link:

- model parameters,
- autocovariances,
- and autocorrelations.

---

# 11.18 Characteristic Roots and Stationarity

The stationarity condition depends on the roots of:

```{math}
:enumerated: false
\phi(z)=0
```

```{admonition} Stationarity Condition
An AR($p$) process is stationary if all roots lie outside the unit circle.
```

```{admonition} Intuition
Stationarity means shocks eventually die out rather than explode.
```

---

# 11.19 Estimating an AR Model in Python

Let's estimate a simple AR(2) model (which was simulated above, $x$).

# Estimate AR model

```{code-cell} python
from statsmodels.tsa.ar_model import AutoReg

model = AutoReg(
    x,
    lags=2
)

results = model.fit()

print(results.summary())
```

---

# 11.19 Estimation in Gretl

## Menu

```text
Model → Time Series → ARIMA
```

## Basic Workflow

1. Plot the series
2. Check stationarity
3. Examine ACF/PACF
4. Choose tentative AR order
5. Estimate the model
6. Check residual diagnostics

```markdown
[GRETL Screenshot Placeholder: AR model estimation dialog]
```

```markdown
[GRETL Screenshot Placeholder: AR(1) output]
```

---

# 11.20 Residual Diagnostics

After fitting an AR model, residuals should resemble white noise.

## Residual ACF

```{code-cell} python
import statsmodels.api as sm

model = sm.tsa.ARIMA(x, order=(2,0,0))
res = model.fit()

plot_acf(res.resid, lags=20)

plt.savefig("figs/ch11/res-acf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Residual ACF](figs/ch11/res-acf.png)

## Ljung–Box Test

```{code-cell} python
from statsmodels.stats.diagnostic import acorr_ljungbox

lb = acorr_ljungbox(res.resid, lags=[10,20], return_df=True)
lb
```

| lag | lb_stat | lb_pvalue |
|---|---:|---:|
| 10 | 4.862906 | 0.900146 |
| 20 | 25.729475 | 0.174934 |

The large p-values suggest that we fail to reject the null hypothesis of no serial correlation.

```{admonition} Goal
A good model should leave little remaining autocorrelation in the residuals.
```

---

# 11.21 Information Criteria

Two commonly used model-selection tools are:

$$
AIC
=
-2\log(\hat L)
+
2k
$$

$$
BIC
=
-2\log(\hat L)
+
k\log n
$$

```{admonition} Interpretation
Lower AIC/BIC values generally indicate better models.
```

---

# 11.22 AR Models in Economics and Finance

AR models appear throughout applied work.

Examples include:

- inflation persistence
- interest-rate smoothing
- GDP growth dynamics
- volatility persistence
- inventory adjustment

```{admonition} Economic Interpretation
AR models capture gradual adjustment rather than instantaneous change.
```

---

# 11.23 Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Ignoring stationarity**  
AR models require stationarity.

**2. Using too many lags**  
Overparameterization may reduce interpretability.

**3. Blind reliance on ACF/PACF**  
Use economic reasoning together with statistical tools.

**4. Ignoring residual diagnostics**  
Residual autocorrelation suggests model misspecification.

**5. Confusing persistence with trend**  
A highly persistent stationary series is not necessarily nonstationary.
```

---

# 11.24 Looking Ahead

In this chapter, we studied autoregressive models, which capture persistence through dependence on past values.

In the next chapter, we study:

- moving average (MA) models
- dependence through past shocks
- finite-memory processes

before combining both ideas into ARMA models.

# Key Takeaways

```{admonition} Summary
- AR models relate the present to the past
- AR(1) exhibits geometric persistence
- Stationarity requires roots outside the unit circle
- Yule–Walker equations connect AR parameters and autocorrelations
- ACF tails off for AR processes
- PACF cuts off after lag $p$
- AR models are central building blocks in time series analysis
```