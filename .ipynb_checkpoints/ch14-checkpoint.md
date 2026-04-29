---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 14 — ARIMA Models

In the previous chapters, we studied:

- autoregressive (AR) models
- moving average (MA) models
- ARMA models

These models assume that the underlying process is stationary.

However, many economic and financial time series are not stationary.

Examples include:

- stock prices
- GDP
- exchange rates
- price indices
- money supply

These series often exhibit:

- trends
- persistent drift
- changing variance
- unit roots

To model such series, we extend ARMA models using differencing.

This leads to the **ARIMA model**.

```{admonition} Central Idea
ARIMA models combine:

- autoregression (AR)
- moving averages (MA)
- integration through differencing (I)
```

---

## Learning Objectives

By the end of this chapter, you should be able to:

- understand the motivation behind ARIMA models
- distinguish stationary and integrated processes
- understand the role of differencing
- define ARIMA($p,d,q$) models
- interpret the Box–Jenkins workflow
- estimate ARIMA models in Gretl
- evaluate residual diagnostics and model fit

---

# 14.1 Why ARIMA Models?

Many real-world time series exhibit strong persistence and nonstationarity.

For example:

- stock prices drift over time
- GDP tends to grow
- inflation may exhibit persistent trends

Applying stationary ARMA models directly to such data may produce misleading results.

```{admonition} Key Idea
ARIMA models first transform nonstationary data into stationary data using differencing.
```

---

# 14.2 Differencing Revisited

Recall the first difference operator:

```{math}
:enumerated: false
\Delta x_t
=
x_t - x_{t-1}
```

Differencing removes persistent stochastic trends.

## Example: Random Walk

Suppose:

```{math}
:enumerated: false
x_t = x_{t-1} + w_t
```

Then:

```{math}
:enumerated: false
\Delta x_t = w_t
```

which is white noise.

```{admonition} Important
Differencing converts many nonstationary processes into stationary ones.
```

---

# 14.3 Integrated Processes

```{admonition} Definition
A series is:

- $I(0)$ if stationary
- $I(1)$ if first differences are stationary
- $I(2)$ if second differences are stationary
```

## First Difference

```{math}
:enumerated: false
\Delta x_t = x_t - x_{t-1}
```

## Second Difference

```{math}
:enumerated: false
\Delta^2 x_t
=
\Delta(\Delta x_t)
```

or:

```{math}
:enumerated: false
\Delta^2 x_t
=
x_t - 2x_{t-1} + x_{t-2}
```

```{admonition} Practical Note
Most macroeconomic and financial series encountered in practice are either:

- $I(0)$
- or $I(1)$
```

---

# 14.4 The ARIMA Model

```{admonition} Definition
An ARIMA($p,d,q$) model consists of:

- AR order $p$
- differencing order $d$
- MA order $q$
```

## General Form

After differencing $d$ times:

```{math}
\phi(B)(1-B)^d x_t
=
\theta(B)w_t
```

where:

- $\phi(B)$ is the AR polynomial
- $(1-B)^d$ is the differencing operator
- $\theta(B)$ is the MA polynomial

---

# 14.5 Understanding the Components

## AR Component

Captures persistence through past values.

## I Component

Captures nonstationarity through differencing.

## MA Component

Captures temporary propagation of shocks.

```{admonition} Intuition
ARIMA models describe both:

- long-run stochastic movement
- short-run dependence structure
```

---

# 14.6 Example: ARIMA(0,1,0)

Consider:

```{math}
:enumerated: false
(1-B)x_t = w_t
```

or:

```{math}
:enumerated: false
x_t - x_{t-1} = w_t
```

Thus:

```{math}
:enumerated: false
x_t = x_{t-1} + w_t
```

```{admonition} Key Observation
ARIMA(0,1,0) is simply a random walk.
```

---

# 14.7 Example: ARIMA(1,1,0)

Suppose:

```{math}
:enumerated: false
(1-\phi B)(1-B)x_t = w_t
```

or equivalently:

```{math}
:enumerated: false
\Delta x_t
=
\phi \Delta x_{t-1}
+
w_t
```

```{admonition} Intuition
The differenced series follows an AR(1) process.
```

---

# 14.8 Example: ARIMA(0,1,1)

Suppose:

```{math}
:enumerated: false
(1-B)x_t
=
(1+\theta B)w_t
```

Then:

```{math}
:enumerated: false
\Delta x_t
=
w_t + \theta w_{t-1}
```

```{admonition} Interpretation
The differenced series follows an MA(1) process.
```

---

# 14.9 Simulating an ARIMA Process

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

n = 400

phi = 0.7

w = np.random.normal(size=n)

dx = np.zeros(n)

for t in range(1,n):
    dx[t] = phi*dx[t-1] + w[t]

x = np.cumsum(dx)

fig, ax = plt.subplots(2,1, figsize=(10,6))

ax[0].plot(dx)
ax[0].set_title("Differenced Series")

ax[1].plot(x)
ax[1].set_title("Integrated Series")

plt.tight_layout()

plt.savefig("figs/ch14/ARIMA.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![ARMA](figs/ch14/ARIMA.png)

```{admonition} Observation
The differenced series may appear stationary even though the original series is nonstationary.
```

---

# 14.10 The Box–Jenkins Methodology

Classical ARIMA modeling follows the Box–Jenkins approach.

## Step 1 — Identification

- plot the data
- check stationarity
- difference if necessary
- examine ACF/PACF

## Step 2 — Estimation

Estimate candidate ARIMA models.

## Step 3 — Diagnostic Checking

Check whether residuals resemble white noise.

## Step 4 — Forecasting

Generate forecasts and evaluate performance.

```{admonition} Key Idea
ARIMA modeling is iterative rather than mechanical.
```

---

# 14.11 Identifying Differencing Order

A series may require differencing if:

- it exhibits persistent trends
- ACF decays very slowly
- unit root tests fail to reject nonstationarity

```{admonition} Warning
Overdifferencing can create unnecessary noise and distort dynamics.
```

---

# 14.12 Under-Differencing vs Over-Differencing

## Under-Differencing

- residual nonstationarity remains
- strong persistence persists

## Over-Differencing

- introduces excess noise
- may create artificial negative autocorrelation

```{admonition} Practical Advice
Use the smallest amount of differencing necessary to achieve stationarity.
```

---

# 14.13 ACF and PACF in ARIMA Modeling

After differencing:

- examine ACF
- examine PACF
- identify possible AR and MA orders


## Typical Patterns

| Model | ACF | PACF |
|---|---|---|
| AR($p$) | tails off | cuts off |
| MA($q$) | cuts off | tails off |
| ARMA($p,q$) | tails off | tails off |

```{admonition} Important
Identification should be performed on the stationary differenced series.
```

---

# 14.14 Estimation in Gretl

## Menu

```text
Model → Time Series → ARIMA
```

## Typical Workflow

1. plot the series
2. test for unit roots
3. difference if needed
4. inspect ACF/PACF
5. estimate candidate models
6. compare AIC/BIC
7. check residuals

```markdown
[GRETL Screenshot Placeholder: ARIMA estimation dialog]
```

```markdown
[GRETL Screenshot Placeholder: ARIMA output]
```

---

# 14.15 Residual Diagnostics

Residuals should resemble white noise.

## Residual ACF

```{code-cell} python
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

model = sm.tsa.ARIMA(x, order=(1,1,0))
res = model.fit()

plot_acf(res.resid, lags=20)
plt.show()
```

---

## Ljung–Box Test

```{code-cell} python
from statsmodels.stats.diagnostic import acorr_ljungbox

acorr_ljungbox(res.resid, lags=[10,20], return_df=True)
```

---

```{admonition} Goal
Residual autocorrelation should be small if the model captures the dependence structure adequately.
```

---

# 14.16 Information Criteria

Model selection often uses:

- AIC
- BIC

## Akaike Information Criterion

```{math}
AIC
=
-2\log(\hat L)
+
2k
```

## Bayesian Information Criterion

$$
BIC
=
-2\log(\hat L)
+
k\log n
$$

---

```{admonition} Interpretation
Lower AIC or BIC values generally indicate better-fitting models after penalizing complexity.
```

---

# 14.17 Forecasting with ARIMA Models

Once estimated, ARIMA models can generate forecasts.

## Multi-Step Forecasts

Forecasts are generated recursively using:

- past observations
- estimated parameters
- projected future dynamics

```{admonition} Important
Forecast uncertainty increases as the forecast horizon expands.
```

---

# 14.18 ARIMA Models in Economics and Finance

ARIMA models are widely used for:

- GDP forecasting
- inflation forecasting
- demand forecasting
- exchange rates
- inventory management
- energy consumption

```{admonition} Economic Interpretation
ARIMA models focus primarily on statistical forecasting rather than structural economic relationships.
```

---

# 14.19 Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Ignoring nonstationarity**  
ARIMA models require appropriate differencing.

**2. Overdifferencing**  
Too much differencing may destroy useful structure.

**3. Overfitting**  
Adding excessive AR or MA terms may hurt forecasting performance.

**4. Ignoring residual diagnostics**  
Residual autocorrelation suggests misspecification.

**5. Blindly following automatic procedures**  
Model selection should involve judgment and interpretation.
```

---

# 14.20 Looking Ahead

In this chapter, we extended ARMA models to handle nonstationary series through differencing.

We now move to forecasting and forecast evaluation, where we study:

- static vs dynamic forecasting
- forecast accuracy
- RMSE, MAE, MAPE
- Theil’s U statistics

# Key Takeaways

```{admonition} Summary
- ARIMA models extend ARMA models to nonstationary data
- Differencing removes stochastic trends
- ARIMA combines AR, differencing, and MA components
- Model identification follows the Box–Jenkins methodology
- Residual diagnostics are central to ARIMA modeling
- ARIMA models are widely used for forecasting
```