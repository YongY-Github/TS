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

These models assume that the underlying process is **stationary**.

However, many economic and financial time series are not stationary.

Examples include:

- stock prices  
- GDP  
- exchange rates  
- price indices  

These series often exhibit:

- trends  
- persistent drift  
- unit roots  

```{admonition} Central Problem
How can we model time series that are not stationary?
```

The solution is to **transform the data before modeling**.

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

```{admonition} Big Picture
ARIMA modeling follows a simple but powerful logic:

Nonstationary data  
→ difference the data  
→ obtain stationarity  
→ apply ARMA modeling  

This is the foundation of modern time series analysis.
```

---

# 14.3 Integrated Processes

```{admonition} Intuition

An $I(1)$ process behaves like a random walk:

- shocks accumulate over time  
- the series does not return to a fixed level  

Differencing removes this accumulation and restores stability.
```

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

```{admonition} Key Insight

- Under-differencing → residual nonstationarity  
- Over-differencing → introduces unnecessary noise  

Goal: difference just enough — no more, no less
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

plt.savefig("figs/ch14/acf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![ACF](figs/ch14/acf.png)

---

## Ljung–Box Test

```{code-cell} python
from statsmodels.stats.diagnostic import acorr_ljungbox

acorr_ljungbox(res.resid, lags=[10,20], return_df=True)
```

```
|  | lb_stat   | lb_pvalue |
|---------|------------|-----------|
| 10      | 4.729113   | 0.908524  |
| 20      | 25.169131  | 0.195037  |

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

# Concept Check

### Basic

1. What is an ARIMA model?

2. What does the “I” in ARIMA represent?

3. What does differencing do to a time series?

---

### Intuition

4. Why are many economic time series nonstationary?

5. Why is it problematic to apply ARMA models to nonstationary data?

6. What is the idea behind transforming data before modeling?

---

### Intermediate

7. What does it mean for a series to be:

   - $I(0)$  
   - $I(1)$  

8. What is the difference between first and second differencing?

9. Why is most real-world data $I(1)$ rather than $I(2)$?

---

### ARIMA Structure

10. What do $p$, $d$, and $q$ represent in ARIMA($p,d,q$)?

11. What happens after differencing is applied?

---

### Challenge

12. Suppose a series becomes stationary after differencing once.

   - What does this imply?

---

# Interpretation & Practice

1. A time series shows a strong upward trend.

   - What transformation might be needed?

2. After differencing, the series fluctuates around zero.

   - What does this suggest?

3. A series exhibits very slow ACF decay.

   - What does this indicate?

4. After differencing, ACF shows AR-type behavior.

   - What does this suggest?

5. A series still appears nonstationary after differencing once.

   - What might you do next?

---

### Finance Interpretation

6. Stock prices are nonstationary.

   - Why are returns preferred for modeling?

7. A return series appears stationary.

   - Why is this useful?

---

### Challenge

8. A model fits well but uses $d=2$.

   - Why might this be problematic?

---

# Model Selection (AIC & BIC)

1. Suppose you estimate two ARIMA models:

| Model | AIC | BIC |
|---|---:|---:|
| ARIMA(1,1,1) | 520 | 540 |
| ARIMA(2,1,2) | 510 | 560 |

---

- Which model is preferred according to AIC?
- Which model is preferred according to BIC?
- Why might these criteria disagree?

---

2. Suppose you estimate:

| Model | AIC | BIC |
|---|---:|---:|
| ARIMA(1,1,0) | 600 | 610 |
| ARIMA(3,1,2) | 590 | 640 |

---

- Which model has better fit?
- Which model penalizes complexity more?
- Which would you choose, and why?

---

3. Explain the intuition behind:

```{math}
:enumerated: false
AIC = -2\log L + 2k
```

```{math}
:enumerated: false
BIC = -2\log L + k \log n
```

---

- What does the first term measure?
- What does the second term penalize?

---

4. Why does BIC typically select simpler models than AIC?

---

### Interpretation

5. A model has very low AIC but performs poorly out-of-sample.

- What might be happening?
- Why is model validation important?

---

### Challenge

6. Suppose you keep adding lags to improve fit.

- What happens to AIC?
- What happens to BIC?
- Why is this important for model selection?

---

# Numerical Practice

### Differencing

1. Given:

```{math}
:enumerated: false
x_t = 100, 105, 111, 118
```

- Compute $\Delta x_t$

---

2. Compute second differences.

---

### Identification

3. Suppose:

- ACF decays slowly  
- series trends  

- What transformation is needed?

---

4. Suppose after differencing:

- ACF cuts off after lag 1  

- What model is suggested?

---

### Model Structure

5. Interpret:

```{math}
:enumerated: false
ARIMA(1,1,1)
```

- What is being modeled?

---

### Diagnostics

6. Residuals still show autocorrelation.

- What does this imply?

---

### Challenge

7. Suppose you over-difference a series.

- What happens?
- Why is this problematic?

---

# Appendix 14A — Understanding Differencing and Integration

## A.1 First Difference

```{math}
:enumerated: false
\Delta x_t = x_t - x_{t-1}
```

This removes linear stochastic trends.

---

## A.2 Random Walk Example

```{math}
:enumerated: false
x_t = x_{t-1} + w_t
```

Then:

```{math}
:enumerated: false
\Delta x_t = w_t
```

```{admonition} Insight
Differencing removes accumulated shocks.
```

---

## A.3 Second Difference

```{math}
:enumerated: false
\Delta^2 x_t = x_t - 2x_{t-1} + x_{t-2}
```

Used for stronger nonstationarity.

---

## A.4 Why Differencing Works

Nonstationary series accumulate shocks:

```{math}
:enumerated: false
x_t = \sum w_t
```

Differencing removes this accumulation:

```{math}
:enumerated: false
\Delta x_t = w_t
```

---

## A.5 Practical Interpretation

```{admonition} Interpretation

- Levels → long-term evolution  
- Differences → short-term changes  

In finance:

- prices → nonstationary  
- returns → stationary  
```