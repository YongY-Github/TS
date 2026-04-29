---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 12 — Moving Average Models

In the previous chapter, we studied autoregressive (AR) models, where the current value depends on past values of the series itself.

We now turn to another important class of time series models:

```{admonition} Central Idea
In moving average (MA) models, the present depends on past shocks.
```

This distinction is extremely important.

- AR models describe persistence through past values
- MA models describe persistence through past disturbances

Together, AR and MA models form the foundation of Box–Jenkins time series analysis.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- distinguish MA models from moving-average smoothing
- understand the intuition behind MA($q$) processes
- derive the mean and variance of MA(1)
- derive the autocorrelation structure of MA models
- understand invertibility intuitively
- interpret ACF and PACF patterns
- estimate MA models in Gretl

---

# 12.1 Moving Average Smoothing vs MA(q) Models

Before introducing MA($q$) models, we must clarify an important distinction.

Earlier in the book, we studied **moving average smoothing**, where nearby observations are averaged to reduce noise.

For example:

```{math}
:enumerated: false
\tilde{x}_t
=
\frac{1}{3}(x_{t-1}+x_t+x_{t+1})
```

This is a smoothing method.

```{admonition} Important
Moving average smoothing is NOT the same thing as a moving average stochastic process.
```

In contrast, a moving average time series model expresses the current value as a combination of:

- current shocks
- past shocks

These are entirely different concepts, despite the similar terminology.

---

# 12.2 The Basic Idea

Suppose shocks influence the process for more than one period.

For example:

- a policy shock may affect GDP growth for several quarters
- news may affect financial markets temporarily
- supply disruptions may influence inflation for several months

MA models capture this temporary propagation of shocks.

```{admonition} Intuition
In MA models, shocks have temporary but possibly multi-period effects.
```

---

# 12.3 The MA(1) Model

```{admonition} Definition
A moving average process of order 1, denoted MA(1), is:

```{math}
x_t
=
\theta w_{t-1}
+
w_t
```

where:

- $w_t \sim wn(0,\sigma_w^2)$
- $\theta$ measures the influence of the previous shock

Unlike AR models:

- MA models do NOT depend on past values of $x_t$
- they depend on past disturbances

---

# 12.4 Intuition Behind MA(1)

Suppose a positive shock occurs today.

Then:

- the shock affects today's observation directly
- part of the shock also affects tomorrow through $\theta w_t$
- after that, the effect disappears

```{admonition} Key Insight
MA models have finite memory.

Shocks eventually disappear completely.
```

---

# 12.5 Simulating MA(1)

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

n = 400

w = np.random.normal(size=n)

theta_values = [0.3, 0.8, -0.8]

fig, ax = plt.subplots(3,1, figsize=(10,8))

for i, theta in enumerate(theta_values):

    x = np.zeros(n)

    for t in range(1,n):
        x[t] = theta*w[t-1] + w[t]

    ax[i].plot(x, lw=1)
    ax[i].set_title(rf"MA(1): $\theta={theta}$")

plt.tight_layout()

plt.savefig("figs/ch12/MA1.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![MA 1](figs/ch12/MA1.png)

```{admonition} Observation
Positive and negative values of $\theta$ produce different dependence patterns.
```

---

# 12.6 Mean of MA(1)

Taking expectations:

```{math}
:enumerated: false
E[x_t]
=
\theta E[w_{t-1}]
+
E[w_t]
```

Since white noise has mean zero:

```{math}
:enumerated: false
E[x_t] = 0
```

```{admonition} Result
The mean of an MA(1) process is zero.
```

(assuming no constant term)

---

# 12.7 Variance of MA(1)

Starting from:

```{math}
:enumerated: false
x_t
=
\theta w_{t-1}
+
w_t
```

take variances:

```{math}
:enumerated: false
Var(x_t)
=
\theta^2 Var(w_{t-1})
+
Var(w_t)
```

since:

```{math}
:enumerated: false
Cov(w_t,w_{t-1})=0
```

Thus:

```{math}
:enumerated: false
Var(x_t)
=
(1+\theta^2)\sigma_w^2
```

```{admonition} Result
For MA(1):

$$
Var(x_t)
=
(1+\theta^2)\sigma_w^2
$$
```

---

# 12.8 Autocovariance Function

Recall:

```{math}
:enumerated: false
\gamma(h)
=
Cov(x_t,x_{t-h})
```

For MA(1):

```{math}
:enumerated: false
x_t = \theta w_{t-1} + w_t
```

and:

```{math}
:enumerated: false
x_{t-1}
=
\theta w_{t-2}
+
w_{t-1}
```

Thus:

```{math}
:enumerated: false
\gamma(1)
=
Cov(x_t,x_{t-1})
```

Only one term survives:

```{math}
:enumerated: false
\gamma(1)
=
\theta \sigma_w^2
```

For:

```{math}
:enumerated: false
h>1
```

there are no overlapping shocks, so:

```{math}
:enumerated: false
\gamma(h)=0
```

---

# 12.9 Autocorrelation Function (ACF)

Using:

```{math}
:enumerated: false
\rho(h)
=
\frac{\gamma(h)}{\gamma(0)}
```

we obtain:

```{math}
:enumerated: false
\rho(1)
=
\frac{\theta}{1+\theta^2}
```

and:

```{math}
:enumerated: false
\rho(h)=0
\quad \text{for } h>1
```

```{admonition} Key Result
The ACF of MA(1) cuts off after lag 1.
```

This is the mirror image of AR(1), where the ACF tails off gradually.

## Simulated ACF

```{code-cell} python
from statsmodels.graphics.tsaplots import plot_acf

np.random.seed(123)

n = 500
theta = 0.8

w = np.random.normal(size=n)

x = np.zeros(n)

for t in range(1,n):
    x[t] = w[t] + theta*w[t-1]

plot_acf(x, lags=20)

plt.savefig("figs/ch12/acf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![ACF](figs/ch12/acf.png)

```{admonition} Observation
The ACF becomes negligible after lag 1.
```

---

# 12.10 Partial Autocorrelation Function (PACF)

Unlike the ACF, the PACF of MA processes typically tails off gradually.


```{admonition} Fundamental Result
For MA($q$) models:

- ACF cuts off after lag $q$
- PACF tails off gradually
```

## Simulated PACF

```{code-cell} python
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(x, lags=20, method='ywm')

plt.savefig("figs/ch12/pacf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![PACF](figs/ch12/pacf.png)

```{admonition} Identification Rule
MA(1):

- ACF cuts off after lag 1
- PACF tails off gradually
```

---

# 12.11 The MA(2) Model

We can extend the model to include two lagged shocks:

```{math}
:enumerated: false
x_t
=
w_t
+
\theta_1 w_{t-1}
+
\theta_2 w_{t-2}
```

```{admonition} Intuition
MA(2) allows shocks to influence the process for three periods:

- today,
- next period,
- and one additional lag.
```

---

# 12.12 General MA(q) Models

```{admonition} Definition
An MA($q$) process is:

$$
x_t
=
w_t
+
\theta_1 w_{t-1}
+
\theta_2 w_{t-2}
+
\cdots
+
\theta_q w_{t-q}
$$
```

## Key Property

The autocorrelation function satisfies:

```{math}
:enumerated: false
\rho(h)=0
\quad \text{for } h>q
```

```{admonition} Key Insight
MA processes have finite memory because shocks affect the system for only a limited number of periods.
```

---

# 12.13 Invertibility

Different MA parameter values can sometimes generate identical autocorrelation structures.

To ensure uniqueness, we impose an invertibility condition.

```{admonition} Invertibility Condition
An MA(1) process is invertible if:

$$
|\theta|<1
$$
```

## Intuition

Invertibility ensures that the process can be represented uniquely in terms of past observations.

```{admonition} Practical Note
Invertibility for MA models plays a role similar to stationarity for AR models.
```

---

# 12.14 AR vs MA Models

| Feature | AR Model | MA Model |
|---|---|---|
| Depends on | past values | past shocks |
| Memory | potentially infinite | finite |
| ACF | tails off | cuts off |
| PACF | cuts off | tails off |

```{admonition} Important
Understanding the difference between AR and MA behavior is central to Box–Jenkins model identification.
```

---

# 12.15 Estimation in Gretl

## Menu

```text
Model → Time Series → ARIMA
```

## Basic Workflow

1. Examine stationarity
2. Inspect ACF/PACF
3. Choose tentative MA order
4. Estimate model
5. Check residual diagnostics

```markdown
[GRETL Screenshot Placeholder: MA model estimation]
```

```markdown
[GRETL Screenshot Placeholder: MA(1) output]
```

---

# 12.16 Residual Diagnostics

After fitting an MA model, residuals should resemble white noise.

## Residual ACF

```{code-cell} python
:tags: [hide-input]
import statsmodels.api as sm

model = sm.tsa.ARIMA(x, order=(0,0,1))
res = model.fit()

plot_acf(res.resid, lags=20)

plt.savefig("figs/ch12/res_acf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Residual ACF](figs/ch12/res_acf.png)

## Ljung–Box Test



## Ljung–Box Test

```{code-cell} python
from statsmodels.stats.diagnostic import acorr_ljungbox

lb = acorr_ljungbox(res.resid, lags=[10,20], return_df=True)
lb
```

| lag | lb_stat | lb_pvalue |
|---|---:|---:|
| 10 | 7.663762 | 0.661642 |
| 20 | 22.431194 | 0.317578 |

The large p-values suggest that we fail to reject the null hypothesis of no serial correlation.

```{admonition} Goal
A good MA model should leave little remaining autocorrelation in the residuals.
```

---

# 12.17 Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Confusing smoothing MA with stochastic MA models**  
These are entirely different concepts.

**2. Overinterpreting sample cutoffs**  
Sample ACFs are noisy and may not cut off perfectly.

**3. Ignoring invertibility**  
Noninvertible models may create identification problems.

**4. Relying mechanically on ACF/PACF**  
Economic interpretation still matters.

**5. Ignoring residual diagnostics**  
Residual autocorrelation suggests misspecification.
```

---

# 12.18 Looking Ahead

In this chapter, we studied moving average models, where dependence arises through past shocks.

In the next chapter, we combine:

- autoregressive dynamics
- moving average dynamics

into the more flexible ARMA framework.

# Key Takeaways

```{admonition} Summary
- MA models depend on past shocks
- MA processes have finite memory
- The ACF of MA($q$) cuts off after lag $q$
- The PACF tails off gradually
- Invertibility ensures unique representation
- AR and MA models exhibit fundamentally different dependence structures
```