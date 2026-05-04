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

This distinction is extremely important:

- AR models describe persistence through past values  
- MA models describe dependence through past disturbances  

Together, AR and MA models form the foundation of Box–Jenkins time series analysis.

---

# Learning Objectives

By the end of this chapter, you should be able to:

- distinguish MA models from moving-average smoothing  
- understand the intuition behind MA($q$) processes  
- derive the mean and variance of MA(1)  
- understand the autocorrelation structure  
- interpret ACF and PACF patterns  
- understand invertibility intuitively  
- estimate MA models  

---

# 12.1 Moving Average Smoothing vs MA(q) Models

Earlier, we studied moving average smoothing:

```{math}
:enumerated: false
\tilde{x}_t = \frac{1}{3}(x_{t-1}+x_t+x_{t+1})
```

This is a filtering method.

```{admonition} Important
Moving average smoothing is NOT the same as a moving average stochastic process.
```

In MA models, the current value is driven by:

- current shocks  
- past shocks  

---

# 12.2 The Basic Idea

MA models capture how shocks affect the system for several periods.

Examples:

- policy shocks affecting output  
- financial news impacting markets  
- supply disruptions affecting inflation  

```{admonition} Intuition
Shocks have temporary but possibly multi-period effects.
```

```{admonition} Shock Propagation (Key Insight)

Suppose a shock occurs at time $t$.

In an MA(1):

- it affects $x_t$ directly  
- it affects $x_{t+1}$ through $\theta w_t$  
- after that, its effect disappears  

Shock → short-lived ripple → gone
```

---

# 12.3 The MA(1) Model

```{admonition} Definition
An MA(1) process is:

$$
x_t = w_t + \theta w_{t-1}
$$
```

where:

- $w_t \sim wn(0,\sigma^2)$  
- $\theta$ controls the effect of past shocks  

---

# 12.4 Intuition Behind MA(1)

Unlike AR models:

- MA models do NOT depend on past values  
- they depend on past disturbances  

```{admonition} Key Insight
MA models have finite memory — shocks eventually disappear completely.
```

---

# 12.5 Simulating MA(1)

```{code-cell} python
:tags: [hide-input]

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
        x[t] = w[t] + theta*w[t-1]

    ax[i].plot(x)
    ax[i].set_title(rf"MA(1), $\theta={theta}$")

plt.tight_layout()
plt.savefig("figs/ch12/ma1.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![MA1](figs/ch12/ma1.png)

```{admonition} Observation
Different values of $\theta$ produce distinct dependence patterns.
```

---

# 12.6 Mean

```{math}
:enumerated: false
E[x_t] = 0
```

(assuming zero mean shocks)

---

# 12.7 Variance

```{math}
:enumerated: false
Var(x_t) = (1+\theta^2)\sigma_w^2
```

---

# 12.8 Autocovariance

```{math}
:enumerated: false
\gamma(1) = \theta \sigma_w^2
```

```{math}
:enumerated: false
\gamma(h) = 0 \quad \text{for } h>1
```

---

# 12.9 Autocorrelation Function (ACF)

```{math}
:enumerated: false
\rho(1) = \frac{\theta}{1+\theta^2}
```

```{math}
:enumerated: false
\rho(h) = 0 \quad \text{for } h>1
```

```{admonition} Key Result
The ACF of MA(1) cuts off after lag 1.
```

## Simulation

```{code-cell} python
:tags: [hide-input]

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(x, lags=20)

plt.savefig("figs/ch12/acf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![ACF](figs/ch12/acf.png)

---

# 12.10 Partial Autocorrelation (PACF)


```{code-cell} python
:tags: [hide-input]

from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(x, lags=20)

plt.savefig("figs/ch12/pacf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![PACF](figs/ch12/pacf.png)


```{admonition} Fundamental Result

For MA($q$):

- ACF cuts off after lag $q$  
- PACF tails off gradually  
```

```{admonition} Intuition

In MA models, dependence is driven by a finite number of shocks.

However, indirect relationships across time still exist, causing the PACF to decay rather than cut off sharply.
```

---

# 12.11 MA(2)

```{math}
:enumerated: false
x_t = w_t + \theta_1 w_{t-1} + \theta_2 w_{t-2}
```

```{admonition} Intuition
Shocks influence the process for a limited number of periods.
```

---

# 12.12 General MA(q)

```{math}
:enumerated: false
x_t = w_t + \theta_1 w_{t-1} + \cdots + \theta_q w_{t-q}
```

```{admonition} Key Insight
MA models have finite memory.
```

---

# 12.13 Invertibility

```{admonition} Condition
An MA(1) is invertible if:

$$
|\theta| < 1
$$
```

```{admonition} Intuition

Invertibility ensures that the model can be uniquely represented.

Without it:

- different parameter values produce identical behavior  
- estimation becomes ambiguous  

This plays a role similar to stationarity in AR models.
```

---

# 12.14 AR vs MA

| Feature | AR | MA |
|---|---|---|
| Depends on | past values | past shocks |
| Memory | infinite | finite |
| ACF | tails off | cuts off |
| PACF | cuts off | tails off |

---

# 12.15 Estimation

```{code-cell} python
:tags: [hide-input]

import statsmodels.api as sm

model = sm.tsa.ARIMA(x, order=(0,0,1))
res = model.fit()

print(res.summary())
```

``` verbatim
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                  400
Model:                 ARIMA(0, 0, 1)   Log Likelihood                -565.364
Date:                Mon, 04 May 2026   AIC                           1136.727
Time:                        21:54:51   BIC                           1148.702
Sample:                             0   HQIC                          1141.469
                                - 400                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -0.0099      0.010     -0.954      0.340      -0.030       0.010
ma.L1         -0.7942      0.032    -24.900      0.000      -0.857      -0.732
sigma2         0.9865      0.070     14.130      0.000       0.850       1.123
===================================================================================
Ljung-Box (L1) (Q):                   0.06   Jarque-Bera (JB):                 0.29
Prob(Q):                              0.80   Prob(JB):                         0.86
Heteroskedasticity (H):               0.69   Skew:                             0.07
Prob(H) (two-sided):                  0.03   Kurtosis:                         3.00
===================================================================================
```

---

# 12.16 Diagnostics

```{code-cell} python
:tags: [hide-input]

from statsmodels.stats.diagnostic import acorr_ljungbox

acorr_ljungbox(res.resid, lags=[10,20], return_df=True)
```

``` verbatim
| |  lb_stat   | lb_pvalue |
|---------|------------|-----------|
| 10      | 4.539711   | 0.919734  |
| 20      | 23.589777  | 0.260771  |

```

```{admonition} Goal
Residuals should resemble white noise.
```

---

# 12.17 Common Mistakes

```{admonition} Warning

- Confusing smoothing with stochastic MA  
- Overinterpreting ACF cutoffs  
- Ignoring invertibility  
- Ignoring diagnostics  
```

---

# 12.18 Looking Ahead

Next, we combine AR and MA models into ARMA models.

---

# Key Takeaways

```{admonition} Summary

- MA models depend on past shocks  
- MA models have finite memory  
- ACF cuts off  
- PACF tails off  
- Invertibility ensures uniqueness  
```

# Concept Check

### Basic

1. What is a moving average (MA) model?

2. What is the key difference between:

   - moving average smoothing  
   - MA($q$) stochastic models  

---

### Intuition

3. In an MA(1) model, how long does a shock affect the system?

4. What does it mean for an MA process to have **finite memory**?

5. How does the parameter $\theta$ affect the behavior of the series?

---

### Intermediate

6. Why do MA models depend on past shocks rather than past values?

7. What happens to the effect of a shock after $q$ periods in an MA($q$) model?

8. What is invertibility in MA models?

---

### ACF & PACF

9. What pattern does the ACF of an MA(1) process exhibit?

10. What pattern does the PACF of an MA(1) process exhibit?

11. Why does the ACF “cut off” for MA models?

---

### Finance Insight

12. Give an example of a real-world situation where shocks have temporary effects.

13. Why might MA models be useful for modeling short-term market reactions?

---

### Challenge

14. Suppose $\theta$ is very large (in absolute value).

   - What might happen to the variability of the series?
   - Why might invertibility become important?

---

# Interpretation & Practice

1. A time series reacts strongly to shocks but quickly stabilizes.

   - What type of model might describe this?
   - Why?

---

2. ACF shows:

- large spike at lag 1  
- near zero afterward  

   - What model is suggested?

---

3. PACF shows gradual decay.

   - What type of model might this indicate?

---

4. ACF does NOT cut off sharply.

   - Why might this happen in practice even if the true model is MA(1)?

---

5. A series shows:

- temporary reaction to shocks  
- no long-term persistence  

   - Is this more consistent with AR or MA behavior?

---

### Finance Interpretation

6. A stock reacts to news for one or two days and then stabilizes.

   - What type of model could describe this?
   - Why?

---

7. A volatility series shows long-lasting effects of shocks.

   - Would MA models be sufficient?
   - Why or why not?

---

### Challenge

8. A model fits the ACF pattern well but produces poor forecasts.

   - What might be missing?
   - Why might combining AR and MA be useful?

---

# Numerical Practice

### MA(1) Construction

1. Consider:

```{math}
:enumerated: false
x_t = w_t + 0.5 w_{t-1}
```

with shocks:

```{math}
:enumerated: false
w_t = 2, -1, 3
```

---

- Compute $x_1, x_2, x_3$

---

### Comparing Effects

2. Repeat with:

```{math}
:enumerated: false
x_t = w_t + 0.9 w_{t-1}
```

- Compare the magnitude of fluctuations  
- What changes?

---

### Finite Memory

3. Suppose a shock of +5 occurs at time $t$ in an MA(1).

- What is its effect on:
  - $x_t$  
  - $x_{t+1}$  
  - $x_{t+2}$  

---

### ACF Interpretation

4. Suppose you observe:

- $\rho(1) = 0.4$  
- $\rho(2) \approx 0$  
- $\rho(3) \approx 0$  

---

- What model is suggested?

---

### Model Identification

5. You observe:

- ACF cuts off after lag 2  
- PACF decays gradually  

---

- What model is suggested?

---

### Estimation Output

6. Suppose an estimated MA(1) model is:

```{math}
:enumerated: false
x_t = w_t + 0.7 w_{t-1}
```

---

- Is the series highly influenced by past shocks?
- How long do shocks persist?

---

### Diagnostics

7. After estimating an MA model, residuals show:

- significant autocorrelation  

---

- What does this imply?
- What should you do next?

---

### Challenge

8. Suppose $\theta = 1.2$.

- Does this violate invertibility?
- Why might this be problematic?

---

9. Suppose you incorrectly use an MA(1) model when the true process is AR(1).

- What patterns might appear in the residuals?

---

# Appendix 12A — Mathematical Details of MA Models

This appendix provides additional insight into the structure of moving average (MA) models.

The goal is to explain *why* the key results hold, not just state them.

---

## A.1 The MA(1) Process

Consider:

```{math}
:enumerated: false
x_t = w_t + \theta w_{t-1}
```

where $w_t \sim wn(0,\sigma^2)$.

---

## A.2 Mean

Taking expectations:

```{math}
:enumerated: false
E[x_t] = E[w_t] + \theta E[w_{t-1}] = 0
```

So:

```{math}
:enumerated: false
E[x_t] = 0
```

---

## A.3 Variance

```{math}
:enumerated: false
Var(x_t) = Var(w_t + \theta w_{t-1})
```

Because shocks are independent:

```{math}
:enumerated: false
Var(x_t) = Var(w_t) + \theta^2 Var(w_{t-1})
```

```{math}
:enumerated: false
= (1 + \theta^2)\sigma^2
```

```{admonition} Insight
Variance increases with the magnitude of $\theta$, reflecting stronger influence of past shocks.
```

---

## A.4 Autocovariance

We compute:

```{math}
:enumerated: false
\gamma(h) = Cov(x_t, x_{t-h})
```

---

### Lag 1

```{math}
:enumerated: false
x_t = w_t + \theta w_{t-1}
```

```{math}
:enumerated: false
x_{t-1} = w_{t-1} + \theta w_{t-2}
```

Only the shared term \( w_{t-1} \) contributes:

```{math}
:enumerated: false
\gamma(1) = Cov(\theta w_{t-1}, w_{t-1}) = \theta \sigma^2
```

---

### Lag 2

```{math}
:enumerated: false
x_t = w_t + \theta w_{t-1}
```

```{math}
:enumerated: false
x_{t-2} = w_{t-2} + \theta w_{t-3}
```

No overlapping shocks:

```{math}
:enumerated: false
\gamma(2) = 0
```

---

### General Result

```{math}
:enumerated: false
\gamma(h) =
\begin{cases}
(1+\theta^2)\sigma^2 & h = 0 \\
\theta \sigma^2 & h = 1 \\
0 & h > 1
\end{cases}
```

---

## A.5 Autocorrelation Function (ACF)

```{math}
:enumerated: false
\rho(h) = \frac{\gamma(h)}{\gamma(0)}
```

So:

```{math}
:enumerated: false
\rho(1) = \frac{\theta}{1+\theta^2}
```

```{math}
:enumerated: false
\rho(h) = 0 \quad \text{for } h>1
```

```{admonition} Key Insight
The ACF cuts off because shocks do not overlap beyond lag 1.
```

---

## A.6 Why the ACF Cuts Off

This is the defining feature of MA models.

At lag 1:

- both $x_t$ and $x_{t-1}$ contain $w_{t-1}$

At lag 2:

- no common shocks exist  

```{admonition} Intuition
Autocorrelation requires shared shocks.

Once shocks no longer overlap, correlation disappears.
```

---

## A.7 Invertibility (Deeper Insight)

Consider:

```{math}
:enumerated: false
x_t = w_t + \theta w_{t-1}
```

We can rearrange:

```{math}
:enumerated: false
w_t = x_t - \theta w_{t-1}
```

Substituting repeatedly:

```{math}
:enumerated: false
w_t = x_t - \theta x_{t-1} + \theta^2 x_{t-2} - \cdots
```

Thus:

```{math}
:enumerated: false
x_t = \sum_{j=0}^{\infty} (-\theta)^j x_{t-j} + w_t
```

This expresses the MA process as an **infinite AR process**.

---

### Condition

For convergence:

```{math}
:enumerated: false
|\theta| < 1
```

```{admonition} Key Insight
Invertibility ensures that the MA model can be uniquely expressed in terms of past observations.
```

---

## A.8 Why Invertibility Matters

Without invertibility:

- multiple values of $\theta$ produce the same ACF  
- parameters cannot be uniquely identified  

```{admonition} Big Picture

AR models need **stationarity** for stability.

MA models need **invertibility** for uniqueness.

```

---

## A.9 General MA(q)

```{math}
:enumerated: false
x_t = w_t + \theta_1 w_{t-1} + \cdots + \theta_q w_{t-q}
```

Key properties:

- ACF cuts off after lag $q$  
- PACF decays gradually  

```{admonition} Insight
Finite dependence in shocks leads to finite autocorrelation.
```