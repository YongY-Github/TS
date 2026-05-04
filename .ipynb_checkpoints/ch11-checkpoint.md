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

```{admonition} Central Idea
The present is often related to the recent past.
```

Many economic and financial variables adjust gradually rather than instantaneously. Inflation, unemployment, interest rates, exchange rates, and GDP growth all tend to exhibit inertia or persistence.

Autoregressive models capture this gradual adjustment process mathematically.

```{admonition} Big Picture
AR models provide a structured way to model persistence:

- weak persistence → small $\phi$
- strong persistence → $\phi$ close to 1

They form the foundation for modern time series analysis.
```

---

# Learning Objectives

By the end of this chapter, you should be able to:

- understand the logic of autoregressive models
- define AR($p$) processes
- interpret persistence
- understand stationarity conditions
- interpret ACF and PACF patterns
- simulate AR models
- estimate simple AR models
- perform basic diagnostics

---

# 11.1 The Basic Idea

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
- If $\phi$ is close to zero → little dependence  
- If $\phi$ is large → strong persistence  
- If $\phi$ is negative → oscillation  
```

---

# 11.2 The AR(1) Model

```{admonition} Definition
An AR(1) process is:

$$
x_t = \phi x_{t-1} + \mu + w_t
$$
```

## Mean

```{math}
:enumerated: false
E[x_t] = \frac{\mu}{1-\phi}
```

---

# 11.3 Mean-Centered Form

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t
```

---

# 11.4 Recursive Representation (Key Insight)

By repeated substitution:

```{math}
:enumerated: false
x_t = \sum_{j=0}^{\infty} \phi^j w_{t-j}
```

```{admonition} Key Insight
An AR(1) is an infinite weighted sum of past shocks, with geometrically declining weights.
```

---

# 11.5 Stationarity

```{admonition} Condition
Stationary if:

$$
|\phi| < 1
$$
```

```{admonition} Intuition
Shocks must fade over time for the series to remain stable.
```

---

# 11.6 Simulation

```{code-cell} python
:tags: [hide-input]

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

    ax[i].plot(x)
    ax[i].set_title(f"AR(1), phi={phi}")

plt.tight_layout()

plt.savefig("figs/ch10/persistence.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Persistence](figs/ch10/persistence.png)

```{admonition} Observation
As $\phi \to 1$, persistence increases dramatically.
```

---

# 11.7 Variance

```{math}
:enumerated: false
Var(x_t) = \frac{\sigma_w^2}{1-\phi^2}
```

```{admonition} Interpretation
Variance grows rapidly as $\phi$ approaches 1.
```

---

# 11.8 ACF of AR(1)

```{math}
:enumerated: false
\rho(h) = \phi^h
```

```{admonition} Key Result
ACF decays geometrically.
```

---

# 11.9 Simulated ACF

```{code-cell} python
:tags: [hide-input]

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(x, lags=30)

plt.savefig("figs/ch10/acf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![ACF](figs/ch10/acf.png)


```{admonition} Observation
ACF tails off gradually.
```

---

# 11.10 PACF of AR(1)

```{admonition} Key Insight
PACF isolates direct effects.
```

```{admonition} Result
AR(1):

- PACF large at lag 1  
- near zero afterward  
```

```{admonition} Intuition
Only lag 1 directly affects $x_t$.
```

---

# 11.11 AR(2) Model

```{math}
:enumerated: false
x_t = \phi_1 x_{t-1} + \phi_2 x_{t-2} + w_t
```

```{admonition} Intuition
AR(2) allows:

- cycles  
- oscillations  
- richer dynamics  
```

---

# 11.12 AR(2) Simulation

```{code-cell} python
:tags: [hide-input]

np.random.seed(123)

phi1, phi2 = 1.0, -0.6
n = 1000

w = np.random.normal(size=n)
x = np.zeros(n)

for t in range(2,n):
    x[t] = phi1*x[t-1] + phi2*x[t-2] + w[t]

plt.plot(x)
plt.title("AR(2)")

plt.savefig("figs/ch10/ar2.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![AR2](figs/ch10/ar2.png)


---

# 11.13 ACF and PACF for AR(2)

```{code-cell} python
:tags: [hide-input]

from statsmodels.graphics.tsaplots import plot_pacf

plot_acf(x, lags=30)
plot_pacf(x, lags=30)

plt.savefig("figs/ch11/ar2_acf_pacf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![AR2 acf pacf](figs/ch11/ar2_acf_pacf.png)


```{admonition} Identification Rule
AR(2):

- ACF tails off  
- PACF cuts off at lag 2  
```

---

# 11.14 General AR(p)

```{math}
:enumerated: false
x_t = \sum_{i=1}^p \phi_i x_{t-i} + w_t
```

---

# 11.15 Estimation in Python

```{code-cell} python
:tags: [hide-input]

from statsmodels.tsa.ar_model import AutoReg

model = AutoReg(x, lags=2)
res = model.fit()

print(res.summary())
```

```verbatim
                            AutoReg Model Results                             
==============================================================================
Dep. Variable:                      y   No. Observations:                 1000
Model:                     AutoReg(2)   Log Likelihood               -1416.520
Method:               Conditional MLE   S.D. of innovations              1.000
Date:                Mon, 04 May 2026   AIC                           2841.040
Time:                        21:40:24   BIC                           2860.663
Sample:                             2   HQIC                          2848.499
                                 1000                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -0.0386      0.032     -1.218      0.223      -0.101       0.024
y.L1           1.0000      0.026     38.890      0.000       0.950       1.050
y.L2          -0.5850      0.026    -22.746      0.000      -0.635      -0.535
                                    Roots                                    
=============================================================================
                  Real          Imaginary           Modulus         Frequency
-----------------------------------------------------------------------------
AR.1            0.8547           -0.9894j            1.3074           -0.1366
AR.2            0.8547           +0.9894j            1.3074            0.1366
-----------------------------------------------------------------------------
```

---

# 11.16 Residual Diagnostics

```{code-cell} python
:tags: [hide-input]

from statsmodels.stats.diagnostic import acorr_ljungbox

lb = acorr_ljungbox(res.resid, lags=[10,20], return_df=True)
lb
```

``` verbatim
|  | lb_stat   | lb_pvalue |
|---------|------------|-----------|
| 10      | 6.476702   | 0.773750  |
| 20      | 18.268325  | 0.569737  |

```

```{admonition} Goal
Residuals should resemble white noise.
```

---

# 11.17 Information Criteria

```{math}
:enumerated: false
AIC = -2\log L + 2k
```

```{math}
:enumerated: false
BIC = -2\log L + k\log n
```

---

# 11.18 Common Mistakes

```{admonition} Warning

- Ignoring stationarity  
- Too many lags  
- Ignoring diagnostics  
- Confusing persistence with trend  
```

---

# 11.19 Looking Ahead

Next:

- MA models  
- ARMA models  
- full modeling workflow  

---

# Key Takeaways

```{admonition} Summary

- AR models capture persistence  
- Stationarity requires $|\phi| < 1$  
- ACF tails off  
- PACF cuts off  
- AR models are foundational  
```

# Concept Check

## Basic

1. What is an autoregressive (AR) model?

2. What does the parameter $\phi$ represent in an AR(1) model?

3. What is the role of the error term $w_t$?

---

## Intuition

4. What does it mean for a time series to exhibit persistence?

5. How does the value of $\phi$ affect the behavior of the series?

6. What happens when $\phi$ is:

   - close to zero  
   - close to one  
   - negative  

---

## Intermediate

7. What is the stationarity condition for an AR(1) model?

8. Why must shocks decay over time in a stationary process?

9. What is the difference between:

   - AR(1)  
   - AR(2)  

---

## ACF & PACF

10. What pattern does the ACF of an AR(1) process exhibit?

11. What pattern does the PACF of an AR(1) process exhibit?

12. Why does the PACF “cut off” for an AR process?

---

## Finance Insight

13. Why are AR models useful in modeling economic or financial time series?

14. Why is strong persistence sometimes mistaken for a trend?

---

## Challenge

15. Suppose $\phi = 0.98$.

   - Is the process stationary?
   - How would it behave in practice?

---

# Interpretation & Practice

1. A time series shows gradual adjustment after shocks.

   - What type of model might describe this?
   - Why?

2. A series appears highly persistent but does not trend upward.

   - What does this suggest about $\phi$?
   - Is the series likely stationary?

3. ACF decays slowly and smoothly.

   - What type of process might this indicate?

4. PACF shows a large spike at lag 1 and near zero afterward.

   - What model is suggested?

5. ACF shows a wavy, oscillating pattern.

   - What type of model might generate this?

6. After fitting an AR model, residuals still show autocorrelation.

   - What does this imply?
   - What should you do?

---

## Finance Interpretation

7. A return series shows small but positive autocorrelation.

   - What does this imply about predictability?
   - Is this consistent with market efficiency?

8. A volatility series shows strong persistence.

   - Why might an AR-type structure be useful?

---

### Challenge

9. A model fits the data well but produces poor forecasts.

   - What might be wrong?
   - What does this suggest about overfitting?

---

# Numerical Practice

## AR(1) Simulation Logic

1. Consider the AR(1) process:

```{math}
:enumerated: false
x_t = 0.5 x_{t-1} + w_t
```

with:

- $x_0 = 10$
- $w_t = 2, -1, 3$

---

- Compute $x_1, x_2, x_3$

---

2. Repeat with:

```{math}
:enumerated: false
x_t = 0.9 x_{t-1} + w_t
```

- Compare results  
- What changes?

---

## Persistence

3. Suppose:

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t
```

---

- If $\phi = 0.2$, how quickly do shocks disappear?  
- If $\phi = 0.9$, how quickly do shocks disappear?

---

### ACF Interpretation

4. Suppose you observe:

- $\rho(1) = 0.8$  
- $\rho(2) = 0.64$  
- $\rho(3) = 0.51$  

---

- What pattern is this?
- What does it suggest about $\phi$?

---

## Model Identification

5. You observe:

- ACF gradually decays  
- PACF cuts off after lag 2  

---

- What model is suggested?

---

## Estimation Output

6. Suppose an estimated AR(1) model gives:

```{math}
:enumerated: false
x_t = 0.85 x_{t-1} + w_t
```

---

- Is the series highly persistent?
- Is it stationary?
- What does this imply for forecasting?

---

### Diagnostics

7. Suppose the residuals from an AR model show:

- significant autocorrelation  

---

- What does this imply?
- What should you do next?

---

### Challenge

8. Suppose $\phi > 1$.

- What happens to the process?
- Why is this problematic?

---

9. Suppose you include too many lags in an AR model.

- What is the risk?
- How can information criteria help?

---

# Appendix 11A — Mathematical Details of AR Models

This appendix provides additional insight into the properties of autoregressive models.

These results are not required for basic understanding, but they help explain *why* AR models behave the way they do.

---

## A.1 Recursive Representation of AR(1)

Consider:

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t
```

Substitute repeatedly:

```{math}
:enumerated: false
x_t = \phi(\phi x_{t-2} + w_{t-1}) + w_t
= \phi^2 x_{t-2} + \phi w_{t-1} + w_t
```

Continuing:

```{math}
:enumerated: false
x_t = \sum_{j=0}^{\infty} \phi^j w_{t-j}
```

```{admonition} Key Insight
An AR(1) process is an infinite weighted sum of past shocks.
```

---

## A.2 Stationarity Condition

From the recursive form:

```{math}
:enumerated: false
x_t = \sum_{j=0}^{\infty} \phi^j w_{t-j}
```

For this to converge, we require:

```{math}
:enumerated: false
|\phi| < 1
```

```{admonition} Interpretation
If $|\phi| \geq 1$, past shocks do not decay, and the process becomes unstable.
```

---

## A.3 Mean of AR(1)

Consider:

```{math}
:enumerated: false
x_t = \phi x_{t-1} + \mu + w_t
```

Take expectations:

```{math}
:enumerated: false
E[x_t] = \phi E[x_{t-1}] + \mu
```

In equilibrium:

```{math}
:enumerated: false
E[x_t] = E[x_{t-1}] = \bar{x}
```

So:

```{math}
:enumerated: false
\bar{x} = \phi \bar{x} + \mu
```

```{math}
:enumerated: false
\bar{x} = \frac{\mu}{1-\phi}
```

---

## A.4 Variance of AR(1)

From:

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t
```

Take variance:

```{math}
:enumerated: false
Var(x_t) = \phi^2 Var(x_{t-1}) + \sigma_w^2
```

In steady state:

```{math}
:enumerated: false
Var(x_t) = \sigma_x^2
```

So:

```{math}
:enumerated: false
\sigma_x^2 = \phi^2 \sigma_x^2 + \sigma_w^2
```

```{math}
:enumerated: false
\sigma_x^2 = \frac{\sigma_w^2}{1-\phi^2}
```

```{admonition} Insight
Variance increases as $\phi$ approaches 1, reflecting stronger persistence.
```

---

## A.5 Autocovariance Function

We define:

```{math}
:enumerated: false
\gamma(h) = Cov(x_t, x_{t-h})
```

Multiply the AR(1) equation by $x_{t-h}$:

```{math}
:enumerated: false
Cov(x_t, x_{t-h}) = \phi Cov(x_{t-1}, x_{t-h})
```

Thus:

```{math}
:enumerated: false
\gamma(h) = \phi \gamma(h-1)
```

Iterating:

```{math}
:enumerated: false
\gamma(h) = \phi^h \gamma(0)
```

---

## A.6 Autocorrelation Function

Since:

```{math}
:enumerated: false
\rho(h) = \frac{\gamma(h)}{\gamma(0)}
```

we obtain:

```{math}
:enumerated: false
\rho(h) = \phi^h
```

```{admonition} Key Result
The ACF of an AR(1) decays geometrically.
```

---

## A.7 Yule–Walker Equation (AR(1))

At lag 0:

```{math}
:enumerated: false
\gamma(0) = \phi^2 \gamma(0) + \sigma_w^2
```

Rearranging:

```{math}
:enumerated: false
\gamma(0) (1 - \phi^2) = \sigma_w^2
```

This gives the variance result above.

---

## A.8 AR(2): Intuition and Dynamics

Consider:

```{math}
:enumerated: false
x_t = \phi_1 x_{t-1} + \phi_2 x_{t-2} + w_t
```

Unlike AR(1), this process can generate:

- oscillations  
- cycles  
- damped fluctuations  

```{admonition} Insight
The interaction between $\phi_1$ and $\phi_2$ determines whether the series:

- converges smoothly  
- oscillates  
- becomes unstable  
```

---

## A.9 Characteristic Equation

For AR(p):

```{math}
:enumerated: false
x_t = \phi_1 x_{t-1} + \cdots + \phi_p x_{t-p} + w_t
```

We define the characteristic equation:

```{math}
:enumerated: false
1 - \phi_1 z - \phi_2 z^2 - \cdots - \phi_p z^p = 0
```

````{admonition} Stationarity Condition
All roots must lie outside the unit circle:

```{math}
:enumerated: false
|z| > 1
```
````

---

## A.10 Why This Matters

```{admonition} Big Picture

These results explain:

- why AR models are stable when $|\phi| < 1$  
- why persistence creates smooth dynamics  
- why ACF patterns emerge  
- why PACF helps identify model order  

Understanding these foundations helps avoid treating AR models as “black boxes.”
```