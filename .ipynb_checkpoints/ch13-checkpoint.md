---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 13 — ARMA Models

In the previous chapters, we studied:

- autoregressive (AR) models  
- moving average (MA) models  

Each captures a different type of dependence:

- AR models depend on past values  
- MA models depend on past shocks  

In practice, many time series exhibit **both types simultaneously**.

This motivates the ARMA model.

```{admonition} Central Idea
ARMA models combine:

- persistence through past values  
- dependence through past shocks  
```

---

# Learning Objectives

By the end of this chapter, you should be able to:

- understand the motivation behind ARMA models  
- define ARMA($p,q$) processes  
- interpret AR and MA components jointly  
- understand stationarity and invertibility  
- interpret ACF and PACF behavior  
- estimate ARMA models  
- perform diagnostics  

---

# 13.1 Why Combine AR and MA?

AR models:
- capture persistence  

MA models:
- capture temporary shock propagation  

But real-world data often show both.

```{admonition} Intuition
ARMA models allow both the history of the series and the history of shocks to influence the present.
```

```{admonition} Big Picture

AR → gradual decay of shocks  
MA → short-lived shock effects  

ARMA → both mechanisms operate together  
```

---

# 13.2 The ARMA(1,1) Model

```{admonition} Definition

$$
x_t = \phi x_{t-1} + \theta w_{t-1} + w_t
$$
```

## Interpretation

The present depends on:

- past values (persistence)  
- current shocks  
- past shocks  

```{admonition} Key Insight
ARMA models blend persistence and shock propagation in a single framework.
```

---

# 13.3 Simulation

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

n = 400
phi = 0.7
theta = 0.5

w = np.random.normal(size=n)
x = np.zeros(n)

for t in range(1,n):
    x[t] = phi*x[t-1] + w[t] + theta*w[t-1]

plt.figure(figsize=(10,4))
plt.plot(x, lw=1)
plt.title(r"ARMA(1,1): $\phi=0.7,\ \theta=0.5$")

plt.savefig("figs/ch13/arma11.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![ARMA11](figs/ch13/arma11.png)


```{admonition} Observation
ARMA processes often look smoother and more realistic than pure AR or MA models.
```

---

# 13.4 Mean

Without a constant:

```{math}
E[x_t] = 0
```

---

# 13.5 Stationarity and Invertibility

ARMA requires:

### Stationarity (AR part)

```{math}
|\phi| < 1
```

### Invertibility (MA part)

```{math}
|\theta| < 1
```

```{admonition} Important
Both conditions must hold.
```

---

# 13.6 Backshift Form

```{math}
(1-\phi B)x_t = (1+\theta B)w_t
```

---

# 13.7 Infinite MA Representation

```{math}
x_t = \frac{\theta(B)}{\phi(B)} w_t
```

```{admonition} Intuition
A stationary ARMA model can be viewed as an infinite weighted sum of past shocks.
```

---

# 13.8 Infinite AR Representation

```{math}
w_t = \frac{\phi(B)}{\theta(B)} x_t
```

```{admonition} Intuition
Invertibility allows shocks to be reconstructed from observed data.
```

---

# 13.9 ACF and PACF Behavior

```{admonition} Key Property
For ARMA models:

- ACF tails off  
- PACF tails off  
```

## Why?

```{admonition} Intuition

AR part → creates persistence → slow decay  

MA part → adds short-run shock structure  

Together → no sharp cutoff patterns  
```

---

# 13.10 Simulated ACF/PACF

```{code-cell} python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(2,1, figsize=(8,6))

plot_acf(x, lags=30, ax=ax[0])
plot_pacf(x, lags=30, method='ywm', ax=ax[1])

plt.tight_layout()

plt.savefig("figs/ch13/arma11_acf_pacf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![ARMA11 ACF PACF](figs/ch13/arma11_acf_pacf.png)

---

# 13.11 AR vs MA vs ARMA

| Model | ACF | PACF |
|---|---|---|
| AR | tails off | cuts off |
| MA | cuts off | tails off |
| ARMA | tails off | tails off |

```{admonition} Insight
ARMA removes the clean identification patterns of pure AR or MA models.
```

---

# 13.12 Identification in Practice

Steps:

1. visualize data  
2. ensure stationarity  
3. inspect ACF/PACF  
4. estimate candidates  
5. check diagnostics  
6. compare models  

```{admonition} Reality
Model identification is partly statistical and partly judgment.
```

---

# 13.13 Estimation

```{code-cell} python
import statsmodels.api as sm

model = sm.tsa.ARIMA(x, order=(1,0,1))
res = model.fit()

print(res.summary())
```

``` verbatim
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                  400
Model:                 ARIMA(1, 0, 1)   Log Likelihood                -564.412
Date:                Mon, 04 May 2026   AIC                           1136.824
Time:                        22:05:59   BIC                           1152.790
Sample:                             0   HQIC                          1143.147
                                - 400                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -0.2646      0.248     -1.068      0.286      -0.750       0.221
ar.L1          0.6901      0.044     15.840      0.000       0.605       0.775
ma.L1          0.5423      0.049     11.081      0.000       0.446       0.638
sigma2         0.9803      0.069     14.148      0.000       0.845       1.116
===================================================================================
Ljung-Box (L1) (Q):                   0.07   Jarque-Bera (JB):                 0.11
Prob(Q):                              0.80   Prob(JB):                         0.95
Heteroskedasticity (H):               0.70   Skew:                             0.04
Prob(H) (two-sided):                  0.04   Kurtosis:                         3.01
===================================================================================
```

---

# 13.14 Diagnostics

```{code-cell} python
from statsmodels.stats.diagnostic import acorr_ljungbox

acorr_ljungbox(res.resid, lags=[10,20], return_df=True)
```

``` verbatim
|  | lb_stat   | lb_pvalue |
|---------|------------|-----------|
| 10      | 4.229114   | 0.936420  |
| 20      | 25.470103  | 0.184034  |

```

```{admonition} Goal
Residuals should resemble white noise.
```

---

# 13.15 Information Criteria

```{math}
AIC = -2\log L + 2k
```

```{math}
BIC = -2\log L + k\log n
```

---

# 13.16 Applications

ARMA models are used for:

- inflation  
- GDP growth  
- exchange rates  
- interest rates  
- demand forecasting  

---

# 13.17 Common Mistakes

```{admonition} Warning

- ignoring stationarity  
- overfitting  
- relying only on ACF/PACF  
- ignoring diagnostics  
- small sample issues  
```

---

# 13.18 Looking Ahead

Next:

```{admonition} Preview
ARIMA models extend ARMA to nonstationary series using differencing.
```

---

# Key Takeaways

```{admonition} Summary

- ARMA combines AR and MA dynamics  
- AR → persistence  
- MA → short-run shock effects  
- ACF & PACF both tail off  
- diagnostics are essential  
```

# Concept Check

### Basic

1. What is an ARMA model?

2. What are the two components of an ARMA model?

3. What does the AR component capture?

4. What does the MA component capture?

---

### Intuition

5. Why do many real-world time series require both AR and MA components?

6. How does an ARMA model differ from a pure AR or pure MA model?

7. What happens when both persistence and short-term shocks influence a series?

---

### Intermediate

8. What are the stationarity and invertibility conditions for an ARMA(1,1) model?

9. Why must both conditions hold?

10. What is the backshift representation of an ARMA model?

---

### ACF & PACF

11. What pattern does the ACF of an ARMA model exhibit?

12. What pattern does the PACF of an ARMA model exhibit?

13. Why do ARMA models not show sharp cutoffs in ACF or PACF?

---

### Interpretation

14. Why is model identification more difficult for ARMA models than for AR or MA models?

15. Why should ACF and PACF be used cautiously in ARMA identification?

---

### Challenge

16. Suppose a time series exhibits both:

- strong persistence  
- short-lived shock effects  

   - Why is ARMA a natural model?

---

# Interpretation & Practice

1. A time series shows:

- gradual decay in ACF  
- gradual decay in PACF  

   - What type of model might this suggest?
   - Why?

2. ACF and PACF do not show clear cutoff patterns.

   - Why might this happen?
   - What modeling approach would you take?

3. A series appears smoother than pure AR but still persistent.

   - What might this indicate?

4. A model captures persistence but leaves short-term fluctuations unexplained.

   - What component might be missing?

5. A model captures shocks well but fails to capture persistence.

   - What component might be missing?

---

### Finance Interpretation

6. A financial time series shows:

- short-term reaction to news  
- longer-term persistence  

   - Why might ARMA be appropriate?

7. A return series appears unpredictable but slightly autocorrelated.

   - What type of structure might exist?

---

### Diagnostics

8. After estimating an ARMA model, residuals still show autocorrelation.

   - What does this imply?
   - What should you do next?

---

### Challenge

9. Two ARMA models fit equally well visually.

   - How would you choose between them?

---

# Numerical Practice

### ARMA Construction

1. Consider:

```{math}
:enumerated: false
x_t = 0.5 x_{t-1} + w_t + 0.3 w_{t-1}
```

with:

- $x_0 = 0$  
- $w_t = 2, -1, 3$  

---

- Compute $x_1, x_2, x_3$

---

### Comparing Models

2. Compare:

- AR(1): $x_t = 0.7 x_{t-1} + w_t$  
- MA(1): $x_t = w_t + 0.7 w_{t-1}$  

---

- Which shows longer persistence?
- Which shows finite shock effects?

---

### ACF Interpretation

3. Suppose you observe:

- ACF decays gradually  
- PACF also decays gradually  

---

- What model class is suggested?

---

### Identification

4. You observe:

- ACF does not cut off  
- PACF does not cut off  

---

- Why is ARMA a candidate model?

---

### Estimation Output

5. Suppose:

```{math}
:enumerated: false
x_t = 0.8 x_{t-1} + w_t + 0.4 w_{t-1}
```

---

- Is the series persistent?
- Are shocks temporary or long-lasting?
- What does this imply about dynamics?

---

### Diagnostics

6. Suppose residuals show:

- significant autocorrelation  

---

- What does this imply?
- What should be done?

---

### Model Selection

7. Suppose two models produce:

- Model A: lower AIC  
- Model B: lower BIC  

---

- What does this suggest?
- How would you decide?

---

### Challenge

8. Suppose an ARMA model fits the data well but performs poorly in forecasting.

- What might be the issue?
- Why is validation important?

---

9. Suppose $\phi$ is close to 1 and $\theta$ is large.

- What type of behavior would you expect?
- Why might this resemble a near-nonstationary process?

---

10. A time series shows:

- moderate persistence  
- small but noticeable short-term fluctuations  
- no clear ACF cutoff  

What model would you try first? Why?

---

# Appendix 13A — Additional Insight

## A.1 Why ACF Tails Off

AR component generates:

- gradual decay in correlations  

MA component adds:

- short-term structure  

Combined → no cutoff  

---

## A.2 Infinite Representations

ARMA can be expressed as:

- infinite MA (if stationary)  
- infinite AR (if invertible)  

```{admonition} Insight
ARMA bridges the two worlds: value dependence and shock dependence.
```