---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 13 — ARMA Models

In the previous chapters, we studied:

- autoregressive (AR) models
- moving average (MA) models

Each captures a different type of dependence structure:

- AR models depend on past values
- MA models depend on past shocks

In practice, many time series exhibit both types of behavior simultaneously.

This motivates the **ARMA model**.

```{admonition} Central Idea
ARMA models combine:

- persistence through past values
- dependence through past shocks
```

ARMA models are among the most important tools in classical time series analysis and form the core of the Box–Jenkins methodology.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- understand the motivation behind ARMA models
- define ARMA($p,q$) processes
- interpret AR and MA components jointly
- understand stationarity and invertibility conditions
- interpret ACF and PACF behavior
- identify ARMA patterns visually
- estimate ARMA models in Gretl

---

# 13.1 Why Combine AR and MA Components?

AR models are useful for capturing persistence through past observations.

MA models are useful for capturing temporary propagation of shocks.

However, many real-world series exhibit both features.

## Examples

### Inflation

- depends on past inflation
- responds gradually to shocks

### GDP Growth

- exhibits persistence
- affected by temporary disturbances

### Financial Variables

- returns may exhibit short-lived shock effects
- volatility and persistence may coexist

```{admonition} Intuition
ARMA models allow both the history of the series and the history of shocks to influence the present.
```

---

# 13.2 The ARMA(1,1) Model

The simplest ARMA model combines:

- one autoregressive term
- one moving average term

```{admonition} Definition: ARMA(1,1)
An ARMA(1,1) process is:

$$
x_t
=
\phi x_{t-1}
+
\theta w_{t-1}
+
w_t
$$

where:

- $w_t \sim wn(0,\sigma_w^2)$
- $\phi$ is the AR parameter
- $\theta$ is the MA parameter
```

## Interpretation

The current value depends on:

- the previous observation
- the current shock
- the previous shock

```{admonition} Key Insight
ARMA models combine persistence and shock propagation in a single framework.
```

---

# 13.3 Simulating an ARMA(1,1) Process

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

    x[t] = (
        phi*x[t-1]
        + w[t]
        + theta*w[t-1]
    )

plt.figure(figsize=(10,4))
plt.plot(x, lw=1)
plt.title(r"ARMA(1,1): $\phi=0.7,\ \theta=0.5$")
plt.xlabel("Time")
plt.ylabel("$x_t$")

plt.savefig("figs/ch13/ARMA11.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![ARMA 1,1](figs/ch13/ARMA11.png)

```{admonition} Observation
ARMA processes often appear smoother and more realistic than pure AR or MA models alone.
```

---

# 13.4 Mean of ARMA(1,1)

Assuming no constant term:

```{math}
:enumerated: false
x_t
=
\phi x_{t-1}
+
\theta w_{t-1}
+
w_t
```

Taking expectations:

```{math}
:enumerated: false
E[x_t]
=
\phi E[x_{t-1}]
```

Under stationarity:

```{math}
:enumerated: false
E[x_t]=0
```

```{admonition} Result
An ARMA model without a constant has mean zero.
```

---

# 13.5 Stationarity and Invertibility

ARMA models inherit two important conditions:

## Stationarity

The AR component must be stationary.

For ARMA(1,1):

```{math}
:enumerated: false
|\phi|<1
```

## Invertibility

The MA component must be invertible.

For MA(1):

```{math}
:enumerated: false
|\theta|<1
```

```{admonition} Important
ARMA models require BOTH:

- stationarity of the AR component
- invertibility of the MA component
```

---

# 13.6 Backshift Representation

The backshift operator provides compact notation.

Recall:

```{math}
:enumerated: false
Bx_t = x_{t-1}
```

## ARMA(1,1)

We can write:

```{math}
:enumerated: false
(1-\phi B)x_t
=
(1+\theta B)w_t
```

## General ARMA(p,q)

````{admonition} Definition
An ARMA($p,q$) process satisfies:

$$
\phi(B)x_t
=
\theta(B)w_t
$$

where:

```{math}
:enumerated: false
\phi(B)
=
1-\phi_1B-\cdots-\phi_pB^p
```

and:

```{math}
:enumerated: false
\theta(B)
=
1+\theta_1B+\cdots+\theta_qB^q
```
````

```{admonition} Intuition
The AR polynomial governs persistence.

The MA polynomial governs shock propagation.
```

---

# 13.7 Infinite MA Representation

If the AR part is stationary:

```{math}
:enumerated: false
x_t
=
\frac{\theta(B)}{\phi(B)}w_t
```

which can often be expanded as an infinite MA process.


```{admonition} Key Insight
Stationary ARMA models can often be viewed as infinite weighted sums of past shocks.
```

---

# 13.8 Infinite AR Representation

If the MA part is invertible:

```{math}
:enumerated: false
w_t
=
\frac{\phi(B)}{\theta(B)}x_t
```

which can often be expanded as an infinite AR process.

```{admonition} Intuition
Invertibility allows shocks to be reconstructed from observed data.
```

---

# 13.9 ACF and PACF Behavior

Unlike pure AR or MA models, ARMA models do not exhibit simple cutoff behavior.

---

```{admonition} Fundamental Property
For ARMA models:

- ACF usually tails off
- PACF usually tails off
```

## Interpretation

This makes ARMA identification more challenging than pure AR or MA identification.

---

# 13.10 Simulated ACF and PACF

```{code-cell} python
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

fig, ax = plt.subplots(2,1, figsize=(8,6))

plot_acf(x, lags=30, ax=ax[0])
plot_pacf(x, lags=30, method='ywm', ax=ax[1])

plt.tight_layout()

plt.savefig("figs/ch13/ARMA_acf_pacf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![ARMA ACF PACF](figs/ch13/ARMA_acf_pacf.png)

```{admonition} Observation
Both the ACF and PACF typically decay gradually in ARMA models.
```

---

# 13.11 AR vs MA vs ARMA

| Model | ACF | PACF |
|---|---|---|
| AR($p$) | tails off | cuts off |
| MA($q$) | cuts off | tails off |
| ARMA($p,q$) | tails off | tails off |

```{admonition} Important
ARMA models blur the clean identification patterns of pure AR and MA models.
```

---

# 13.12 Identification in Practice

The Box–Jenkins methodology typically follows these steps:

1. visualize the series
2. ensure stationarity
3. inspect ACF/PACF
4. estimate candidate models
5. compare diagnostics
6. evaluate forecasting performance

```{admonition} Practical Reality
Model identification is partly statistical and partly interpretive.
```

---

# 13.13 Estimation in Gretl

## Menu

```text
Model → Time Series → ARIMA
```

## Typical Workflow

1. Select AR order
2. Select MA order
3. Estimate parameters
4. Check significance
5. Examine residual diagnostics

```markdown
[GRETL Screenshot Placeholder: ARMA estimation dialog]
```

```markdown
[GRETL Screenshot Placeholder: ARMA output]
```

---

# 13.14 Residual Diagnostics

A good ARMA model should leave residuals resembling white noise.

## Residual ACF

```{code-cell} python
:tags: [hide-input]
import statsmodels.api as sm

model = sm.tsa.ARIMA(x, order=(1,0,1))
res = model.fit()

plot_acf(res.resid, lags=20)

plt.savefig("figs/ch13/res_acf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Residual ACF](figs/ch13/res_acf.png)

---

## Ljung–Box Test

```{code-cell} python
from statsmodels.stats.diagnostic import acorr_ljungbox

lb = acorr_ljungbox(res.resid, lags=[10,20], return_df=True)
lb
```

| lag | lb_stat | lb_pvalue |
|---|---:|---:|
| 10 | 4.229114 | 0.936420 |
| 20 | 25.470103 | 0.184034 |

The large p-values suggest that we fail to reject the null hypothesis of no serial correlation.

```{admonition} Goal
Residual autocorrelation should be small if the model captures the dependence structure adequately.
```

---

# 13.15 Information Criteria

Model selection often uses:

- AIC
- BIC

## Akaike Information Criterion

$$
AIC
=
-2\log(\hat L)
+
2k
$$

## Bayesian Information Criterion

$$
BIC
=
-2\log(\hat L)
+
k\log n
$$

```{admonition} Interpretation
Lower AIC or BIC values generally indicate better-fitting models after penalizing complexity.
```

---

# 13.16 ARMA Models in Economics and Finance

ARMA models are widely used for:

- inflation dynamics
- GDP growth
- exchange rates
- interest rates
- demand forecasting
- inventory management

```{admonition} Economic Interpretation
ARMA models capture gradual adjustment together with temporary shock effects.
```

---

# 13.17 Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Ignoring stationarity**  
ARMA models require stationary behavior.

**2. Overfitting**  
Too many parameters may reduce forecasting performance.

**3. Blind reliance on ACF/PACF**  
Identification is often imperfect in finite samples.

**4. Ignoring residual diagnostics**  
Residual autocorrelation suggests misspecification.

**5. Overinterpreting short samples**  
Small samples can make ARMA patterns difficult to identify.
```

---

# 13.18 Looking Ahead

In this chapter, we combined AR and MA dynamics into the ARMA framework.

However, many real-world economic and financial series remain nonstationary.

In the next chapter, we extend ARMA models to handle nonstationarity through differencing:

```{admonition} Preview
ARIMA models combine:

- autoregression,
- moving averages,
- and integration (differencing).
```

# Key Takeaways

```{admonition} Summary
- ARMA models combine AR and MA dynamics
- AR terms capture persistence
- MA terms capture temporary shock propagation
- Both ACF and PACF usually tail off in ARMA models
- ARMA models are central tools in Box–Jenkins analysis
- Residual diagnostics are essential in model evaluation
```