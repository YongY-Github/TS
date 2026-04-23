---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 21 — Error Correction Models (ECM)

In the previous chapters:

- **Chapter 17** showed that regressions in levels can be **spurious**  
- **Chapter 18** introduced **dynamic models (ARDL)**  
- **Chapter 20** showed that **cointegration restores meaning** via stationary residuals  

We now bring everything together.

> How do variables adjust when they deviate from a long-run equilibrium?

The answer is given by the **Error Correction Model (ECM)**.

## Learning Objectives

By the end of this chapter, you should be able to:

- understand the intuition behind ECM  
- derive ECM from a cointegrating relationship  
- interpret the error correction term  
- estimate ECM in GRETL  
- connect ECM to ARDL models  

## 21.1 From Cointegration to Adjustment

If $y_t$ and $x_t$ are cointegrated:

$$
y_t = \alpha + \beta x_t + e_t
$$

then $e_t$ represents deviations from the long-run equilibrium.

```{admonition} Key Idea
When variables deviate from equilibrium, adjustment forces tend to bring them back.
````

## 21.2 The Error Correction Model

Define:

$$
e_{t-1} = y_{t-1} - \alpha - \beta x_{t-1}
$$

The ECM is:

$$
\Delta y_t = \gamma_0 + \gamma_1 \Delta x_t + \lambda e_{t-1} + u_t
$$

```{admonition} Definition
An ECM explains short-run changes as a function of:

- current changes  
- past deviations from equilibrium  
```

## 21.3 Interpretation

* $\gamma_1$ → short-run effect
* $\lambda$ → speed of adjustment

```{admonition} Key Interpretation
- $\lambda < 0$ → convergence to equilibrium  
- larger $|\lambda|$ → faster adjustment  
```

Example:

* $\lambda = -0.5$ → 50% of deviation corrected each period

### Example

* $\lambda = -0.5$ → 50% of deviation corrected each period

## 21.4 Intuition

```{admonition} Intuition
The ECM says:

- variables may drift apart in the short run  
- but are pulled back toward equilibrium over time  
```

This combines:

* **short-run dynamics**
* **long-run equilibrium**

## 21.5 Estimation in GRETL (Step-by-Step)

We now estimate the ECM using the same data as in Chapter 20.

### Step 1: Estimate Long-Run Relationship

```gretl
ols aus const usa
```

### Step 2: Save Residuals

```gretl
series ehat = $uhat
```

### Step 3: Construct ECM Regression

```gretl
series d_usa = diff(usa)
series d_aus = diff(aus)
series ehat_1 = ehat(-1)

ols d_aus const d_usa ehat_1
```

Or using the **menu**

```gretl
Model 2: OLS, using observations 1970:2-2000:4 (T = 123)
Dependent variable: d_aus

             coefficient   std. error   t-ratio   p-value 
  --------------------------------------------------------
  const        0.211535    0.0710333     2.978    0.0035   ***
  d_usa        0.567700    0.0983739     5.771    6.29e-08 ***
  ehat_1      −0.138852    0.0426256    −3.257    0.0015   ***

Mean dependent var   0.499554   S.D. dependent var   0.649528
Sum squared resid    37.68901   S.E. of regression   0.560424
R-squared            0.267750   Adjusted R-squared   0.255546
F(2, 120)            21.93920   P-value(F)           7.58e-09
Log-likelihood      −101.7863   Akaike criterion     209.5725
Schwarz criterion    218.0091   Hannan-Quinn         212.9994
rho                  0.021551   Durbin-Watson        1.913618
```

```{math}
:enumerated: false
\begin{gather*}
\widehat{\text{d\_aus}} = \underset{(0.071)}{0.212} + \underset{(0.098)}{0.568} \text{ d\_usa} - \underset{(0.043)}{0.139} \text{ ehat}_{t-1} \\
T = 123 \quad \bar{R}^2 = 0.2555 \quad F(2, 120) = 21.939 \quad \hat{\sigma} = 0.560 \\
\text{\scriptsize (standard errors in parentheses)}
\end{gather*}
```

## 21.6 Interpreting GRETL Output

Focus on:

### Coefficient on $\Delta x_t$

* short-run effect

### Coefficient on $ehat_{t-1}$

* should be **negative and significant**

```{admonition} Interpretation Rule
If the coefficient on the error correction term is:

- negative → equilibrium is stable  
- statistically significant → adjustment mechanism exists  
```

## 21.6 Connection to ARDL

Recall the ARDL(1,1) model:

```{math}
:enumerated: false
y_t = \alpha + \phi y_{t-1} + \beta_0 x_t + \beta_1 x_{t-1} + u_t
```

This can be rewritten as:

```{math}
:enumerated: false
\Delta y_t = \gamma \Delta x_t + \lambda (y_{t-1} - \beta x_{t-1}) + u_t
```

```{admonition} Deep Insight
The ECM is not a new model — it is a **reparameterization of ARDL** that makes the long-run relationship explicit.
```

This means:

* ARDL → short-run + long-run
* ECM → makes this structure explicit

## 21.8 Common Pitfalls

```{admonition} Common Mistakes
:class: warning

**1. Ignoring cointegration**  
ECM requires a valid long-run relationship.

**2. Wrong sign of $\lambda$**  
Positive $\lambda$ implies divergence.

**3. Misinterpreting coefficients**  
Short-run and long-run effects are different.

**4. Omitting dynamics**  
ECM may require additional lags.

**5. Small sample issues**  
Estimates may be unstable.
```
## 21.9 Big Picture

```{admonition} Big Picture
ECM provides a unified framework:

- short-run dynamics  
- long-run equilibrium  
- gradual adjustment over time  
````

## 21.10 Summary

```{admonition} Key Takeaways
- ECM links cointegration and dynamics  
- deviations from equilibrium drive adjustment  
- the coefficient $\lambda$ measures speed of adjustment  
- ECM is derived from ARDL  
```

## Looking Ahead

So far, we have focused on **bivariate relationships**.

In the next part of the book, we extend these ideas to **multivariate systems**, including:

* VAR models
* impulse response functions
* VECM

---

## Appendix 21A — Deriving the ECM from an ARDL Model

In this appendix, we show how the **Error Correction Model (ECM)** arises directly from an ARDL model.

### A.1 Start with ARDL(1,1)

Consider:

```{math}
:enumerated: false
y_t = \alpha + \phi y_{t-1} + \beta_0 x_t + \beta_1 x_{t-1} + u_t
```

### A.2 Subtract $y_{t-1}$

Subtract $y_{t-1}$ from both sides:

```{math}
:enumerated: false
y_t - y_{t-1}
= \alpha + \phi y_{t-1} + \beta_0 x_t + \beta_1 x_{t-1} - y_{t-1} + u_t
```

### A.3 Express in Differences

Recall:

```{math}
:enumerated: false
\Delta y_t = y_t - y_{t-1}
```

So:

```{math}
:enumerated: false
\Delta y_t
= \alpha + (\phi - 1)y_{t-1} + \beta_0 x_t + \beta_1 x_{t-1} + u_t
```

### A.4 Separate the Change in $x_t$

Add and subtract $\beta_0 x_{t-1}$:

```{math}
:enumerated: false
\Delta y_t
= \alpha + (\phi - 1)y_{t-1}
+ \beta_0 (x_t - x_{t-1})
+ (\beta_0 + \beta_1)x_{t-1}
+ u_t
```

### A.5 Recognize $\Delta x_t$

Since:

```{math}
:enumerated: false
\Delta x_t = x_t - x_{t-1}
```

we obtain:

```{math}
:enumerated: false
\Delta y_t
= \alpha + \beta_0 \Delta x_t
+ (\phi - 1)y_{t-1}
+ (\beta_0 + \beta_1)x_{t-1}
+ u_t
```

### A.6 Rearranging

Group the level terms:

```{math}
:enumerated: false
\Delta y_t
= \alpha + \beta_0 \Delta x_t
+ (\phi - 1)\left[y_{t-1} - \frac{\beta_0 + \beta_1}{1 - \phi} x_{t-1}\right]
+ u_t
```

### A.7 Final ECM Form

Define:

- $\lambda = \phi - 1$
- $\theta = \frac{\beta_0 + \beta_1}{1 - \phi}$

Then:

```{math}
:enumerated: false
\Delta y_t
= \alpha + \beta_0 \Delta x_t
+ \lambda (y_{t-1} - \theta x_{t-1})
+ u_t
```

```{admonition} Key Result
The ECM arises naturally from ARDL, with:

- $\beta_0$ → short-run effect  
- $\theta$ → long-run relationship  
- $\lambda$ → speed of adjustment  
````

### A.8 Interpretation

* $(y_{t-1} - \theta x_{t-1})$ → deviation from equilibrium
* $\lambda$ → how quickly the system corrects

```{admonition} Insight
The ECM is not a new model — it is a **reparameterization of ARDL** that separates short-run dynamics from long-run equilibrium.
```
