---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 20 — Cointegration and Long-Run Relationships

In Chapter 17, we saw that regressions involving nonstationary variables can produce **spurious results**.

In Chapter 18, we introduced dynamic models (ARDL), which help capture short-run dependence.

However, an important question remains:

> **Can nonstationary variables still have a meaningful long-run relationship?**

The answer is **yes** — and this is the idea behind **cointegration**.

## Learning Objectives

By the end of this chapter, you should be able to:

- understand the concept of cointegration  
- distinguish spurious regression from cointegrated relationships  
- implement the Engle–Granger test in GRETL  
- interpret residual stationarity  
- understand the idea of long-run equilibrium  

## 20.1 Motivation: Spurious vs Meaningful Relationships

Recall from Chapter 17:

If we regress two nonstationary series:

```{math}
:enumerated: false
y_t = \alpha + \beta x_t + e_t
```

we may obtain:

- high $R^2$  
- significant t-statistics  
- but **completely meaningless results**

```{admonition} Key Problem
Nonstationary variables can move together purely by chance, leading to **spurious regression**.
````

But sometimes, variables move together **because they are linked by economic forces**.

Examples:

* consumption and income
* prices in related markets
* exchange rates and interest rates

## 20.2 What Is Cointegration?

```{admonition} Definition: Cointegration
Two nonstationary series $x_t$ and $y_t$ are **cointegrated** or **co-trending** if a linear combination of them is stationary.
```

Formally:

* $x_t \sim I(1)$
* $y_t \sim I(1)$

but:

$$
e_t = y_t - \beta x_t
$$

is **stationary** or $I(0)$.

---

```{admonition} Key Idea
Cointegration means that although variables wander over time, they do not drift too far apart.
```

## 20.3 Intuition: Long-Run Equilibrium

Even if $x_t$ and $y_t$ individually behave like random walks:

* their difference may be stable
* deviations from equilibrium are temporary


```{admonition} Intuition
Cointegrated variables are tied together by a **long-run equilibrium relationship**.
```

---

## 20.4 Spurious Regression vs Cointegration

This is a crucial distinction.

| Case                | Residuals     | Interpretation              |
| ------------------- | ------------- | --------------------------- |
| Spurious regression | Nonstationary | No meaningful relationship  |
| Cointegration       | Stationary    | Long-run equilibrium exists |

---

```{admonition} Diagnostic Principle
The key to distinguishing spurious regression from cointegration is:

→ **Are the residuals stationary?**
```

## 20.5 Engle–Granger Two-Step Procedure

We now describe a practical method for testing cointegration.

Get data from Gretl:

`File → Open data → Sample file...` and select `gdp` data from the `POE 4th ed.` database.

which contains the variables:

```gretl
usa     real GDP of USA
aus     real GDP of Australia
```

---

### Step 1: Estimate Long-Run Relationship

Estimate:

$$
aus_t = \alpha + \beta usa_t + e_t
$$

---

#### Gretl Command

```gretl
ols aus const usa
```

```gretl
Model 1: OLS, using observations 1970:1-2000:4 (T = 124)
Dependent variable: aus

             coefficient   std. error   t-ratio    p-value 
  ---------------------------------------------------------
  const       −1.07237     0.403225      −2.659   0.0089    ***
  usa          1.00099     0.00610028   164.1     5.85e-145 ***

Mean dependent var   62.72528   S.D. dependent var   17.65155
Sum squared resid    172.8638   S.E. of regression   1.190343
R-squared            0.995489   Adjusted R-squared   0.995452
F(1, 122)            26925.45   P-value(F)           5.8e-145
Log-likelihood      −196.5462   Akaike criterion     397.0924
Schwarz criterion    402.7329   Hannan-Quinn         399.3837
rho                  0.860968   Durbin-Watson        0.272654

```

### Step 2: Extract Residuals

Save residuals:

```gretl
series uhat = $uhat
```

```{figure} figs/ch20/gretl_1.png
:name: fig-residual
:width: 70%
:align: center

Residual
```

### Step 3: Test Residual Stationarity

Perform an ADF test:

#### Menu

Select `uhat`

`Variable → Unit root tests → Augmented Dickey-Fuller test`

#### Command

```gretl
adf 1 uhat
```

```gretl
[Augmented Dickey-Fuller test for uhat
testing down from 25 lags, criterion AIC
sample size 123
unit-root null hypothesis: a = 1

  test with constant 
  including 0 lags of (1-L)uhat
  model: (1-L)y = b0 + (a-1)*y(-1) + e
  estimated value of (a - 1): -0.139029
  test statistic: tau_c(1) = -3.03875
  asymptotic p-value 0.03145
  1st-order autocorrelation coeff. for e: -0.007

  with constant and trend 
  including 0 lags of (1-L)uhat
  model: (1-L)y = b0 + b1*t + (a-1)*y(-1) + e
  estimated value of (a - 1): -0.138869
  test statistic: tau_ct(1) = -3.01769
  asymptotic p-value 0.1272
  1st-order autocorrelation coeff. for e: -0.008
```

## 20.6 Hypotheses

$$
H_0: \text{Residuals have a unit root (no cointegration)}
$$

$$
H_1: \text{Residuals are stationary (cointegration)}
$$

```{admonition} Decision Rule
- Reject $H_0$ → residuals stationary → cointegration  
- Fail to reject $H_0$ → spurious regression  
```

## 20.7 Interpretation

If residuals are stationary:

* deviations from equilibrium are temporary
* variables move together in the long run


```{admonition} Key Insight
Cointegration restores meaning to regressions involving nonstationary variables.
```

## 20.8 Important Caveats

```{admonition} Important
The Engle–Granger method:

- assumes a single cointegrating relationship  
- depends on which variable is treated as dependent  
```

## 20.9 Two Approaches to Cointegration

So far, we have used the **Engle–Granger approach**, which is based on testing whether residuals are stationary.

An alternative approach is based on **dynamic models**, using ARDL.

```{admonition} Two Approaches to Cointegration
- **Residual-based approach**: Engle–Granger  
- **Model-based approach**: ARDL bounds test  

Both aim to detect long-run relationships, but they use different strategies.
```

## 20.10 Cointegration via ARDL (Bounds Testing)

While the Engle–Granger method is intuitive, it has an important limitation:

* it requires all variables to be $I(1)$

In practice, some variables may be $I(0)$ and others $I(1)$.

```{admonition} Advantage
The ARDL bounds approach can be used when variables are a **mixture of $I(0)$ and $I(1)$**.
```

### 20.10.1 From ARDL to ECM

Consider:

$$
y_t = \alpha + \phi y_{t-1} + \beta_0 x_t + \beta_1 x_{t-1} + u_t
$$

This can be rewritten as:

$$
\Delta y_t = \gamma \Delta x_t + \lambda_1 y_{t-1} + \lambda_2 x_{t-1} + u_t
$$

```{admonition} Key Insight
A long-run relationship exists if the lagged level terms are jointly significant.
```

### 20.10.2 The Bounds Test

We test:

$$
H_0: \lambda_1 = \lambda_2 = 0
$$

```{admonition} Interpretation
- $H_0$ → no cointegration  
- reject $H_0$ → cointegration exists  
```

### 20.10.3 Decision Rule

| F-statistic   | Conclusion       |
| ------------- | ---------------- |
| < lower bound | No cointegration |
| > upper bound | Cointegration    |
| In between    | Inconclusive     |

```{admonition} Key Insight
The bounds test avoids requiring knowledge of the exact integration order.
```

### 20.10.4 Implementation in Gretl


---

#### Step 1: Estimate ARDL Model

#### Menu

Model → Time series → ARDL

* Select dependent variable ($y$)
* Select regressors ($x$)
* Choose lag lengths

---

```markdown id="ardl_setup_placeholder"
[GRETL Screenshot Placeholder: ARDL specification window]
```

---

#### Step 2: Perform Bounds Test

In the ARDL output window:

* select option for **Bounds test for cointegration**

---

```markdown id="ardl_bounds_output_placeholder"
[GRETL Screenshot Placeholder: Bounds test output]
```

---

#### Command (example)

```gretl id="ardl_cmd"
ardl 2 2 y x
```

Then:

```gretl id="bounds_cmd"
ecm
```

(or use GUI for bounds test output)

---

### 20.11.6 Interpreting GRETL Output

Focus on:

* F-statistic
* critical value bounds

---

```{admonition} Interpretation Rule
- If F-statistic > upper bound → cointegration  
- If F-statistic < lower bound → no cointegration  
- Otherwise → inconclusive  
```

---

### 20.11.7 Important Conditions

```{admonition} Important
:class: warning

The ARDL bounds test requires:

- variables are not $I(2)$  
- correct model specification  
- appropriate lag selection  
```

---

### 20.11.8 Comparison with Engle–Granger

| Feature                          | Engle–Granger | ARDL Bounds |
| -------------------------------- | ------------- | ----------- |
| Requires all variables $I(1)$    | Yes           | No          |
| Based on residual stationarity   | Yes           | No          |
| Uses dynamic model               | No            | Yes         |
| Handles mixed integration orders | No            | Yes         |

---

```{admonition} Two Approaches to Cointegration
- **Residual-based approach**: Engle–Granger  
- **Model-based approach**: ARDL bounds test  

Both aim to detect long-run relationships, but they use different strategies.
```

---

### 20.10.5 Summary of ARDL Approach

```{admonition} Key Takeaways
- ARDL allows testing for cointegration within a dynamic model  
- works with mixed $I(0)/I(1)$ variables  
- complements the Engle–Granger approach  
```

---

## 20.11 Summary

```{admonition} Key Takeaways
- Cointegration allows meaningful relationships between nonstationary variables  
- The key diagnostic is **residual stationarity**  
- Engle–Granger and ARDL provide complementary approaches  
- Cointegration implies a long-run equilibrium  
```

---

## Looking Ahead

Cointegration tells us that a long-run relationship exists.

In the next chapter, we show how this relationship governs **short-run adjustments** through the **Error Correction Model (ECM)**.

---

## Appendix 20A — The ARDL Bounds Test (Conceptual Overview)

This appendix provides a simplified explanation of the ARDL bounds test for cointegration.

### A.1 Starting Point

Consider the ECM representation:

```{math}
:enumerated: false
\Delta y_t = \gamma \Delta x_t + \lambda_1 y_{t-1} + \lambda_2 x_{t-1} + u_t
```

### A.2 The Key Question

We want to know:

> Do the level terms matter?

### A.3 Hypothesis

```{math}
:enumerated: false
H_0: \lambda_1 = \lambda_2 = 0
```

```{math}
:enumerated: false
H_1: \text{At least one is nonzero}
```

```{admonition} Interpretation
- If both coefficients are zero → no long-run relationship  
- If at least one is nonzero → cointegration exists  
````

### A.4 Why Two Critical Values?

The test statistic depends on whether variables are:

* stationary ($I(0)$)
* nonstationary ($I(1)$)

Since we may not know this exactly, the bounds test provides:

* a **lower bound** (all variables $I(0)$)
* an **upper bound** (all variables $I(1)$)

### A.5 Decision Rule

* Below lower bound → no cointegration
* Above upper bound → cointegration
* Between bounds → inconclusive

```{admonition} Key Insight
The bounds test allows inference **without knowing the exact integration order** of variables.
```

### A.6 Intuition

* If lagged levels do not matter → no long-run link
* If they matter → system has equilibrium structure

```{admonition} Big Picture
The ARDL bounds test checks whether **long-run information is embedded in the dynamic model**.
```
