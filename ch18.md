---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 18 — Dynamic Models

## 18.1 From Spurious Regression to Dynamics

In the previous chapter, we saw that regressing non-stationary variables can lead to **spurious results**.

A common solution was to difference the data:

```{math}
:enumerated: false
\Delta y_t = \alpha + \beta \Delta x_t + u_t
````

```{admonition} Key Insight
This is not just a statistical fix — it is a **dynamic model**.

The model now explains how **changes in $x_t$ affect changes in $y_t$**.
```

---

```{admonition} Important Clarification
The differenced model focuses on **short-run relationships**: how changes in one variable affect changes in another.
```

---

## 18.2 What is a Dynamic Model?

A **dynamic model** allows current values to depend on:

* past values
* past shocks
* changes over time

---

### Static vs Dynamic

**Static model:**

```{math}
:enumerated: false
y_t = \alpha + \beta x_t + u_t
```

**Dynamic model:**

```{math}
:enumerated: false
y_t = \alpha + \beta x_t + \gamma y_{t-1} + u_t
```

---

```{admonition} Key Idea
Dynamic models incorporate **time dependence explicitly**, allowing past values to influence current outcomes.
```

---

## 18.3 Distributed Lag Models

A natural extension is the **distributed lag model**:

$$
y_t = \alpha + \beta_0 x_t + \beta_1 x_{t-1} + \beta_2 x_{t-2} + u_t
$$

---

```{admonition} Interpretation
Past values of $x_t$ continue to influence $y_t$ over time.

This captures delayed effects, which are common in economics.
```

---

## 18.4 Autoregressive Distributed Lag (ARDL)

We now combine:

* lagged dependent variable
* lagged independent variables

---

### ARDL(p, q)

$$
y_t = \alpha + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{j=0}^{q} \beta_j x_{t-j} + u_t
$$

---

```{admonition} Key Model
ARDL models are one of the most flexible and widely used dynamic models in economics.

They allow both **persistence** (through $y_{t-i}$) and **distributed effects** (through $x_{t-j}$).
```

---

## 18.5 Gretl Implementation

---

### Step 1: Load Data

#### Menu

`File → Open data → Sample file...` and select `denmark` data from the `Gretl` database.

The variables are:

```gretl
LRM     log of real money supply, M2
LRY     log of real income
IBO     bond rate
IDE     bank deposit rate
```

```{figure} figs/ch18/gretl_1.png
:name: fig-denmark
:width: 90%
:align: center

Denmark Macro Data
```

### Step 2: Estimate ARDL(1,1)

#### Menu

`Model → OLS`

* Dependent: `LRM`
* Regressors: `LRM(-1) LRY LRY(-1)`

To get the lags, click the `lags...` icon in the **specify model** box. 

#### Command

```gretl
ols LRM const LRM(-1) LRY LRY(-1)
```

---

```gretl
Model 1: OLS, using observations 1974:2-1987:3 (T = 54)
Dependent variable: LRM

             coefficient   std. error   t-ratio    p-value 
  ---------------------------------------------------------
  const        0.125046    0.342856      0.3647   0.7169   
  LRM_1        1.00090     0.0580381    17.25     1.09e-022 ***
  LRY          0.630431    0.173817      3.627    0.0007    ***
  LRY_1       −0.652328    0.170176     −3.833    0.0004    ***


Mean dependent var   11.75666   S.D. dependent var   0.152858
Sum squared resid    0.044114   S.E. of regression   0.029703
R-squared            0.964377   Adjusted R-squared   0.962240
F(3, 50)             451.2022   P-value(F)           3.49e-36
Log-likelihood       115.3464   Akaike criterion    −222.6929
Schwarz criterion   −214.7370   Hannan-Quinn        −219.6246
rho                 −0.186568   Durbin's h          −1.515760
```
which can be written as:

```{math}
:enumerated: false
\begin{gather*}
\widehat{\text{LRM}}_t = \underset{(0.343)}{0.125} + \underset{(0.058)}{1} \text{LRM}_{t-1} + \underset{(0.1734)}{0.63} \text{LRY}_t - \underset{(0.17)}{0.65} \text{LRY}_{t-1}  \\
T = 54 \quad \overline{R}^2 = 0.9622 \quad F(3, 50) = 451.20 \quad \hat{\sigma} = 0.029703 \\
\text{\footnotesize (standard errors in parentheses)}
\end{gather*}
```

## 18.6 Diagnosing ARDL Models

After estimating an ARDL model, it is important to check whether the model is well specified.

Recall that a correctly specified dynamic model should leave **no predictable structure in the residuals**.

```{admonition} Key Principle
For a well-specified ARDL model, the residuals $\hat{u}_t$ should behave like **white noise**.
````

### 18.6.1 What Do We Mean by White Noise?

Residuals are white noise if they satisfy:

* mean zero
* constant variance
* **no autocorrelation**

```{admonition} Intuition
If residuals are not white noise, then there is still information in the data that the model has failed to capture.
```

#### Step 1: Visual Inspection

#### Residual Plot

#### Menu

`Graph → Time series plot`

Select residuals (e.g. `uhat`). Or right click on variable and select `Time series plot`. 

#### Command

```gretl
gnuplot uhat --time-series
```

```{figure} figs/ch18/gretl_2.png
:name: fig-residd
:width: 70%
:align: center

Residual
```

```{admonition} What to Look For
- No obvious trend  
- No cycles or repeating patterns  
- Fluctuations around zero  
```

#### Step 2: Autocorrelation Function (ACF)

#### Correlogram

#### Menu

Select `uhat`.

`Variable → Correlogram`

Or right click!

#### Command

```gretl
corrgm uhat
```

```{figure} figs/ch18/gretl_3.png
:name: fig-residual-plot
:width: 90%
:align: center

Correlogram
```

```{admonition} Interpretation
All autocorrelations should lie within the confidence bands.

Significant spikes suggest missing dynamics.
```

#### Step 3: Formal Test for Serial Correlation

#### Breusch–Godfrey or Ljung-Box Q Test

#### Menu

In the **Model** window
`Tests → Autocorrelation`

---

```gretl
Breusch-Godfrey test for autocorrelation up to order 4
OLS, using observations 1974:2-1987:3 (T = 54)
Dependent variable: uhat

             coefficient   std. error   t-ratio    p-value
  --------------------------------------------------------
  const       0.342059     0.336486      1.017     0.3147 
  LRY        −0.00513435   0.162471     −0.03160   0.9749 
  LRY_1       0.0773596    0.173429      0.4461    0.6576 
  LRM_1      −0.0657127    0.0631492    −1.041     0.3035 
  uhat_1     −0.0607536    0.158039     −0.3844    0.7024 
  uhat_2      0.427085     0.147320      2.899     0.0057  ***
  uhat_3      0.110868     0.162276      0.6832    0.4979 
  uhat_4      0.240594     0.160901      1.495     0.1417 

  Unadjusted R-squared = 0.283633

Test statistic: LMF = 4.553230,
with p-value = P(F(4,46) > 4.55323) = 0.0035

Alternative statistic: TR^2 = 15.316197,
with p-value = P(Chi-square(4) > 15.3162) = 0.00409

Ljung-Box Q' = 23.0858,
with p-value = P(Chi-square(4) > 23.0858) = 0.000122
```

```{admonition} Hypotheses
- $H_0$: no serial correlation  
- $H_1$: serial correlation present  
```

```{admonition} Decision Rule
If the p-value is small, reject $H_0$ → model is misspecified.
```

### 18.6.2 What If Residuals Are Not White Noise?

If residuals exhibit autocorrelation, this indicates that the model is incomplete.

---

```{admonition} Common Fixes
- Add more lags of $y_t$  
- Add more lags of $x_t$  
- Reconsider model specification  
```

---

```{admonition} Key Insight
Failure of residual diagnostics usually means the model has not captured the full dynamic structure of the data.
```

---

### 18.6.3 Important Distinction

```{admonition} Important Distinction
In this chapter, ARDL is used as a **dynamic model**.

At this stage, we require residuals to be **white noise**, but we do not yet require them to be stationary in the cointegration sense.

The issue of stationarity of residuals becomes crucial in the next chapter on cointegration.
```

### 18.6.4 Summary

```{admonition} Diagnostic Checklist
After estimating an ARDL model:

1. Plot residuals  
2. Examine the correlogram  
3. Perform a serial correlation test  
4. Adjust the model if needed  
```

## 18.7 Short-Run vs Long-Run Effects

In ARDL models:

- $\beta_0$ → **immediate (short-run) effect**  
- $\beta_1, \beta_2$ → **delayed effects**  
- $\phi_i$ → **persistence (dependence on past values of $y_t$)**  

```{admonition} Key Insight
Dynamic models distinguish between:

- **short-run responses** (immediate and lagged effects), and  
- **long-run equilibrium effects** (cumulative impact over time).
````

### Long-Run Effect

If the system is stable, the long-run relationship is:

```{math}
:enumerated: false
\text{Long-run multiplier} = \frac{\sum \beta_j}{1 - \sum \phi_i}
```

```{admonition} Interpretation
The long-run multiplier measures the **total cumulative effect** of a change in $x_t$ on $y_t$ after all dynamic adjustments have taken place.
```

## 18.8 Dynamic Interpretation

Consider an economic example:

* an increase in interest rates today
* output does not adjust immediately
* effects unfold gradually over time

```{admonition} Intuition
Dynamic models capture how shocks **propagate over time**, rather than affecting variables instantaneously.
```

## 18.9 Connection to Differencing

Recall from Chapter 17:

```{math}
:enumerated: false
\Delta y_t = \beta \Delta x_t + u_t
```

This can be viewed as a **restricted dynamic model**:

* no lagged levels
* no persistence
* no long-run structure

```{admonition} Key Insight
Differencing focuses purely on **short-run changes**, removing long-run information from the data.
```

## 18.10 Why Differencing Is Not Enough

While differencing solves the spurious regression problem, it comes at a cost:

* loss of long-run relationships
* inability to model equilibrium behavior
* oversimplification of dynamics

```{admonition} Motivation
We need a framework that captures both:

- short-run changes  
- long-run equilibrium  
```

## 18.11 Looking Ahead: ECM

This leads to the **Error Correction Model (ECM)**:

```{math}
:enumerated: false
\Delta y_t = \beta \Delta x_t + \gamma (y_{t-1} - \theta x_{t-1}) + u_t
```

```{admonition} Key Idea
ECM combines:

- short-run dynamics (through $\Delta x_t$), and  
- long-run equilibrium (through $y_{t-1} - \theta x_{t-1}$)
```

```{admonition} Intuition (Optional)
Think of two variables connected by a long-run relationship as being tied together by a rubber band.

They may drift apart in the short run, but forces exist that pull them back together.
```

## 18.12 From Dynamics to Long-Run Relationships

Dynamic models help us understand **how variables adjust over time**, but they do not yet resolve the issue of nonstationarity.

```{admonition} Key Observation
Differencing avoids spurious regression by making the data stationary — but it may also remove meaningful long-run information.
```

```{admonition} The Core Question
Have we removed too much information by differencing?
```

## 18.13 A Fundamental Tension

```{admonition} The Core Problem
- Using levels → risk of spurious regression  
- Using differences → loss of long-run relationships  
```

👉 Neither approach is fully satisfactory on its own.

## 18.14 A Way Forward

Is it possible to:

* retain long-run relationships (levels), and
* avoid spurious regression?

```{admonition} Preview
Yes — if a particular combination of variables is stationary.

This concept is called **cointegration**.
```

## 18.15 Looking Ahead

In the next chapter, we will see that:

* some non-stationary variables are **linked in the long run**
* deviations from equilibrium are **temporary**

```{math}
:enumerated: false
\Delta y_t = \beta \Delta x_t + \gamma (y_{t-1} - \theta x_{t-1}) + u_t
```

```{admonition} Big Picture
Dynamic models describe short-run changes.

Cointegration explains long-run equilibrium.

ECM combines both.
```

## Key Takeaways

* ARDL models capture both immediate and delayed effects
* Dynamic models describe how shocks propagate over time
* Differencing isolates short-run movements but removes long-run structure
* A complete framework must incorporate both short-run and long-run behavior
* ECM provides this bridge

---

## Appendix 18A — Dynamic Models and Long-Run Effects

### A.1 A Simple ARDL(1,1)

```{math}
:enumerated: false
y_t = \alpha + \phi y_{t-1} + \beta_0 x_t + \beta_1 x_{t-1} + u_t
```

### A.2 Steady-State (Long-Run Equilibrium)

```{math}
:enumerated: false
y_t = y_{t-1} = y, \quad x_t = x_{t-1} = x
```

### A.3 Long-Run Relationship

```{math}
:enumerated: false
y = \frac{\alpha}{1 - \phi} + \frac{\beta_0 + \beta_1}{1 - \phi} x
```

```{admonition} Key Result
Long-run multiplier:

```

```{math}
:enumerated: false
\frac{\beta_0 + \beta_1}{1 - \phi}
```

### A.4 Dynamic Adjustment

```{math}
:enumerated: false
\Delta y_t = (\phi - 1)y_{t-1} + \beta_0 x_t + \beta_1 x_{t-1} + u_t
```

### A.5 Stability Condition

```{math}
:enumerated: false
|\phi| < 1
```

```{admonition} Interpretation
If $|\phi| < 1$, shocks decay over time and the system converges to equilibrium.
```

### A.6 Link to ECM

```{math}
:enumerated: false
\Delta y_t = \beta \Delta x_t + \gamma (y_{t-1} - \theta x_{t-1}) + u_t
```

---

```{admonition} Big Picture
ARDL models naturally contain both:

- short-run dynamics  
- long-run equilibrium  
```
