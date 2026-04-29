---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 18 — Dynamic Regression and ARDL Models

In the previous chapter, we saw that regressions involving nonstationary variables can produce **spurious results**.

One common response was to difference the data:

```{math}
:enumerated: false
\Delta y_t = \alpha + \beta \Delta x_t + u_t
```

At first glance, this may look like a technical fix. But it is more than that.

```{admonition} Key Insight
A regression in differences is a **dynamic model**.

It explains how **changes in $x_t$ affect changes in $y_t$**.
```

This chapter introduces dynamic regression models more generally. These models allow us to study how relationships unfold over time.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- explain what a dynamic regression model is
- distinguish static and dynamic models
- understand distributed lag models
- understand autoregressive distributed lag (ARDL) models
- interpret short-run and long-run effects
- diagnose residual autocorrelation
- explain why differencing may remove long-run information
- understand how dynamic models lead naturally to cointegration and ECM

---

# 18.1 From Spurious Regression to Dynamics

Recall the differenced regression:

```{math}
:enumerated: false
\Delta y_t = \alpha + \beta \Delta x_t + u_t
```

This model no longer explains the level of $y_t$ using the level of $x_t$. Instead, it explains how changes in one variable are related to changes in another.

```{admonition} Important Clarification
The differenced model focuses on **short-run relationships**.
```

For example:

- how changes in income affect changes in consumption
- how changes in interest rates affect changes in investment
- how changes in exchange rates affect changes in exports

This is already a dynamic way of thinking.

---

# 18.2 What Is a Dynamic Model?

A **dynamic model** allows current outcomes to depend on time-related information, such as:

- past values of the dependent variable
- past values of explanatory variables
- current and past changes
- past shocks

## Static vs Dynamic Models

A static model is:

```{math}
:enumerated: false
y_t = \alpha + \beta x_t + u_t
```

This says that $y_t$ responds immediately to $x_t$.

A dynamic model might be:

```{math}
:enumerated: false
y_t = \alpha + \beta x_t + \gamma y_{t-1} + u_t
```

Here, $y_t$ depends not only on $x_t$, but also on its own past value.

```{admonition} Key Idea
Dynamic models incorporate **time dependence explicitly**.
```

---

# 18.3 Why Dynamics Matter

Many economic relationships do not adjust instantly.

For example:

- consumption may respond gradually to income
- investment may respond slowly to interest rates
- prices may adjust with delays
- money demand may depend on past money balances

```{admonition} Intuition
Dynamic models allow effects to unfold over time rather than occurring all at once.
```

This makes them especially useful in economics, finance, and business forecasting.

---

# 18.4 Distributed Lag Models

A natural way to introduce dynamics is to include current and past values of an explanatory variable.

A simple distributed lag model is:

```{math}
:enumerated: false
y_t = \alpha + \beta_0 x_t + \beta_1 x_{t-1} + \beta_2 x_{t-2} + u_t
```

Here:

- $\beta_0$ measures the immediate effect of $x_t$
- $\beta_1$ measures the effect of last period’s $x$
- $\beta_2$ measures the effect from two periods ago

```{admonition} Interpretation
Past values of $x_t$ continue to influence $y_t$ over time.
```

## Short-Run and Cumulative Effects

In the distributed lag model:

```{math}
:enumerated: false
y_t = \alpha + \beta_0 x_t + \beta_1 x_{t-1} + \beta_2 x_{t-2} + u_t
```

the short-run effect is:

```{math}
:enumerated: false
\beta_0
```

The cumulative effect over three periods is:

```{math}
:enumerated: false
\beta_0 + \beta_1 + \beta_2
```

```{admonition} Key Insight
Distributed lag models separate immediate effects from delayed effects.
```

---

# 18.5 Autoregressive Distributed Lag Models

We now combine two ideas:

- lagged dependent variables
- lagged explanatory variables

This gives the **autoregressive distributed lag model**, or **ARDL**.

---

```{admonition} Definition
An ARDL($p,q$) model is:

$$
y_t
=
\alpha
+
\sum_{i=1}^{p}\phi_i y_{t-i}
+
\sum_{j=0}^{q}\beta_j x_{t-j}
+
u_t
$$
```

where:

- $p$ is the number of lags of $y_t$
- $q$ is the number of lags of $x_t$
- $\phi_i$ captures persistence in $y_t$
- $\beta_j$ captures current and lagged effects of $x_t$

```{admonition} Key Model
ARDL models are flexible dynamic regression models.

They allow both:

- **persistence** through lagged $y_t$
- **distributed effects** through current and lagged $x_t$
```

---

# 18.6 Interpreting an ARDL Model

Consider a simple ARDL(1,1):

```{math}
:enumerated: false
y_t = \alpha + \phi y_{t-1} + \beta_0 x_t + \beta_1 x_{t-1} + u_t
```

The coefficients have different roles:

- $\beta_0$ → immediate effect
- $\beta_1$ → delayed effect
- $\phi$ → persistence in $y_t$

```{admonition} Intuition
The lagged dependent variable allows the effect of a shock to persist over time.
```

---

# 18.7 Short-Run and Long-Run Effects

Dynamic models distinguish between:

- short-run responses
- long-run cumulative effects

For the ARDL(1,1):

```{math}
:enumerated: false
y_t = \alpha + \phi y_{t-1} + \beta_0 x_t + \beta_1 x_{t-1} + u_t
```

the immediate short-run effect is:

```{math}
:enumerated: false
\beta_0
```

If the system is stable, the long-run multiplier is:

```{math}
:enumerated: false
\frac{\beta_0 + \beta_1}{1-\phi}
```

More generally:

```{math}
:enumerated: false
\text{Long-run multiplier}
=
\frac{\sum_j \beta_j}{1-\sum_i \phi_i}
```

```{admonition} Interpretation
The long-run multiplier measures the total effect of a permanent change in $x_t$ after all dynamic adjustment has taken place.
```

---

# 18.8 Dynamic Interpretation

Consider an increase in interest rates.

Its effect on output may not be immediate:

- firms may delay investment decisions
- households may adjust spending gradually
- banks may adjust lending conditions over time

A dynamic model allows this adjustment path to be represented explicitly.

```{admonition} Intuition
Dynamic models capture how shocks **propagate over time**, rather than affecting variables instantaneously.
```

---

# 18.9 Differencing as a Restricted Dynamic Model

Recall from Chapter 17:

```{math}
:enumerated: false
\Delta y_t = \beta \Delta x_t + u_t
```

This can be viewed as a restricted dynamic model:

- no lagged levels
- no persistence term
- no long-run structure

```{admonition} Key Insight
Differencing focuses on short-run changes, but it removes information about long-run relationships in levels.
```

This is useful for avoiding spurious regression, but it may discard economically meaningful long-run information.

---

# 18.10 Why Differencing Is Not Enough

While differencing often solves the spurious regression problem, it comes at a cost.

Differencing may remove:

- long-run relationships
- equilibrium behavior
- information about levels
- gradual adjustment mechanisms

```{admonition} Motivation
We need a framework that captures both:

- short-run changes
- long-run equilibrium
```

This motivates cointegration and the error correction model.

---

# 18.11 Looking Ahead: ECM

If two variables share a long-run equilibrium relationship, we may want to model both:

1. short-run changes
2. adjustment back toward the long-run relationship

This leads to the **Error Correction Model (ECM)**:

```{math}
:enumerated: false
\Delta y_t
=
\beta \Delta x_t
+
\gamma (y_{t-1} - \theta x_{t-1})
+
u_t
```

```{admonition} Key Idea
ECM combines:

- short-run dynamics through $\Delta x_t$
- long-run equilibrium through $y_{t-1}-\theta x_{t-1}$
```

```{admonition} Intuition
Think of two variables connected by a long-run relationship as being tied together by a rubber band.

They may drift apart in the short run, but forces exist that pull them back together.
```

We return to this idea in later chapters.

---

# 18.12 Gretl Example: ARDL Model

We now estimate a simple ARDL model using the `denmark` dataset in GRETL.

## Step 1: Load the Data

### Menu

`File → Open data → Sample file...`

Select the `denmark` data from the GRETL database.

The variables include:

```gretl
LRM     log of real money supply, M2
LRY     log of real income
IBO     bond rate
IDE     bank deposit rate
```

```{figure} figs/ch18/gretl_1.png
:name: fig-denmark-data
:width: 90%
:align: center

Denmark macroeconomic data
```

## Step 2: Estimate ARDL(1,1)

We estimate:

```{math}
:enumerated: false
LRM_t
=
\alpha
+
\phi LRM_{t-1}
+
\beta_0 LRY_t
+
\beta_1 LRY_{t-1}
+
u_t
```

### Menu

`Model → Ordinary Least Squares`

- Dependent variable: `LRM`
- Regressors: `LRM(-1) LRY LRY(-1)`

To create lags, click the `lags...` icon in the model specification window.

### Command

```gretl
ols LRM const LRM(-1) LRY LRY(-1)
```

Example output:

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

This can be written as:

```{math}
:enumerated: false
\begin{gather*}
\widehat{LRM}_t
=
\underset{(0.343)}{0.125}
+
\underset{(0.058)}{1.001} LRM_{t-1}
+
\underset{(0.174)}{0.630} LRY_t
-
\underset{(0.170)}{0.652} LRY_{t-1}
\\
T = 54
\quad
\overline{R}^2 = 0.9622
\quad
F(3,50)=451.20
\quad
\hat{\sigma}=0.0297
\\
\text{\footnotesize (standard errors in parentheses)}
\end{gather*}
```

```{admonition} Interpretation
The current value of real money supply depends strongly on its own lag and also on current and lagged real income.
```

---

# 18.13 Diagnosing ARDL Models

After estimating an ARDL model, we need to check whether the model has captured the relevant dynamics.

```{admonition} Key Principle
For a well-specified dynamic model, the residuals should behave like white noise.
```

If residuals still contain autocorrelation, the model is missing some dynamic structure.

## What Do We Mean by White Noise Residuals?

Residuals should show:

- mean near zero
- roughly constant variance
- no autocorrelation

```{admonition} Intuition
If residuals are not white noise, then there is still information in the data that the model has failed to capture.
```

## Step 1: Plot Residuals

### Menu

`Graph → Time series plot`

Select residuals, such as `uhat`.

You can also right click on the residual variable and select `Time series plot`.

### Command

```gretl
gnuplot uhat --time-series
```

```{figure} figs/ch18/gretl_2.png
:name: fig-ardl-residuals
:width: 70%
:align: center

Residual plot
```

```{admonition} What to Look For
- no obvious trend
- no cycles or repeating patterns
- fluctuations around zero
```

## Step 2: Residual Correlogram

### Menu

Select `uhat`.

`Variable → Correlogram`

You can also right click on `uhat` and select `Correlogram`.

### Command

```gretl
corrgm uhat
```

```{figure} figs/ch18/gretl_3.png
:name: fig-ardl-residual-correlogram
:width: 90%
:align: center

Correlogram of residuals
```

```{admonition} Interpretation
All autocorrelations should lie within the confidence bands.

Significant spikes suggest missing dynamics.
```

## Step 3: Formal Test for Serial Correlation

In the model window:

`Tests → Autocorrelation`

GRETL reports tests such as the Breusch–Godfrey test and Ljung–Box Q test.

Example output:

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
- $H_1$: serial correlation is present
```

```{admonition} Decision Rule
If the p-value is small, reject $H_0$.

This suggests that the model has not fully captured the dynamic structure of the data.
```

## What If Residuals Are Not White Noise?

If residuals show autocorrelation, possible responses include:

- add more lags of $y_t$
- add more lags of $x_t$
- reconsider the model specification
- check whether variables are nonstationary
- consider cointegration or ECM if levels matter

```{admonition} Key Insight
Failure of residual diagnostics usually means the model has not captured the full dynamic structure of the data.
```

## Important Distinction

```{admonition} Important Distinction
In this chapter, ARDL is used as a **dynamic regression model**.

At this stage, we want residuals to behave like **white noise**.

In the cointegration chapter, residual stationarity becomes a separate and crucial issue.
```

---

# 18.14 Practical Checklist

```{admonition} Diagnostic Checklist
After estimating a dynamic regression or ARDL model:

1. Plot residuals
2. Examine the residual correlogram
3. Perform a serial correlation test
4. Adjust the lag structure if needed
5. Recheck residual diagnostics
```

---

# 18.15 Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Treating a static model as enough**  
Many economic relationships unfold over time.

**2. Adding lags mechanically**  
Lag choice should be guided by diagnostics and economic reasoning.

**3. Ignoring residual autocorrelation**  
Residual autocorrelation means the model has missed dynamics.

**4. Confusing short-run and long-run effects**  
Immediate effects and cumulative effects are not the same.

**5. Forgetting the cost of differencing**  
Differencing may remove meaningful long-run information.
```

---

# 18.16 Looking Ahead

Dynamic models help us understand how variables adjust over time.

But they do not fully resolve the issue of nonstationarity.

```{admonition} The Core Problem
- Using levels may lead to spurious regression.
- Using differences may remove long-run relationships.
```

Is it possible to retain long-run information while avoiding spurious regression?

Yes — if a particular combination of nonstationary variables is stationary.

This concept is called **cointegration**, which we study later.

# Key Takeaways

```{admonition} Summary
- Differenced regressions are dynamic models.
- Distributed lag models capture delayed effects.
- ARDL models combine persistence and distributed lag effects.
- Short-run and long-run effects are different.
- Residuals should behave like white noise.
- Differencing avoids spurious regression but may remove long-run information.
- Dynamic models lead naturally to cointegration and ECM.
```

---

# Appendix 18A — Dynamic Models and Long-Run Effects

This appendix shows how long-run effects arise from a simple ARDL model.

---

## A.1 A Simple ARDL(1,1)

Consider:

```{math}
:enumerated: false
y_t = \alpha + \phi y_{t-1} + \beta_0 x_t + \beta_1 x_{t-1} + u_t
```

---

## A.2 Steady-State Long-Run Equilibrium

In the long run, suppose:

```{math}
:enumerated: false
y_t = y_{t-1}=y
\quad
\text{and}
\quad
x_t = x_{t-1}=x
```

Substitute these into the ARDL(1,1) model:

```{math}
:enumerated: false
y = \alpha + \phi y + \beta_0 x + \beta_1 x
```

So:

```{math}
:enumerated: false
y = \alpha + \phi y + (\beta_0+\beta_1)x
```

Rearranging:

```{math}
:enumerated: false
(1-\phi)y = \alpha + (\beta_0+\beta_1)x
```

Therefore:

```{math}
:enumerated: false
y
=
\frac{\alpha}{1-\phi}
+
\frac{\beta_0+\beta_1}{1-\phi}x
```

---

```{admonition} Key Result
For ARDL(1,1), the long-run multiplier is:

$$
\frac{\beta_0+\beta_1}{1-\phi}
$$
```

---

## A.3 Stability Condition

For the long-run expression to be meaningful, the process must be dynamically stable.

For ARDL(1,1), this requires:

```{math}
:enumerated: false
|\phi| < 1
```

```{admonition} Interpretation
If $|\phi|<1$, shocks decay over time and the system can converge toward a long-run relationship.
```

---

## A.4 Dynamic Adjustment

Start again from:

```{math}
:enumerated: false
y_t = \alpha + \phi y_{t-1} + \beta_0 x_t + \beta_1 x_{t-1} + u_t
```

Subtract $y_{t-1}$ from both sides:

```{math}
:enumerated: false
\Delta y_t
=
\alpha
+
(\phi-1)y_{t-1}
+
\beta_0 x_t
+
\beta_1 x_{t-1}
+
u_t
```

Since:

```{math}
:enumerated: false
x_t = x_{t-1} + \Delta x_t
```

we can write:

```{math}
:enumerated: false
\Delta y_t
=
\alpha
+
\beta_0 \Delta x_t
+
(\phi-1)y_{t-1}
+
(\beta_0+\beta_1)x_{t-1}
+
u_t
```

This expression begins to reveal the link between ARDL and ECM.

---

## A.5 Link to ECM

An ECM has the form:

```{math}
:enumerated: false
\Delta y_t
=
\beta \Delta x_t
+
\gamma (y_{t-1}-\theta x_{t-1})
+
u_t
```

```{admonition} Big Picture
ARDL models naturally contain both:

- short-run dynamics
- long-run equilibrium
```

The ECM representation makes this separation explicit.