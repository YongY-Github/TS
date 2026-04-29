---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 17 — Spurious Regression

In earlier chapters, we learned that many time series are nonstationary. We also saw that random walks can drift over time even when they are generated only by random shocks.

We now study one of the most important warnings in applied time series regression:

```{admonition} Key Idea
Spurious regression occurs when two unrelated time series appear to be statistically related.
```

This problem is especially common when working with nonstationary data, such as random walks or trending variables.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- explain what spurious regression means
- understand why nonstationary series can produce misleading regression results
- recognize warning signs such as high $R^2$ and persistent residuals
- use residual diagnostics to detect problems
- understand why differencing may solve the problem
- distinguish spurious regression from cointegration

---

# 17.1 The Basic Problem

Suppose we regress one time series on another:

$$
y_t = \alpha + \beta x_t + u_t
$$

In ordinary regression analysis, a statistically significant $\beta$ might suggest that $x_t$ is related to $y_t$.

But with time series data, especially nonstationary data, this conclusion can be misleading.

```{admonition} Central Warning
A regression can look statistically impressive even when the variables are completely unrelated.
```

This is the problem of **spurious regression**.

---

# 17.2 Why Nonstationarity Creates Trouble

Consider two independent random walks:

```{math}
:enumerated: false
x_t = x_{t-1} + e_t
```

and

```{math}
:enumerated: false
y_t = y_{t-1} + v_t
```

where $e_t$ and $v_t$ are independent white noise processes.

By construction, there is no true relationship between $x_t$ and $y_t$.

Yet both series may drift over time.

```{admonition} Intuition
Random walks can appear to trend even when the trend is entirely accidental.
```

If two unrelated random walks drift in similar directions over a sample period, a regression may interpret that common movement as evidence of a relationship.

---

# 17.3 Symptoms of Spurious Regression

A spurious regression often produces:

- high $R^2$
- significant t-statistics
- low Durbin–Watson statistic
- persistent residuals
- residual autocorrelation

```{admonition} Warning
:class: warning

A high $R^2$ does not necessarily imply a meaningful relationship when the variables are nonstationary.
```

This is one reason why time series regression requires additional diagnostic checks.

---

# 17.4 A Simple Simulation

Let us simulate two independent random walks.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(124)

n = 200

u = np.random.standard_normal(n)
v = np.random.standard_normal(n)

x = np.cumsum(u)
y = np.cumsum(v)

plt.figure(figsize=(10, 5))
plt.plot(x, label="x")
plt.plot(y, label="y")
plt.title("Two Independent Random Walks")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()

plt.savefig("figs/ch17/rw.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![RW](figs/ch17/rw.png)

```{admonition} Observation
The two series may appear to move together, even though they are independent.
```

---

# 17.5 Regressing One Random Walk on Another

Now regress $y_t$ on $x_t$:

```{math}
:enumerated: false
y_t = \alpha + \beta x_t + u_t
```

```{code-cell} python
import statsmodels.api as sm

X = sm.add_constant(x)
model = sm.OLS(y, X)
results = model.fit()

print(results.summary())
```

```verbatim
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.001
Model:                            OLS   Adj. R-squared:                 -0.004
Method:                 Least Squares   F-statistic:                    0.2312
Date:                Wed, 29 Apr 2026   Prob (F-statistic):              0.631
Time:                        14:10:42   Log-Likelihood:                -607.38
No. Observations:                 200   AIC:                             1219.
Df Residuals:                     198   BIC:                             1225.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          5.4880      0.469     11.692      0.000       4.562       6.414
x1             0.0249      0.052      0.481      0.631      -0.077       0.127
==============================================================================
Omnibus:                       15.351   Durbin-Watson:                   0.038
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               16.030
Skew:                          -0.654   Prob(JB):                     0.000330
Kurtosis:                       2.538   Cond. No.                         12.0
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

The output will often show a statistically significant coefficient, even though the two variables were generated independently.

```{admonition} Surprising Result
You may observe:

- a high $R^2$
- a statistically significant coefficient
- an apparently strong relationship
```

This is the essence of spurious regression.

---

# 17.6 Why This Is Problematic

Both $x_t$ and $y_t$ are nonstationary random walks.

Standard regression inference relies on assumptions that are violated in this setting.

```{admonition} Key Insight
Standard regression tools are designed for stable probabilistic relationships. Random walks do not have stable means and variances.
```

This can lead to:

- misleading t-statistics
- invalid p-values
- false confidence in the estimated relationship

---

# 17.7 Residual Diagnostics

A key diagnostic is to examine the residuals.

If a regression relationship is meaningful, the residuals should be stationary.

If the residuals remain nonstationary, the regression has not captured a stable relationship.

```{admonition} Diagnostic Principle
For a regression between nonstationary variables to be meaningful, the residuals should be stationary.
```

---

## Residual Plot

```{code-cell} python
uhat = results.resid

plt.figure(figsize=(10, 4))
plt.plot(uhat, linewidth=1.5)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Residuals from Spurious Regression")
plt.xlabel("Time")
plt.ylabel("Residual")

plt.savefig("figs/ch17/res.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Residual](figs/ch17/res.png)


## Residual ACF

```{code-cell} python
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(uhat, lags=40)
plt.title("ACF of Residuals")

plt.savefig("figs/ch17/res_acf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Residual ACF](figs/ch17/res_acf.png)

```{admonition} Warning
:class: warning

Persistent residual autocorrelation is a warning sign that the regression is spurious.
```

---

# 17.8 Unit Root Testing

To formally check nonstationarity, we can perform unit root tests.

For the original series, we expect:

- $x_t$ to be nonstationary
- $y_t$ to be nonstationary

For a spurious regression, residuals are often nonstationary as well.

```{code-cell} python
from statsmodels.tsa.stattools import adfuller

def adf_summary(series, name):
    result = adfuller(series)
    print(f"ADF test for {name}")
    print(f"ADF statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    print()

adf_summary(x, "x")
adf_summary(y, "y")
adf_summary(uhat, "residuals")
```

```verbatim
ADF test for x
ADF statistic: -0.7378
p-value: 0.8367
Critical values:
  1%: -3.4638
  5%: -2.8763
  10%: -2.5746

ADF test for y
ADF statistic: -0.6210
p-value: 0.8662
Critical values:
  1%: -3.4636
  5%: -2.8762
  10%: -2.5746

ADF test for residuals
ADF statistic: -0.5837
p-value: 0.8746
Critical values:
  1%: -3.4636
  5%: -2.8762
  10%: -2.5746
```

```{admonition} Interpretation
If the ADF test fails to reject the unit-root null, the series should be treated as nonstationary.
```

---

# 17.9 Fixing the Problem: Differencing

A common solution is to difference the data.

Instead of estimating:

```{math}
:enumerated: false
y_t = \alpha + \beta x_t + u_t
```

we estimate:

```{math}
:enumerated: false
\Delta y_t = \alpha + \beta \Delta x_t + u_t
```

where:

```{math}
:enumerated: false
\Delta x_t = x_t - x_{t-1}
```

and:

```{math}
:enumerated: false
\Delta y_t = y_t - y_{t-1}
```

```{admonition} Key Idea
Differencing removes stochastic trends and focuses on short-run changes.
```

---

## Regression in Differences

```{code-cell} python
d_x = np.diff(x)
d_y = np.diff(y)

D_X = sm.add_constant(d_x)

diff_model = sm.OLS(d_y, D_X)
diff_results = diff_model.fit()

print(diff_results.summary())
```

```verbatim
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.005
Model:                            OLS   Adj. R-squared:                 -0.000
Method:                 Least Squares   F-statistic:                    0.9755
Date:                Wed, 29 Apr 2026   Prob (F-statistic):              0.325
Time:                        14:14:36   Log-Likelihood:                -279.21
No. Observations:                 199   AIC:                             562.4
Df Residuals:                     197   BIC:                             569.0
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         -0.0405      0.070     -0.576      0.565      -0.179       0.098
x1             0.0719      0.073      0.988      0.325      -0.072       0.215
==============================================================================
Omnibus:                        0.339   Durbin-Watson:                   1.967
Prob(Omnibus):                  0.844   Jarque-Bera (JB):                0.181
Skew:                          -0.067   Prob(JB):                        0.914
Kurtosis:                       3.063   Cond. No.                         1.08
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

```{admonition} Observation
After differencing, the apparent relationship usually disappears.
```

This is what we should expect, because the two random walks were generated from independent shocks.

---

# 17.10 Interpretation

Spurious regression arises because nonstationary series can move together over time even when there is no true relationship.

```{admonition} Key Lesson
Trending behavior can create artificial correlation.
```

Differencing often solves the problem by removing the stochastic trend.

However, differencing is not always the final answer.

Sometimes nonstationary variables really do move together because of a long-run equilibrium relationship.

This leads to the idea of **cointegration**.

---

# 17.11 Connection to Cointegration

There is one important exception to the warning about regressions in levels.

```{admonition} Important
If two nonstationary variables have stationary residuals, the regression may be meaningful.

This is called **cointegration**.
```

In other words:

- nonstationary residuals → spurious regression
- stationary residuals → possible cointegration

We return to this idea in Chapter 20.

---

# 17.12 GRETL Example

We now reproduce the simulation and regression in GRETL.

## Step 1: Create Data in GRETL

### Menu

1. Set up for data entry:  
   `File → New data set`

2. Select:  
   `Time series: T = 200`

3. Choose frequency:  
   `Other`

This creates an `index` variable.

Then generate two white-noise variables:

`Add → Random variable → Normal distribution`

Generate:

- `u`
- `v`

Then create cumulative sums:

`Add → Define new variable`

Use:

```text
x = cum(u)
y = cum(v)
```

### Command

```gretl
nulldata 200
set seed 124

series u = normal()
series v = normal()

series x = cum(u)
series y = cum(v)
```

## Step 2: Plot the Series

### Menu

`Graph → Time series plot`

Select:

- `x`
- `y`

### Command

```gretl
gnuplot x y --time-series --with-lines
```

```{figure} figs/ch17/gretl_1.png
:name: fig-random-walks
:width: 90%
:align: center

Two independent random walks
```

## Step 3: Run the Spurious Regression

### Menu

`Model → Ordinary Least Squares`

- Dependent variable: `y`
- Independent variable: `x`

### Command

```gretl
ols y const x
```

Example output:

```gretl
Model 1: OLS, using observations 1-200
Dependent variable: y

             coefficient   std. error   t-ratio    p-value 
  ---------------------------------------------------------
  const       0.578591     0.295706      1.957    0.0518    *
  x           0.660102     0.0333565    19.79     8.40e-049 ***

Mean dependent var  −4.061555   S.D. dependent var   4.385945
Sum squared resid    1285.510   S.E. of regression   2.548033
R-squared            0.664188   Adjusted R-squared   0.662492
F(1, 198)            391.6158   P-value(F)           8.40e-49
Log-likelihood      −469.8470   Akaike criterion     943.6941
Schwarz criterion    950.2907   Hannan-Quinn         946.3636
```

```{admonition} Interpretation
The regression appears highly significant, but this is misleading because $x$ and $y$ are independent random walks.
```

## Step 4: Save and Plot Residuals

### Menu

In the regression window:

`Save → Residuals`

### Command

```gretl
series uhat = $uhat
gnuplot uhat --time-series --with-lines
```

```{figure} figs/ch17/gretl_2.png
:name: fig-spurious-residuals
:width: 70%
:align: center

Residuals from the spurious regression
```

The residuals still show persistent behavior.

## Step 5: Check Residual Autocorrelation

### Menu

`Variable → Correlogram`

Select `uhat`.

If you do not see this option, go to:

`Data → Dataset structure...`

and select:

`Time series → Other`

You can also right click on `uhat` and select `Correlogram`.

### Command

```gretl
corrgm uhat
```

```{figure} figs/ch17/gretl_3.png
:name: fig-spurious-residual-correlogram
:width: 90%
:align: center

Correlogram of residuals
```

## Step 6: ADF Tests

### Menu

`Variable → Unit root tests → Augmented Dickey-Fuller`

Test:

- `x`
- `y`
- `uhat`

### Command

```gretl
adf 0 x
adf 0 y
adf 0 uhat
```

It is often better to use the menu because GRETL can help determine an appropriate lag length for the ADF test.

Example output for `x`:

```gretl
Augmented Dickey-Fuller test for x
testing down from 14 lags, criterion AIC
sample size 199
unit-root null hypothesis: a = 1

  test with constant 
  including 0 lags of (1-L)x
  model: (1-L)y = b0 + (a-1)*y(-1) + e
  estimated value of (a - 1): -0.0224933
  test statistic: tau_c(1) = -1.57309
  asymptotic p-value 0.4964
```

Example output for `y`:

```gretl
Augmented Dickey-Fuller test for y
testing down from 14 lags, criterion AIC
sample size 199
unit-root null hypothesis: a = 1

  test with constant 
  including 0 lags of (1-L)y
  estimated value of (a - 1): -0.0198401
  test statistic: tau_c(1) = -1.22352
  asymptotic p-value 0.6666
```

Example output for `uhat`:

```gretl
Augmented Dickey-Fuller test for uhat
testing down from 14 lags, criterion AIC
sample size 195
unit-root null hypothesis: a = 1

  test with constant 
  including 4 lags of (1-L)uhat
  estimated value of (a - 1): -0.117113
  test statistic: tau_c(1) = -2.93493
  asymptotic p-value 0.04143
```

```{admonition} Note
In finite samples, residual-based unit-root tests can sometimes give borderline results.

The broader lesson remains: regressions involving nonstationary variables require careful residual diagnostics.
```

## Step 7: Difference the Data

### Menu

Select `x` and `y`, right click, and choose:

`Add differences`

### Command

```gretl
series d_x = diff(x)
series d_y = diff(y)
```

## Step 8: Re-estimate the Regression in Differences

### Menu

`Model → Ordinary Least Squares`

- Dependent variable: `d_y`
- Independent variable: `d_x`

### Command

```gretl
ols d_y const d_x
```

Example output:

```gretl
Model 2: OLS, using observations 2-200 (T = 199)
Dependent variable: d_y

             coefficient   std. error   t-ratio   p-value
  -------------------------------------------------------
  const      −0.0721970    0.0706927    −1.021    0.3084 
  d_x        −0.0574247    0.0647031    −0.8875   0.3759 

Mean dependent var  −0.068429   S.D. dependent var   0.994910
Sum squared resid    195.2089   S.E. of regression   0.995444
R-squared            0.003982   Adjusted R-squared  -0.001073
F(1, 197)            0.787676   P-value(F)           0.375886
Log-likelihood      −280.4549   Akaike criterion     564.9099
Schwarz criterion    571.4965   Hannan-Quinn         567.5757
rho                  0.000807   Durbin-Watson        1.978732
```

```{admonition} Observation
After differencing, the relationship disappears.
```

---

# 17.13 Practical Checklist

```{admonition} Practical Checklist
When working with time series regressions:

1. Plot the variables  
2. Check for trends and persistence  
3. Test for unit roots  
4. Estimate the regression  
5. Examine residuals  
6. Check residual autocorrelation  
7. Test whether residuals are stationary  
8. Consider differencing or cointegration  
```

---

# 17.14 Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Trusting high $R^2$ too quickly**  
A high $R^2$ may simply reflect common trending behavior.

**2. Ignoring stationarity**  
Regression with nonstationary variables can be misleading.

**3. Skipping residual diagnostics**  
Residual behavior is crucial.

**4. Differencing mechanically**  
Differencing removes long-run information, so it should be used carefully.

**5. Forgetting cointegration**  
Some regressions in levels are meaningful if residuals are stationary.
```

---

# 17.15 Looking Ahead

Spurious regression teaches us a central lesson:

```{admonition} Big Lesson
With time series data, regression is not enough. We must also understand the time-series properties of the variables.
```

In the next chapter, we introduce dynamic regression models, including distributed lag models and ARDL models.

These models allow us to study how variables affect each other over time.

# Key Takeaways

```{admonition} Summary
- Spurious regression occurs when unrelated nonstationary variables appear related.
- Random walks can drift together by chance.
- High $R^2$ and significant coefficients can be misleading.
- Residual diagnostics are essential.
- Differencing often removes spurious relationships.
- Cointegration provides an important exception.
```

---

# Appendix 17A — Why Nonstationarity Leads to Spurious Regression

This appendix provides an intuitive but slightly more formal explanation of why regressions involving nonstationary time series can produce misleading results.

---

## A.1 Setup

Consider two independent random walks:

```{math}
:enumerated: false
x_t = x_{t-1} + e_t, \quad e_t \sim \text{WN}(0, \sigma_u^2)
```

```{math}
:enumerated: false
y_t = y_{t-1} + v_t, \quad v_t \sim \text{WN}(0, \sigma_v^2)
```

Assume:

- $e_t$ and $v_t$ are independent
- there is **no true relationship** between $x_t$ and $y_t$

```{admonition} Key Point
By construction, $x_t$ and $y_t$ are completely unrelated.
```

---

## A.2 Accumulation of Shocks

A random walk can be written as:

```{math}
:enumerated: false
x_t = \sum_{s=1}^t e_s
```

so its variance is:

```{math}
:enumerated: false
\text{Var}(x_t) = t \sigma_e^2
```

```{admonition} Key Observation
The variance of a random walk **grows over time** — it does not remain constant.
```

This is the defining feature of **nonstationarity**.

---

## A.3 Persistent Trending Behavior

Because shocks accumulate:

- both $x_t$ and $y_t$ tend to drift over time
- they exhibit persistent trending behavior

Even though these trends are random, they can look systematic in finite samples.

```{admonition} Intuition
Random walks often appear to trend, even when driven purely by chance.
```

---

## A.4 The Regression Problem

Now consider the regression:

```{math}
:enumerated: false
y_t = \alpha + \beta x_t + e_t
```

The OLS estimator is:

```{math}
:enumerated: false
\hat{\beta} = \frac{\sum x_t y_t}{\sum x_t^2}
```

This expression is written in simplified mean-zero form. The same intuition applies when an intercept is included.

---

## A.5 Why the Estimator Misbehaves

Even though $x_t$ and $y_t$ are independent:

- both contain persistent trends
- large values of $x_t$ may coincide with large values of $y_t$
- both $\sum x_t^2$ and $\sum x_t y_t$ can grow because shocks accumulate

```{admonition} Key Insight
Trending behavior induces **apparent correlation**, even when variables are unrelated.
```

---

## A.6 Failure of Standard Inference

Standard regression theory assumes:

- constant variance
- weak dependence
- stable distributions over time

But with nonstationary data:

- variance increases with time
- shocks have long-lasting effects
- observations are highly dependent

```{admonition} Important
Standard t-tests and F-tests are **not valid** when variables are nonstationary.
```

---

## A.7 Residual Behavior

If the regression were meaningful, the residuals should be stationary.

However:

```{math}
:enumerated: false
u_t = y_t - \hat{\alpha} - \hat{\beta}x_t
```

often inherits nonstationarity from $y_t$ and $x_t$.

```{admonition} Key Diagnostic
Nonstationary residuals are a hallmark of **spurious regression**.
```

---

## A.8 Big Picture

```{admonition} Summary
Spurious regression arises because:

- nonstationary series drift over time
- random trends can align by chance
- OLS interprets this as a meaningful relationship
- standard inference breaks down
```

---

## A.9 How to Fix the Problem

There are two main approaches:

- difference the data, focusing on short-run relationships
- test for cointegration, recovering possible long-run relationships

```{admonition} Looking Ahead
In later chapters, we will see that if a linear combination of nonstationary variables is stationary, the relationship is **not spurious**. This is called **cointegration**.
```