---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 17 — Spurious Regression

## 17.1 Introduction

In time series analysis, regression can sometimes produce **misleading results** when applied to non-stationary data.

```{admonition} Key Idea
Spurious regression occurs when two unrelated time series appear to be statistically related.
````

This problem is especially common when working with **random walks** or trending variables.

---

## 17.2 A Simulation in GRETL

We begin by generating two independent random walks.

---

#### Menu

### Step 1: Create Data in GRETL

1. Set up for data entry: `File → New data set` and select `Time series: T=200` with time series frequency as `Other`. This will create an `index` variable.

3.   Go to: `Add → Random variable → Normal distribution`

4. Generate two variables:

   * `u`
   * `v`

5. Then create cumulative sums: `Add → Define new variable`

   * `x = cum(u)`
   * `y = cum(v)`

### Step 2: Plot the Series

`Graph → Time series plot`

Select `x` and `y`.

---

#### Command

```gretl
nulldata 200
set seed 124

series u = normal()
series v = normal()

series x = cum(u)
series y = cum(v)

gnuplot x y --time-series --with-lines
```

:::{dropdown} Python Script

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

# 1. Initialize dataset size and set the random seed
n = 200
np.random.seed(124)

# 2. Generate series u and v from a standard normal distribution
u = np.random.standard_normal(n)
v = np.random.standard_normal(n)

# 3. Create random walks x and y using the cumulative sum
x = np.cumsum(u)
y = np.cumsum(v)

# 4. Plot the results as a time series
plt.figure(figsize=(10, 5))
plt.plot(x, label='x', color='blue')
plt.plot(y, label='y', color='red')

# Formatting the plot
plt.title('Random Walk Series (x and y)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
````

:::

---

```{figure} figs/ch17/gretl_1.png
:name: fig-random_walks
:width: 90%
:align: center

Two random walks
```

---

```{admonition} Observation
The two series may appear to move together, even though they are independent.
```

---

## 17.3 Spurious Regression

Now regress $y_t$ on $x_t$.

---

### Step 3: Run Regression

#### Menu

`Model → Ordinary Least Squares`

* Dependent variable: `y`
* Independent variable: `x`

---

#### Command

```gretl
ols y const x
```

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

---

```{admonition} Surprising Result
You will often observe:
- high $R^2$  
- statistically significant coefficient  
```

:::{dropdown} Python Script
#### Python

```{code-cell} python
import statsmodels.api as sm

# 1. Statsmodels requires adding a constant term for the intercept
# This transforms x from a 1D array into a 2D array with a column of ones
X = sm.add_constant(x)

# 2. Fit the Ordinary Least Squares (OLS) model
# The order is (dependent variable, independent variable)
model = sm.OLS(y, X)
results = model.fit()

# 3. Print the comprehensive regression results
print(results.summary())

# Optional: Access specific values
print(f"\nCoefficient for x: {results.params[1]:.4f}")
print(f"\nThe t-statistic for x is: {results.tvalues[1]:.4f}")
```
:::

---

## 17.4 Why Is This Problematic?

Both $x_t$ and $y_t$ are **non-stationary random walks**.

```{admonition} Key Insight
Standard regression assumptions require stationarity — which is violated here.
```

This leads to:

* misleading t-statistics
* invalid inference
* false sense of relationship

---

## 17.5 Residual Diagnostics

A crucial step is to examine the residuals.

---

### Step 4: Save Residuals

#### Menu

In regression window:

`Save → Residuals`

#### Command

```gretl
series uhat = $uhat
```

### Step 5: Plot Residuals

#### Menu

`Graph → Time series plot`

Select `uhat`.

#### Command

```gretl
gnuplot uhat --time-series --with-lines
```

:::{dropdown} Python Script

#### Python

```{code-cell} python
import matplotlib.pyplot as plt

# 1. Capture the residuals (the Python version of $uhat)
uhat = results.resid

# 2. Plot the residuals as a time series
plt.figure(figsize=(10, 4))
plt.plot(uhat, color='purple', linewidth=1.5)

# Adding the '0' line for reference (common in residual plots)
plt.axhline(0, color='black', linestyle='--', linewidth=1)

# Formatting
plt.title('Residuals (uhat) from OLS Regression')
plt.xlabel('Observation')
plt.ylabel('Residual Value')
plt.grid(True, alpha=0.3)
plt.show()
```
:::

---

```{figure} figs/ch17/gretl_2.png
:name: fig-resid
:width: 70%
:align: center

Residual plot
```

Clearly this looks nonstatinary. But let's do a formal check.

---

### Step 6: Check Autocorrelation

#### Menu

`Variable → Correlogram`

Select `uhat`.

If you do not see this option, go to `Data → Dataset structure...` and select `Time series`... `Other`. You can also right click on `uhat` and select `Correlogram`.
 
---

#### Command

```gretl
corrgm uhat
```

```{figure} figs/ch17/gretl_3.png
:name: fig-correlogram
:width: 90%
:align: center

Correlogram for residual
```

---

```{admonition} Warning
:class: warning

Residuals typically show strong persistence → evidence of spurious regression.
```

---

## 17.6 Unit Root Testing

To confirm non-stationarity, we perform a unit root test.

---

### Step 7: ADF Test

#### Menu

`Variable → Unit root tests → Augmented Dickey-Fuller`

Test:

* `x`
* `y`
* `uhat`

---

#### Command

```gretl
adf 0 x
adf 0 y
adf 0 uhat
```
It is better to use the **menu** because Gretl will determine the approproiate lag for the ADF test. 

---

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
  1st-order autocorrelation coeff. for e: -0.030

  with constant and trend 
  including 0 lags of (1-L)x
  model: (1-L)y = b0 + b1*t + (a-1)*y(-1) + e
  estimated value of (a - 1): -0.0780285
  test statistic: tau_ct(1) = -2.78519
  asymptotic p-value 0.2027
  1st-order autocorrelation coeff. for e: -0.001
```

```gretl
Augmented Dickey-Fuller test for y
testing down from 14 lags, criterion AIC
sample size 199
unit-root null hypothesis: a = 1

  test with constant 
  including 0 lags of (1-L)y
  model: (1-L)y = b0 + (a-1)*y(-1) + e
  estimated value of (a - 1): -0.0198401
  test statistic: tau_c(1) = -1.22352
  asymptotic p-value 0.6666
  1st-order autocorrelation coeff. for e: 0.012

  with constant and trend 
  including 0 lags of (1-L)y
  model: (1-L)y = b0 + b1*t + (a-1)*y(-1) + e
  estimated value of (a - 1): -0.0553024
  test statistic: tau_ct(1) = -2.3086
  asymptotic p-value 0.4287
  1st-order autocorrelation coeff. for e: 0.027
```

```gretl
Augmented Dickey-Fuller test for uhat
testing down from 14 lags, criterion AIC
sample size 195
unit-root null hypothesis: a = 1

  test with constant 
  including 4 lags of (1-L)uhat
  model: (1-L)y = b0 + (a-1)*y(-1) + ... + e
  estimated value of (a - 1): -0.117113
  test statistic: tau_c(1) = -2.93493
  asymptotic p-value 0.04143
  1st-order autocorrelation coeff. for e: 0.004
  lagged differences: F(4, 189) = 2.860 [0.0248]

  with constant and trend 
  including 4 lags of (1-L)uhat
  model: (1-L)y = b0 + b1*t + (a-1)*y(-1) + ... + e
  estimated value of (a - 1): -0.117152
  test statistic: tau_ct(1) = -2.93311
  asymptotic p-value 0.1519
  1st-order autocorrelation coeff. for e: 0.003
  lagged differences: F(4, 188) = 2.821 [0.0264]
```

:::{dropdown} Python Script
#### Python

```{code-cell} python
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# Using the residuals (uhat) from our previous regression
# lags=40 is a common default for 200 observations
plot_acf(uhat, lags=40)
plt.title("ACF of Residuals (uhat)")
plt.show()

# To get the raw numerical values:
from statsmodels.tsa.stattools import acf
acf_values = acf(uhat, nlags=40)
print(f"First 5 ACF values: {acf_values[:5]}")
```

```{code-cell} python
from statsmodels.tsa.stattools import adfuller

# Run the test on the residuals
adf_result = adfuller(uhat)

# Formatting the output for readability
print('ADF Statistic: %f' % adf_result[0])
print('p-value: %f' % adf_result[1])
print('Critical Values:')
for key, value in adf_result[4].items():
    print('\t%s: %.3f' % (key, value))

# Interpretation
if adf_result[1] < 0.05:
    print("\nResult: Reject the null hypothesis (Series is stationary)")
else:
    print("\nResult: Fail to reject the null hypothesis (Series is non-stationary)")
```

:::

---

```{admonition} Interpretation
- $x$ and $y$: non-stationary  
- residuals: (often) non-stationary  
```

---

## 17.7 Fixing the Problem: Differencing

We now difference the data.

### Step 8: Create Differences

#### Menu

Select `x` and `y`, right click and `Add differences`

---

#### Command

```gretl
series d_x = diff(x)
series d_y = diff(y)
```

### Step 9: Re-estimate Regression

#### Menu

`Model → OLS`

* Dependent: `d_y`
* Independent: `d_x`

---

#### Command

```gretl
ols d_y const d_x
```

```markdown
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

:::{dropdown} Python Script
#### Python

```{code-cell} python
# import numpy as np
# import statsmodels.api as sm

# 1. Create first differences: d_x = x_t - x_{t-1}
# Note: np.diff(x) results in an array of length 199 (if n was 200)
d_x = np.diff(x)
d_y = np.diff(y)

# 2. Add the constant (intercept) to the independent variable
# This is the Python equivalent of 'const' in gretl
D_X = sm.add_constant(d_x)

# 3. Run the OLS regression
# We are regressing the change in y on the change in x
diff_model = sm.OLS(d_y, D_X)
diff_results = diff_model.fit()

# 4. Show the results
print(diff_results.summary())
```
:::

---

```{admonition} Observation
After differencing:
- coefficients usually become insignificant  
- relationship disappears  
```

---

## 17.8 Interpretation

```{admonition} Key Lesson
Spurious regression arises because trending series move together over time, not because they are truly related.
```

---

## 17.9 Connection to Cointegration

There is one important exception:

```{admonition} Important
If residuals are stationary, the regression may be meaningful — this is called **cointegration**.
```

---

## 17.10 Summary Workflow (GRETL)

```{admonition} Practical Checklist
1. Plot the data  
2. Run regression  
3. Examine residuals  
4. Check autocorrelation  
5. Perform unit root tests  
6. Difference the data if necessary  
```

---

## Key Takeaways

* Spurious regression is common with non-stationary data
* High $R^2$ does not imply a true relationship
* Residual diagnostics are essential
* Differencing often resolves the issue
* Cointegration provides an alternative framework

---

This is a great opportunity — both versions are strong but with slightly different strengths:

* your earlier version → **more formal (variance growth, estimator behavior)**
* my later version → **clean intuition + narrative clarity**

👉 The goal is to **blend them into one coherent appendix**:

* keep the **formal backbone**
* preserve the **intuitive storyline**
* avoid redundancy

---

## Appendix 17A — Why Nonstationarity Leads to Spurious Regression

This appendix provides an intuitive but slightly more formal explanation of why regressions involving non-stationary time series can produce misleading results.

### A.1 Setup

Consider two independent random walks:

```{math}
:enumerated: false
x_t = x_{t-1} + u_t, \quad u_t \sim \text{WN}(0, \sigma_u^2)
````

```{math}
:enumerated: false
y_t = y_{t-1} + v_t, \quad v_t \sim \text{WN}(0, \sigma_v^2)
```

Assume:

* $u_t$ and $v_t$ are independent
* there is **no true relationship** between $x_t$ and $y_t$

```{admonition} Key Point
By construction, $x_t$ and $y_t$ are completely unrelated.
```

### A.2 Accumulation of Shocks

A random walk can be written as:

```{math}
:enumerated: false
x_t = \sum_{s=1}^t u_s
```

so its variance is:

```{math}
:enumerated: false
\text{Var}(x_t) = t \sigma_u^2
```

```{admonition} Key Observation
The variance of a random walk **grows over time** — it does not remain constant.
```

This is the defining feature of **non-stationarity**.

### A.3 Persistent Trending Behavior

Because shocks accumulate:

* both $x_t$ and $y_t$ tend to **drift over time**
* they exhibit **persistent trending behavior**

Even though these trends are random, they can look systematic in finite samples.

```{admonition} Intuition
Random walks often appear to trend, even when driven purely by chance.
```

### A.4 The Regression Problem

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

### A.5 Why the Estimator Misbehaves

Even though $x_t$ and $y_t$ are independent:

* both contain **persistent trends**
* large values of $x_t$ tend to coincide with large values of $y_t$

As the sample grows:

* $\sum x_t^2$ increases
* $\sum x_t y_t$ also increases due to shared trending behavior

```{admonition} Key Insight
Trending behavior induces **apparent correlation**, even when variables are unrelated.
```

### A.6 Failure of Standard Inference

Standard regression theory assumes:

* constant variance
* weak dependence
* stable distributions over time

But with non-stationary data:

* variance increases with $t$
* shocks have long-lasting effects
* observations are highly dependent

```{admonition} Important
Standard t-tests and F-tests are **not valid** when variables are non-stationary.
```

### A.7 Residual Behavior

If the regression were meaningful, the residuals should be stationary.

However:

```{math}
:enumerated: false
e_t = y_t - \hat{\alpha} - \hat{\beta} x_t
```

inherits non-stationarity from $y_t$ and $x_t$.

```{admonition} Key Diagnostic
Non-stationary residuals are a hallmark of **spurious regression**.
```

### A.8 Big Picture

```{admonition} Summary
Spurious regression arises because:

- non-stationary series drift over time  
- random trends can align by chance  
- OLS interprets this as a meaningful relationship  
- standard inference breaks down  
```

### A.9 How to Fix the Problem

There are two main approaches:

* difference the data (focus on short-run relationships)
* test for cointegration (recover long-run relationships)


```{admonition} Looking Ahead
In later chapters, we will see that if a linear combination of non-stationary variables is stationary, the relationship is **not spurious** — this is called **cointegration**.
```