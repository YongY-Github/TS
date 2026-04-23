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

### Step 1: Create Data in GRETL

#### Menu

1. Go to: `Add → Random variable → Normal distribution`

2. Generate two variables:

   * `u`
   * `v`

3. Then create cumulative sums: `Add → Define new variable`

   * `x = cum(u)`
   * `y = cum(v)`

### Step 2: Plot the Series

#### Menu

`Graph → Time series plot`

Select `x` and `y`.

---

:::{dropdown} Script
#### GRETL

```gretl
nulldata 200
set seed 1234

series u = normal()
series v = normal()

series x = cum(u)
series y = cum(v)

gnuplot x y --time-series --with-lines
```

#### Python

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

# 1. Initialize dataset size and set the random seed
n = 200
np.random.seed(1234)

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

```{image} figs/ch17/gretl_1.png
:width: 90%
:align: center
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
  const        3.75874     0.394810       9.520   6.24e-018 ***
  x           −0.466912    0.0453499    −10.30    3.49e-020 ***

Mean dependent var   6.740362   S.D. dependent var   4.690510
Sum squared resid    2851.546   S.E. of regression   3.794963
R-squared            0.348691   Adjusted R-squared   0.345401
F(1, 198)            106.0031   P-value(F)           3.49e-20
Log-likelihood      −549.5176   Akaike criterion     1103.035
Schwarz criterion    1109.632   Hannan-Quinn         1105.705
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

### Step 5: Plot Residuals

#### Menu

`Graph → Time series plot`

Select `uhat`.

:::{dropdown} Script
#### GRETL

```gretl
series uhat = $uhat
gnuplot uhat --time-series --with-lines
```

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

```{image} figs/ch17/gretl_2.png
:name: fig-residual-plot
:width: 90%
:align: center

Residual plot
```

Clearly this looks nonstatinary. But let's do a formal check.

---

### Step 6: Check Autocorrelation

#### Menu

`Variable → Correlogram`

Select `uhat`.

---

#### Command

```gretl
corrgm uhat
```

```{image} figs/ch17/gretl_3.png
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

---

```gretl
? adf 0 x

Dickey-Fuller test for x
sample size 199
unit-root null hypothesis: a = 1

  test with constant 
  model: (1-L)y = b0 + (a-1)*y(-1) + e
  estimated value of (a - 1): -0.0168864
  test statistic: tau_c(1) = -1.32174
  p-value 0.6194

  with constant and trend 
  model: (1-L)y = b0 + b1*t + (a-1)*y(-1) + e
  estimated value of (a - 1): -0.0124333
  test statistic: tau_ct(1) = -0.923675
  p-value 0.9503

? adf 0 y

Dickey-Fuller test for y
sample size 199
unit-root null hypothesis: a = 1

  test with constant 
  model: (1-L)y = b0 + (a-1)*y(-1) + e
  estimated value of (a - 1): -0.02404
  test statistic: tau_c(1) = -1.5531
  p-value 0.5047

  with constant and trend 
  model: (1-L)y = b0 + b1*t + (a-1)*y(-1) + e
  estimated value of (a - 1): -0.0163954
  test statistic: tau_ct(1) = -0.945463
  p-value 0.9477

? adf 0 uhat

Dickey-Fuller test for uhat
sample size 199
unit-root null hypothesis: a = 1

  test with constant 
  model: (1-L)y = b0 + (a-1)*y(-1) + e
  estimated value of (a - 1): -0.0471964
  test statistic: tau_c(1) = -2.17431
  p-value 0.2164

  with constant and trend 
  model: (1-L)y = b0 + b1*t + (a-1)*y(-1) + e
  estimated value of (a - 1): -0.0466435
  test statistic: tau_ct(1) = -2.02618
  p-value 0.5831
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
Model 2: OLS, using observations 2-200 (n = 199)
Dependent variable: d_y

             coefficient   std. error   t-ratio   p-value
  -------------------------------------------------------
  const      0.000742608   0.0725424    0.01024   0.9918 
  d_x        0.0625065     0.0680560    0.9185    0.3595 

Mean dependent var   0.000505   S.D. dependent var   1.022925
Sum squared resid    206.2991   S.E. of regression   1.023330
R-squared            0.004264   Adjusted R-squared  -0.000791
F(1, 197)            0.843562   P-value(F)           0.359503
Log-likelihood      −285.9530   Akaike criterion     575.9060
Schwarz criterion    582.4926   Hannan-Quinn         578.5717
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

## Appendix 17A — Why Nonstationarity Leads to Spurious Regression

This appendix provides a more formal explanation of why spurious regression arises when working with non-stationary time series.

### A.1 Setup

Consider two independent random walks:

```{math}
:enumerated: false
x_t = x_{t-1} + u_t, \quad u_t \sim \text{WN}(0, \sigma_u^2)
```

```{math}
:enumerated: false
y_t = y_{t-1} + v_t, \quad v_t \sim \text{WN}(0, \sigma_v^2)
```

Assume:

- $u_t$ and $v_t$ are independent  
- there is **no true relationship** between $x_t$ and $y_t$  

### A.2 Growth of Variance

For a random walk:

```{math}
:enumerated: false
x_t = \sum_{s=1}^t u_s
```

so:

```{math}
:enumerated: false
\text{Var}(x_t) = t \sigma_u^2
```

```{admonition} Key Observation
The variance of a random walk grows over time — it does not remain constant.
````

This is the defining feature of **non-stationarity**.

### A.3 Regression Problem

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

### A.4 Why the Estimator Misbehaves

Even though $x_t$ and $y_t$ are independent:

* both are sums of persistent shocks
* both exhibit strong trending behavior

As $t$ increases:

* $\sum x_t^2$ grows rapidly
* $\sum x_t y_t$ also grows due to shared trending patterns

---

```{admonition} Key Insight
Trending behavior induces **apparent correlation** even when none exists.
```

### A.5 Failure of Standard Asymptotics

In standard regression theory:

* variables are stationary
* variances are constant
* central limit theorem applies

But here:

* variance grows with $t$
* dependence structure is strong
* standard t-statistics no longer follow a t-distribution

---

```{admonition} Important
The usual statistical tests are **not valid** under non-stationarity.
```

### A.6 Residual Behavior

More importantly, if the regression were meaningful, residuals should be stationary.

However, in spurious regression:

```{math}
:enumerated: false
e_t = y_t - \hat{\alpha} - \hat{\beta} x_t
```

inherits non-stationarity from $y_t$ and $x_t$.

```{admonition} Key Diagnostic
Non-stationary residuals are a hallmark of spurious regression.
```

### A.7 Differencing as a Solution

Taking first differences:

```{math}
:enumerated: false
\Delta x_t = u_t, \quad \Delta y_t = v_t
```

which are stationary.

Regression in differences:

```{math}
:enumerated: false
\Delta y_t = \beta_1 \Delta x_t + \varepsilon_t
```

now satisfies standard assumptions.

---

### A.8 Big Picture

```{admonition} Summary
Spurious regression arises because:

- non-stationary series exhibit persistent trends  
- these trends create artificial correlation  
- standard regression theory breaks down  
```

---

### A.9 Looking Ahead

In the next chapter, we will see an important exception:

👉 If a linear combination of non-stationary variables is stationary, then the relationship is **not spurious** — this is called **cointegration**.

---
