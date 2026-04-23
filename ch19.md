---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 19 - Predictive Causality (Granger Causality)

In the previous chapter, we introduced **dynamic models (ARDL)** and emphasized the importance of capturing temporal dependence.

In this chapter, we ask a natural question:

> **Does one variable help predict another?**

This idea is formalized using **Granger causality**.

## Learning Objectives

By the end of this chapter, you should be able to:

- understand the idea of predictive (Granger) causality  
- distinguish Granger causality from true causality  
- implement Granger causality tests in GRETL  
- interpret test results carefully  
- understand limitations and common pitfalls  

---

## 19.1 What Is Granger Causality?

Suppose we are interested in whether a variable $x_t$ helps predict another variable $y_t$.

We say that:

```{admonition} Definition: Granger Causality
Variable $x_t$ **Granger-causes** $y_t$ if past values of $x_t$ help predict $y_t$, beyond what is possible using only past values of $y_t$.
````

### A Simple Comparison

We compare two models:

### Model 1 (restricted)

$$
y_t = \alpha + \sum_{i=1}^{p} \phi_i y_{t-i} + e_t
$$

### Model 2 (unrestricted)

$$
y_t = \alpha + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{j=1}^{q} \beta_j x_{t-j} + u_t
$$

By the F-test:

$$
F = \frac{(SSR_R - SSR_U) / q}{SSR_U / (n - k - 1)}
$$

where 

* $SSR_R$ is the sum of square residuals for the restricted model
* $SSR_U$ is the sum of square residuals for the unrestricted model

```{admonition} Key Idea
If including lagged values of $x_t$ improves prediction of $y_t$, then $x_t$ Granger-causes $y_t$.
```

---

## 19.2 Intuition

Granger causality is about **information and prediction**, not true causality.

```{admonition} Important
Granger causality does **not** mean that $x_t$ truly causes $y_t$ in a structural or economic sense.

It only means that $x_t$ contains useful predictive information about $y_t$.
```

### Example (Economic Intuition)

* Interest rates → inflation
* Income → consumption
* Money supply → output

In each case, we ask:

👉 Do past values of one variable help forecast another?

## 19.3 Hypothesis Testing Framework

We test:

$$
H_0: \beta_1 = \beta_2 = \cdots = \beta_q = 0
$$

$$
H_1: \text{At least one } \beta_j \neq 0
$$

---

```{admonition} Interpretation
- Fail to reject $H_0$ → no Granger causality  
- Reject $H_0$ → evidence of Granger causality  
```

---

## 19.4 Implementation in GRETL

### Menu

#### Step 1: Estimate the Model

`File → Open data → Sample file...` and select `jgm-data` data from the `Gretl` database.

* Select variables:

```gretl
pi_c     Inflation rate based on the CPI
r_s      Short term iterest rate
```

#### Step 2: Restricted model AR(p)

`Model → Univariate time series → ARIMA lag selection`

Dependent variable `pi_c`  and try AR = 5:

```gretl
Function evaluations: 23
Evaluations of gradient: 10

Model 1: ARMA, using observations 1952-1994 (T = 43)
Estimated using AS 197 (exact ML)
Dependent variable: pi_c
Standard errors based on Hessian

             coefficient   std. error      z       p-value 
  ---------------------------------------------------------
  const       3.82979       1.40057      2.734    0.0062    ***
  phi_1       1.08168       0.162053     6.675    2.47e-011 ***
  phi_2      −0.253378      0.246877    −1.026    0.3047   
  phi_3       0.0181131     0.166423     0.1088   0.9133   

Mean dependent var   4.246077   S.D. dependent var   3.195937
Mean of innovations  0.014738   S.D. of innovations  1.494649
R-squared            0.776339   Adjusted R-squared   0.765156
Log-likelihood      −79.07128   Akaike criterion     168.1426
Schwarz criterion    176.9486   Hannan-Quinn         171.3899

                        Real  Imaginary    Modulus  Frequency
  -----------------------------------------------------------
  AR
    Root  1           1.2659     0.0000     1.2659     0.0000
    Root  2           6.3614    -1.7735     6.6040    -0.0433
    Root  3           6.3614     1.7735     6.6040     0.0433
  -----------------------------------------------------------
```

The above suggest one lag, i.e. AR(1).

Estimate this AR(1)

`Model → Univariate time series → ARIMA`

```gretl
Function evaluations: 23
Evaluations of gradient: 10

Model 2: ARMA, using observations 1952-1994 (T = 43)
Estimated using AS 197 (exact ML)
Dependent variable: pi_c
Standard errors based on Hessian

             coefficient   std. error     z       p-value 
  --------------------------------------------------------
  const       3.53530      1.64385       2.151   0.0315    **
  phi_1       0.873605     0.0693047    12.61    1.97e-036 ***

Mean dependent var   4.246077   S.D. dependent var   3.195937
Mean of innovations  0.034572   S.D. of innovations  1.532847
R-squared            0.764867   Adjusted R-squared   0.764867
Log-likelihood      −80.10105   Akaike criterion     166.2021
Schwarz criterion    171.4857   Hannan-Quinn         168.1505

                        Real  Imaginary    Modulus  Frequency
  -----------------------------------------------------------
  AR
    Root  1           1.1447     0.0000     1.1447     0.0000
  -----------------------------------------------------------
```

Note your SSR:

`Save → Error sum of squares` as scalar = 101.033673577098

#### Step 3: Unrestricted model

`Model → Multivariate time series → VAR lag selection`

Dependent variable `pi_c`  and `r_s` as exogenous variable and maximum lag = 5:

```gretl
VAR system, maximum lag order 5

The asterisks below indicate the best (that is, minimized) values
of the respective information criteria, AIC = Akaike criterion,
BIC = Schwarz Bayesian criterion and HQC = Hannan-Quinn criterion.

lags        loglik    p(LR)       AIC          BIC          HQC

   1     -68.39186             3.757466     3.886749     3.803464 
   2     -65.81966  0.02332    3.674719*    3.847096*    3.736049*
   3     -65.81635  0.93515    3.727176     3.942648     3.803839 
   4     -65.76655  0.75230    3.777187     4.035753     3.869182 
   5     -65.76616  0.97785    3.829798     4.131458     3.937126 
```

AIC (BIC and HQC) suggest 2 lags is optimum.

`Model → Univariate time series → ARIMA`

* Dependent: `pi_c`
* Regressors: `r_s(-1) r_s(-2)`

Use `lags...` in the model box (for regressors) and set AR = 1.

```gretl
Function evaluations: 23
Evaluations of gradient: 10

Model 3: ARMAX, using observations 1954-1994 (T = 41)
Estimated using AS 197 (exact ML)
Dependent variable: pi_c
Standard errors based on Hessian

             coefficient   std. error     z       p-value 
  --------------------------------------------------------
  const        2.48133     1.73771       1.428   0.1533   
  phi_1        0.886579    0.0736422    12.04    2.22e-033 ***
  r_s_1        0.342550    0.111572      3.070   0.0021    ***
  r_s_2       −0.197852    0.109574     −1.806   0.0710    *

Mean dependent var   4.415833   S.D. dependent var   3.153664
Mean of innovations  0.076028   S.D. of innovations  1.288293
R-squared            0.829694   Adjusted R-squared   0.820731
Log-likelihood      −69.33345   Akaike criterion     148.6669
Schwarz criterion    157.2348   Hannan-Quinn         151.7868

                        Real  Imaginary    Modulus  Frequency
  -----------------------------------------------------------
  AR
    Root  1           1.1279     0.0000     1.1279     0.0000
  -----------------------------------------------------------
```

Note your SSR for the unrestricted model:

`Save → Error sum of squares` as scalar = 68.0476155829897

#### Step 4: Perform Granger Causality Test

So we now have:

SSR unrestricted and SSR restricted, which you can place into the formula for F-test:

```{math}
:enumerated: false
F = \frac{(SSR_R - SSR_U) / q}{SSR_U / (n - k - 1)} = \frac{(101.03-68.05)/2}{101.03/(41-3-1)} = 6.04
```

At 0.05 level of significance, the critical value is F(2,37) = 3.25192.

So we reject the null and coclude that `r_s` or short term interst rates **granger causes** `pi_c` or CPI inflation rates.

#### VAR alternative

An alternative is to invoke the VAR command in Gretl:

`Model → Multivariate time series → Vector Autoregression` then select **lag order = 2**  and place both `pi_c` and `r_c` as endogenous variables.

#### Command

```gretl
var 2 pi_c r_c
```

```gretl
VAR system, lag order 2
OLS estimates, observations 1954-1994 (T = 41)
Log-likelihood = -142.63358
Determinant of covariance matrix = 3.6037735
AIC = 7.4455
BIC = 7.8635
HQC = 7.5977
Portmanteau test: LB(10) = 30.8463, df = 32 [0.5248]

Equation 1: pi_c

             coefficient   std. error   t-ratio   p-value 
  --------------------------------------------------------
  const       1.15247       0.435623     2.646    0.0120   **
  pi_c_1      0.900227      0.152258     5.913    9.10e-07 ***
  pi_c_2      0.0613022     0.160992     0.3808   0.7056  
  r_s_1       0.255695      0.134001     1.908    0.0644   *
  r_s_2      −0.401963      0.123225    −3.262    0.0024   ***

Mean dependent var   4.415833   S.D. dependent var   3.153664
Sum squared resid    63.26925   S.E. of regression   1.325699
R-squared            0.840962   Adjusted R-squared   0.823291
F(4, 36)             47.59014   P-value(F)           6.84e-14
rho                  0.121414   Durbin-Watson        1.734941

F-tests of zero restrictions:

All lags of pi_c             F(2, 36) =   43.758 [0.0000]
All lags of r_s              F(2, 36) =   5.6225 [0.0075]
All vars, lag 2              F(2, 36) =   6.1167 [0.0052]

Equation 2: r_s

... <output cut>
```

Note here that the `All lags of r_s` is `F(2, 36) =   5.6225 [0.0075]`: Same concludion - Reject the null!

## 19.5 Interpreting Results

The output provides an F-test (or Wald test).

```{admonition} Example Interpretation
If the p-value is less than 0.05:

→ Reject $H_0$  
→ Conclude that $x_t$ Granger-causes $y_t$
```

### Possible Outcomes

| Result            | Interpretation             |
| ----------------- | -------------------------- |
| $x \rightarrow y$ | $x$ helps predict $y$      |
| $y \rightarrow x$ | reverse causality          |
| both              | feedback relationship      |
| neither           | no predictive relationship |

## 19.6 Choice of Lag Length

Lag selection is crucial.

```{admonition} Practical Advice
- Too few lags → omitted dynamics  
- Too many lags → loss of degrees of freedom  
```

### In GRETL

`Model → Univariate time series → ARIMA lag selection`

OR

`Model → Multivariate time series → VAR lag selection`

## 19.7 Stationarity Matters

Granger causality tests require **stationary data**.

```{admonition} Important
If variables are nonstationary, Granger causality tests can produce misleading results.
```

### What to Do

* If variables are $I(1)$ → difference them
* or move to cointegration framework (next chapter)

```{admonition} Connection to Previous Chapter
Spurious regression arises when nonstationary variables are used in levels.

Granger causality can suffer from similar issues if stationarity is ignored.
```

## 19.8 Common Pitfalls

```{admonition} Common Mistakes
:class: warning

**1. Confusing correlation with causation**  
Granger causality is not true causality.

**2. Ignoring stationarity**  
Nonstationary data can produce false results.

**3. Incorrect lag length**  
Results depend heavily on lag choice.

**4. Omitted variables**  
Missing variables can distort conclusions.

**5. Small sample size**  
Tests may lack power.
```

## 19.9 Economic Interpretation

Even when Granger causality is detected:

* it does **not** imply policy effectiveness
* it does **not** reveal mechanisms
* it does **not** establish structural relationships

```{admonition} Big Picture
Granger causality is best viewed as a **tool for forecasting and temporal ordering**, not as proof of economic causation.
```

## 19.10 Summary

```{admonition} Key Takeaways
- Granger causality tests predictive relationships  
- it is based on lagged information  
- it requires stationarity  
- it does not imply true causation  
```

## Looking Ahead

Granger causality focuses on predictive relationships in stationary data.

However, many economic time series are nonstationary.

In the next chapter, we introduce **cointegration**, which allows us to study **long-run relationships between nonstationary variables**.

---

## Appendix 19A — Testing Linear Restrictions in Regression

In this appendix, we explain how to test whether certain coefficients in a regression model are equal to zero.

This framework is widely used in econometrics and underlies tests such as the **Granger causality test**.

### A.1 The Basic Idea

Suppose we estimate a regression model:

```{math}
:enumerated: false
y_t = \alpha + \beta_1 x_{1t} + \beta_2 x_{2t} + \cdots + \beta_k x_{kt} + u_t
```

We may be interested in testing whether some of these variables matter.

### A.2 A Joint Hypothesis

For example, we might test:

```{math}
:enumerated: false
H_0: \beta_2 = \beta_3 = 0
```

This means that variables $x_{2t}$ and $x_{3t}$ have **no effect** on $y_t$.

```{admonition} Key Idea
Testing multiple coefficients at once is called a **test of joint (linear) restrictions**.
```

### A.3 Restricted vs Unrestricted Models

To test this hypothesis, we compare two models:

#### Unrestricted model (U)

Includes all variables:

```{math}
:enumerated: false
y_t = \alpha + \beta_1 x_{1t} + \beta_2 x_{2t} + \beta_3 x_{3t} + u_t
```

#### Restricted model (R)

Imposes the restrictions:

```{math}
:enumerated: false
y_t = \alpha + \beta_1 x_{1t} + u_t
```

```{admonition} Intuition
The restricted model removes variables that are assumed to have no effect under $H_0$.
```

### A.4 Comparing the Models

We now ask:

> Does removing these variables make the model significantly worse?

To answer this, we compare the **sum of squared residuals (SSR)**:

* $SSR_U$ → from the unrestricted model
* $SSR_R$ → from the restricted model

```{admonition} Key Insight
If the restricted model fits much worse (higher SSR), then the excluded variables are important.
```

### A.5 The F-Test

The test statistic is:

```{math}
:enumerated: false
F = \frac{(SSR_R - SSR_U)/q}{SSR_U/(T - k - 1)}
```

where:

* $q$ = number of restrictions
* $T$ = number of observations
* $k$ = number of regressors in the unrestricted model

```{admonition} Interpretation
- Large $F$ → reject $H_0$ → variables matter  
- Small $F$ → fail to reject $H_0$ → variables do not add explanatory power  
```

### A.6 Connection to Granger Causality

In Chapter 19, we used this framework to test:

```{math}
:enumerated: false
H_0: \text{lagged values of } x_t \text{ do not affect } y_t
```

This is simply a test that:

* several lag coefficients are jointly equal to zero

```{admonition} Big Picture
Granger causality is an application of a general idea:

→ testing whether a group of variables improves the model.
```

### A.7 Intuition in Words

The logic of the test is simple:

* If the excluded variables truly do not matter
  → removing them should not change the model much

* If they do matter
  → removing them worsens the fit significantly

```{admonition} Final Insight
Tests of linear restrictions provide a systematic way to determine whether additional variables contribute meaningful information.
```
