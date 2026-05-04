---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 19 — Predictive Causality (Granger Causality)

In the previous chapter, we introduced **dynamic regression models** and emphasized the importance of lagged relationships.

We now ask a natural question:

```{admonition} Central Question
Does one variable help predict another?
```

This idea is formalized through **Granger causality**.

Despite the name, Granger causality is not about deep philosophical or structural causality. It is about **predictive content**.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- explain the idea of Granger causality
- distinguish predictive causality from true causality
- understand restricted and unrestricted models
- interpret the F-test for Granger causality
- choose lag length carefully
- implement Granger causality tests in Gretl
- understand common limitations and pitfalls

---

# 19.1 What Is Granger Causality?

Suppose we want to know whether $x_t$ helps predict $y_t$.

We say that $x_t$ **Granger-causes** $y_t$ if past values of $x_t$ help forecast $y_t$ after controlling for past values of $y_t$.

```{admonition} Definition: Granger Causality
Variable $x_t$ Granger-causes $y_t$ if lagged values of $x_t$ contain useful predictive information for $y_t$ beyond the information already contained in lagged values of $y_t$.
```

---

# 19.2 Prediction, Not True Causation

Granger causality is about forecasting.

It asks:

> Do past values of $x_t$ improve prediction of $y_t$?

It does **not** automatically answer:

> Does $x_t$ structurally or economically cause $y_t$?

```{admonition} Important
Granger causality does **not** prove true causality.

It only shows that one variable contains useful predictive information about another variable.
```

## Examples

We might ask:

- Do interest rates help predict inflation?
- Does income help predict consumption?
- Does money supply help predict output?
- Do exchange rates help predict exports?

In each case, the question is predictive:

```{admonition} Key Question
Do past values of one variable improve forecasts of another?
```

---

# 19.3 Restricted and Unrestricted Models

To test whether $x_t$ Granger-causes $y_t$, we compare two models.

## Restricted Model

The restricted model uses only past values of $y_t$:

```{math}
:enumerated: false
y_t
=
\alpha
+
\sum_{i=1}^{p}\phi_i y_{t-i}
+
u_t
```

This model assumes that past values of $x_t$ do **not** help predict $y_t$.

## Unrestricted Model

The unrestricted model includes both past values of $y_t$ and past values of $x_t$:

```{math}
:enumerated: false
y_t
=
\alpha
+
\sum_{i=1}^{p}\phi_i y_{t-i}
+
\sum_{j=1}^{q}\beta_j x_{t-j}
+
u_t
```

```{admonition} Key Idea
If adding lagged values of $x_t$ improves prediction of $y_t$, then $x_t$ Granger-causes $y_t$.
```

---

# 19.4 Hypothesis Testing Framework

The null hypothesis is that lagged values of $x_t$ do not help predict $y_t$:

```{math}
:enumerated: false
H_0:
\beta_1
=
\beta_2
=
\cdots
=
\beta_q
=
0
```

The alternative is:

```{math}
:enumerated: false
H_1:
\text{at least one } \beta_j \neq 0
```

```{admonition} Interpretation
- Fail to reject $H_0$ → no evidence that $x_t$ Granger-causes $y_t$
- Reject $H_0$ → evidence that $x_t$ Granger-causes $y_t$
```

---

# 19.5 The F-Test Intuition

The Granger causality test compares:

- the restricted model
- the unrestricted model

If the unrestricted model fits much better, then lagged values of $x_t$ add useful predictive information.

The F-statistic is:

```{math}
:enumerated: false
F
=
\frac{(SSR_R - SSR_U)/q}{SSR_U/(T-k)}
```

where:

- $SSR_R$ is the sum of squared residuals from the restricted model
- $SSR_U$ is the sum of squared residuals from the unrestricted model
- $q$ is the number of restrictions
- $T$ is the number of observations
- $k$ is the number of parameters in the unrestricted model

```{admonition} Intuition
If removing lagged $x_t$ variables makes the model much worse, then those lagged variables contain useful predictive information.
```

---

# 19.6 Possible Outcomes

Granger causality can run in one direction, both directions, or neither direction.

| Result | Interpretation |
|---|---|
| $x \rightarrow y$ | $x$ helps predict $y$ |
| $y \rightarrow x$ | $y$ helps predict $x$ |
| both directions | feedback relationship |
| neither direction | no predictive relationship |

```{admonition} Important
A feedback relationship does not necessarily mean simultaneous causation. It means both variables contain useful lagged predictive information about each other.
```

---

# 19.7 Lag Length Matters

Lag selection is crucial.

Too few lags may omit important dynamics.

Too many lags may reduce degrees of freedom and make the test less precise.

```{admonition} Practical Advice
- Too few lags → omitted dynamics
- Too many lags → loss of degrees of freedom
```

In practice, lag length can be guided by:

- AIC
- BIC
- HQC
- residual diagnostics
- economic reasoning

---

# 19.8 Stationarity Matters

Granger causality tests are usually applied to stationary series.

```{admonition} Important
If variables are nonstationary, Granger causality tests can produce misleading results.
```

If variables are $I(1)$, common options include:

- difference the variables
- test for cointegration
- use an error correction framework if cointegration exists

```{admonition} Connection
Spurious regression arises when nonstationary variables are used carelessly in levels.

Granger causality tests can suffer from similar problems if stationarity is ignored.
```

---

# 19.9 Economic Interpretation

Even when Granger causality is detected, we should interpret the result carefully.

It does not necessarily imply:

- policy effectiveness
- a structural mechanism
- true economic causality

```{admonition} Big Picture
Granger causality is best viewed as a tool for forecasting and temporal ordering, not as proof of structural causation.
```

---

# 19.10 Gretl Example: Interest Rates and Inflation

We now implement a Granger causality test in GRETL using the `jgm-data` dataset. This follows the example in your notes. :contentReference[oaicite:0]{index=0}

We ask:

> Do short-term interest rates help predict CPI inflation?

## Step 1: Load the Data

### Menu

`File → Open data → Sample file...`

Select `jgm-data` from the GRETL database.

We use:

```gretl
pi_c     inflation rate based on the CPI
r_s      short-term interest rate
```

# 19.11 Restricted Model

The restricted model predicts inflation using only its own past values.

In your notes, lag selection suggests an AR(1) model for `pi_c`.

## Menu

`Model → Univariate time series → ARIMA lag selection`

Dependent variable:

```gretl
pi_c
```

Try a maximum AR lag of 5.

Example output suggests that one lag is sufficient.

Then estimate:

`Model → Univariate time series → ARIMA`

with AR order 1.

## Restricted Model Output

```gretl
Model 2: ARMA, using observations 1952-1994 (T = 43)
Estimated using AS 197 (exact ML)
Dependent variable: pi_c

             coefficient   std. error     z       p-value 
  --------------------------------------------------------
  const       3.53530      1.64385       2.151   0.0315    **
  phi_1       0.873605     0.0693047    12.61    1.97e-036 ***

Mean dependent var   4.246077   S.D. dependent var   3.195937
Mean of innovations  0.034572   S.D. of innovations  1.532847
R-squared            0.764867   Adjusted R-squared   0.764867
Log-likelihood      −80.10105   Akaike criterion     166.2021
Schwarz criterion    171.4857   Hannan-Quinn         168.1505
```

Save the error sum of squares:

`Save → Error sum of squares`

In the example:

```{math}
:enumerated: false
SSR_R = 101.0337
```

# 19.12 Unrestricted Model

The unrestricted model includes lagged interest rates.

We use two lags of `r_s`, based on lag selection.

## Lag Selection

### Menu

`Model → Multivariate time series → VAR lag selection`

Use:

- dependent variable: `pi_c`
- predictor: `r_s`
- maximum lag: 5

Example output:

```gretl
VAR system, maximum lag order 5

lags        loglik    p(LR)       AIC          BIC          HQC

   1     -68.39186             3.757466     3.886749     3.803464 
   2     -65.81966  0.02332    3.674719*    3.847096*    3.736049*
   3     -65.81635  0.93515    3.727176     3.942648     3.803839 
   4     -65.76655  0.75230    3.777187     4.035753     3.869182 
   5     -65.76616  0.97785    3.829798     4.131458     3.937126 
```

The information criteria suggest two lags.-

## Estimate the Unrestricted Model

### Menu

`Model → Univariate time series → ARIMA`

Use:

- dependent variable: `pi_c`
- AR order: 1
- regressors: `r_s(-1) r_s(-2)`

Use the `lags...` icon in the model box to create lagged regressors.

## Output

```gretl
Model 3: ARMAX, using observations 1954-1994 (T = 41)
Estimated using AS 197 (exact ML)
Dependent variable: pi_c

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
```

Save the unrestricted error sum of squares.

In the example:

```{math}
:enumerated: false
SSR_U = 68.0476
```

---

# 19.13 Manual F-Test

We now compute:

```{math}
:enumerated: false
F
=
\frac{(SSR_R - SSR_U)/q}{SSR_U/(T-k)}
```

Using:

- $SSR_R = 101.0337$
- $SSR_U = 68.0476$
- $q = 2$
- $T = 41$
- $k = 4$

we get:

```{math}
:enumerated: false
F
=
\frac{(101.0337 - 68.0476)/2}{68.0476/(41-4)}
=
8.97
```

```{admonition} Interpretation
The F-statistic is large, so we reject the null hypothesis.

Short-term interest rates appear to Granger-cause CPI inflation in this example.
```

---

# 19.14 VAR Alternative in Gretl

A more direct way is to estimate a VAR and use GRETL’s built-in tests.

## Menu

`Model → Multivariate time series → Vector Autoregression`

Choose:

- lag order: 2
- endogenous variables: `pi_c` and `r_s`

## Command

```gretl
var 2 pi_c r_s
```

Example output:

```gretl
VAR system, lag order 2
OLS estimates, observations 1954-1994 (T = 41)

Equation 1: pi_c

             coefficient   std. error   t-ratio   p-value 
  --------------------------------------------------------
  const       1.15247       0.435623     2.646    0.0120   **
  pi_c_1      0.900227      0.152258     5.913    9.10e-07 ***
  pi_c_2      0.0613022     0.160992     0.3808   0.7056  
  r_s_1       0.255695      0.134001     1.908    0.0644   *
  r_s_2      −0.401963      0.123225    −3.262    0.0024   ***

F-tests of zero restrictions:

All lags of pi_c             F(2, 36) =   43.758 [0.0000]
All lags of r_s              F(2, 36) =   5.6225 [0.0075]
All vars, lag 2              F(2, 36) =   6.1167 [0.0052]
```

The line:

```gretl
All lags of r_s              F(2, 36) =   5.6225 [0.0075]
```

tests whether lagged values of `r_s` help predict `pi_c`.

Since the p-value is small, we reject the null.

```{admonition} Conclusion
In this example, short-term interest rates Granger-cause CPI inflation.
```

---

# 19.15 Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Confusing predictive causality with true causality**  
Granger causality does not prove structural causality.

**2. Ignoring stationarity**  
Nonstationary variables can produce misleading Granger tests.

**3. Choosing lag length mechanically**  
Lag choice should reflect diagnostics and economic reasoning.

**4. Omitting relevant variables**  
A third variable may drive both series.

**5. Overinterpreting small samples**  
Granger tests can have low power in small samples.
```

---

# 19.16 Looking Ahead

Granger causality focuses on predictive relationships, usually in stationary data.

However, many economic time series are nonstationary.

In the next chapter, we introduce **cointegration**, which allows us to study meaningful long-run relationships between nonstationary variables.

# Key Takeaways

```{admonition} Summary
- Granger causality is about prediction, not true causation.
- $x_t$ Granger-causes $y_t$ if lagged $x_t$ improves forecasts of $y_t$.
- The test compares restricted and unrestricted models.
- Lag length matters.
- Stationarity matters.
- Gretl can implement Granger tests directly using VAR models.
```

# Concept Check

## Basic

1. What is Granger causality?

2. What does it mean for $x_t$ to Granger-cause $y_t$?

3. What is the difference between predictive causality and true causality?

---

## Intuition

4. Why can a variable help predict another without truly causing it?

5. Why is Granger causality fundamentally a forecasting concept?

6. Why is it important to include lagged values of $y_t$ in the test?

---

## Models

7. What is the difference between:

   - restricted model  
   - unrestricted model  

8. What is the null hypothesis in a Granger causality test?

9. What does it mean to reject the null hypothesis?

---

## Testing

10. What does the F-test compare?

11. What does a large F-statistic indicate?

---

## Lag Length

12. Why does lag length matter in Granger causality testing?

13. What happens if too few lags are used?

14. What happens if too many lags are used?

---

## Stationarity

15. Why must variables be stationary before applying Granger causality tests?

---

## Challenge

16. Can Granger causality exist in both directions?  
   - What does this imply?

---

# Interpretation & Practice

1. A test finds that $x_t$ Granger-causes $y_t$.

   - What does this mean?
   - What does it NOT mean?

2. A test fails to reject the null.

   - What conclusion can you draw?
   - What can you NOT conclude?

3. Both $x_t$ and $y_t$ Granger-cause each other.

   - What type of relationship might this indicate?

4. A model includes too few lags.

   - How might this affect the test?

5. A model includes too many lags.

   - What problem might arise?

---

## Stationarity Interpretation

6. You run a Granger test on nonstationary variables in levels.

   - What is the risk?

7. After differencing, the Granger result disappears.

   - What does this suggest?

---

## Economic Interpretation

8. Interest rates Granger-cause inflation.

   - Why should we interpret this result cautiously?

---

## Challenge

9. A third variable affects both $x_t$ and $y_t$.

   - How could this affect Granger causality results?

---

# Numerical Practice

## Understanding the Test

1. Suppose:

- $SSR_R = 120$
- $SSR_U = 80$
- $q = 2$
- $T = 50$
- $k = 4$

---

- Compute the F-statistic.

---

## Interpretation

2. The F-statistic is large and statistically significant.

- What is your decision?
- What does it imply?

---

## Model Comparison

3. Suppose:

- restricted model fits poorly  
- unrestricted model fits much better  

---

- What does this suggest?

---

## Lag Structure

4. Suppose only one lag is included, but the true relationship requires two lags.

- What happens to the test?

---

## Stationarity

5. Suppose both variables are random walks.

- Why might the test be misleading?

---

## VAR Output Interpretation

6. Suppose Gretl reports:

```
All lags of x: F(2,36) = 5.2 [0.01]
```

- What does this mean?
- What is your conclusion?

---

## Challenge

7. Suppose:

- $x_t$ does not Granger-cause $y_t$  
- but $y_t$ Granger-causes $x_t$

- How would you interpret this?

---

8. Suppose:

- results change dramatically when lag length changes  

- What does this suggest?

---

# Appendix 19A — Testing Linear Restrictions in Regression

This appendix explains the general idea behind tests such as the Granger causality test.

---

## A.1 The Basic Idea

Suppose we estimate:

```{math}
:enumerated: false
y_t
=
\alpha
+
\beta_1 x_{1t}
+
\beta_2 x_{2t}
+
\cdots
+
\beta_k x_{kt}
+
u_t
```

We may want to test whether some variables matter jointly.

---

# A.2 A Joint Hypothesis

For example:

```{math}
:enumerated: false
H_0:
\beta_2
=
\beta_3
=
0
```

This means that $x_{2t}$ and $x_{3t}$ have no effect on $y_t$ after controlling for the other variables.

```{admonition} Key Idea
Testing multiple coefficients at once is called a test of **joint restrictions**.
```

---

# A.3 Restricted and Unrestricted Models

The unrestricted model includes all variables:

```{math}
:enumerated: false
y_t
=
\alpha
+
\beta_1 x_{1t}
+
\beta_2 x_{2t}
+
\beta_3 x_{3t}
+
u_t
```

The restricted model imposes the null hypothesis:

```{math}
:enumerated: false
y_t
=
\alpha
+
\beta_1 x_{1t}
+
u_t
```

```{admonition} Intuition
The restricted model removes variables that are assumed to have no effect under the null hypothesis.
```

---

# A.4 Comparing the Models

We compare the sum of squared residuals:

- $SSR_U$ from the unrestricted model
- $SSR_R$ from the restricted model

If the restricted model fits much worse, then the excluded variables are probably important.

```{admonition} Key Insight
If removing variables greatly increases the residual sum of squares, those variables add useful explanatory power.
```

---

# A.5 The F-Test

The test statistic is:

```{math}
:enumerated: false
F
=
\frac{(SSR_R - SSR_U)/q}{SSR_U/(T-k)}
```

where:

- $q$ is the number of restrictions
- $T$ is the number of observations
- $k$ is the number of parameters in the unrestricted model

```{admonition} Interpretation
- Large $F$ → reject $H_0$
- Small $F$ → fail to reject $H_0$
```

---

# A.6 Connection to Granger Causality

In Granger causality testing, the null is:

```{math}
:enumerated: false
H_0:
\text{lagged values of } x_t \text{ do not help predict } y_t
```

This is simply a joint test that several lag coefficients are equal to zero.

```{admonition} Big Picture
Granger causality is an application of a general regression idea:

testing whether a group of variables improves the model.
```

---

# A.7 Intuition in Words

The logic is simple:

- if excluded variables do not matter, removing them should not change the model much
- if excluded variables do matter, removing them worsens the fit

```{admonition} Final Insight
Tests of joint restrictions provide a systematic way to decide whether additional variables contribute useful information.
```