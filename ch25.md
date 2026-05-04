---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 25 — ARCH Models

In earlier chapters, we studied models for the **mean** of a time series.

For example:

- AR models explained persistence,
- ARIMA models explained trends and dynamics,
- VAR models explained interactions across variables.

But financial time series often display another important feature.

```{admonition} Key Idea
Volatility changes over time.
```

Periods of calm are often followed by periods of turbulence.

Large shocks tend to cluster together.

```{admonition} Definition
Volatility clustering refers to the tendency for large shocks to be followed by large shocks and small shocks to be followed by small shocks.
```

ARCH models were developed precisely to model this behavior.

```{admonition} Central Question
Can we model changing volatility through time?
```

This chapter introduces:

- conditional heteroskedasticity,
- volatility clustering,
- ARCH models,
- ARCH testing,
- and volatility forecasting.

The discussion emphasizes intuition and financial applications.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- explain volatility clustering
- distinguish homoskedasticity from heteroskedasticity
- understand conditional variance
- explain the intuition of ARCH models
- estimate ARCH models
- test for ARCH effects
- interpret ARCH coefficients
- visualize time-varying volatility

---

# 25.1 Volatility in Financial Markets

Financial returns rarely fluctuate with constant intensity.

Instead:

- calm periods are followed by calm periods,
- turbulent periods are followed by turbulent periods.

Large returns tend to cluster together.

## Example: Stock Returns

During financial crises:

- returns fluctuate dramatically,
- uncertainty rises,
- volatility increases sharply.

During stable periods:

- returns fluctuate less,
- volatility remains low.

```{admonition} Observation
Volatility itself often displays persistence.
```

This feature appears repeatedly in:

- stock returns,
- exchange rates,
- cryptocurrency markets,
- commodity prices.

---

# 25.2 Constant Variance vs Time-Varying Variance

Classical regression models often assume:

```{math}
:enumerated: false
Var(e_t)=\sigma^2
```

where:

- the variance is constant,
- and does not change over time.

```{admonition} Definition
Homoskedasticity means the variance remains constant through time.
```

## The Problem

Financial returns often violate this assumption.

Instead:

```{math}
:enumerated: false
Var(e_t)
```

changes over time.

```{admonition} Definition
Heteroskedasticity means the variance changes through time.
```

```{admonition} Important
ARCH models treat changing variance as something to model rather than something to ignore.
```

---

# 25.3 Conditional Variance

ARCH models focus on conditional variance rather than unconditional variance.

```{admonition} Definition
Conditional variance is the variance expected at time $t$ given information available up to time $t-1$.
```

## Intuition

```{admonition} Intuition
Today's volatility depends on yesterday's information.
```

If yesterday experienced a large shock, today may also be volatile.

## Example

Suppose markets experienced a crash yesterday.

Would you expect volatility today to be:

- unusually low?
- or unusually high?

Most investors would expect volatility to remain elevated.

ARCH models formalize this intuition.

---

# 25.4 Volatility Clustering

Volatility clustering is one of the most important empirical features of financial returns.

A plot of returns often shows:

- quiet periods,
- followed by bursts of volatility.

```{admonition} Key Idea
Large shocks tend to be followed by large shocks.

Small shocks tend to be followed by small shocks.
```

The sign may change:

- positive returns,
- negative returns,

but the magnitude tends to persist.

---

# 25.5 A Simple ARCH(1) Model

The ARCH model was introduced by Robert Engle in 1982.

The simplest ARCH model is ARCH(1).

## Mean Equation

Suppose returns follow:

```{math}
:enumerated: false
r_t=\mu+e_t
```

where:

- $\mu$ is the average return,
- $e_t$ is the error term.

## ARCH Variance Equation

The ARCH(1) variance equation is:

```{math}
:enumerated: false
h_t = \alpha_0 + \alpha_1 e_{t-1}^2
```

where:

- $h_t$ = conditional variance,
- $e_{t-1}^2$ = previous squared shock,
- $\alpha_0>0$,
- $\alpha_1 \ge 0$.

---

# 25.6 Intuition of ARCH

Suppose yesterday produced a large return shock:

```{math}
:enumerated: false
e_{t-1}^2
```

Then the model predicts:

```{math}
:enumerated: false
h_t
```

will increase.

```{admonition} Key Insight
ARCH models explain volatility using past squared shocks.
```

Large shocks yesterday imply elevated volatility today.

---

# 25.7 Why Squared Errors?

Why do we square the errors?

Because volatility concerns magnitude rather than direction.

Both:

- large positive shocks,
- and large negative shocks

increase volatility.

Squaring removes the sign.

## Example

| Shock | Squared Shock |
|---|---|
| 3 | 9 |
| -3 | 9 |

Both imply high volatility.

---

# 25.8 ARCH as Volatility Memory

ARCH models create persistence in volatility.

Suppose:

- yesterday experienced a large shock,
- therefore today's variance rises,
- which increases the probability of large future shocks.

This creates volatility clustering naturally.

```{admonition} Observation
ARCH models generate periods of persistent turbulence and persistent calm.
```

---

# 25.9 Simulating ARCH Data in Python

We now simulate an ARCH(1) process.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

T = 500

alpha0 = 1
alpha1 = 0.8

e = np.zeros(T)
h = np.zeros(T)

z = np.random.normal(size=T)

h[0] = alpha0

for t in range(1, T):

    h[t] = alpha0 + alpha1 * e[t-1]**2

    e[t] = np.sqrt(h[t]) * z[t]

plt.figure(figsize=(10,4))

plt.plot(e)

plt.title("Simulated ARCH(1) Process")

plt.savefig("figs/ch25/arch.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![ARCH](figs/ch25/arch.png)


---

```{admonition} Observation
Notice how periods of high volatility cluster together.
```

Even though the shocks are random, the variance evolves systematically through time.

---

# 25.10 ARCH Effects in Real Financial Data

ARCH effects are extremely common in:

- stock returns,
- exchange rates,
- commodity prices,
- cryptocurrency markets.

Financial returns often display:

- volatility clustering,
- fat tails,
- changing risk over time.

---

## Example from Financial Markets

```markdown
[Figure Placeholder: Volatility clustering in stock returns]
```

---

# 25.11 Testing for ARCH Effects

Before estimating ARCH models, we usually test whether ARCH effects exist.

```{admonition} Definition
The ARCH LM test is a statistical test used to detect ARCH effects and time-varying volatility.
```

The test was developed by Engle.

## Intuition Behind the ARCH Test

If volatility is constant:

```{math}
:enumerated: false
e_t^2
```

should display little serial dependence.

But if volatility clusters, large squared residuals tend to follow large squared residuals.

```{admonition} Key Idea
ARCH effects imply autocorrelation in squared residuals.
```
## Steps in the ARCH Test

Suppose we estimate:

```{math}
:enumerated: false
r_t=\mu+e_t
```

using OLS.

## Step 1 — Obtain Residuals

Compute residuals:

```{math}
:enumerated: false
\hat e_t
```

## Step 2 — Square the Residuals

Compute:

```{math}
:enumerated: false
\hat e_t^2
```

## Step 3 — Estimate Auxiliary Regression

Estimate:

```{math}
:enumerated: false
\hat e_t^2
=
\alpha_0
+
\alpha_1 \hat e_{t-1}^2
+
u_t
```

More generally, multiple lags may be included.

## Step 4 — Compute the LM Statistic

```{admonition} Important
The ARCH LM statistic is:

```{math}
:enumerated: false
TR^2
```

where:

- $T$ = sample size,
- $R^2$ = coefficient of determination from the auxiliary regression.

## Step 5 — Hypothesis Test

Under the null hypothesis:

```{math}
:enumerated: false
H_0:
\alpha_1=0
```

the statistic follows approximately:

```{math}
:enumerated: false
\chi^2(q)
```

where:

- $q$ = number of ARCH lags.

## Interpreting the ARCH Test

| Result | Interpretation |
|---|---|
| small p-value | evidence of ARCH |
| large p-value | little evidence of ARCH |

```{admonition} Important
Rejecting the null suggests time-varying volatility.
```

## Example of an ARCH Test

Suppose we obtain:

| Statistic | Value |
|---|---|
| LM statistic | 62.15 |
| p-value | 0.000 |

```{admonition} Interpretation
A very small p-value suggests strong evidence of ARCH effects and time-varying volatility.
```

This implies volatility is not constant through time.

---

# 25.12 ARCH Estimation in Python

We now estimate an ARCH model using Python.

```{code-cell} python
# !pip install arch

import yfinance as yf
import numpy as np
from arch import arch_model

sp500 = yf.download("^GSPC", start="2018-01-01", auto_adjust=False)

returns = 100 * np.log(
    sp500["Adj Close"] /
    sp500["Adj Close"].shift(1)
).dropna()

model = arch_model(
    returns,
    vol="ARCH",
    p=1
)

results = model.fit()

print(results.summary())
```

``` verbatim
[*********************100%***********************]  1 of 1 completed
Iteration:      1,   Func. Count:      5,   Neg. LLF: 22080.30900905849
Iteration:      2,   Func. Count:     13,   Neg. LLF: 700493.8997438303
Iteration:      3,   Func. Count:     19,   Neg. LLF: 3163.8892644355155
Iteration:      4,   Func. Count:     25,   Neg. LLF: 3124.9447385811245
Iteration:      5,   Func. Count:     29,   Neg. LLF: 3124.944202980584
Iteration:      6,   Func. Count:     33,   Neg. LLF: 3124.9442004194407
Iteration:      7,   Func. Count:     36,   Neg. LLF: 3124.9442004194448
Optimization terminated successfully    (Exit mode 0)
            Current function value: 3124.9442004194407
            Iterations: 7
            Function evaluations: 36
            Gradient evaluations: 7
                      Constant Mean - ARCH Model Results                      
==============================================================================
Dep. Variable:                  ^GSPC   R-squared:                       0.000
Mean Model:             Constant Mean   Adj. R-squared:                  0.000
Vol Model:                       ARCH   Log-Likelihood:               -3124.94
Distribution:                  Normal   AIC:                           6255.89
Method:            Maximum Likelihood   BIC:                           6272.82
                                        No. Observations:                 2091
Date:                Thu, Apr 30 2026   Df Residuals:                     2090
Time:                        09:45:06   Df Model:                            1
                                Mean Model                                
==========================================================================
                 coef    std err          t      P>|t|    95.0% Conf. Int.
--------------------------------------------------------------------------
mu             0.1090  2.578e-02      4.226  2.380e-05 [5.843e-02,  0.159]
                            Volatility Model                            
========================================================================
                 coef    std err          t      P>|t|  95.0% Conf. Int.
------------------------------------------------------------------------
omega          0.7970  5.646e-02     14.115  3.052e-45 [  0.686,  0.908]
alpha[1]       0.4822  9.694e-02      4.974  6.562e-07 [  0.292,  0.672]
========================================================================

Covariance estimator: robust
```

---

# 25.13 Plotting Conditional Volatility

We now plot the estimated conditional volatility.

```{code-cell} python
vol = results.conditional_volatility

vol.plot(figsize=(10,4))

plt.title("Estimated ARCH Volatility")

plt.savefig("figs/ch25/arch_vol.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![ARCH Conditional Volatility](figs/ch25/arch_vol.png)


```{admonition} Interpretation
Periods of market turbulence correspond to higher estimated volatility.
```

---

# 25.14 ARCH vs GARCH

ARCH models are useful but sometimes require many lag terms.

GARCH models solve this problem by including lagged variance terms.

We will study GARCH models in the next chapter.

## ARCH(1)

```{math}
:enumerated: false
h_t = \alpha_0 + \alpha_1 e_{t-1}^2
```

## GARCH(1,1)

```{math}
:enumerated: false
h_t = \omega + \alpha_1 e_{t-1}^2 + \beta_1 h_{t-1}
```

```{admonition} Looking Ahead
GARCH models allow volatility itself to evolve dynamically through time.
```

---

# 25.15 Gretl Example: Testing for ARCH

GRETL makes ARCH testing very straightforward.

---

## Step 1 — Estimate a Regression Model

Menu:

```text
Model → Ordinary Least Squares
```

---

## Step 2 — Run ARCH Test

From the model window:

```text
Tests → ARCH
```

Choose the number of ARCH lags.

---

```markdown
[GRETL Screenshot Placeholder: ARCH test dialog]
```

---

## Example Output

```text
Null hypothesis: no ARCH effect is present

LM statistic = 62.15

p-value = 0.000
```

```{admonition} Interpretation
Rejecting the null hypothesis suggests that volatility is not constant through time.
```

ARCH effects are present.

---

# 25.16 Gretl Example: Estimating ARCH

To estimate an ARCH(1) model:

```text
Model → Time Series → GARCH
```

Then set:

- GARCH order = 0
- ARCH order = 1

---

```markdown
[GRETL Screenshot Placeholder: ARCH estimation dialog]
```

---

# 25.17 Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Confusing volatility with direction**  
ARCH models volatility, not average returns.

**2. Ignoring volatility clustering**  
Financial returns rarely have constant variance.

**3. Forgetting positivity constraints**  
Variance must remain positive.

**4. Overinterpreting individual shocks**  
ARCH models focus on systematic volatility behavior.

**5. Confusing ARCH with autocorrelation in returns**  
ARCH concerns autocorrelation in squared residuals.
```

---

# 25.18 Looking Ahead

ARCH models introduced the idea of modeling volatility dynamically.

However, pure ARCH models often require many lag terms.

The next chapter introduces:

- GARCH models,
- volatility persistence,
- long-run variance,
- and volatility forecasting.

# Key Takeaways

```{admonition} Summary
- Financial volatility often changes over time.
- Volatility clustering is common in financial markets.
- ARCH models explain conditional variance using past squared shocks.
- ARCH effects imply autocorrelation in squared residuals.
- The ARCH LM test detects time-varying volatility.
- ARCH models are foundational tools in financial econometrics.
- GARCH models extend ARCH by modeling volatility persistence directly.
```

# Concept Check

## Basic

1. What is volatility?

2. What is the difference between:

   - mean  
   - variance  

3. What does homoskedasticity mean?

4. What does heteroskedasticity mean?

---

## Intuition

5. What is volatility clustering?

6. Why do financial returns often display clustering in volatility?

7. Why do large shocks tend to be followed by large shocks?

8. Why is volatility easier to “see” than to model?

---

## ARCH Structure

9. What is the key idea behind an ARCH model?

10. What does the equation

```{math}
:enumerated: false
h_t = \alpha_0 + \alpha_1 e_{t-1}^2
```

represent?

11. Why are squared residuals used?

12. Suppose volatility spikes after a large shock.

- What does this suggest?

---

## Interpretation

13. What does a large value of $\alpha_1$ imply?

14. What does it mean if $\alpha_1 = 0$?

---

## Challenge

15. Can volatility be predictable even if returns are not?
16. You analyze stock returns and find:

- no autocorrelation in returns  
- strong autocorrelation in squared returns  

---

- What does this imply?
- Why might an ARCH model be appropriate?

---

# Interpretation & Practice

1. A return series shows:

- periods of calm  
- followed by periods of turbulence  

   - What does this suggest?

2. Residuals show no autocorrelation, but squared residuals do.

   - What does this imply?

3. A model assumes constant variance, but volatility clearly changes.

   - What problem arises?

4. An ARCH model is estimated and $\alpha_1$ is significant.

   - What does this indicate?

---

## Economic Interpretation

5. A financial crisis leads to large return shocks.

   - What does the ARCH model predict for future volatility?

6. A period of calm persists.

   - What does the model predict?

---

## Challenge

7. Why might volatility clustering be important for risk management?

---

# Numerical Practice

## Squared Shocks

1. Suppose shocks are:

```{math}
:enumerated: false
2, -3, 1
```

- Compute squared shocks.

---

## ARCH Equation

2. Suppose:

```{math}
:enumerated: false
h_t = 1 + 0.5 e_{t-1}^2
```

and:

```{math}
:enumerated: false
e_{t-1} = 2
```

- Compute $h_t$.

---

3. Suppose:

```{math}
:enumerated: false
e_{t-1} = -2
```

- Compute $h_t$ again.
- What do you observe?

---

## Interpretation

4. Suppose $\alpha_1 = 0.8$.

- What does this imply about volatility persistence?

---

5. Suppose $\alpha_1 = 0.1$.

- How does this differ?

---

## Stability

6. What happens if $\alpha_1 \ge 1$?

---

## Challenge

7. Suppose:

- small shocks yesterday  
- small variance today  

- What does the model predict for tomorrow?

---

# ARCH Testing

1. What is the purpose of the ARCH LM test?

2. What is the null hypothesis?

---

## Interpretation

3. Suppose:

- LM statistic = 45  
- p-value = 0.000  

- What is your conclusion?

4. Suppose:

- p-value = 0.40  

- What does this imply?

---

## Conceptual

5. Why does the ARCH test use squared residuals?

---

### Challenge

6. Why is autocorrelation in squared residuals important?

---

# Graph Interpretation

### Volatility Clustering

Consider the following simulated return series:

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

T = 300
alpha0 = 1
alpha1 = 0.8

e = np.zeros(T)
h = np.zeros(T)

z = np.random.normal(size=T)

h[0] = alpha0

for t in range(1, T):
    h[t] = alpha0 + alpha1 * e[t-1]**2
    e[t] = np.sqrt(h[t]) * z[t]

plt.figure(figsize=(10,4))
plt.plot(e)
plt.title("Simulated ARCH(1) Returns")
plt.axhline(0, linestyle='--', linewidth=1)
plt.tight_layout()

plt.savefig("figs/ch25/rtn_Q.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Rtn](figs/ch25/rtn_Q.png)

---

1. What feature of financial data does this illustrate?

2. Identify periods of:

   - high volatility  
   - low volatility  

3. Why is this inconsistent with constant variance?

4. Why might a standard regression model fail to capture this behavior?

---

```{admonition} Hint
Focus on clustering of large and small movements, not the direction.
```

---

# Appendix 25A — ARCH(1) Stability Condition

For the ARCH(1) model:

```{math}
:enumerated: false
h_t = \alpha_0 + \alpha_1 e_{t-1}^2
```

the parameter restrictions are:

```{math}
:enumerated: false
\alpha_0>0
```

and:

```{math}
:enumerated: false
0 \le \alpha_1 < 1
```

The condition:

```{math}
:enumerated: false
\alpha_1<1
```

ensures the variance process remains stable.

If:

```{math}
:enumerated: false
\alpha_1 \ge 1
```

volatility may become explosive.

---

# Appendix 25B — Why Financial Returns Often Display Fat Tails

ARCH processes naturally generate:

- clusters of volatility,
- periods of calm,
- and occasional extreme observations.

Even if the underlying shocks are normal, the resulting returns may appear fat-tailed.

```{admonition} Key Insight
Time-varying volatility alone can generate unusually large observations and apparent fat tails.
```

This is one reason ARCH and GARCH models became so influential in finance.
