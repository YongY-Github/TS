---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 22 — VAR Models

In earlier chapters, we studied models involving a single time series.

For example:

- AR models explained persistence in one variable,
- ARIMA models modeled trends and differencing,
- forecasting models projected future values from past observations,
- ECM models described long-run equilibrium adjustment.

These models are powerful when the focus is a single variable through time.

But many economic and financial systems involve several variables evolving together.

Examples include:

- inflation and interest rates,
- GDP growth and unemployment,
- exchange rates and inflation,
- stock returns and volatility,
- oil prices and inflation.

Economic systems often involve:

- feedback,
- interaction,
- and dynamic dependence across variables.

```{admonition} Central Question
How can we model several interacting time series simultaneously?
```

Vector autoregression (VAR) models were developed precisely for this purpose.

VAR models allow multiple variables, feedback channels, and dynamic interactions to be modeled jointly.

This chapter introduces:

- multivariate dynamics,
- vector autoregressions,
- lag selection,
- stability,
- and VAR forecasting.

The emphasis remains intuition-first and applications-oriented.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- explain the intuition behind VAR models
- distinguish univariate and multivariate models
- estimate VAR models
- select lag lengths
- interpret lagged interactions
- perform basic VAR forecasting
- interpret VAR results economically

---

# 22.1 Why Multivariate Models?

Economic variables rarely move independently.

For example:

- inflation influences interest rates,
- interest rates influence investment,
- investment influences GDP,
- GDP influences unemployment.

```{admonition} Key Idea
Economic systems involve feedback and interaction across variables.
```

A univariate AR model ignores these interactions.

VAR models attempt to capture them directly.

---

# 22.2 From AR Models to VAR Models

Recall the AR(1) model:

```{math}
:enumerated: false
y_t
=
\alpha
+
\phi y_{t-1}
+
e_t
```

The current value depends on:

- its own past,
- plus a shock.

## Extending to Multiple Variables

Suppose we have two variables:

- inflation,
- interest rate.

A VAR allows each variable to depend on:

- its own lags,
- and lags of the other variable.

---

# 22.3 A Simple VAR(1)

A two-variable VAR(1) may be written:

```{math}
:enumerated: false
\pi_t
=
\alpha_1
+
\beta_{11}\pi_{t-1}
+
\beta_{12}r_{t-1}
+
u_{1t}
```

```{math}
:enumerated: false
r_t
=
\alpha_2
+
\beta_{21}\pi_{t-1}
+
\beta_{22}r_{t-1}
+
u_{2t}
```

where:

- $\pi_t$ = inflation,
- $r_t$ = interest rate.

```{admonition} Intuition
Each variable depends on its own history and the history of other variables.
```

---

# 22.4 Why VAR Models Became Popular

VAR models became highly influential in macroeconomics and finance because they provide a flexible framework for studying dynamic interactions across variables.

Christopher Sims argued that many macroeconomic systems should be modeled jointly rather than through isolated single equations.

```{admonition} Key Insight
VAR models treat all variables as potentially endogenous.
```

This contrasts with traditional regression models where some variables are treated as purely explanatory.

---

# 22.5 Endogenous and Exogenous Variables

In many regression models:

- some variables are explanatory,
- others are dependent.

VAR models are different.

```{admonition} Definition
In a VAR model, all variables are treated as endogenous.
```

This means:

- inflation may affect interest rates,
- but interest rates may also affect inflation.

The relationship is dynamic and simultaneous through time.

---

# 22.6 Reduced Form VARs

The most common VAR specification is the reduced form VAR.

A reduced form VAR expresses each variable as a function of lagged values of all variables in the system.

## Example

Suppose we model:

- inflation,
- unemployment,
- interest rate.

Then each equation includes:

- lags of inflation,
- lags of unemployment,
- lags of interest rates.

---

# 22.7 VAR(p) Models

A VAR with multiple lags is written as VAR($p$).

For example:

- VAR(1),
- VAR(2),
- VAR(4).

## General Form

```{math}
:enumerated: false
Y_t
=
A_0
+
A_1 Y_{t-1}
+
A_2 Y_{t-2}
+
\cdots
+
A_p Y_{t-p}
+
u_t
```

where:

- $Y_t$ is a vector of variables,
- $A_i$ are coefficient matrices,
- $u_t$ is a vector of shocks.

```{admonition} Important
VAR models may contain many parameters very quickly.
```

As the number of variables and lags increases, estimation becomes more complex.

---

# 22.8 Stationarity in VAR Models

Stationarity remains important in multivariate time series models.

Standard VAR models are typically estimated using stationary variables.

If variables are nonstationary:

- spurious relationships may emerge,
- statistical inference becomes unreliable,
- and forecasts may become unstable.

```{admonition} Important
Stationarity remains crucial in multivariate models.
```

## Common Approaches

If variables are nonstationary:

- difference the data,
- or use cointegration and VECM methods.

We studied unit roots and cointegration earlier in the book.

Differencing may remove important long-run information when variables are cointegrated.

---

# 22.9 Example: Inflation and Interest Rates

Suppose inflation rises persistently.

Central banks may respond by:

- increasing interest rates.

But higher interest rates may later reduce:

- inflation,
- investment,
- economic activity.

VAR models attempt to capture these dynamic feedback effects.

---

# 22.10 Estimating a VAR in Python

We now estimate a simple VAR using financial data.

The system contains two variables:

- SPY daily log returns,
- a rolling 20-day volatility proxy.

This example is useful because returns and volatility often interact dynamically.

For example:

- large negative returns may be followed by higher volatility,
- high-volatility periods may affect subsequent returns,
- and both variables may display persistence through time.

```{admonition} Note
This is a reduced-form VAR example. It is useful for studying dynamic dependence, but it should not be interpreted as a structural causal model.
```

```{code-cell} python
import yfinance as yf
import pandas as pd
import numpy as np

from statsmodels.tsa.api import VAR

# Download SPY data
spy = yf.download(
    "SPY",
    start="2015-01-01",
    auto_adjust=False
)

# Extract adjusted close prices
prices = spy["Adj Close"]

# Compute daily log returns in percent
returns = 100 * np.log(
    prices / prices.shift(1)
)

# Rolling 20-day volatility proxy
volatility = returns.rolling(20).std()

# Combine variables
data = pd.concat(
    [returns, volatility],
    axis=1
)

data.columns = [
    "Returns",
    "Volatility"
]

data = data.dropna()

# Estimate VAR(2)
model = VAR(data)

results = model.fit(2)

print(results.summary())
```

``` verbatim
  Summary of Regression Results   
==================================
Model:                         VAR
Method:                        OLS
Date:           Thu, 07, May, 2026
Time:                     16:08:54
--------------------------------------------------------------------
No. of Equations:         2.00000    BIC:                   -4.95229
Nobs:                     2830.00    HQIC:                  -4.96572
Log likelihood:          -983.966    FPE:                 0.00692024
AIC:                     -4.97330    Det(Omega_mle):      0.00689585
--------------------------------------------------------------------
Results for equation Returns
================================================================================
                   coefficient       std. error           t-stat            prob
--------------------------------------------------------------------------------
const                 0.003909         0.037664            0.104           0.917
L1.Returns           -0.116688         0.018935           -6.162           0.000
L1.Volatility        -0.517839         0.271001           -1.911           0.056
L2.Returns            0.051612         0.018954            2.723           0.006
L2.Volatility         0.572478         0.271108            2.112           0.035
================================================================================

Results for equation Volatility
================================================================================
                   coefficient       std. error           t-stat            prob
--------------------------------------------------------------------------------
const                 0.009302         0.002573            3.615           0.000
L1.Returns           -0.008181         0.001294           -6.324           0.000
L1.Volatility         1.176947         0.018513           63.574           0.000
L2.Returns           -0.006876         0.001295           -5.310           0.000
L2.Volatility        -0.186193         0.018520          -10.053           0.000
================================================================================

Correlation matrix of residuals
               Returns  Volatility
Returns       1.000000   -0.122110
Volatility   -0.122110    1.000000
```

```{admonition} Observation
Each equation contains lagged values of both variables.
```
---

# 22.11 Reading VAR Output

A VAR output table may initially appear overwhelming.

Remember:

- each equation has its own dynamics,
- each variable depends on lagged values of all variables,
- and feedback effects accumulate through time.

Rather than focusing immediately on individual coefficients, it is often more useful to ask broader questions:

- Which variables appear persistent?
- Which cross-variable effects appear strongest?
- Do relationships appear economically plausible?
- Do shocks appear temporary or long-lasting?

```{admonition} Important
VAR interpretation usually emphasizes system dynamics rather than isolated coefficients.
```
This is one reason impulse response analysis becomes especially important in multivariate time series models.

---

# 22.12 Choosing Lag Lengths

One important decision is:

```text
How many lags should be included?
```

Too few lags:

- omit important dynamics.

Too many lags:

- waste degrees of freedom,
- increase estimation noise.

---

## Information Criteria

VAR lag lengths are often selected using:

- AIC,
- BIC,
- HQIC.

## Intuition

These criteria balance:

- model fit,
- and model complexity.

```{admonition} Definition
Information criteria penalize models that become excessively complex.
```

---

# 22.13 Lag Selection in Python

```{code-cell} python
lag_selection = model.select_order(10)

print(lag_selection.summary())
```

``` verbatim
 VAR Order Selection (* highlights the minimums)  
==================================================
       AIC         BIC         FPE         HQIC   
--------------------------------------------------
0      -0.7217     -0.7174      0.4859     -0.7201
1       -4.917      -4.904    0.007321      -4.912
2       -4.967      -4.946    0.006962      -4.960
3       -4.991      -4.962    0.006797      -4.981
4       -5.028      -4.990    0.006550      -5.015
5       -5.046      -4.999    0.006436      -5.029
6       -5.060      -5.006    0.006343      -5.041
7       -5.071      -5.007    0.006278      -5.048
8       -5.078      -5.006    0.006232      -5.052
9       -5.089     -5.009*    0.006166     -5.060*
10     -5.090*      -5.002   0.006155*      -5.058
--------------------------------------------------
```

```{admonition} Interpretation
Different criteria may suggest different lag lengths.
```

In practice:

- economic intuition,
- diagnostics,
- and forecasting performance

also matter.

---

# 22.14 Stability of VAR Models

A stable VAR system eventually absorbs shocks.

After temporary disturbances:

- variables gradually return toward typical behavior,
- dynamic effects decay,
- and forecasts remain bounded.

An unstable system may instead produce explosive dynamics.

```{admonition} Important
Stable VAR systems are generally required for meaningful interpretation and forecasting.
```

## Intuition

Suppose a shock temporarily increases inflation.

In a stable system:

- inflation eventually settles,
- interest rates stabilize,
- and the system returns toward normal dynamics.

In an unstable system:

- responses may grow larger over time,
- feedback effects may amplify,
- and forecasts may explode unrealistically.

## Stability and Dynamic Systems

VAR models are dynamic systems.

Small changes today may influence future periods through recursive feedback.

A stable system prevents these effects from growing indefinitely.

This idea becomes especially important when studying:

- impulse responses,
- dynamic forecasting,
- and long-horizon simulations.


# 22.15 Inverse Roots and Stability

One common way to assess VAR stability is through inverse roots plots.

A stable VAR requires all inverse roots to lie inside the unit circle.

```{admonition} Key Idea
If inverse roots lie inside the unit circle, shocks eventually die out.
```

## Intuition

The unit circle acts like a stability boundary.

Inside the circle → stable dynamics
Outside the circle → unstable or explosive dynamics


## Add this inverse-roots code

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

# Statsmodels reports roots of the VAR characteristic polynomial.
# Stability requires these roots to lie outside the unit circle.
# Inverse roots should therefore lie inside the unit circle.

inverse_roots = 1 / results.roots

fig, ax = plt.subplots(figsize=(6,6))

circle = plt.Circle(
    (0, 0),
    1,
    fill=False,
    linestyle="--"
)

ax.add_artist(circle)

ax.scatter(
    inverse_roots.real,
    inverse_roots.imag
)

ax.axhline(0, linewidth=1)
ax.axvline(0, linewidth=1)

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect("equal")

ax.set_title("Inverse Roots of the VAR Model")
ax.set_xlabel("Real")
ax.set_ylabel("Imaginary")

plt.savefig("figs/ch22/inverse_roots.png", dpi=300, bbox_inches="tight")
plt.close()
```

![Inverse Root](figs/ch22/inverse_roots.png)

---

# 22.16 Forecasting with VAR Models

VAR models are widely used for forecasting because they incorporate:

- multiple variables,
- dynamic interactions,
- and recursive feedback effects.

Unlike univariate models, VAR forecasts allow information from several variables to influence future predictions jointly.

```{admonition} Key Idea
VAR models treat forecasting as a system-wide problem rather than a single-equation problem.
```

---

# 22.17 Dynamic Forecasting in VAR Models

VAR forecasts are recursive.

Forecasts for future periods depend partly on previous forecasts.

For example:

```{math}
:enumerated: false
\hat Y_{t+2}
```

depends partly on:


```{math}
:enumerated: false
\hat Y_{t+1}
```

This recursive structure allows VAR models to capture rich dynamic interactions across variables.

## Intuition

Suppose stock-market volatility rises unexpectedly.

The VAR may predict:

- lower future returns,
- which then influence future volatility,
- which then feed back into later forecasts.

This creates dynamic forecast paths rather than isolated predictions.

# 22.18 VAR Forecasting in Python

VAR models are widely used for forecasting.

Because they incorporate:

- multiple variables,
- interactions,
- and dynamic feedback,

they often outperform simple univariate models.

---

## Example: VAR Forecasting

```{code-cell} python
forecast = results.forecast(
    data.values[-2:],
    steps=10
)

forecast_df = pd.DataFrame(
    forecast,
    columns=data.columns
)

print(forecast_df)
```

``` verbatim
    Returns  Volatility
0 -0.007599    0.600904
1  0.130212    0.588281
2  0.027692    0.588780
3  0.039283    0.591609
4  0.031460    0.595456
5  0.032598    0.599441
6  0.032200    0.603460
7  0.032506    0.607443
8  0.032688    0.611383
9  0.032922    0.615275
```

---

# 22.19 Beyond Coefficients: Dynamic Responses

VAR coefficient tables alone are often difficult to interpret economically.

Because variables interact dynamically through time, economists frequently analyze systems using:

- impulse response functions (IRFs),
- and forecast error variance decomposition (FEVD).

These tools help trace:

- how shocks propagate,
- how long effects persist,
- and which shocks matter most.

The next chapter develops these ideas in detail.

---

# 22.20 Common Applications of VAR Models

VARs are widely used in:

- macroeconomics,
- monetary policy,
- finance,
- exchange-rate analysis,
- business-cycle analysis.

## Financial Applications

Examples include:

- stock returns and volatility,
- exchange rates and interest rates,
- oil prices and inflation.

---

# 22.21 Gretl Example: Estimating a VAR

Gretl provides built-in tools for VAR estimation.

---

## Step 1

Load multiple time series.

Example:

- inflation,
- interest rate,
- GDP growth.

---

## Step 2

Menu:

```text
Model → Time Series → VAR
```

---

## Step 3

Choose:

- variables,
- lag length,
- deterministic terms.

---

```markdown
[GRETL Screenshot Placeholder: VAR specification window]
```

---

## Gretl Diagnostics

After estimation, GRETL provides:

- residual diagnostics,
- stability tests,
- impulse responses,
- variance decomposition.

---

```markdown
[GRETL Screenshot Placeholder: VAR output]
```

---

# 22.23 Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Ignoring stationarity**  
Nonstationary variables may produce misleading results.

**2. Using too many lags**  
Large VARs may overfit badly.

**3. Interpreting coefficients mechanically**  
VAR interpretation usually focuses on dynamic responses.

**4. Ignoring identification issues**  
Impulse responses may depend heavily on assumptions.

**5. Confusing correlation with causation**  
Reduced-form VARs do not automatically identify causal relationships.
```

# 22.24 Looking Ahead

VAR models provide a flexible framework for studying interacting time series systems.

But coefficient tables alone rarely provide the clearest economic interpretation.

The next chapter introduces impulse response functions (IRFs), which trace how shocks propagate dynamically through multivariate systems.

We will move from:

```{admonition} Observation
Estimating systems.
```

---

# Key Takeaways

```{admonition} Summary
- VAR models analyze multiple interacting time series jointly.
- Each variable depends on its own lags and the lags of other variables.
- VARs treat variables in the system as endogenous.
- Stationarity remains important in VAR analysis.
- Lag length affects model fit, complexity, and forecasting performance.
- Stable VAR systems are generally required for meaningful interpretation and forecasting.
- VAR models are useful tools for system-wide forecasting.
- VARs provide the foundation for later analysis of dynamic shock propagation.
```

# Concept Check

## Basic

1. What is a VAR model?

2. How does a VAR differ from a univariate AR model?

3. What does it mean for all variables in a VAR to be endogenous?

---

## Intuition

4. Why are multivariate models important in economics?

5. What type of interactions can VAR models capture?

6. Why is it unrealistic to model macroeconomic variables in isolation?

---

## Structure

7. In a VAR(1), what variables appear on the right-hand side?

8. What does VAR($p$) mean?

9. Why does the number of parameters increase quickly in VAR models?

---

## Stationarity

10. Why is stationarity important in VAR models?

11. What are the consequences of estimating a VAR with nonstationary data?

---

## Challenge

12. Why are VAR coefficients often difficult to interpret directly?

---

# Interpretation & Practice

1. A VAR model shows that:

- inflation depends on past interest rates  
- interest rates depend on past inflation  

   - What type of relationship does this suggest?

---

2. A VAR is estimated with too few lags.

   - What might happen?

---

3. A VAR is estimated with too many lags.

   - What problem might arise?

---

4. A model includes multiple variables with strong feedback.

   - Why might a VAR be appropriate?

---

## Lag Selection

5. AIC suggests 5 lags, BIC suggests 2 lags.

   - Why might these differ?
   - Which might you prefer?

---

## Stationarity

6. Variables appear to be nonstationary.

   - What should you do before estimating a VAR?

---

# Numerical Practice

## VAR Interpretation

1. Consider a VAR(1):

```{math}
:enumerated: false
y_t = 0.5y_{t-1} + 0.3x_{t-1}
```

```{math}
:enumerated: false
x_t = 0.2y_{t-1} + 0.6x_{t-1}
```

- Does $y_t$ depend on past $x_t$?
- Does $x_t$ depend on past $y_t$?

---

## Lag Structure

2. Suppose you increase lag length from 1 to 4.

- What happens to:
  - number of parameters?
  - estimation complexity?

---

## Information Criteria

3. Suppose:

| Lag | AIC | BIC |
|----|----:|----:|
| 1  | -4.5 | -4.4 |
| 2  | -4.8 | -4.6 |
| 3  | -4.9 | -4.5 |

- Which lag is chosen by AIC?
- Which by BIC?

---

## Interpretation

4. Suppose a variable responds strongly to its own lag.

- What does this suggest?

---

## Diagnostics

5. Residuals show autocorrelation.

- What does this imply?
- What should be done?

---

## Challenge

6. Suppose a VAR includes 5 variables and 4 lags.

- Why might this be problematic?

---

## Interpretation

7. A coefficient on lagged inflation is positive and significant.

   - What does this imply?
   - Why might interpretation still be limited?

---

## Challenge

8. Why should VAR results be interpreted using IRFs rather than raw coefficients?

9. Forecasting with VAR

- Why might VAR models outperform univariate models?
- What makes VAR forecasts dynamic?

---

## Interpretation

Forecasts depend on previous forecasts.

- Why?
- Why might VAR forecasts still perform poorly in practice?

---

# Appendix 22A — Why VARs Became Influential

VARs became highly influential because they offered a systematic way to model rich dynamics across multiple time series.

Earlier macroeconomic models often relied heavily on strong theoretical restrictions.

VARs instead emphasized:

- empirical dynamics,
- flexible interactions,
- and forecasting performance.

This made them especially attractive for applied macroeconomics and finance.

---

# Appendix 22B — Dynamic Forecasting

VAR forecasts are dynamic because future values depend recursively on previous forecasts.

For example:

```{math}
:enumerated: false
\hat Y_{t+2}
```

depends partly on:

```{math}
:enumerated: false
\hat Y_{t+1}
```

This recursive structure is one reason VAR forecasting can capture complex dynamic interactions across variables.