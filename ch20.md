---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 20 — Cointegration and Long-Run Relationships

In Chapter 17, we saw that regressions involving nonstationary variables can produce **spurious results**.

In Chapter 18, we introduced dynamic models such as ARDL, which capture short-run dependence and adjustment dynamics.

An important question now arises:

```{admonition} Central Question
Can nonstationary variables still have a meaningful long-run relationship?
```

The answer is yes.

This idea is called **cointegration**.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- explain the idea of cointegration
- distinguish spurious regression from cointegrated relationships
- understand long-run equilibrium
- implement the Engle–Granger procedure in GRETL
- interpret residual stationarity
- understand the logic of the ARDL bounds test

---

# 20.1 Motivation: Spurious vs Meaningful Relationships

Recall from Chapter 17:

If we regress two unrelated nonstationary variables,

```{math}
:enumerated: false
y_t = \alpha + \beta x_t + e_t
```

we may obtain:

- high $R^2$
- significant t-statistics
- apparently strong relationships

even when the variables are unrelated.

```{admonition} Key Problem
Nonstationary variables may move together purely by chance, leading to **spurious regression**.
```

However, some nonstationary variables genuinely move together because they are linked by economic forces.

Examples include:

- consumption and income
- exchange rates and prices
- interest rates and inflation
- prices in related financial markets

```{admonition} Key Idea
Some nonstationary variables drift over time together because they share a long-run equilibrium relationship.
```

---

# 20.2 What Is Cointegration?

Suppose:

- $x_t$ is nonstationary
- $y_t$ is nonstationary

but a particular linear combination is stationary.

Then the variables are cointegrated.

```{admonition} Definition: Cointegration
Two variables are cointegrated if they are individually nonstationary, but some linear combination of them is stationary.
```

Formally:

```{math}
:enumerated: false
x_t \sim I(1),
\qquad
y_t \sim I(1)
```

but:

```{math}
:enumerated: false
e_t = y_t - \beta x_t \sim I(0)
```

```{admonition} Key Insight
Cointegration means the variables may wander over time, but they do not drift too far apart in the long run.

Individually: variables wander  
Together: they are tied by an equilibrium
```

---

# 20.3 Intuition: Long-Run Equilibrium

Even if two variables individually behave like random walks, their difference may remain stable.

Cointegration means that while variables may drift over time, they do not drift arbitrarily far apart.

This suggests that some equilibrium force ties them together.

```{admonition} Intuition
Cointegrated variables are connected by a long-run equilibrium relationship.
```

```{admonition} Intuition (Optional)
Think of two variables connected by a rubber band.

They may drift apart temporarily, but forces exist that pull them back together over time.
```

Examples:

- consumption cannot permanently diverge from income
- exchange rates and relative prices remain linked in the long run
- stock prices of related firms may move together over time

---

# 20.4 Spurious Regression vs Cointegration

This distinction is fundamental.

| Case | Residuals | Interpretation |
|---|---|---|
| Spurious regression | Nonstationary | No meaningful relationship |
| Cointegration | Stationary | Long-run equilibrium exists |

```{admonition} Diagnostic Principle
The key question is:

→ Are the residuals stationary?
```

If residuals are stationary, the regression may be meaningful despite nonstationarity in the original variables.

---

# 20.5 The Engle–Granger Two-Step Procedure

We now describe the classic Engle–Granger procedure for testing cointegration.

We use quarterly GDP data from Gretl.

## Step 1: Load the Data

### Menu

`File → Open data → Sample file...`

Select:

```text
gdp
```

from the `POE 4th ed.` database.

The dataset contains:

```gretl
usa     real GDP of USA
aus     real GDP of Australia
```

## Step 2: Estimate the Long-Run Relationship

Estimate:

```{math}
:enumerated: false
aus_t
=
\alpha
+
\beta usa_t
+
e_t
```

## Gretl Command

```gretl
ols aus const usa
```

## Output

```gretl
Model 1: OLS, using observations 1970:1-2000:4 (T = 124)
Dependent variable: aus

             coefficient   std. error   t-ratio    p-value 
  ---------------------------------------------------------
  const       −1.07237     0.403225      −2.659   0.0089    ***
  usa          1.00099     0.00610028   164.1     5.85e-145 ***

Mean dependent var   62.72528
R-squared            0.995489
Durbin-Watson        0.272654
```

```{admonition} Observation
The regression appears extremely strong.

But high $R^2$ alone does not prove a meaningful relationship.
```

## Step 3: Extract the Residuals

Save the residuals:

```gretl
series uhat = $uhat
```

```markdown
[GRETL Screenshot Placeholder: Residual series]
```

```{admonition} Key Step
The residuals contain the estimated deviations from long-run equilibrium.
```

## Step 4: Test Residual Stationarity

We now test whether the residuals are stationary.

This is the crucial step.


## Menu

Select `uhat`.

Then:

`Variable → Unit root tests → Augmented Dickey-Fuller`

## Gretl Command

```gretl
adf 1 uhat
```

## Example Output

```gretl
Augmented Dickey-Fuller test for uhat
unit-root null hypothesis: a = 1

test statistic: tau_c(1) = -3.03875
asymptotic p-value 0.03145
```

---

# 20.6 Hypotheses

We test:

```{math}
:enumerated: false
H_0:
\text{Residuals contain a unit root}
```

against:

```{math}
:enumerated: false
H_1:
\text{Residuals are stationary}
```

```{admonition} Decision Rule
- Reject $H_0$ → residuals stationary → cointegration
- Fail to reject $H_0$ → no cointegration
```

---

# 20.7 Interpretation

If residuals are stationary:

- deviations from equilibrium are temporary
- variables move together in the long run
- the regression is not spurious

```{admonition} Key Insight
Cointegration restores meaning to regressions involving nonstationary variables.
```

---

# 20.8 Why Residual Stationarity Matters

Suppose:

```{math}
:enumerated: false
e_t
=
y_t
-
\beta x_t
```

is stationary.

Then although:

- $x_t$ may drift
- $y_t$ may drift

their deviations from equilibrium remain bounded.

```{admonition} Big Picture
Cointegration means that the variables share a common long-run stochastic trend.
```

---

# 20.9 Important Caveats

The Engle–Granger procedure has several limitations.

```{admonition} Important
The Engle–Granger approach:

- assumes a single cointegrating relationship
- depends on which variable is treated as dependent
- requires variables to be $I(1)$
```

In multivariate systems, more advanced methods may be preferable.

---

# 20.10 Cointegration and Dynamic Models

Cointegration naturally connects to the ARDL framework from Chapter 18.

Recall the ARDL model:

```{math}
:enumerated: false
y_t
=
\alpha
+
\phi y_{t-1}
+
\beta_0 x_t
+
\beta_1 x_{t-1}
+
u_t
```

This model contains both:

- short-run dynamics
- long-run structure

```{admonition} Key Connection
ARDL models can be used to test for long-run relationships between variables.
```

---

# 20.11 Cointegration via ARDL (Bounds Testing)

The ARDL bounds approach provides an alternative to Engle–Granger.

An important advantage is flexibility.

```{admonition} Advantage
The ARDL bounds approach allows variables to be a mixture of:

- $I(0)$
- $I(1)$
```

as long as none are $I(2)$.

---

# 20.12 From ARDL to ECM

Consider:

```{math}
:enumerated: false
y_t
=
\alpha
+
\phi y_{t-1}
+
\beta_0 x_t
+
\beta_1 x_{t-1}
+
u_t
```

This can be rewritten as:

```{math}
:enumerated: false
\Delta y_t
=
\gamma \Delta x_t
+
\lambda_1 y_{t-1}
+
\lambda_2 x_{t-1}
+
u_t
```

```{admonition} Key Insight
Cointegration exists if the lagged level terms are jointly significant.
```

---

# 20.13 The Bounds Test

We test:

```{math}
:enumerated: false
H_0:
\lambda_1
=
\lambda_2
=
0
```

against:

```{math}
:enumerated: false
H_1:
\text{At least one coefficient is nonzero}
```

## Decision Rule

| F-statistic | Conclusion |
|---|---|
| below lower bound | no cointegration |
| above upper bound | cointegration |
| between bounds | inconclusive |

```{admonition} Key Idea
The bounds test avoids requiring exact knowledge of the integration order of the variables.
```

---

# 20.14 Implementing ARDL Bounds Testing in GRETL

## Step 1: Estimate an ARDL Model

### Menu

`Model → Time series → ARDL`

Choose:

- dependent variable
- regressors
- lag lengths

```markdown
[GRETL Screenshot Placeholder: ARDL specification window]
```

## Step 2: Perform Bounds Test

From the ARDL output window:

- select bounds test option

```markdown
[GRETL Screenshot Placeholder: Bounds test output]
```

## Example Command

```gretl
ardl 2 2 y x
```

Then:

```gretl
ecm
```

---

# 20.15 Comparing Engle–Granger and ARDL

| Feature | Engle–Granger | ARDL Bounds |
|---|---|---|
| Requires all variables $I(1)$ | Yes | No |
| Residual-based | Yes | No |
| Dynamic model based | No | Yes |
| Allows mixed $I(0)/I(1)$ variables | No | Yes |

```{admonition} Two Approaches
- Engle–Granger focuses on residual stationarity.
- ARDL bounds testing focuses on long-run information embedded in a dynamic model.
```

---

# 20.16 Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Ignoring integration order**  
Variables should not be $I(2)$.

**2. Assuming high $R^2$ implies cointegration**  
Residual stationarity is the key diagnostic.

**3. Forgetting lag selection**  
Dynamic misspecification affects cointegration tests.

**4. Confusing correlation with equilibrium**  
Cointegration implies a stable long-run relationship, not merely correlation.

**5. Ignoring economic theory**  
Cointegration tests should be guided by economic reasoning.
```

---

# 20.17 Looking Ahead

Cointegration tells us that a long-run equilibrium relationship exists.

But how do variables adjust when they deviate from equilibrium?

This leads naturally to the **Error Correction Model (ECM)**.


# Key Takeaways

```{admonition} Summary
- Cointegration allows meaningful relationships between nonstationary variables.
- Cointegrated variables share a long-run equilibrium relationship.
- Residual stationarity distinguishes cointegration from spurious regression.
- Engle–Granger provides a residual-based approach.
- ARDL bounds testing provides a dynamic-model-based approach.
```

# Concept Check

## Basic

1. What is cointegration?

2. What does it mean for two variables to be $I(1)$?

3. What does it mean for a linear combination of variables to be $I(0)$?

---

## Intuition

4. Why can two nonstationary variables still have a meaningful relationship?

5. What is meant by a long-run equilibrium?

6. Explain the “rubber band” analogy for cointegration.

---

## Spurious vs Cointegration

7. What distinguishes a spurious regression from a cointegrated relationship?

8. Why is a high $R^2$ not sufficient evidence of cointegration?

9. What role do residuals play in diagnosing cointegration?

---

## Engle–Granger Procedure

10. What are the two steps in the Engle–Granger method?

11. What is the null hypothesis in the residual-based test?

12. What does it mean to reject the null hypothesis?

---

## ARDL and Bounds Testing

13. How does the ARDL bounds approach differ from Engle–Granger?

14. What is the key hypothesis tested in the bounds test?

---

## Challenge

15. Can cointegration exist if one variable is $I(0)$ and the other is $I(1)$?

---

# Interpretation & Practice

1. A regression between two variables produces:

- high $R^2$
- significant coefficients
- nonstationary residuals  

   - What does this imply?

2. Residuals from a regression are stationary.

   - What does this suggest?

3. Two variables are both $I(1)$, but their difference is stationary.

   - What does this imply?

4. ADF test on residuals gives p-value = 0.02.

   - What is your conclusion?

5. ADF test on residuals gives p-value = 0.60.

   - What is your conclusion?

---

## ARDL Interpretation

6. In an ARDL model, lagged level terms are jointly significant.

   - What does this imply?

7. Bounds test F-statistic is above the upper bound.

   - What is your conclusion?

---

## Economic Interpretation

8. Consumption and income are cointegrated.

   - What does this imply about their relationship?

---

## Challenge

9. A regression is significant in levels but insignificant in differences.

   - What might this suggest?

---

# Numerical Practice

## Residual-Based Logic

1. Suppose:

- $x_t \sim I(1)$  
- $y_t \sim I(1)$  
- residuals $\hat{e}_t \sim I(0)$  

---

- What is your conclusion?

---

## ADF Interpretation Table

2. Consider:

| Series | ADF p-value |
|-------|------------:|
| $x_t$ | 0.85 |
| $y_t$ | 0.78 |
| residuals | 0.03 |

---

- Are $x_t$ and $y_t$ stationary?
- Are residuals stationary?
- What does this imply?

---

3. Now consider:

| Series | ADF p-value |
|-------|------------:|
| $x_t$ | 0.90 |
| $y_t$ | 0.88 |
| residuals | 0.72 |

---

- What is your conclusion?

---

## Engle–Granger

4. Explain why testing residuals is central to the Engle–Granger procedure.

---

## Bounds Test

5. Suppose:

- F-statistic = 6.5  
- upper bound = 5.0  

---

- What is your conclusion?

---

## Interpretation

6. Suppose cointegration exists.

- What does this imply about long-run behavior?

---

## Challenge

7. Suppose two variables are cointegrated.

- What happens if they deviate from equilibrium?
- What concept does this lead to?

8. You regress:

- exchange rate  
- price level  

You find:

- strong relationship  
- stationary residuals  

---

- What does this imply?
- Why is this not spurious?

---

# Appendix 20A — The ARDL Bounds Test (Conceptual Overview)

This appendix provides a simplified explanation of the ARDL bounds approach.

---

# A.1 Starting Point

Consider:

```{math}
:enumerated: false
\Delta y_t
=
\gamma \Delta x_t
+
\lambda_1 y_{t-1}
+
\lambda_2 x_{t-1}
+
u_t
```

---

# A.2 The Key Question

Do the lagged level terms matter?

That is:

```{math}
:enumerated: false
H_0:
\lambda_1
=
\lambda_2
=
0
```

---

# A.3 Interpretation

If both coefficients are zero:

- no long-run relationship exists

If at least one coefficient is nonzero:

- long-run equilibrium exists

```{admonition} Key Insight
Cointegration means that lagged levels help explain current changes.
```

---

# A.4 Why Two Critical Values?

The asymptotic distribution depends on whether variables are:

- stationary ($I(0)$)
- nonstationary ($I(1)$)

The bounds approach therefore provides:

- lower critical values
- upper critical values

---

# A.5 Decision Rule

- Below lower bound → no cointegration
- Above upper bound → cointegration
- Between bounds → inconclusive

```{admonition} Big Picture
The bounds test allows inference without requiring exact knowledge of the integration order of the variables.
```