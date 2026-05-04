---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 21 — Error Correction Models (ECM)

In previous chapters:

- **Chapter 17** showed that regressions involving nonstationary variables can be **spurious**
- **Chapter 18** introduced **dynamic models (ARDL)**
- **Chapter 20** showed that **cointegration restores meaning** through stationary residuals

We now bring these ideas together.

```{admonition} Central Question
How do variables adjust when they deviate from a long-run equilibrium?
```

The answer is given by the **Error Correction Model (ECM)**.

This chapter follows and extends the structure in your notes. :contentReference[oaicite:0]{index=0}

---

## Learning Objectives

By the end of this chapter, you should be able to:

- explain the intuition behind ECM
- understand the role of the error correction term
- distinguish short-run and long-run effects
- estimate ECMs in Gretl
- connect ECM to ARDL and cointegration
- interpret adjustment dynamics

---

# 21.1 From Cointegration to Adjustment

Suppose two variables are cointegrated:

```{math}
:enumerated: false
y_t
=
\alpha
+
\beta x_t
+
e_t
```

Then:

```{math}
:enumerated: false
e_t
=
y_t
-
\alpha
-
\beta x_t
```

measures deviations from the long-run equilibrium relationship.

```{admonition} Key Idea
If variables are cointegrated, deviations from equilibrium are temporary rather than permanent.
```

---

# 21.2 Why an Adjustment Mechanism Is Needed

If the variables drift apart, economic forces may push them back toward equilibrium.

Examples:

- consumption adjusts toward income
- exchange rates adjust toward purchasing power parity
- money demand adjusts toward long-run monetary equilibrium

```{admonition} Intuition
Cointegrated variables behave like they are tied together by a rubber band.

They may drift apart temporarily, but adjustment forces pull them back together.
```

---

# 21.3 The Error Correction Model

The ECM combines:

- short-run dynamics
- long-run equilibrium adjustment

Define:

```{math}
:enumerated: false
e_{t-1}
=
y_{t-1}
-
\alpha
-
\beta x_{t-1}
```

Then the ECM is:

```{math}
:enumerated: false
\Delta y_t
=
\gamma_0
+
\gamma_1 \Delta x_t
+
\lambda e_{t-1}
+
u_t
```

```{admonition} Definition
An Error Correction Model explains current changes as a function of:

- short-run changes
- past deviations from long-run equilibrium
```

---

# 21.4 Interpreting the ECM

Each component has a distinct interpretation.

## Short-Run Effect

```{math}
:enumerated: false
\gamma_1 \Delta x_t
```

captures the immediate effect of changes in $x_t$ on changes in $y_t$.

## Error Correction Term

```{math}
:enumerated: false
\lambda e_{t-1}
```

captures adjustment toward equilibrium.

```{admonition} Key Interpretation
The coefficient $\lambda$ measures the speed of adjustment back toward equilibrium.
```

---

# 21.5 The Sign of the Adjustment Coefficient

The sign of $\lambda$ is crucial.

```{admonition} Stability Condition
For a stable equilibrium:

$$
\lambda < 0
$$
```

Why?

Suppose:

```{math}
:enumerated: false
e_{t-1} > 0
```

meaning:

```{math}
:enumerated: false
y_{t-1}
>
\alpha
+
\beta x_{t-1}
```

Then a negative $\lambda$ pushes:

```{math}
:enumerated: false
\Delta y_t < 0
```

helping restore equilibrium.

```{admonition} Interpretation
- $\lambda < 0$ → convergence toward equilibrium
- $\lambda > 0$ → divergence away from equilibrium
```

---

# 21.6 Speed of Adjustment

The magnitude of $\lambda$ measures how quickly adjustment occurs.

Examples:

| $\lambda$ | Interpretation |
|---|---|
| $-0.1$ | slow adjustment |
| $-0.5$ | moderate adjustment |
| $-1$ | rapid adjustment |

```{admonition} Example
If:

$$
\lambda = -0.5
$$

then approximately 50% of disequilibrium is corrected each period.
```

---

# 21.7 Short-Run vs Long-Run Dynamics

ECM separates two different forces.

```{admonition} Two Components
- $\Delta x_t$ → short-run movements
- $e_{t-1}$ → long-run equilibrium correction
```

This is one of the major strengths of ECM.

---

# 21.8 Estimating ECM in Gretl

We now estimate an ECM using the same data as in Chapter 20.

This follows your notes closely.

## Step 1: Estimate Long-Run Relationship

Estimate:

```gretl
ols aus const usa
```

## Step 2: Save Residuals

Save the residuals from the long-run regression:

```gretl
series ehat = $uhat
```

These residuals estimate deviations from equilibrium.

## Step 3: Construct Differenced Variables

Create first differences:

```gretl
series d_usa = diff(usa)
series d_aus = diff(aus)
```

Create lagged residuals:

```gretl
series ehat_1 = ehat(-1)
```

## Step 4: Estimate the ECM

Estimate:

```gretl
ols d_aus const d_usa ehat_1
```

## Example Output

```gretl
Model 2: OLS, using observations 1970:2-2000:4 (T = 123)
Dependent variable: d_aus

             coefficient   std. error   t-ratio   p-value 
  --------------------------------------------------------
  const        0.211535    0.0710333     2.978    0.0035   ***
  d_usa        0.567700    0.0983739     5.771    6.29e-08 ***
  ehat_1      −0.138852    0.0426256    −3.257    0.0015   ***
```

```{math}
:enumerated: false
\widehat{\Delta aus_t}
=
0.212
+
0.568 \Delta usa_t
-
0.139 ehat_{t-1}
```

```markdown
[GRETL Screenshot Placeholder: ECM estimation output]
```

## Interpreting the Output

## Coefficient on $\Delta usa_t$

```{math}
:enumerated: false
0.568
```

This measures the short-run effect.

```{admonition} Interpretation
A one-unit increase in $\Delta usa_t$ is associated with an immediate increase of approximately 0.568 in $\Delta aus_t$.
```

## Coefficient on $ehat_{t-1}$

```{math}
:enumerated: false
-0.139
```

This is the adjustment coefficient.

```{admonition} Interpretation
About 13.9% of disequilibrium is corrected each period.
```

Since the coefficient is:

- negative
- statistically significant

there is evidence of stable equilibrium adjustment.

---

# 21.9 Why ECM Matters

ECM solves a major problem in time series analysis.

Recall the tension:

| Approach | Problem |
|---|---|
| levels regression | spurious regression |
| differencing | loss of long-run information |

ECM combines both.

```{admonition} Big Picture
ECM allows us to model:

- short-run changes
- long-run equilibrium
- gradual adjustment over time
```

---

# 21.10 Connection to ARDL

Recall the ARDL(1,1) model:

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
\lambda
(y_{t-1} - \theta x_{t-1})
+
u_t
```

```{admonition} Deep Insight
The ECM is not a completely different model.

It is a reparameterization of ARDL that makes the equilibrium structure explicit.
```

---

# 21.11 ECM and Cointegration

ECM is meaningful only when cointegration exists.

```{admonition} Important
If variables are not cointegrated, the error correction term does not represent a valid equilibrium relationship.
```

Thus:

- cointegration justifies ECM
- ECM operationalizes cointegration

---

# 21.12 Residual Diagnostics

As with all dynamic models, residuals should behave like white noise.

After estimating the ECM:

- plot residuals
- examine correlograms
- test for serial correlation

## Gretl Menu

From model window:

`Tests → Autocorrelation`

```markdown
[GRETL Screenshot Placeholder: ECM residual diagnostics]
```

```{admonition} Goal
A correctly specified ECM should leave little remaining serial correlation in the residuals.
```

---

# 21.13 Common Pitfalls

```{admonition} Common Mistakes
:class: warning

**1. Ignoring cointegration**  
ECM requires a valid long-run relationship.

**2. Wrong sign of adjustment coefficient**  
Positive adjustment implies instability.

**3. Confusing short-run and long-run effects**  
These represent different mechanisms.

**4. Omitting dynamic structure**  
Additional lags may be necessary.

**5. Blind mechanical estimation**  
Economic theory should guide specification.
```

---

## 21.14 Looking Ahead

So far, we have focused mainly on bivariate relationships.

In the next part of the book, we move to multivariate systems:

- VAR models
- impulse response functions
- VECM

where multiple variables interact dynamically over time.

---

# Key Takeaways

```{admonition} Summary
- ECM combines short-run dynamics and long-run equilibrium.
- The error correction term measures deviations from equilibrium.
- The adjustment coefficient measures speed of convergence.
- ECM arises naturally from ARDL models.
- Cointegration provides the foundation for ECM.
```

# Concept Check

## Basic

1. What is an Error Correction Model (ECM)?

2. What is the role of the error correction term?

3. What does the coefficient $\lambda$ represent?

---

## Intuition

4. Why do cointegrated variables require an adjustment mechanism?

5. What does it mean for deviations from equilibrium to be temporary?

6. Explain the “rubber band” analogy in the context of ECM.

---

## Structure

7. What are the two main components of an ECM?

8. What does $\Delta x_t$ capture?

9. What does $e_{t-1}$ capture?

---

## Short-Run vs Long-Run

10. What is the difference between:

   - short-run effect  
   - long-run equilibrium  

11. Why is it important to distinguish between the two?

---

## Stability

12. Why must the adjustment coefficient $\lambda$ be negative?

13. What happens if $\lambda > 0$?

---

## Challenge

14. Suppose $\lambda = -0.9$.

   - What does this imply about adjustment speed?

---

# Interpretation & Practice

1. A model shows:

- significant short-run effect  
- insignificant error correction term  

   - What does this imply?

2. The error correction term is:

- negative  
- statistically significant  

   - What does this indicate?

3. The error correction term is positive.

   - What does this imply?

4. A model shows:

- weak short-run effect  
- strong error correction  

   - How would you interpret this?

---

## Cointegration Link

5. Why is ECM only valid when variables are cointegrated?

6. If variables are not cointegrated, what happens to the error correction term?

---

## ARDL Link

7. How is ECM related to ARDL models?

---

## Economic Interpretation

8. Suppose consumption and income are cointegrated.

- What does the ECM tell us about adjustment?

---

## Challenge

9. A model fits well in differences but ignores the error correction term.

- What might be missing?

---

# Numerical Practice

## ECM Interpretation

1. Suppose:

```{math}
:enumerated: false
\Delta y_t = 0.5 \Delta x_t - 0.4 e_{t-1}
```

- What is the short-run effect?
- What is the speed of adjustment?

---

## Adjustment Speed

2. If:

```{math}
:enumerated: false
\lambda = -0.2
```

- What proportion of disequilibrium is corrected each period?

---

3. If:

```{math}
:enumerated: false
\lambda = -0.8
```

- How does this compare?

---

## Sign Interpretation

4. Suppose:

```{math}
:enumerated: false
\lambda = 0.3
```

- What does this imply?
- Is the system stable?

---

## Model Interpretation

5. Suppose ECM estimation gives:

- $\gamma_1 = 0.6$  
- $\lambda = -0.15$  

---

- Interpret both coefficients.

---

## Diagnostics

6. Residuals from ECM show autocorrelation.

- What does this imply?
- What should be done?

---

## Challenge

7. Suppose:

- strong cointegration  
- but $\lambda$ is very small  

---

- What does this imply about adjustment dynamics?

---

8. Suppose:

- $\lambda$ is large in magnitude  
- but short-run effects are weak  

---

- What does this imply about the system?

---

9. You estimate an ECM for exchange rate and price level.

You find:

- $\lambda = -0.3$  
- significant  

---

- What does this imply about adjustment toward purchasing power parity?
- How quickly does the system return to equilibrium?

---

# Appendix 21A — Deriving the ECM from an ARDL Model

This appendix shows how ECM arises naturally from an ARDL model.

---

# A.1 Start with ARDL(1,1)

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

---

# A.2 Subtract $y_{t-1}$

Subtract $y_{t-1}$ from both sides:

```{math}
:enumerated: false
y_t - y_{t-1}
=
\alpha
+
\phi y_{t-1}
+
\beta_0 x_t
+
\beta_1 x_{t-1}
-
y_{t-1}
+
u_t
```

Since:

```{math}
:enumerated: false
\Delta y_t = y_t - y_{t-1}
```

we obtain:

```{math}
:enumerated: false
\Delta y_t
=
\alpha
+
(\phi - 1)y_{t-1}
+
\beta_0 x_t
+
\beta_1 x_{t-1}
+
u_t
```

---

# A.3 Introduce $\Delta x_t$

Add and subtract $\beta_0 x_{t-1}$:

```{math}
:enumerated: false
\Delta y_t
=
\alpha
+
(\phi - 1)y_{t-1}
+
\beta_0(x_t - x_{t-1})
+
(\beta_0 + \beta_1)x_{t-1}
+
u_t
```

Recognize:

```{math}
:enumerated: false
\Delta x_t = x_t - x_{t-1}
```

Therefore:

```{math}
:enumerated: false
\Delta y_t
=
\alpha
+
\beta_0 \Delta x_t
+
(\phi - 1)y_{t-1}
+
(\beta_0 + \beta_1)x_{t-1}
+
u_t
```

---

# A.4 Rearranging

Factor the level terms:

```{math}
:enumerated: false
\Delta y_t
=
\alpha
+
\beta_0 \Delta x_t
+
(\phi - 1)
\left[
y_{t-1}
-
\frac{\beta_0 + \beta_1}{1-\phi}x_{t-1}
\right]
+
u_t
```

Define:

```{math}
:enumerated: false
\lambda = \phi - 1
```

and:

```{math}
:enumerated: false
\theta
=
\frac{\beta_0 + \beta_1}{1-\phi}
```

Then:

```{math}
:enumerated: false
\Delta y_t
=
\alpha
+
\beta_0 \Delta x_t
+
\lambda
(y_{t-1} - \theta x_{t-1})
+
u_t
```

---

# A.5 Final Interpretation

The ECM contains:

| Component | Interpretation |
|---|---|
| $\Delta x_t$ | short-run effect |
| $y_{t-1} - \theta x_{t-1}$ | disequilibrium term |
| $\lambda$ | speed of adjustment |

```{admonition} Final Insight
The ECM separates:

- short-run dynamics
- long-run equilibrium adjustment

within a single unified framework.
```

