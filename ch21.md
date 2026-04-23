---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 21 — Error Correction Models (ECM)

In the previous chapters:

- **Chapter 17** showed that regressions in levels can be **spurious**  
- **Chapter 18** introduced **dynamic models (ARDL)**  
- **Chapter 20** showed that **cointegration restores meaning** via stationary residuals  

We now bring everything together.

> How do variables adjust when they deviate from a long-run equilibrium?

The answer is given by the **Error Correction Model (ECM)**.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- understand the intuition behind ECM  
- derive ECM from a cointegrating relationship  
- interpret the error correction term  
- estimate ECM in GRETL  
- connect ECM to ARDL models  

---

## 21.1 From Cointegration to Adjustment

Suppose $y_t$ and $x_t$ are cointegrated:

$$
y_t = \alpha + \beta x_t + e_t
$$

with:

- $e_t$ stationary  
- $e_t$ representing deviations from long-run equilibrium  

---

```{admonition} Key Idea
If $y_t$ and $x_t$ deviate from equilibrium, economic forces tend to push them back.
````

---

## 21.2 The Error Correction Mechanism

Define:

$$
e_{t-1} = y_{t-1} - \alpha - \beta x_{t-1}
$$

This is the **lagged equilibrium error**.

---

We model short-run changes:

$$
\Delta y_t = \gamma_0 + \gamma_1 \Delta x_t + \lambda e_{t-1} + u_t
$$

---

```{admonition} Definition: Error Correction Model
An ECM explains changes in a variable as a function of:

- short-run changes  
- past deviations from long-run equilibrium  
```

---

## 21.3 Interpretation of the ECM

Each component has a clear meaning.

---

### Short-run effect

$$
\gamma_1 \Delta x_t
$$

→ immediate impact of changes in $x$

---

### Error correction term

$$
\lambda e_{t-1}
$$

---

```{admonition} Key Interpretation
- $\lambda < 0$ → system returns to equilibrium  
- magnitude of $\lambda$ → speed of adjustment  
```

---

### Example

* $\lambda = -0.5$ → 50% of deviation corrected each period

---

## 21.4 Intuition

```{admonition} Intuition
The ECM says:

- variables may drift apart in the short run  
- but are pulled back toward equilibrium over time  
```

---

This combines:

* **short-run dynamics**
* **long-run equilibrium**

---

## 21.5 Estimation in GRETL (Step-by-Step)

---

### Step 1: Estimate Long-Run Relationship

```gretl
ols y const x
```

---

```markdown
[GRETL Screenshot Placeholder: long-run regression]
```

---

### Step 2: Save Residuals

```gretl
series ehat = $uhat
```

---

```markdown
[GRETL Screenshot Placeholder: residual series]
```

---

### Step 3: Construct ECM Regression

```gretl
series dy = diff(y)
series dx = diff(x)
series ehat_1 = ehat(-1)

ols dy const dx ehat_1
```

---

```markdown
[GRETL Screenshot Placeholder: ECM regression output]
```

---

## 21.6 Interpreting GRETL Output

Focus on:

### Coefficient on $\Delta x_t$

* short-run effect

### Coefficient on $ehat_{t-1}$

* should be **negative and significant**

---

```{admonition} Interpretation Rule
If the coefficient on the error correction term is:

- negative → equilibrium is stable  
- statistically significant → adjustment mechanism exists  
```

---

## 21.7 ECM from ARDL (Very Important Connection)

Recall ARDL(1,1):

$$
y_t = \alpha + \phi y_{t-1} + \beta_0 x_t + \beta_1 x_{t-1} + u_t
$$

---

Rewriting gives:

$$
\Delta y_t = \gamma \Delta x_t + \lambda (y_{t-1} - \beta x_{t-1}) + u_t
$$

---

```{admonition} Deep Insight
An ARDL model **implicitly contains an ECM representation**.
```

---

This means:

* ARDL → short-run + long-run
* ECM → makes this structure explicit

---

## 21.8 Common Pitfalls

```{admonition} Common Mistakes
:class: warning

**1. Ignoring cointegration**  
ECM requires a valid long-run relationship.

**2. Wrong sign of $\lambda$**  
Positive $\lambda$ implies divergence.

**3. Misinterpreting coefficients**  
Short-run and long-run effects are different.

**4. Omitting dynamics**  
ECM may require additional lags.

**5. Small sample issues**  
Estimates may be unstable.
```

---

## 21.9 Big Picture

```{admonition} Big Picture
ECM provides a unified framework:

- short-run fluctuations  
- long-run equilibrium  
- dynamic adjustment  
```

---

## 21.10 Summary

```{admonition} Key Takeaways
- ECM links cointegration and dynamics  
- deviations from equilibrium drive adjustments  
- the error correction term measures speed of adjustment  
- ECM can be derived from ARDL  
```

---

## Looking Ahead

So far, we have focused on **bivariate relationships**.

In the next part of the book, we extend these ideas to **multivariate systems**, including:

* VAR models
* impulse response functions
* VECM
