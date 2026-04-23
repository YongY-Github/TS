---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 20 — Cointegration and Long-Run Relationships

In Chapter 17, we saw that regressions involving nonstationary variables can produce **spurious results**.

In Chapter 18, we introduced dynamic models (ARDL), which help capture short-run dependence.

However, an important question remains:

> **Can nonstationary variables still have a meaningful long-run relationship?**

The answer is **yes** — and this is the idea behind **cointegration**.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- understand the concept of cointegration  
- distinguish spurious regression from cointegrated relationships  
- implement the Engle–Granger test in GRETL  
- interpret residual stationarity  
- understand the idea of long-run equilibrium  

---

## 20.1 Motivation: Spurious vs Meaningful Relationships

Recall from Chapter 17:

If we regress two nonstationary series:

$$
y_t = \alpha + \beta x_t + e_t
$$

we may obtain:

- high $R^2$  
- significant t-statistics  
- but **completely meaningless results**

---

```{admonition} Key Problem
Nonstationary variables can move together purely by chance, leading to **spurious regression**.
````

---

But sometimes, variables move together **because they are linked by economic forces**.

Examples:

* consumption and income
* prices in related markets
* exchange rates and interest rates

---

## 20.2 What Is Cointegration?

```{admonition} Definition: Cointegration
Two nonstationary series $x_t$ and $y_t$ are **cointegrated** if a linear combination of them is stationary.
```

Formally:

* $x_t \sim I(1)$
* $y_t \sim I(1)$

but:

$$
e_t = y_t - \beta x_t
$$

is **stationary** ($I(0)$)

---

```{admonition} Key Idea
Cointegration means that although variables wander over time, they do not drift too far apart.
```

---

## 20.3 Intuition: Long-Run Equilibrium

Even if $x_t$ and $y_t$ individually behave like random walks:

* their difference may be stable
* deviations from equilibrium are temporary

---

```{admonition} Intuition
Cointegrated variables are tied together by a **long-run equilibrium relationship**.
```

---

## 20.4 Spurious Regression vs Cointegration

This is a crucial distinction.

| Case                | Residuals     | Interpretation              |
| ------------------- | ------------- | --------------------------- |
| Spurious regression | Nonstationary | No meaningful relationship  |
| Cointegration       | Stationary    | Long-run equilibrium exists |

---

```{admonition} Diagnostic Principle
The key to distinguishing spurious regression from cointegration is:

→ **Are the residuals stationary?**
```

---

## 20.5 Engle–Granger Two-Step Procedure

We now describe a practical method for testing cointegration.

---

### Step 1: Estimate Long-Run Relationship

Estimate:

$$
y_t = \alpha + \beta x_t + e_t
$$

---

#### GRETL Command

```gretl
ols y const x
```

---

```markdown
[GRETL Screenshot Placeholder: OLS output]
```

---

### Step 2: Extract Residuals

Save residuals:

```gretl
series uhat = $uhat
```

---

```markdown
[GRETL Screenshot Placeholder: residual series]
```

---

### Step 3: Test Residual Stationarity

Perform an ADF test:

#### Menu

Variable → Unit root tests → ADF

Select `uhat`

---

#### Command

```gretl
adf 1 uhat
```

---

```markdown
[GRETL Screenshot Placeholder: ADF test output]
```

---

## 20.6 Hypotheses

$$
H_0: \text{Residuals have a unit root (no cointegration)}
$$

$$
H_1: \text{Residuals are stationary (cointegration)}
$$

---

```{admonition} Decision Rule
- Reject $H_0$ → residuals stationary → cointegration  
- Fail to reject $H_0$ → spurious regression  
```

---

## 20.7 Interpretation

If residuals are stationary:

* deviations from equilibrium are temporary
* variables move together in the long run

---

```{admonition} Key Insight
Cointegration restores meaning to regressions involving nonstationary variables.
```

---

## 20.8 Important Caveats

```{admonition} Important
The Engle–Granger method:

- assumes a single cointegrating relationship  
- depends on which variable is treated as dependent  
```

---

## 20.9 Relationship to ARDL

Recall Chapter 18:

* ARDL captures **short-run dynamics**
* cointegration captures **long-run equilibrium**

---

```{admonition} Connection
If variables are cointegrated, an ARDL model can be rewritten as an **error correction model (ECM)**.
```

---

## 20.10 Common Pitfalls

```{admonition} Common Mistakes
:class: warning

**1. Skipping unit root testing**  
Always verify that variables are $I(1)$ before testing cointegration.

**2. Using levels blindly**  
Not all regressions in levels imply cointegration.

**3. Misinterpreting residual tests**  
Stationary residuals are essential.

**4. Ignoring lag structure in ADF test**  
Incorrect lag choice can distort results.

**5. Small samples**  
Cointegration tests may have low power.
```

---

## 20.11 Summary

```{admonition} Key Takeaways
- Cointegration allows meaningful relationships between nonstationary variables  
- The key test is whether residuals are stationary  
- Engle–Granger provides a simple two-step procedure  
- Cointegration implies a long-run equilibrium  
```

---

## Looking Ahead

Cointegration tells us that a long-run relationship exists.

But how do variables adjust when they deviate from this equilibrium?

In the next chapter, we introduce the **Error Correction Model (ECM)**, which combines:

* short-run dynamics
* long-run equilibrium adjustment
