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

---

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

---

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

---

```{admonition} Key Idea
If including lagged values of $x_t$ improves prediction of $y_t$, then $x_t$ Granger-causes $y_t$.
```

---

## 19.2 Intuition

Granger causality is about **information and prediction**, not true causality.

---

```{admonition} Important
Granger causality does **not** mean that $x_t$ truly causes $y_t$ in a structural or economic sense.

It only means that $x_t$ contains useful predictive information about $y_t$.
```

---

### Example (Economic Intuition)

* Interest rates → inflation
* Income → consumption
* Money supply → output

In each case, we ask:

👉 Do past values of one variable help forecast another?

---

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

### Step 1: Estimate the Model

#### Menu

Model → Time series → VAR

* Select variables: $y$, $x$
* Choose lag length (e.g., 2 or 4)

---

```markdown
[GRETL Screenshot Placeholder: VAR setup window]
```

---

### Step 2: Perform Granger Causality Test

#### Menu

Model → Tests → Granger causality

---

```markdown
[GRETL Screenshot Placeholder: Granger causality output]
```

---

#### Command

```gretl
var 2 y x
granger 2 y x
```

---

## 19.5 Interpreting Results

The output provides an F-test (or Wald test).

---

```{admonition} Example Interpretation
If the p-value is less than 0.05:

→ Reject $H_0$  
→ Conclude that $x_t$ Granger-causes $y_t$
```

---

### Possible Outcomes

| Result            | Interpretation             |
| ----------------- | -------------------------- |
| $x \rightarrow y$ | $x$ helps predict $y$      |
| $y \rightarrow x$ | reverse causality          |
| both              | feedback relationship      |
| neither           | no predictive relationship |

---

## 19.6 Choice of Lag Length

Lag selection is crucial.

---

```{admonition} Practical Advice
- Too few lags → omitted dynamics  
- Too many lags → loss of degrees of freedom  
```

---

### In GRETL

Model → Lag selection → AIC / BIC

---

```markdown
[GRETL Screenshot Placeholder: Lag selection criteria]
```

---

## 19.7 Stationarity Matters

Granger causality tests require **stationary data**.

---

```{admonition} Important
If variables are nonstationary, Granger causality tests can produce misleading results.
```

---

### What to Do

* If variables are $I(1)$ → difference them
* or move to cointegration framework (next chapter)

---

```{admonition} Connection to Previous Chapter
Spurious regression arises when nonstationary variables are used in levels.

Granger causality can suffer from similar issues if stationarity is ignored.
```

---

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

---

## 19.9 Economic Interpretation

Even when Granger causality is detected:

* it does not imply policy effectiveness
* it does not reveal mechanisms
* it does not establish structural relationships

---

```{admonition} Big Picture
Granger causality is best viewed as a **tool for forecasting and temporal ordering**, not as proof of economic causation.
```

---

## 19.10 Summary

```{admonition} Key Takeaways
- Granger causality tests predictive relationships  
- it is based on lagged information  
- it requires stationarity  
- it does not imply true causation  
```

---

## Looking Ahead

Granger causality focuses on predictive relationships in stationary data.

However, many economic time series are nonstationary.

In the next chapter, we introduce **cointegration**, which allows us to study **long-run relationships between nonstationary variables**.
