---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 9 — ACF and PACF

In previous chapters, we introduced dependence, persistence, and stationarity.

We now develop two of the most important tools in time series analysis:

- the **autocorrelation function (ACF)**
- the **partial autocorrelation function (PACF)**

These tools help us understand:

- dependence across time  
- persistence  
- lag structure  
- model identification  

---

# Learning Objectives

By the end of this chapter, you should be able to:

- understand autocorrelation and autocovariance  
- interpret ACF and PACF plots  
- recognize dependence patterns  
- distinguish white noise and persistent processes  
- use ACF/PACF for model identification  

---

# 9.1 Correlation Across Time

In time series analysis, we study relationships such as:

```{math}
:enumerated: false
Corr(X_t, X_{t-h})
```

where \(h\) is the **lag**.

```{admonition} Definition
Autocorrelation measures dependence between a variable and its past values.
```

---

# 9.2 Autocovariance and ACF

For a weakly stationary process:

```{math}
:enumerated: false
\gamma(h) = Cov(X_t, X_{t-h})
```

The autocorrelation function (ACF) is:

```{math}
:enumerated: false
\rho(h) = \frac{\gamma(h)}{\gamma(0)}
```

where \(\gamma(0) = Var(X_t)\).

````{admonition} Key Property
The autocorrelation satisfies:

```{math}
:enumerated: false
-1 \leq \rho(h) \leq 1
```

````

```{admonition} Key Idea
The ACF measures how strongly the present relates to the past.
```

---

# 9.3 White Noise: No Dependence

## Simulation

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

np.random.seed(10101)

wn = np.random.normal(0, 1, 300)

plt.figure(figsize=(10,4))
plt.plot(wn)
plt.title("White Noise")
plt.savefig("figs/ch9/wn.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![White Noise](figs/ch9/wn.png)

## ACF and PACF

```{code-cell} python
:tags: [hide-input]

fig, ax = plt.subplots(1,2, figsize=(10,4))

plot_acf(wn, lags=30, ax=ax[0])
ax[0].set_title("ACF (White Noise)")

plot_pacf(wn, lags=30, ax=ax[1], method="ywm")
ax[1].set_title("PACF (White Noise)")

plt.savefig("figs/ch9/wn_acf_pacf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![ACF PACF](figs/ch9/wn_acf_pacf.png)

```{admonition} Observation
White noise shows no structure:

- ACF ≈ 0  
- PACF ≈ 0  
```

---

# 9.4 Persistent Series: Random Walk

## Simulation

```{code-cell} python
:tags: [hide-input]

np.random.seed(123)

w = np.random.normal(0, 1, 300)
x = np.cumsum(w)

plt.figure(figsize=(10,4))
plt.plot(x)
plt.title("Random Walk")

plt.savefig("figs/ch9/rw.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Random Walk](figs/ch9/rw.png)

## ACF and PACF

```{code-cell} python
:tags: [hide-input]

fig, ax = plt.subplots(1,2, figsize=(10,4))

plot_acf(x, lags=40, ax=ax[0])
ax[0].set_title("ACF (Random Walk)")

plot_pacf(x, lags=40, ax=ax[1], method="ywm")
ax[1].set_title("PACF (Random Walk)")

plt.tight_layout()

plt.savefig("figs/ch9/rw_acf_pacf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![ACF PACF](figs/ch9/rw_acf_pacf.png)


```{admonition} Key Insight
Persistent series show:

- slowly decaying ACF  
- strong dependence across time  
```

```{admonition} Deep Insight
A random walk can appear to have patterns even though it is driven entirely by random shocks.

This makes it easy to mistake randomness for structure.
```

---

# 9.5 Interpreting ACF Patterns

### White Noise
- ACF ≈ 0 at all lags  

### Persistent Series
- ACF decays slowly  

### Oscillatory Behavior
- ACF alternates signs  

```{admonition} Rule of Thumb
ACF reveals persistence and dependence patterns.
```

---

# 9.6 Partial Autocorrelation (PACF)

The ACF measures total dependence, including indirect effects.

The PACF isolates **direct dependence**.

```{admonition} Definition
The PACF measures the relationship between $X_t$ and $X_{t-h}$, controlling for intermediate lags.
```

---

# 9.7 ACF vs PACF (Intuition)

```{admonition} Intuition

- ACF → total dependence across time  
- PACF → direct dependence at each lag  

PACF helps identify the true lag structure of a process.
```

---

# 9.8 Visual Comparison

```{code-cell} python
:tags: [hide-input]

np.random.seed(42)

# AR(1)
phi = 0.7
ar = np.zeros(300)

for t in range(1, 300):
    ar[t] = phi * ar[t-1] + np.random.normal()

fig, ax = plt.subplots(2,2, figsize=(10,6))

plot_acf(ar, lags=30, ax=ax[0,0])
ax[0,0].set_title("ACF (AR(1))")

plot_pacf(ar, lags=30, ax=ax[0,1], method="ywm")
ax[0,1].set_title("PACF (AR(1))")

plot_acf(wn, lags=30, ax=ax[1,0])
ax[1,0].set_title("ACF (White Noise)")

plot_pacf(wn, lags=30, ax=ax[1,1], method="ywm")
ax[1,1].set_title("PACF (White Noise)")

plt.tight_layout()

plt.savefig("figs/ch9/acf_pacf_.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![ACF PACF](figs/ch9/acf_pacf_.png)

```{admonition} Observation

- White noise → no structure  
- AR(1) →  
  - ACF decays  
  - PACF cuts off after lag 1  
```

---

# 9.9 Model Identification (Rule of Thumb)

| Model | ACF | PACF |
|------|-----|------|
| AR(p) | decays | cuts off at p |
| MA(q) | cuts off at q | decays |
| ARMA | decays | decays |
| White noise | none | none |

```{admonition} Preview
These patterns guide ARIMA model selection.
```

---

# 9.10 Confidence Bands

ACF/PACF plots include approximate bounds:

```{math}
:enumerated: false
\pm \frac{2}{\sqrt{T}}
```

```{admonition} Interpretation
Spikes outside these bands may indicate significant autocorrelation.
```

```{admonition} Caution
Some spikes may exceed the bounds by chance.
```

---

# 9.11 Economic and Financial Insight

- Macroeconomic variables → persistent  
- Financial returns → weak autocorrelation  
- Volatility → highly persistent  

```{admonition} Important
Low autocorrelation does not imply absence of structure.
```

---

# 9.12 Looking Ahead

ACF and PACF help diagnose dependence patterns.

Next, we study:

- unit roots  
- nonstationarity  
- differencing  

which explain whether persistence is temporary or permanent.

---

# Key Takeaways

```{admonition} Summary

- ACF measures dependence across time  
- PACF isolates direct dependence  
- White noise shows no structure  
- Persistent series show slow decay  
- ACF and PACF are essential tools for model identification  
```

# Concept Check

### Basic

1. What is autocorrelation?

2. What does the autocorrelation function (ACF) measure?

3. What is a lag?

---

### Intuition

4. Why do we examine correlations across time in a time series?

5. What does a high autocorrelation at lag 1 suggest?

6. What does it mean if autocorrelations decay slowly?

---

### Intermediate

7. What is the difference between:

   - ACF  
   - PACF  

8. Why is PACF useful in time series analysis?

9. What does it mean if autocorrelations are close to zero at all lags?

---

### Interpretation

10. What pattern would you expect in the ACF of:

   - white noise  
   - a persistent series  

---

### Challenge

11. Suppose the ACF shows strong values at many lags.

   - What does this suggest about the series?
   - Why might this indicate nonstationarity?

---

# Interpretation & Practice

1. ACF values are close to zero at all lags.

   - What type of process is this likely?
   - Why?

2. ACF shows a large value at lag 1, then quickly drops to zero.

   - What does this suggest about dependence?
   - What type of process might generate this?

3. ACF decays slowly over many lags.

   - What does this indicate?
   - Why might this suggest nonstationarity?

4. PACF shows a sharp cutoff after lag 1.

   - What does this imply?
   - Why is this useful for model identification?

5. ACF alternates between positive and negative values.

   - What type of behavior might this indicate?
   - What does this suggest about the series dynamics?

---

### Finance Interpretation

6. A return series shows very low autocorrelation.

   - What does this imply about predictability?
   - How does this relate to market efficiency?

7. A volatility series shows strong autocorrelation.

   - What does this suggest?
   - Why is this important in finance?

---

### Challenge

8. Suppose both ACF and PACF show no clear pattern.

   - What type of process might this be?
   - Why is model identification difficult in this case?

---

# Numerical Practice

### Identifying Autocorrelation

1. Consider the following series:

   2, 4, 6, 8, 10  

- Does this series show dependence over time?
- Would you expect high autocorrelation?
- Why?

---

2. Consider:

   3, −1, 2, −2, 1, 0  

- Does this series appear random?
- Would you expect autocorrelation to be high or low?

---

### Lag Interpretation

3. Suppose:

```{math}
:enumerated: false
Corr(X_t, X_{t-1}) = 0.8
```

- What does this imply?
- Is the series highly dependent?

---

4. Suppose:

```{math}
:enumerated: false
Corr(X_t, X_{t-1}) \approx 0
```

- What does this suggest about predictability?

---

### ACF Patterns

5. Match the pattern:

- ACF ≈ 0 for all lags  
- ACF slowly decays  
- ACF cuts off quickly  

To the likely process:

- white noise  
- persistent series  
- short-memory process  

---

### Challenge

6. Suppose a series has:

- strong autocorrelation at lag 1  
- weak autocorrelation afterward  

---

- What type of model might describe this?
- Why?

---


---

# Appendix 9A — Understanding PACF

The PACF can be interpreted as the correlation between:

- $X_t$  
- and $X_{t-h}$

after removing the effect of intermediate lags.

This can be obtained by:

1. Regressing $X_t$ on intermediate lags  
2. Regressing $X_{t-h}$ on the same variables  
3. Taking the correlation between residuals  

```{admonition} Insight
PACF isolates direct dependence at each lag.
```