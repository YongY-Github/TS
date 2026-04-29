---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 9 — ACF and PACF

In the previous chapters, we introduced dependence, persistence, and stationarity.

We now develop two of the most important tools in time series analysis:

- the **autocorrelation function (ACF)**
- the **partial autocorrelation function (PACF)**

These tools help us understand:

- dependence across time
- persistence
- lag structure
- model identification

ACF and PACF plots are fundamental in practical time series analysis and are widely used in ARIMA modeling.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- understand autocovariance and autocorrelation
- interpret ACF plots
- understand partial autocorrelation
- interpret PACF plots
- recognize common dependence patterns
- distinguish white noise, AR, and MA behavior visually

---

## 9.1 Correlation Across Time

In ordinary statistics, correlation measures dependence between two variables.

In time series analysis, we instead study dependence between:

- a series today
- the same series in the past

For example:

```{math}
:enumerated: false
Corr(X_t, X_{t-1})
```

```{math}
:enumerated: false
Corr(X_t, X_{t-2})
```

and more generally:

```{math}
:enumerated: false
Corr(X_t, X_{t-h})
```

where $h$ is called the **lag**.

```{admonition} Definition
The dependence between a time series and its lagged values is called **serial correlation** or **autocorrelation**.
```

---

## 9.2 Autocovariance Function

For a weakly stationary process:

```{math}
\gamma(h)
=
Cov(X_t, X_{t-h})
```

is called the **autocovariance function**.

```{admonition} Important
Under stationarity, autocovariance depends only on the lag $h$, not on time $t$.
```

### Lag 0

At lag zero:

```{math}
:enumerated: false
\gamma(0)
=
Var(X_t)
```

### Positive and Negative Covariance

- Positive autocovariance → observations move together
- Negative autocovariance → observations move oppositely

---

## 9.3 Autocorrelation Function (ACF)

The autocorrelation function standardizes autocovariance.

```{admonition} Definition
The autocorrelation at lag $h$ is:

$$
\rho(h)
=
\frac{\gamma(h)}{\gamma(0)}
$$
```

Thus:

```{math}
:enumerated: false
-1 \leq \rho(h) \leq 1
```

```{admonition} Intuition
The ACF measures how strongly the series is related to its own past values.
```

---

## 9.4 ACF of White Noise

For white noise:

$$
\rho(h) = 0
\quad \text{for } h \neq 0
$$

because white noise contains no serial dependence.

### Simulating White Noise

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10101)

wn = np.random.normal(0, 2, 500)

plt.figure(figsize=(10,4))
plt.plot(wn, lw=1)
plt.title("Simulated White Noise")
plt.xlabel("Time")
plt.ylabel("$w_t$")

plt.savefig("figs/ch7/wn.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![White Noise](figs/ch7/wn.png)

---

## 9.5 ACF Plot of White Noise

```{code-cell} python
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(wn, lags=30)

plt.savefig("figs/ch9/acf_wn.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![ACF White Noise](figs/ch9/acf_wn.png)

```{admonition} Observation
The ACF of white noise fluctuates randomly around zero with no systematic pattern.
```

## 9.6 ACF of a Random Walk

Now consider a persistent process such as a random walk.

### Simulating a Random Walk

```{code-cell} python
:tags: [hide-input]

np.random.seed(123)

w = np.random.normal(0, 1, 500)
x = np.cumsum(w)

plt.figure(figsize=(10,4))
plt.plot(x, lw=1)
plt.title("Simulated Random Walk")
plt.xlabel("Time")
plt.ylabel("$x_t$")

plt.savefig("figs/ch7/rw.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Random Walk](figs/ch7/rw.png)

---

## 9.7 ACF of a Random Walk

```{code-cell} python
plot_acf(x, lags=40)
plt.savefig("figs/ch9/acf_rw.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![ACF Random Walk](figs/ch9/acf_rw.png)


```{admonition} Observation
The autocorrelations decay very slowly, indicating strong persistence.
```

```{admonition} Key Insight
Persistent series tend to exhibit slowly decaying autocorrelations.
```

---

## 9.8 Why Persistence Matters

Strong persistence means:

- shocks remain influential for many periods
- future values depend strongly on past values

This is common in:

- macroeconomic variables
- financial prices
- exchange rates

---

## 9.9 Sample ACF

In practice, we estimate autocorrelations from data.

The sample autocorrelation at lag $h$ is:

$$
\hat{\rho}(h)
=
\frac{
\sum_{t=h+1}^T (x_t-\bar{x})(x_{t-h}-\bar{x})
}{
\sum_{t=1}^T (x_t-\bar{x})^2
}
$$

```{admonition} Important
Sample autocorrelations are estimates and therefore subject to sampling variability.
```

---

## 9.10 Interpreting ACF Plots

ACF plots help identify patterns in time series data.

### White Noise

- autocorrelations near zero
- no visible pattern

### Persistent Series

- slowly decaying ACF
- strong dependence

### Oscillatory Behavior

- alternating positive and negative correlations
- cyclical structure

---

## 9.11 Partial Autocorrelation

The ACF measures total correlation across time.

However, some of this correlation may be indirect.

### Example

Suppose:

- $X_t$ depends strongly on $X_{t-1}$
- $X_{t-1}$ depends strongly on $X_{t-2}$

Then:

```{math}
:enumerated: false
Corr(X_t,X_{t-2})
```

may arise indirectly through $X_{t-1}$.

```{admonition} Intuition
The PACF measures the “direct” relationship between $X_t$ and $X_{t-h}$ after controlling for intermediate lags.
```

---

## 9.12 Partial Autocorrelation Function (PACF)

````{admonition} Definition
The partial autocorrelation at lag $h$ measures the correlation between:

```{math}
:enumerated: false
X_t \quad \text{and} \quad X_{t-h}
```

after removing the effects of intermediate lags.
````

---

## 9.13 PACF of White Noise

For white noise:

- PACF values fluctuate randomly around zero
- no systematic structure exists


## 9.14 PACF Plot of White Noise

```{code-cell} python
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(wn, lags=30, method='ywm')

plt.savefig("figs/ch9/pacf_wn.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![PACF](figs/ch9/pacf_wn.png)

---

## 9.15 PACF of a Random Walk

```{code-cell} python
plot_pacf(x, lags=30, method='ywm')

plt.savefig("figs/ch9/pacf_rw.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![PACF Random Walk](figs/ch9/pacf_rw.png)


```{admonition} Observation
Persistent series often show large PACF values at small lags.
```

---

## 9.16 Why ACF and PACF Matter

ACF and PACF are essential because they help identify appropriate time series models.

In later chapters, we will see:

| Model | ACF | PACF |
|---|---|---|
| AR(p) | gradual decay | cutoff |
| MA(q) | cutoff | gradual decay |
| White noise | near zero | near zero |

```{admonition} Preview
ACF and PACF plots are fundamental tools in ARIMA model identification.
```

---

## 9.17 Confidence Bands

ACF and PACF plots usually include confidence bands.

Approximate bounds are:

$$
\pm \frac{2}{\sqrt{T}}
$$

where $T$ is sample size.

```{admonition} Interpretation
Spikes outside the confidence bands may indicate statistically significant autocorrelation.
```

```{admonition} Caution
Individual spikes may occasionally exceed the bounds purely by chance.
```

---

## 9.18 Economic and Financial Intuition

### Macroeconomic Variables

Variables such as:

- inflation
- unemployment
- GDP

often exhibit strong persistence.

### Financial Returns

Daily stock returns often show:

- weak autocorrelation in returns
- stronger dependence in volatility

```{admonition} Important
Lack of autocorrelation in returns does not imply absence of structure in financial markets.
```

This becomes important later in ARCH and GARCH models.

---

## 9.19 Looking Ahead

In this chapter, we introduced the ACF and PACF as tools for understanding dependence across time.

In the next chapter, we study:

- unit roots
- nonstationarity
- differencing
- the Augmented Dickey–Fuller (ADF) test

which help determine whether persistence is temporary or permanent.

## Key Takeaways

```{admonition} Summary
- The ACF measures dependence across lags
- The PACF measures direct lag relationships
- White noise exhibits little autocorrelation
- Persistent series show slowly decaying autocorrelations
- ACF and PACF are central tools for model identification
```