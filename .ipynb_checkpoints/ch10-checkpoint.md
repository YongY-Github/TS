---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 10 — Unit Roots and Differencing

In previous chapters, we studied:

- persistence
- stationarity
- autocorrelation

A central question now arises:

```{admonition} Central Question
Why do some time series drift indefinitely instead of fluctuating around a stable level?
```

The answer lies in the concept of a **unit root**.

```{admonition} Central Idea
Some time series contain a unit root, meaning that shocks have permanent effects.
```

This has major consequences:

- standard regression methods may fail  
- forecasting becomes difficult  
- statistical inference can be misleading  
- transformation (differencing) becomes necessary  

This chapter introduces:

- unit root intuition  
- random walks revisited  
- differencing  
- detrending vs differencing  
- the Augmented Dickey–Fuller (ADF) test  

```{admonition} Deep Insight
Many economic and financial series look predictable because they are persistent.

But persistence is not the same as predictability.

Unit root processes can exhibit strong patterns while remaining fundamentally unpredictable.
```

---

# Learning Objectives

By the end of this chapter, you should be able to:

- understand what a unit root is
- explain why random walks are nonstationary
- distinguish stationary and unit root processes
- difference a time series
- distinguish detrending from differencing
- interpret the ADF test intuitively

---

# 10.1 Persistence Revisited

Recall the random walk:

```{math}
:enumerated: false
x_t = x_{t-1} + w_t
```

where:

```{math}
:enumerated: false
w_t \sim wn(0,\sigma_w^2)
```

Each new observation equals:

- the previous value
- plus a random shock

```{admonition} Key Insight
In a random walk, shocks accumulate permanently over time.
```

---

# 10.2 Simulating a Random Walk

````{dropdown} Python Code
```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

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
````

![Random Walk](figs/ch7/rw.png)

```{admonition} Observation
The series drifts over time and does not fluctuate around a stable mean.
```

---

# 10.3 Why Random Walks Matter

Random walks are central in economics and finance.

Examples include:

- stock prices
- exchange rates
- some macroeconomic aggregates

The random walk model implies:

- strong persistence
- unpredictable long-run movement
- permanent effects of shocks

---

# 10.4 The Unit Root Idea

The random walk can be rewritten as:

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t
```

with:

```{math}
:enumerated: false
\phi = 1
```

```{admonition} Definition
A process has a **unit root** when the coefficient on the lagged variable equals one.
```

```{admonition} Big Picture

Unit root processes behave fundamentally differently from stationary processes:

- shocks do **not fade away**
- the series does **not return to a stable level**
- variability grows over time

```text
Nonstationary → Difference → Stationary → Model
```

---
# 10.5 Why the Unit Root Matters

Consider:

$$
x_t = \phi x_{t-1} + w_t
$$

## Case 1: $|\phi| < 1$

- shocks gradually disappear  
- the series returns toward a stable mean  
- the process is stationary  

## Case 2: $\phi = 1$

- shocks never disappear  
- the series accumulates past shocks  
- the process becomes nonstationary  

```{admonition} Key Insight
The closer $\phi$ is to 1, the more persistent the series becomes.
```

---

# 10.6 Stationary vs Unit Root Processes

### Stationary Process

- stable mean
- stable variance
- temporary shocks

### Unit Root Process

- drifting behavior
- growing variance
- permanent shocks

```{admonition} Important
Unit root processes exhibit a fundamentally different type of persistence from stationary processes.
```

---

# 10.7 Simulating Different Levels of Persistence

```{code-cell} python
:tags: [hide-input]
np.random.seed(123)

n = 500
w = np.random.normal(0,1,n)

x1 = np.zeros(n)
x2 = np.zeros(n)
x3 = np.zeros(n)

phi1 = 0.4
phi2 = 0.9
phi3 = 1.0

for t in range(1,n):
    x1[t] = phi1*x1[t-1] + w[t]
    x2[t] = phi2*x2[t-1] + w[t]
    x3[t] = phi3*x3[t-1] + w[t]

fig, ax = plt.subplots(3,1, figsize=(10,8))

ax[0].plot(x1)
ax[0].set_title(r"$\phi = 0.4$")

ax[1].plot(x2)
ax[1].set_title(r"$\phi = 0.9$")

ax[2].plot(x3)
ax[2].set_title(r"$\phi = 1.0$ (Unit Root)")

plt.tight_layout()

plt.savefig("figs/ch10/ar-sim.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Persistence](figs/ch10/ar-sim.png)


```{admonition} Observation
As $\phi$ approaches 1, persistence becomes increasingly strong.
```

---

# 10.8 Differencing

A common way to remove unit roots is differencing.

```{admonition} Definition
The first difference of a series is:

$$
\Delta x_t = x_t - x_{t-1}
$$
```

---

# 10.9 Differencing a Random Walk

For a random walk:

```{math}
:enumerated: false
x_t = x_{t-1} + w_t
```

Taking first differences:

```{math}
:enumerated: false
\Delta x_t = x_t - x_{t-1} = w_t
```

```{admonition} Key Insight
Differencing removes the accumulated effect of past shocks.

A nonstationary process becomes stationary after differencing.
```

```{admonition} Intuition
Differencing converts a drifting series into one that fluctuates around a stable mean.
```

---

# 10.10 Simulating Differencing

```{code-cell} python
dx = np.diff(x)

plt.figure(figsize=(10,4))
plt.plot(dx, lw=1)
plt.title("First Difference of Random Walk")
plt.xlabel("Time")
plt.ylabel(r"$\Delta x_t$")

plt.savefig("figs/ch10/rw-diff.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![RW Differencing](figs/ch10/rw-diff.png)

```{admonition} Observation
The differenced series fluctuates around a stable mean and appears stationary.
```

---

# 10.11 Integrated Processes

A unit root process is said to be integrated of order one.

```{admonition} Definition
A series is:

- $I(0)$ if stationary
- $I(1)$ if first differences are stationary
```

## Example

| Series | Order |
|---|---|
| White noise | $I(0)$ |
| Random walk | $I(1)$ |

---

# 10.12 Detrending vs Differencing

Nonstationarity may arise for different reasons.

## Deterministic Trend

Suppose:

$$
x_t = \alpha + \beta t + u_t
$$

where $u_t$ is stationary.

Removing the trend may produce stationarity.

## Stochastic Trend

For a random walk:

$$
x_t = x_{t-1} + w_t
$$

detrending alone is insufficient.

Differencing is required.

```{admonition} Important
Trend-stationary and difference-stationary processes require different treatments.
```

---

# 10.13 Visual Comparison

```{code-cell} python
:tags: [hide-input]

np.random.seed(123)

t = np.arange(500)

trend_stationary = 0.03*t + np.random.normal(0,1,500)

plt.figure(figsize=(10,4))
plt.plot(trend_stationary)
plt.title("Trend-Stationary Series")
plt.xlabel("Time")

plt.savefig("figs/ch10/trend.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Deterministic Trend](figs/ch10/trend.png)


```{admonition} Observation
Trend-stationary series fluctuate around a deterministic trend rather than drifting permanently.
```

---

# 10.14 Why Unit Roots Matter

Unit roots affect:

- forecasting
- inference
- regression
- model selection

```{admonition} Key Problem
Regressions involving nonstationary variables may produce misleading statistical relationships.
```

This problem becomes central later in:

- spurious regression
- cointegration
- ECM models

---

# 10.15 The Dickey–Fuller Test

We now need a way to test for unit roots.

Consider:

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t
```

Subtract $x_{t-1}$ from both sides:

```{math}
:enumerated: false
x_t - x_{t-1}
=
(\phi - 1)x_{t-1} + w_t
```

or:

```{math}
:enumerated: false
\Delta x_t
=
\theta x_{t-1} + w_t
```

where:

```{math}
:enumerated: false
\theta = \phi - 1
```

## Hypotheses

```{math}
:enumerated: false
H_0: \theta = 0
\quad \text{(unit root)}
```

```{math}
:enumerated: false
H_1: \theta < 0
\quad \text{(stationary)}
```

```{admonition} Intuition
The Dickey–Fuller test checks whether the series exhibits mean reversion.
```

---

# 10.16 Augmented Dickey–Fuller (ADF) Test

Real-world data often exhibit serial correlation.

```{admonition} Intuition (ADF Test)
The ADF test asks:

**Does the series tend to return to a stable level?**

- If YES → stationary  
- If NO → unit root  

In other words:

Is there *mean reversion*, or does the series drift indefinitely?
```

The Augmented Dickey–Fuller (ADF) test extends the Dickey–Fuller test by including lagged differences:

$$
\Delta x_t
=
\alpha
+
\theta x_{t-1}
+
\sum_{i=1}^p \gamma_i \Delta x_{t-i}
+
u_t
$$

```{admonition} Important
The additional lagged differences help remove serial correlation from the residuals.
```

---

# 10.17 Interpreting the ADF Test

## Reject $H_0$

- evidence against a unit root  
- series likely stationary  

## Fail to Reject $H_0$

- insufficient evidence against a unit root  
- series may be nonstationary  

```{admonition} Important
Failing to reject a unit root does not prove the series is exactly a random walk.

It simply means we do not have strong evidence of stationarity.
```

```{admonition} Practical Tip
Always check BOTH:

- the test statistic vs critical values  
- the p-value  

Small differences can matter in borderline cases.
```
---

# 10.18 ADF Testing in Gretl

## Menu

```text
Variable → Unit root tests → Augmented Dickey-Fuller
```

## Typical Steps

1. Choose variable
2. Include constant and/or trend if appropriate
3. Select lag length
4. Interpret test statistic and p-value

```markdown
[GRETL Screenshot Placeholder: ADF test dialog]
```

```markdown
[GRETL Screenshot Placeholder: ADF test output]
```

---

# 10.19 KPSS Test (Optional)

The KPSS test reverses the hypotheses.

## KPSS Hypotheses

```{math}
:enumerated: false
H_0: \text{stationary}
```

```{math}
:enumerated: false
H_1: \text{unit root}
```

```{admonition} Practical Advice
Many analysts use both:

- ADF test
- KPSS test

to obtain complementary evidence.
```

---

# 10.20 Looking Ahead

In this chapter, we introduced:

- unit roots
- differencing
- ADF testing

We are now ready to build formal stochastic models for stationary time series.

In the next chapters, we study:

- autoregressive (AR) models
- moving average (MA) models
- ARMA and ARIMA models

## Key Takeaways

```{admonition} Summary
- Unit roots imply permanent effects of shocks
- Random walks are classic unit root processes
- Differencing often removes nonstationarity
- Trend-stationary and difference-stationary processes differ fundamentally
- The ADF test helps detect unit roots
```

# Concept Check

### Basic

1. What is a unit root?

2. What does it mean for a time series to be nonstationary?

3. What is a random walk?

---

### Intuition

4. Why do shocks have permanent effects in a unit root process?

5. Why does a random walk not return to a stable level?

6. Why is nonstationarity problematic for modeling?

---

### Intermediate

7. What is the purpose of differencing a time series?

8. What is the difference between:

   - a stationary process  
   - a unit root process  

9. What is the difference between:

   - deterministic trend  
   - stochastic trend  

---

### Finance Insight

10. Why are stock prices often modeled as random walks?

11. Why are returns typically stationary?

---

### Challenge

12. Suppose a series becomes stationary after differencing once.

   - What does this imply about the original series?

---

# Interpretation & Practice

1. A time series shows a strong upward trend and does not return to a stable level.

   - What does this suggest?
   - What might be an appropriate transformation?

---

2. A series appears to “wander” over time.

   - What type of process might this be?
   - What does this imply about shocks?

---

3. After differencing, a series fluctuates around zero.

   - What does this suggest?
   - Why is this useful?

---

4. A regression between two trending series shows a strong relationship.

   - Why might this be misleading?
   - What concept does this illustrate?

---

5. A series has a deterministic upward trend.

   - Would differencing or detrending be more appropriate?
   - Why?

---

### Finance Interpretation

6. A stock price series is nonstationary.

   - Why is modeling prices directly problematic?
   - Why are returns preferred?

---

7. A return series appears stable over time.

   - What does this suggest?
   - Why is this important?

---

### Challenge

8. Suppose you difference a stationary series.

   - What might happen?
   - Why is over-differencing a problem?

---

# Numerical Practice

### Random Walk and Differencing

1. Suppose a random walk is defined as:

```{math}
:enumerated: false
x_t = x_{t-1} + w_t
```

with:

- $x_0 = 100$
- shocks: 2, −1, 3, −2  

---

- Compute $x_1, x_2, x_3, x_4$

---

2. Compute the first differences:

```{math}
:enumerated: false
\Delta x_t = x_t - x_{t-1}
```

---

- What do you observe?
- What does this suggest?

---

### Identifying Nonstationarity

3. Consider the series:

   10, 12, 15, 19, 24  

---

- Does this appear stationary?
- Compute the first differences  
- Does the differenced series look more stable?

---

### Trend vs Difference

4. Suppose:

```{math}
:enumerated: false
x_t = 5 + 0.5t + u_t
```

---

- What type of trend is this?
- Would differencing remove it?
- What would the differenced series look like?

---

### ADF Interpretation (Applied)

Consider the following output from an Augmented Dickey–Fuller (ADF) test:

```text
Augmented Dickey-Fuller Test

Test Statistic:   -1.85
p-value:           0.67
Lags Used:         2
Observations:      197

Critical Values:
  1% level:       -3.46
  5% level:       -2.88
 10% level:       -2.57
```

1. What is the null hypothesis of the ADF test?

2. Compare the test statistic with the critical values.

   - Is the test statistic more negative than the 5% critical value?

3. Based on the p-value and test statistic:

   - Do you reject the null hypothesis?
   - What does this imply about the series?

4. What would you do next before modeling this series?

---

### Second Example

Now consider:

```text
Augmented Dickey-Fuller Test

Test Statistic:   -3.25
p-value:           0.02
Lags Used:         1
Observations:      198

Critical Values:
  1% level:       -3.46
  5% level:       -2.88
 10% level:       -2.57
```

5. Do you reject the null hypothesis at the 5% level?

6. What does this imply about stationarity?

7. Why might the test statistic be compared with critical values rather than relying only on the p-value?

---

### Challenge

8. Suppose the test statistic is close to the critical value.

   - Why might conclusions be uncertain?
   - What additional checks could you perform?

---

9. Suppose you difference a series twice.

- When might this be necessary?
- What is the risk of doing this unnecessarily?

---

10. Suppose two nonstationary series are regressed on each other.

- Why might the results be misleading?
- What concept does this relate to?