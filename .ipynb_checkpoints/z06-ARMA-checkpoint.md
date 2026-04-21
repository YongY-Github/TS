---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# ARMA

## Moving Average Processes

We begin with one of the simplest classes of stochastic processes: the **moving average (MA)** model.  Note that despite its name, this is **not a smoothing method**, but a probabilistic model describing how a time series is generated.

### Definition

A **moving average process of order $q$**, denoted MA($q$), is defined as

$$
x_t = \theta_1 w_{t-1} + \theta_2 w_{t-2} + \cdots + \theta_q w_{t-q} + \mu + w_t,
$$

where $\{w_t\}$ is a white noise process with mean $0$ and variance $\sigma_w^2$,  
$\theta_1, \dots, \theta_q$ are constants, and $\mu$ is the mean (a constant).

```{admonition} Intuition
An MA($q$) process is driven entirely by **current and past shocks**.

A shock affects the system for at most $q$ periods, after which its effect disappears completely.  
For this reason, MA processes are often described as having **finite memory**.
````

---

### The MA(1) Model

To build intuition, consider the MA(1) model:

```{math}
:enumerated: false
x_t = \theta w_{t-1} + \mu + w_t.
```

### Mean

Taking expectations:

```{math}
:enumerated: false
E[x_t] = \theta E[w_{t-1}] + E[\mu] + E[w_t] = \mu.
```

Thus, the mean is constant over time.

### Autocovariance Function

To understand the dependence structure, we compute the autocovariance:

$$
\gamma(h) = \text{Cov}(x_t, x_{t-h}).
$$

#### Case $h = 0$

```{math}
:enumerated: false
\gamma(0) = \text{Var}(x_t)
= \text{Var}(\theta w_{t-1} + w_t).
```

Since $w_t$ is white noise:

* $\text{Var}(w_t) = \sigma_w^2$
* $\text{Var}(w_{t-1}) = \sigma_w^2$
* $\text{Cov}(w_t, w_{t-1}) = 0$

we obtain:

```{math}
:enumerated: false
\gamma(0) = (1 + \theta^2)\sigma_w^2.
```

#### Case $h = 1$

```{math}
:enumerated: false
\gamma(1) = \text{Cov}(x_t, x_{t-1})
= \text{Cov}(\theta w_{t-1} + w_t,; \theta w_{t-2} + w_{t-1}).
```

Expanding and using independence:

* $\text{Cov}(w_t, w_{t-1}) = 0$
* $\text{Cov}(w_t, w_{t-2}) = 0$
* $\text{Cov}(w_{t-1}, w_{t-2}) = 0$
* $\text{Cov}(w_{t-1}, w_{t-1}) = \sigma_w^2$

we get:

```{math}
:enumerated: false
\gamma(1) = \theta \sigma_w^2.
```

#### Case $h > 1$

There are no overlapping terms, so:

```{math}
:enumerated: false
\gamma(h) = 0 \quad \text{for } h > 1.
```

---

::: {admonition} Key Result
For an MA(1) process:

```{math}
:enumerated: false
\gamma(h) =
\begin{cases}
(1 + \theta^2)\sigma_w^2 & h = 0 \\
\theta \sigma_w^2 & h = 1 \\
0 & h > 1
\end{cases}
```
:::


### Autocorrelation Function (ACF)

The autocorrelation is:

$$
\rho(h) = \frac{\gamma(h)}{\gamma(0)}.
$$

Thus:

```{math}
:enumerated: false
\rho(h) =
\begin{cases}
1 & h = 0 \\
\frac{\theta}{1 + \theta^2} & h = 1 \\
0 & h > 1
\end{cases}
```

::: {admonition} Key Idea
For an MA($q$) process, the autocorrelation function **cuts off after lag $q$**:

```{math}
:enumerated: false
\rho(h) = 0 \quad \text{for all } h > q.
```

This sharp cutoff is a key diagnostic feature used in model identification.
:::

---

### Stationarity

The MA(1) process is **weakly stationary** because:

* $E[x_t] = \mu$ (constant mean)
* $\gamma(h)$ depends only on the lag $h$, not on time $t$

In fact:

```{admonition} Important
All MA($q$) processes are automatically stationary — no additional conditions are required.
```

---

### Example

Suppose $w_t \sim N(0,1)$ and $\theta = 5$. Then:

```{math}
:enumerated: false
\gamma(0) = 26, \quad \gamma(1) = 5, \quad \gamma(h) = 0 \text{ for } h > 1.
```

Interestingly, $\theta = 5$ and $\theta = 1/5$ produce the same autocorrelation structure.
We typically prefer the smaller value — a concept related to **invertibility**, which we look at later.

---

### Summary

* MA($q$) models depend on current and past shocks
* They have **finite memory**
* The ACF **cuts off after lag $q$**
* They are always **stationary**

---

```{admonition} Common Pitfall
Do not confuse:

- Moving average **smoothing** (a data processing method), and  
- MA($q$) **stochastic models**

They are fundamentally different concepts.
```
---

## Autoregressive Processes

We now turn to a second major class of linear time series models: **autoregressive processes**.

Unlike moving average models, which express the current observation as a function of current and past **shocks**, autoregressive models express the current observation as a function of its own **past values**. This makes AR models especially useful for capturing persistence over time.

In many economic and financial series, the present tends to resemble the recent past. Inflation this month is often related to inflation last month. Output this quarter is often related to output last quarter. AR models formalize this idea.

---

### The Backshift Operator

Before studying autoregressive models in detail, it is convenient to introduce a compact notation.

The **backshift operator** (or **lag operator**) $B$ is defined by

```{math}
:enumerated: false
B x_t = x_{t-1}.
```

Applying the operator twice gives

```{math}
:enumerated: false
B^2 x_t = x_{t-2},
```

and more generally,

```{math}
:enumerated: false
B^k x_t = x_{t-k}.
```

This notation is useful because it allows us to write time series models more compactly and manipulate them algebraically.

For example, the first difference of a series can be written as

```{math}
:enumerated: false
\Delta x_t = x_t - x_{t-1} = (1-B)x_t.
```

Similarly, the second difference is

```{math}
:enumerated: false
\Delta^2 x_t = \Delta(\Delta x_t) = (1-B)^2 x_t.
````

```{admonition} Why the backshift operator is useful
The backshift operator turns time series equations into polynomial expressions in $B$.

This makes it easier to:
- write AR and MA models compactly,
- derive infinite-series representations,
- study stationarity conditions through roots of polynomials,
- and later derive the Yule--Walker equations more cleanly.
````

---

### Definition of an AR($p$) Process

An **autoregressive process of order $p$**, denoted AR($p$), is defined by

$$
x_t = \phi_1 x_{t-1} + \phi_2 x_{t-2} + \cdots + \phi_p x_{t-p} + \mu + w_t,
$$

where:

* ${w_t}$ is white noise with mean $0$ and variance $\sigma_w^2$,
* $\phi_1, \phi_2, \dots, \phi_p$ are constants,
* and $\mu$ is a constant term.

Using the backshift operator, this can be written as

```{math}
:enumerated: false
x_t = \phi_1 Bx_t + \phi_2 B^2 x_t + \cdots + \phi_p B^p x_t + w_t + \mu,
```

or equivalently,

```{math}
:enumerated: false
(1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p)x_t = w_t + \mu.
```

The polynomial

```{math}
:enumerated: false
\phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p
```

is called the **autoregressive polynomial**.

```{admonition} Intuition
An AR($p$) model says that the current value of the series depends on its own last $p$ values, plus a new random shock.

So while an MA model has finite memory in shocks, an AR model typically has a more persistent internal dynamics.
```

## The AR(1) Model

We now study the simplest autoregressive model in full detail.

An **AR(1)** process is

$$
x_t = \phi x_{t-1} + \mu + w_t.
$$

This model says that today's value depends on:

* a fraction $\phi$ of yesterday's value,
* a new shock $w_t$,
* and a constant term $\mu$.

Because AR(1) is simple but already rich enough to show persistence, stability, and mean reversion, it is one of the most important models in time series analysis.

### Mean of AR(1)

Assume the process is stationary, so that the mean does not depend on $t$. Let

```{math}
:enumerated: false
E[x_t] = m.
```

Taking expectations on both sides of

```{math}
:enumerated: false
x_t = \phi x_{t-1} + \mu + w_t,
```

we get

```{math}
:enumerated: false
E[x_t] = \phi E[x_{t-1}] + E[\mu] + E[w_t].
```

Since $E[w_t]=0$ and $E[x_t]=E[x_{t-1}]=m$ under stationarity,

```{math}
:enumerated: false
m = \phi m + \mu.
```

Rearranging,

```{math}
:enumerated: false
m(1-\phi) = \mu,
```

so the mean is

```{math}
:enumerated: false
m = \frac{\mu}{1-\phi},
```

provided $\phi \neq 1$.

```{admonition} Result
For a stationary AR(1) process,

$$
E[x_t] = \frac{\mu}{1-\phi}.
$$
```

If $\mu=0$, then the mean is zero.

### Mean-Centered Form

It is often simpler to work with the demeaned process. Let

```{math}
:enumerated: false
y_t = x_t - m,
```

where $m = \mu/(1-\phi)$.

Then the AR(1) model becomes

```{math}
:enumerated: false
y_t = \phi y_{t-1} + w_t.
```

So, without loss of generality, we can often study the AR(1) model in the simpler form

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t,
```

with mean zero.

From now on, we will usually work with this mean-zero version when deriving variance and autocorrelation.

---

### Recursive Representation of AR(1)

Starting from

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t,
```

substitute recursively:

```{math}
:enumerated: false
x_{t-1} = \phi x_{t-2} + w_{t-1}.
```

Then

```{math}
:enumerated: false
x_t = \phi(\phi x_{t-2} + w_{t-1}) + w_t
= \phi^2 x_{t-2} + \phi w_{t-1} + w_t.
```

Substituting again,

```{math}
:enumerated: false
x_t = \phi^3 x_{t-3} + \phi^2 w_{t-2} + \phi w_{t-1} + w_t.
```

Continuing in this way, after $k$ substitutions we obtain

```{math}
:enumerated: false
x_t = \phi^k x_{t-k} + \sum_{j=0}^{k-1} \phi^j w_{t-j}.
```

Now, if $|\phi|<1$, then as $k \to \infty$,

```{math}
:enumerated: false
\phi^k x_{t-k} \to 0,
```

and hence

```{math}
:enumerated: false
x_t = \sum_{j=0}^{\infty} \phi^j w_{t-j}.
```

That is,

```{math}
:enumerated: false
x_t = w_t + \phi w_{t-1} + \phi^2 w_{t-2} + \phi^3 w_{t-3} + \cdots
```

```{admonition} Key Idea
A stationary AR(1) process can be written as an infinite moving average:

$$
x_t = \sum_{j=0}^{\infty} \phi^j w_{t-j}.
$$

This shows that AR(1) has an **infinite memory of past shocks**, but the weights decline geometrically when $|\phi|<1$.
```

---

### Stationarity Condition for AR(1)

The infinite-series representation above only makes sense when the geometric coefficients decay. This leads to the fundamental condition:

::: {admonition} Stationarity Condition
An AR(1) process is stationary if

```{math}
:enumerated: false
|\phi| < 1.
```
:::

Why?

If $|\phi|<1$, then:

* the effect of past shocks gets smaller over time,
* the infinite sum converges,
* the mean and variance remain constant over time.

If $|\phi| \geq 1$, the process is not stable in the stationary sense:

* when $\phi=1$, we get a random walk,
* when $|\phi|>1$, shocks grow explosively.

```{admonition} Intuition
The parameter $\phi$ measures persistence.

- If $\phi$ is close to $0$, shocks die out quickly.
- If $\phi$ is close to $1$, shocks persist for a long time.
- If $\phi$ is negative, the series tends to alternate around its mean.
```

---

### Variance of AR(1)

Now derive the variance under the mean-zero stationary AR(1):

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t.
```

Taking variances,

```{math}
:enumerated: false
\operatorname{Var}(x_t)
= \operatorname{Var}(\phi x_{t-1} + w_t).
```

Expand:

```{math}
:enumerated: false
\operatorname{Var}(x_t)
= \phi^2 \operatorname{Var}(x_{t-1}) + \operatorname{Var}(w_t)
+ 2\phi \operatorname{Cov}(x_{t-1}, w_t).
```

Since $w_t$ is white noise and is uncorrelated with past values such as $x_{t-1}$,

```{math}
:enumerated: false
\operatorname{Cov}(x_{t-1}, w_t)=0.
```

Under stationarity, $\operatorname{Var}(x_t)=\operatorname{Var}(x_{t-1})=\gamma(0)$. Hence

```{math}
:enumerated: false
\gamma(0) = \phi^2 \gamma(0) + \sigma_w^2.
```

So,

```{math}
:enumerated: false
\gamma(0)(1-\phi^2)=\sigma_w^2,
```

and therefore

```{math}
:enumerated: false
\gamma(0)=\frac{\sigma_w^2}{1-\phi^2}.
```

Since $\gamma(0)=\operatorname{Var}(x_t)$, we have:

::: {admonition} Result
For a stationary AR(1) process,

```{math}
:enumerated: false
\operatorname{Var}(x_t)=\frac{\sigma_w^2}{1-\phi^2}.
```
:::

This formula only makes sense when $|\phi|<1$, again reflecting the stationarity condition.

---

### Autocovariance Function of AR(1)

We now derive the autocovariance function

```{math}
:enumerated: false
\gamma(h)=\operatorname{Cov}(x_t, x_{t-h}).
```

Start from

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t.
```

Multiply both sides by $x_{t-h}$ and take expectations:

```{math}
:enumerated: false
E[x_t x_{t-h}] = \phi E[x_{t-1}x_{t-h}] + E[w_t x_{t-h}].
```

Since the process has mean zero, this is

```{math}
:enumerated: false
\gamma(h)=\phi \gamma(h-1) + E[w_t x_{t-h}].
```

Now if $h \geq 1$, then $x_{t-h}$ depends only on shocks dated $t-h, t-h-1, \dots$, all of which occur before $w_t$. Since white noise is uncorrelated across time,

```{math}
:enumerated: false
E[w_t x_{t-h}] = 0 \qquad \text{for } h \geq 1.
```

Hence,

```{math}
:enumerated: false
\gamma(h)=\phi \gamma(h-1), \qquad h \geq 1.
```

Apply this repeatedly:

```{math}
:enumerated: false
\gamma(1)=\phi \gamma(0),
```

```{math}
:enumerated: false
\gamma(2)=\phi \gamma(1)=\phi^2 \gamma(0),
```

and in general,

```{math}
:enumerated: false
\gamma(h)=\phi^h \gamma(0), \qquad h=0,1,2,\dots
```

Substituting the variance formula gives

```{math}
:enumerated: false
\gamma(h)=\phi^h \frac{\sigma_w^2}{1-\phi^2}.
```

Because autocovariance is symmetric,

```{math}
:enumerated: false
\gamma(-h)=\gamma(h),
```

so more generally we may write

```{math}
:enumerated: false
\gamma(h)=\phi^{|h|}\frac{\sigma_w^2}{1-\phi^2}.
```

::: {admonition} Result
For a stationary AR(1) process,

```{math}
:enumerated: false
\gamma(h)=\phi^{|h|}\frac{\sigma_w^2}{1-\phi^2}.
```

:::

---

### Autocorrelation Function of AR(1)

The autocorrelation function is

$$
\rho(h)=\frac{\gamma(h)}{\gamma(0)}.
$$

Since $\gamma(h)=\phi^{|h|}\gamma(0)$, it follows immediately that

```{math}
:enumerated: false
\rho(h)=\phi^{|h|}.
```

So the ACF of an AR(1) process decays geometrically.

```{admonition} Key Result
For a stationary AR(1) process,

$$
\rho(h)=\phi^{|h|}.
$$
```

This is one of the most important results in elementary time series analysis.

---

### How to Interpret the ACF

The formula

```{math}
:enumerated: false
\rho(h)=\phi^{|h|}
```

tells us exactly how dependence fades over time.

* If $0<\phi<1$, the autocorrelation decays gradually and monotonically.
* If $\phi$ is close to $1$, the decay is slow, indicating strong persistence.
* If $-1<\phi<0$, the autocorrelation alternates in sign:

```{math}
:enumerated: false
  \rho(1)<0,\quad \rho(2)>0,\quad \rho(3)<0,\dots
```
  so the series tends to oscillate around its mean.

```{admonition} Important Diagnostic Feature
For an AR(1) process, the ACF does **not** cut off sharply. Instead, it **tails off geometrically**.

This is one of the main ways to distinguish AR models from MA models:
- MA($q$): ACF cuts off after lag $q$
- AR(1): ACF tails off
```

---

### A Causal Interpretation

Because the stationary AR(1) can be written as

$$
x_t = w_t + \phi w_{t-1} + \phi^2 w_{t-2} + \cdots,
$$

the current value depends only on the current and past shocks, not on future ones. In this sense the model is **causal**.

```{admonition} Definition: Causal Representation
A process is said to have a causal representation if it can be written as a convergent infinite sum of current and past white-noise shocks.
```

For AR(1), this happens exactly when $|\phi|<1$.

---

### Summary of AR(1)

The AR(1) model

```{math}
:enumerated: false
x_t = \phi x_{t-1} + \mu + w_t 
```

is stationary when $|\phi|<1$. In that case:

```{math}
:enumerated: false
E[x_t]=\frac{\mu}{1-\phi},
```

```{math}
:enumerated: false
\operatorname{Var}(x_t)=\frac{\sigma_w^2}{1-\phi^2},
```

```{math}
:enumerated: false
\gamma(h)=\phi^{|h|}\frac{\sigma_w^2}{1-\phi^2},
```

and

```{math}
:enumerated: false
\rho(h)=\phi^{|h|}.
```

So AR(1) is a simple model of persistence, where shocks have effects that last forever in principle, but with geometrically declining influence.

---

```{admonition} Common Pitfall
Do not say that AR(1) is always stationary.

The model is stationary only when

```{math}
:enumerated: false
|\phi|<1.
```

The case $\phi=1$ is a random walk, which is not stationary.

---

## The AR(2) Model

We now extend the autoregressive framework to allow dependence on two past values.  
This leads to the **autoregressive process of order 2**, or AR(2).

### Definition

An AR(2) process is defined as

$$
x_t = \phi_1 x_{t-1} + \phi_2 x_{t-2} + \mu + w_t,
$$

where $\{w_t\}$ is white noise with variance $\sigma_w^2$.

As before, it is often convenient to work with the mean-zero version:

$$
x_t = \phi_1 x_{t-1} + \phi_2 x_{t-2} + w_t.
$$

---

### Backshift Representation

Using the backshift operator $B$, we can write the AR(2) model as

```{math}
:enumerated: false
(1 - \phi_1 B - \phi_2 B^2)x_t = w_t.
```

The polynomial

```{math}
:enumerated: false
\phi(B) = 1 - \phi_1 B - \phi_2 B^2
```

is called the **autoregressive polynomial**.

### Stationarity and the Characteristic Equation

To understand when the AR(2) process is stationary, we study the polynomial equation

```{math}
:enumerated: false
1 - \phi_1 z - \phi_2 z^2 = 0.
```

This is called the **characteristic equation**.

Let its roots be $z_1$ and $z_2$. Then:

::: {admonition} Stationarity Condition
The AR(2) process is stationary if both roots satisfy

```{math}
:enumerated: false
|z_1| > 1, \quad |z_2| > 1.
```

:::

Equivalently, this can be expressed in terms of the coefficients:

$$
\phi_1 + \phi_2 < 1, \quad \phi_2 - \phi_1 < 1, \quad |\phi_2| < 1.
$$

```{admonition} Intuition
The stationarity condition ensures that the effect of past values does not explode over time.

Just as in AR(1), we require that the system be **stable**, meaning shocks eventually die out.
```

### MA($\infty$) Representation

If the stationarity condition holds, the AR(2) process can be written as an infinite moving average:

```{math}
:enumerated: false
x_t = \sum_{j=0}^{\infty} \psi_j w_{t-j}.
```

The coefficients $\psi_j$ satisfy a recursion:

```{math}
:enumerated: false
\psi_j = \phi_1 \psi_{j-1} + \phi_2 \psi_{j-2},
```

with $\psi_0 = 1$ and $\psi_{-1} = 0$.

```{admonition} Key Idea
An AR(2) process has **infinite memory**, but the weights decay over time when the process is stationary.
```

### Autocovariance Function: Yule–Walker Equations

We now derive the autocovariance structure.

Starting from

```{math}
:enumerated: false
x_t = \phi_1 x_{t-1} + \phi_2 x_{t-2} + w_t,
```

multiply both sides by $x_{t-h}$ and take expectations:

```{math}
:enumerated: false
E[x_t x_{t-h}] = \phi_1 E[x_{t-1} x_{t-h}] + \phi_2 E[x_{t-2} x_{t-h}] + E[w_t x_{t-h}].
```

Using $\gamma(h) = \text{Cov}(x_t, x_{t-h})$ and the fact that $w_t$ is uncorrelated with past values:

* $E[w_t x_{t-h}] = 0$ for $h \geq 1$
* $E[w_t x_t] = \sigma_w^2$

we obtain:

#### For $h = 0$:

```{math}
:enumerated: false
\gamma(0) = \phi_1 \gamma(1) + \phi_2 \gamma(2) + \sigma_w^2
```

#### For $h = 1$:

```{math}
:enumerated: false
\gamma(1) = \phi_1 \gamma(0) + \phi_2 \gamma(1)
```

#### For $h \geq 2$:

```{math}
:enumerated: false
\gamma(h) = \phi_1 \gamma(h-1) + \phi_2 \gamma(h-2)
```

```{admonition} Definition: Yule–Walker Equations
The system

$$
\gamma(h) = \phi_1 \gamma(h-1) + \phi_2 \gamma(h-2)
$$

(with the adjustment at $h=0$) is called the **Yule–Walker equations** for AR(2).
```

### Autocorrelation Function

Dividing through by $\gamma(0)$ gives the autocorrelation recursion:

```{math}
:enumerated: false
\rho(h) = \phi_1 \rho(h-1) + \phi_2 \rho(h-2), \quad h \geq 1.
```

This is a **second-order difference equation**.

### Solving the ACF

To understand the behavior of $\rho(h)$, consider the equation:

```{math}
:enumerated: false
\rho(h) = \phi_1 \rho(h-1) + \phi_2 \rho(h-2).
```

Its solution takes the form

```{math}
:enumerated: false
\rho(h) = c_1 r_1^h + c_2 r_2^h,
```

where $r_1$ and $r_2$ solve

```{math}
:enumerated: false
r^2 - \phi_1 r - \phi_2 = 0.
```

```{admonition} Key Insight
The behavior of the ACF depends on the nature of the roots:

- **Real roots** → monotonic exponential decay  
- **Complex roots** → oscillating (cyclical) decay  
- **Repeated roots** → polynomial × exponential decay
```

### Interpretation

This is where AR(2) becomes much richer than AR(1).

```{admonition} Economic Interpretation
AR(2) can generate:

- smooth persistence (like AR(1))
- oscillations (business cycles, inventory cycles)
- damped cycles (common in macroeconomic data)

This makes AR(2) especially useful in applied economics.
```

### Example

Consider:

```{math}
:enumerated: false
x_t = x_{t-1} - 0.89 x_{t-2} + w_t.
```

The characteristic equation is

```{math}
:enumerated: false
z^2 - z + 0.89 = 0.
```

This has complex roots, implying that the autocorrelation function will exhibit **damped oscillations**.

### Summary

* AR(2) depends on the two most recent observations
* Stationarity depends on the **roots of the characteristic equation**
* The ACF satisfies a **second-order recursion**
* The ACF may:

  * decay smoothly, or
  * oscillate depending on the roots

```{admonition} Important Diagnostic Feature
For AR(2), the ACF **tails off** (does not cut off),  
while the PACF will **cut off after lag 2** (to be shown later).
```

---

## General AR($p$) Processes

We now generalize the autoregressive model to order $p$.

An **autoregressive process of order $p$**, denoted AR($p$), is defined as

$$
x_t = \phi_1 x_{t-1} + \phi_2 x_{t-2} + \cdots + \phi_p x_{t-p} + \mu + w_t,
$$

where $\{w_t\}$ is white noise with variance $\sigma_w^2$.

As before, we work with the mean-zero version:

```{math}
:enumerated: false
x_t = \phi_1 x_{t-1} + \phi_2 x_{t-2} + \cdots + \phi_p x_{t-p} + w_t.
```

### Backshift Representation

Using the backshift operator $B$, we write

```{math}
:enumerated: false
(1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p)x_t = w_t.
```

Define the **autoregressive polynomial**

```{math}
:enumerated: false
\phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p.
```

### Stationarity

The stationarity condition generalizes naturally:

```{admonition} Stationarity Condition
An AR($p$) process is stationary if all roots of

```{math}
:enumerated: false
\phi(z) = 0
```

lie **outside the unit circle**, i.e.

```{math}
:enumerated: false
|z_i| > 1 \quad \text{for all roots } z_i.
```

This ensures that shocks die out over time and that the process has constant mean and variance.


```{admonition} Intuition
AR($p$) models are systems with internal dynamics governed by their own past values.

Stationarity means the system is **stable**: shocks do not accumulate or explode.
```

---

### MA($\infty$) Representation

When the stationarity condition holds, the AR($p$) process can be written as

```{math}
:enumerated: false
x_t = \sum_{j=0}^{\infty} \psi_j w_{t-j},
```

where the coefficients satisfy the recursion:

```{math}
:enumerated: false
\psi_j = \phi_1 \psi_{j-1} + \phi_2 \psi_{j-2} + \cdots + \phi_p \psi_{j-p}.
```

```{admonition} Key Idea
AR($p$) processes have **infinite memory**, but the influence of past shocks decays over time.
```

## Yule–Walker Equations (General Form)

We now derive a key result that connects the parameters $\phi_1, \dots, \phi_p$ to the autocovariance function.

Start from

```{math}
:enumerated: false
x_t = \phi_1 x_{t-1} + \cdots + \phi_p x_{t-p} + w_t.
```

Multiply both sides by $x_{t-h}$ and take expectations:

```{math}
:enumerated: false
E[x_t x_{t-h}] = \phi_1 E[x_{t-1}x_{t-h}] + \cdots + \phi_p E[x_{t-p}x_{t-h}] + E[w_t x_{t-h}].
```

Using $\gamma(h) = \text{Cov}(x_t, x_{t-h})$ and the white noise properties:

* $E[w_t x_{t-h}] = 0$ for $h \geq 1$
* $E[w_t x_t] = \sigma_w^2$

we obtain:

### For $h \geq 1$

```{math}
:enumerated: false
\gamma(h) = \phi_1 \gamma(h-1) + \phi_2 \gamma(h-2) + \cdots + \phi_p \gamma(h-p)
```

### For $h = 0$

```{math}
:enumerated: false
\gamma(0) = \phi_1 \gamma(1) + \cdots + \phi_p \gamma(p) + \sigma_w^2
```

```{admonition} Definition: Yule–Walker Equations
The system

$$
\gamma(h) = \sum_{j=1}^p \phi_j \gamma(h-j)
$$

(with the adjustment at $h=0$) is called the **Yule–Walker equations** for AR($p$).
```

### Autocorrelation Form

Dividing through by $\gamma(0)$ gives:

```{math}
:enumerated: false
\rho(h) = \phi_1 \rho(h-1) + \cdots + \phi_p \rho(h-p), \quad h \geq 1.
```

This is a **difference equation of order $p$**.

```{admonition} Key Insight
The ACF of an AR($p$) process is determined entirely by a recursion of order $p$.

This is why the ACF **does not cut off**, but instead **tails off gradually**.
```

## Partial Autocorrelation Function (PACF)

While the ACF captures overall dependence, it does not isolate **direct effects**.

To see why, consider:

* $x_t$ may be correlated with $x_{t-2}$
* but this may be due to the intermediate variable $x_{t-1}$

We need a way to measure the **direct relationship**, controlling for intermediate lags.

### Definition

The **partial autocorrelation at lag $h$**, denoted $\phi_{hh}$, is defined as

$$
\phi_{hh} = \text{Corr}(x_t, x_{t-h} \mid x_{t-1}, x_{t-2}, \dots, x_{t-h+1}).
$$

That is, it measures the correlation between $x_t$ and $x_{t-h}$ after removing the effect of intermediate lags.

---

```{admonition} Intuition
PACF answers the question:

👉 “Is there a **direct link** between $x_t$ and $x_{t-h}$,  
or is the relationship entirely mediated through intermediate values?”
```

---

### PACF via Regression

One way to compute PACF is:

1. Regress $x_t$ on $x_{t-1}, \dots, x_{t-h+1}$
2. Regress $x_{t-h}$ on the same variables
3. Take the correlation of the residuals

This removes the influence of intermediate lags.

---

### Key Property for AR($p$)

````{admonition} Fundamental Result
For an AR($p$) process:

- The PACF **cuts off after lag $p$**

That is,

```{math}
:enumerated: false
\phi_{hh} = 0 \quad \text{for all } h > p.
```
````

### Why This Happens

In an AR($p$) model:

```{math}
:enumerated: false
x_t = \phi_1 x_{t-1} + \cdots + \phi_p x_{t-p} + w_t,
```

all dependence beyond lag $p$ must pass through the first $p$ lags.

So once we condition on $x_{t-1}, \dots, x_{t-p}$:

* there is no additional direct relationship with $x_{t-h}$ for $h > p$

---

```{admonition} Key Identification Rule
- AR($p$):  
  - ACF **tails off**  
  - PACF **cuts off after lag $p$**

- MA($q$):  
  - ACF **cuts off after lag $q$**  
  - PACF **tails off**
```

## Putting It All Together

We now have a complete toolkit:

* AR models describe persistence through past values
* Yule–Walker equations link parameters to autocovariances
* PACF isolates direct relationships
* ACF and PACF together allow **model identification**

---

```{admonition} Big Picture
Time series modeling is not about memorizing formulas.

It is about recognizing patterns:

- sharp cutoff → MA  
- gradual decay → AR  
- PACF cutoff → AR order  

These patterns guide us in building models from data.
```

---

## ACF, PACF, and Model Identification

We now bring together the key tools developed so far: the **autocorrelation function (ACF)** and the **partial autocorrelation function (PACF)**.

These functions allow us to **identify appropriate models from data**, which is the first step in time series analysis.

## ACF vs PACF: What Do They Measure?

The **ACF** measures the total correlation between $x_t$ and $x_{t-h}$.

The **PACF** measures the *direct* correlation between $x_t$ and $x_{t-h}$, after removing the effect of intermediate lags.

```{admonition} Intuition
- ACF: “How related are these two points overall?”  
- PACF: “Is there a direct link, or is it explained by intermediate steps?”
```

## Characteristic Patterns

```{admonition} ACF–PACF Summary
| Model        | ACF behavior        | PACF behavior       |
|--------------|-------------------|--------------------|
| AR($p$)      | Tails off          | Cuts off after $p$ |
| MA($q$)      | Cuts off after $q$ | Tails off          |
| ARMA($p,q$)  | Tails off          | Tails off          |
```

## Confidence Bands in ACF and PACF

When plotting sample ACF and PACF, we often include horizontal bands around zero.

::: {admonition} Interpretation of Bands
For large samples, approximate 95% confidence bands are given by

```{math}
:enumerated: false
\pm \frac{2}{\sqrt{n}}
```

Spikes outside these bands suggest statistically significant autocorrelation.
:::

```{admonition} Important
Do not over-interpret individual spikes. Focus on **overall patterns** rather than isolated exceedances.
```

## Simulated Examples: Building Intuition

We simulate simple processes to visualize their ACF and PACF patterns.

### Example 1: MA(1)

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

np.random.seed(123)
n = 300
theta = 0.8
w = np.random.normal(size=n+1)

x_ma1 = np.zeros(n)
for t in range(n):
    x_ma1[t] = w[t+1] + theta*w[t]

plt.figure(figsize=(8,3))
plt.plot(x_ma1)
plt.title("MA(1) Series")
plt.xlabel("Time")
plt.ylabel("$x_t$")
plt.show()
```

```{code-cell} python
plt.figure(figsize=(8,3))
plot_acf(x_ma1, lags=20)
plt.title("ACF: MA(1)")
plt.show()
```

```{code-cell} python
plt.figure(figsize=(8,3))
plot_pacf(x_ma1, lags=20, method="ywm")
plt.title("PACF: MA(1)")
plt.show()
```

The ACF cuts off sharply after lag 1, while the PACF tails off.

---

### Example 2: AR(1)

```{code-cell} python
np.random.seed(123)
n = 300
phi = 0.8
w = np.random.normal(size=n)

x_ar1 = np.zeros(n)
for t in range(1, n):
    x_ar1[t] = phi*x_ar1[t-1] + w[t]

plt.figure(figsize=(8,3))
plt.plot(x_ar1)
plt.title("AR(1) Series")
plt.xlabel("Time")
plt.ylabel("$x_t$")
plt.show()
```

```{code-cell} python
plt.figure(figsize=(8,3))
plot_acf(x_ar1, lags=20)
plt.title("ACF: AR(1)")
plt.show()
```

```{code-cell} python
plt.figure(figsize=(8,3))
plot_pacf(x_ar1, lags=20, method="ywm")
plt.title("PACF: AR(1)")
plt.show()
```

The ACF decays gradually, while the PACF cuts off after lag 1.

---

### Example 3: AR(2) with Oscillation

```{code-cell} python
np.random.seed(123)
n = 400
phi1, phi2 = 1.0, -0.6
w = np.random.normal(size=n)

x_ar2 = np.zeros(n)
for t in range(2, n):
    x_ar2[t] = phi1*x_ar2[t-1] + phi2*x_ar2[t-2] + w[t]

plt.figure(figsize=(8,3))
plt.plot(x_ar2)
plt.title("AR(2) Series")
plt.xlabel("Time")
plt.ylabel("$x_t$")
plt.show()
```

```{code-cell} python
plt.figure(figsize=(8,3))
plot_acf(x_ar2, lags=30)
plt.title("ACF: AR(2)")
plt.show()
```

```{code-cell} python
plt.figure(figsize=(8,3))
plot_pacf(x_ar2, lags=30, method="ywm")
plt.title("PACF: AR(2)")
plt.show()
```

The ACF may oscillate, but the PACF still cuts off after lag 2.

---


## Practical Identification Strategy

```{admonition} Box–Jenkins Workflow
1. **Identify** (ACF/PACF, plots)  
2. **Estimate** (fit candidate models)  
3. **Diagnose** (check residuals)  
4. **Refine** (iterate if needed)
```

```{admonition} Practical Advice
Start with simple models: AR(1), AR(2), MA(1), MA(2).
```

---

## Residual Diagnostics

After fitting a model, residuals $\hat{w}_t$ should behave like **white noise**.

---

### Residual Plot

```{code-cell} python
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

model = sm.tsa.ARIMA(x_ar1, order=(1,0,0))
result = model.fit()
resid = result.resid

plt.figure(figsize=(8,3))
plt.plot(resid)
plt.title("Residuals")
plt.show()
```

---

### Residual ACF

```{code-cell} python
plt.figure(figsize=(8,3))
plot_acf(resid, lags=20)
plt.title("Residual ACF")
plt.show()
```

---

### Ljung–Box Test

$$
Q = n(n+2)\sum_{h=1}^{m} \frac{\hat{\rho}_h^2}{n-h}
$$

```{code-cell} python
acorr_ljungbox(resid, lags=[10,20], return_df=True)
```

---

### Q–Q Plot

```{code-cell} python
sm.qqplot(resid, line='45')
plt.title("Q-Q Plot")
plt.show()
```

```{admonition} Diagnostic Checklist
- residual plot  
- residual ACF  
- Ljung–Box test  
- Q–Q plot  
```

---

## Model Selection: AIC and BIC

$$
AIC = -2 \log(\hat{L}) + 2k
$$

$$
BIC = -2 \log(\hat{L}) + k \log n
$$

```{admonition} Interpretation
- Lower AIC/BIC is better  
- BIC penalizes complexity more strongly  
```

---

### Example Comparison

```{code-cell} python
models = {
    "AR(1)": sm.tsa.ARIMA(x_ar1, order=(1,0,0)).fit(),
    "AR(2)": sm.tsa.ARIMA(x_ar1, order=(2,0,0)).fit(),
    "MA(1)": sm.tsa.ARIMA(x_ar1, order=(0,0,1)).fit(),
    "ARMA(1,1)": sm.tsa.ARIMA(x_ar1, order=(1,0,1)).fit(),
}

for name, res in models.items():
    print(f"{name:10s} AIC = {res.aic:8.2f}   BIC = {res.bic:8.2f}")
```

---

```{admonition} Key Principle
A good model should:

1. make theoretical sense  
2. pass residual diagnostics  
3. be parsimonious  
```

---

## Big Picture

Time series modeling is an **iterative process**:

1. Identify
2. Estimate
3. Diagnose
4. Refine

```{admonition} Final Insight
There is no single “correct” model — only useful approximations.
```
---

## ARMA($p,q$): Combining AR and MA

In practice, many time series exhibit both **persistence** and **short-run shock dynamics**.  

- Autoregressive (AR) models capture dependence on past values  
- Moving average (MA) models capture dependence on past shocks  

The **ARMA($p,q$)** model combines both.

### Definition

An ARMA($p,q$) process is defined as

$$
x_t = \phi_1 x_{t-1} + \cdots + \phi_p x_{t-p}
      + \theta_1 w_{t-1} + \cdots + \theta_q w_{t-q} + \mu + w_t,
$$

where $\{w_t\}$ is white noise.

### Backshift Representation

Using the backshift operator $B$, we write

```{math}
:enumerated: false
\phi(B)x_t = \theta(B) w_t,
```

where

```{math}
:enumerated: false
\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p,
```

```{math}
:enumerated: false
\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q.
```

```{admonition} Interpretation
ARMA models combine:

- **internal dynamics** (through past values), and  
- **shock propagation** (through past innovations)

This makes them flexible enough to capture many real-world time series.
````

### Stationarity and Invertibility

For an ARMA($p,q$) process to be well-behaved:

* the AR part must be **stationary**
* the MA part must be **invertible**

```{admonition} Key Conditions
- Stationarity: roots of $\phi(z) = 0$ lie outside the unit circle  
- Invertibility: roots of $\theta(z) = 0$ lie outside the unit circle
```

### ACF and PACF Behavior

Unlike pure AR or MA models, ARMA processes do not exhibit sharp cutoffs.

```{admonition} Key Feature id="arma-pattern"
For ARMA($p,q$):

- ACF: tails off  
- PACF: tails off  

👉 No clear cutoff in either function
```

```{admonition} Intuition
Because ARMA models combine both AR and MA components,  
their dependence structure is more diffuse.

There is no single lag after which correlations vanish.
```

### Identification in Practice

Since neither ACF nor PACF cuts off cleanly, identifying ARMA models is more challenging.

```{admonition} Practical Strategy
- Start with simple models (e.g. ARMA(1,1))  
- Compare alternatives using AIC/BIC  
- Check residual diagnostics carefully  
```

```{admonition} Common Pitfall
Do not try to infer ARMA orders purely from ACF/PACF plots.

👉 Use them as a guide, not as a strict rule.
```

### Example: ARMA(1,1)

A commonly used model is ARMA(1,1):

```{math}
:enumerated: false
x_t = \phi x_{t-1} + \theta w_{t-1} + w_t.
```

This model often captures both:

* persistence (through $\phi$), and
* short-term adjustment (through $\theta$)

with relatively few parameters.

### Summary

* ARMA($p,q$) combines AR and MA dynamics
* both ACF and PACF **tail off**
* identification relies on:

  * patterns in ACF/PACF
  * model comparison (AIC/BIC)
  * residual diagnostics

---

```{admonition} Looking Ahead
In many economic and financial time series, the data are not stationary.

In the next chapter, we introduce **ARIMA models**, which extend ARMA models to handle non-stationary data through differencing.
```
