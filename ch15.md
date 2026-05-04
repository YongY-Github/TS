---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 15 — Forecasting Methods

So far, we have studied how time series behave and how they can be modeled using AR, MA, ARMA, and ARIMA models.

We now turn to one of the main practical goals of time series analysis:

```{admonition} Central Question
How can we use past information to forecast future values?
```

Forecasting is important in economics, business, finance, and policy.

Examples include:

- forecasting inflation
- predicting GDP growth
- projecting demand
- forecasting exchange rates
- estimating future sales
- predicting stock volatility

This chapter introduces the basic logic of forecasting.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- explain what a forecast is
- distinguish in-sample fit from out-of-sample forecasting
- distinguish one-step-ahead and multi-step-ahead forecasts
- understand static and dynamic forecasts
- generate forecasts from simple time series models
- interpret forecast uncertainty
- understand the basic forecasting workflow

---

# 15.1 What Is a Forecast?

A forecast is a prediction of a future value based on information available today.

Suppose we observe:

```{math}
:enumerated: false
x_1, x_2, \dots, x_T
```

A forecast of $x_{T+1}$ is written as:

```{math}
:enumerated: false
\hat{x}_{T+1}
```

More generally, an $h$-step-ahead forecast is:

```{math}
:enumerated: false
\hat{x}_{T+h}
```

```{admonition} Definition
A forecast is an estimate of a future value of a time series, conditional on information currently available.
```

---

# 15.2 Forecasting Is Conditional

Forecasting is always based on available information.

At time $T$, we know:

```{math}
:enumerated: false
x_1, x_2, \dots, x_T
```

but we do not yet know:

```{math}
:enumerated: false
x_{T+1}, x_{T+2}, \dots
```

A forecast is therefore conditional on the information set available at time $T$.

```{admonition} Key Idea
Forecasts are not guesses. They are based on a model and available information.
```

---

# 15.3 Forecast vs Fitted Value

It is important to distinguish a **forecast** from a **fitted value**.

A fitted value is produced for an observation already in the sample.

A forecast is produced for an observation not yet observed.

| Concept | Meaning |
|---|---|
| fitted value | model prediction for an observed data point |
| forecast | model prediction for a future data point |

```{admonition} Important
A model may fit past data well but forecast future data poorly.
```

This is why out-of-sample evaluation is essential.

---

# 15.4 In-Sample vs Out-of-Sample Forecasting

## In-Sample Fit

In-sample fit refers to how well the model explains data already used for estimation.

For example, if we estimate a model using observations $1$ to $T$, then fitted values within this same range are in-sample.

## Out-of-Sample Forecasting

Out-of-sample forecasting evaluates how well the model predicts observations not used in estimation.

```{admonition} Key Principle
Forecasting performance should be judged using data not used to estimate the model.
```

---

# 15.5 Train-Test Split for Time Series

In cross-sectional data, observations are often randomly split into training and testing samples.

In time series, we do **not** randomly shuffle observations.

Instead, we preserve time order.

For example:

- estimate model using observations $1,\dots,T_0$
- forecast observations $T_0+1,\dots,T$
- compare forecasts with actual values

```{admonition} Important
Time series train-test splits must respect chronological order.
```

---

# 15.6 One-Step-Ahead Forecasts

A one-step-ahead forecast predicts the next observation.

At time $T$, the one-step-ahead forecast is:

```{math}
:enumerated: false
\hat{x}_{T+1}
```

For an AR(1) model:

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t
```

the one-step-ahead forecast is:

```{math}
:enumerated: false
\hat{x}_{T+1} = \phi x_T
```

because the best prediction of the future shock is zero:

```{math}
:enumerated: false
E[w_{T+1}] = 0
```

```{admonition} Intuition
Forecasting replaces unknown future shocks with their expected value, usually zero.
```

---

# 15.7 Multi-Step-Ahead Forecasts

A multi-step-ahead forecast predicts more than one period into the future.

For AR(1):

```{math}
:enumerated: false
x_t = \phi x_{t-1} + w_t
```

the two-step-ahead forecast is:

```{math}
:enumerated: false
\hat{x}_{T+2}
=
\phi \hat{x}_{T+1}
=
\phi^2 x_T
```

More generally:

```{math}
:enumerated: false
\hat{x}_{T+h}
=
\phi^h x_T
```

```{admonition} Key Insight
For a stationary AR(1), forecasts gradually return toward the long-run mean.
```

If the mean is not zero, forecasts return toward:

```{math}
:enumerated: false
\frac{\mu}{1-\phi}
```

---

# 15.8 Forecasting a Random Walk

For a random walk:

```{math}
:enumerated: false
x_t = x_{t-1} + w_t
```

the best forecast of tomorrow is today’s value:

```{math}
:enumerated: false
\hat{x}_{T+1} = x_T
```

For any horizon $h$:

```{math}
:enumerated: false
\hat{x}_{T+h} = x_T
```

```{admonition} Key Result
For a random walk, the best forecast of the future level is the current level.
```

---

# 15.9 Forecasting a Random Walk with Drift

For a random walk with drift:

```{math}
:enumerated: false
x_t = \delta + x_{t-1} + w_t
```

the $h$-step-ahead forecast is:

```{math}
:enumerated: false
\hat{x}_{T+h}
=
x_T + h\delta
```

```{admonition} Interpretation
With drift, the forecast path moves upward or downward by the drift amount each period.
```

---

# 15.10 Static vs Dynamic Forecasts

Forecasting software often distinguishes between **static** and **dynamic** forecasts.

## Static Forecasts

A static forecast uses actual lagged values whenever they are available.

For example, in an AR(1):

```{math}
:enumerated: false
\hat{x}_{t}
=
\hat{\phi}x_{t-1}
```

uses the actual value $x_{t-1}$.

## Dynamic Forecasts

A dynamic forecast uses previous forecasts as inputs when actual future values are unavailable.

For example:

```{math}
:enumerated: false
\hat{x}_{t}
=
\hat{\phi}\hat{x}_{t-1}
```

```{admonition} Key Difference
Static forecasts use actual lagged values.

Dynamic forecasts use previously forecast values.
```

---

# 15.11 Why Dynamic Forecasts Become Harder

Dynamic forecasts become less reliable as the horizon increases.

Why?

Because forecast errors accumulate.

```{admonition} Important
Forecast uncertainty increases as the forecast horizon becomes longer.
```

This is especially important for:

- macroeconomic forecasts
- financial forecasts
- demand forecasts
- policy projections

---

# 15.12 Forecast Errors

A forecast error is the difference between the actual value and the forecast:

```{math}
:enumerated: false
e_{T+h}
=
x_{T+h}
-
\hat{x}_{T+h}
```

A good forecast has small errors on average.

```{admonition} Definition
Forecast error measures the gap between the actual value and the predicted value.
```

Forecast evaluation is the focus of the next chapter.

---

# 15.13 Forecast Intervals

A point forecast gives a single predicted value.

But forecasts are uncertain.

A forecast interval gives a range of plausible future values.

```{admonition} Key Idea
Good forecasting should communicate uncertainty, not only a single predicted value.
```

For example, instead of saying:

> Inflation next year will be 3%.

we might say:

> Inflation is forecast to be 3%, with a plausible range from 2% to 4%.

---

# 15.14 Simulating Forecasts from an AR(1)

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

n = 120
phi = 0.8
w = np.random.normal(size=n)

x = np.zeros(n)

for t in range(1, n):
    x[t] = phi*x[t-1] + w[t]

T = 100
h = 20

history = x[:T+1]

forecast = np.zeros(h)
forecast[0] = phi * history[-1]

for i in range(1, h):
    forecast[i] = phi * forecast[i-1]

forecast_index = np.arange(T+1, T+h+1)

plt.figure(figsize=(10,4))
plt.plot(np.arange(T+1), history, label="Observed")
plt.plot(forecast_index, forecast, linestyle="--", label="Forecast")
plt.axvline(T, linestyle=":", color="black")
plt.title("Dynamic Forecast from an AR(1) Model")
plt.xlabel("Time")
plt.ylabel("$x_t$")
plt.legend()

plt.savefig("figs/ch15/AR1_forecast.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![AR1 Forecast](figs/ch15/AR1_forecast.png)


```{admonition} Observation
For a stationary AR(1), forecasts gradually move back toward the long-run mean.
```

---

# 15.15 Forecasting Workflow

A practical forecasting workflow is:

1. plot the data
2. check stationarity
3. choose candidate models
4. estimate models
5. generate forecasts
6. compare forecasts with actual values
7. evaluate forecast errors
8. revise the model if needed

```{admonition} Practical Advice
Forecasting is iterative.

A model should be judged by how well it forecasts, not only by how well it fits the past.
```

---

# 15.16 Gretl Example: Forecasting with ARIMA

We now outline a simple forecasting workflow in GRETL.

---

## Step 1: Estimate a Model

### Menu

`Model → Time series → ARIMA`

Choose:

- dependent variable
- AR order
- differencing order
- MA order

---

```markdown
[GRETL Screenshot Placeholder: ARIMA model specification]
```

---

## Step 2: Generate Forecasts

After estimating the model, in the model window:

`Analysis → Forecasts`

or:

`Model window → Forecasts`

depending on your GRETL version.

---

```markdown
[GRETL Screenshot Placeholder: Forecast dialog]
```

---

## Step 3: Choose Forecast Range

Choose:

- forecast start date
- forecast end date
- static or dynamic forecast
- whether to include forecast intervals

---

```markdown
[GRETL Screenshot Placeholder: Forecast range and options]
```

---

## Step 4: Inspect Forecast Output

GRETL typically provides:

- forecast values
- standard errors
- confidence intervals
- forecast graph

---

```markdown
[GRETL Screenshot Placeholder: Forecast graph]
```

---

```{admonition} Practical Note
In Gretl, forecast options depend on the model type and the available sample range.
```

---

# 15.17 Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Confusing fitted values with forecasts**  
Fitted values describe the past; forecasts predict the future.

**2. Randomly splitting time series data**  
Time series splits should preserve chronological order.

**3. Ignoring forecast uncertainty**  
A point forecast alone can be misleading.

**4. Overvaluing in-sample fit**  
Good fit does not guarantee good forecasting performance.

**5. Forecasting too far ahead mechanically**  
Uncertainty grows with the forecast horizon.
```

---

# 15.18 Looking Ahead

This chapter introduced the basic logic of forecasting.

In the next chapter, we study how to evaluate forecast accuracy.

We will examine:

- bias
- MSE
- RMSE
- MAE
- MAPE
- Theil’s U1 and U2
- Decomposition

# Key Takeaways

```{admonition} Summary
- A forecast is a prediction conditional on available information.
- Fitted values and forecasts are different.
- Time series forecasting must preserve chronological order.
- One-step forecasts use current information.
- Multi-step forecasts rely increasingly on previous forecasts.
- Forecast uncertainty grows with the forecast horizon.
- Forecasting models should be evaluated out of sample.
```

# Concept Check

### Basic

1. What is a forecast?

2. What does it mean for a forecast to be conditional?

3. What is the difference between a fitted value and a forecast?

---

### Intuition

4. Why is forecasting fundamentally uncertain?

5. Why might a model fit past data well but perform poorly in forecasting?

6. Why is out-of-sample evaluation important?

---

### Intermediate

7. What is the difference between:

   - one-step-ahead forecasts  
   - multi-step-ahead forecasts  

8. What is the difference between:

   - static forecasts  
   - dynamic forecasts  

9. Why do forecast errors tend to increase as the horizon grows?

---

### Model-Based Forecasting

10. In an AR(1) model, why is the expected value of the future shock set to zero?

11. Why do AR(1) forecasts converge toward the long-run mean?

---

### Challenge

12. Suppose a model produces very stable forecasts.

   - Does this necessarily mean it is accurate?

---

# Interpretation & Practice

1. A model fits historical data extremely well but performs poorly out-of-sample.

   - What might be the issue?
   - What does this suggest about model evaluation?

2. A forecast for a stationary AR(1) process gradually flattens over time.

   - Why does this happen?

3. A random walk forecast remains constant over time.

   - What does this imply about predictability?

4. A forecast diverges rapidly as the horizon increases.

   - What might this indicate about the underlying model?

5. Dynamic forecasts differ significantly from static forecasts.

   - Why might this happen?

---

### Finance Interpretation

6. A stock price follows a random walk.

   - Why is the best forecast of tomorrow’s price today’s price?

7. A macroeconomic forecast becomes less precise over longer horizons.

   - Why is this expected?

---

### Challenge

8. A model produces very narrow forecast intervals.

   - Why might this be misleading?
   - What could be wrong with the model?

---

# Numerical Practice

### One-Step Forecast

1. Suppose:

```{math}
:enumerated: false
x_t = 0.8 x_{t-1} + w_t
```

and $x_T = 10$.

- Compute the one-step-ahead forecast $\hat{x}_{T+1}$.

---

### Multi-Step Forecast

2. Using the same model:

- Compute $\hat{x}_{T+2}$  
- Compute $\hat{x}_{T+3}$  

- What pattern do you observe?

---

### Random Walk

3. Suppose:

```{math}
:enumerated: false
x_t = x_{t-1} + w_t
```

and $x_T = 50$.

- Compute $\hat{x}_{T+1}$  
- Compute $\hat{x}_{T+5}$  

- What do you observe?

---

### Random Walk with Drift

4. Suppose:

```{math}
:enumerated: false
x_t = x_{t-1} + 2 + w_t
```

and $x_T = 100$.

- Compute $\hat{x}_{T+1}$  
- Compute $\hat{x}_{T+3}$  

---

### Forecast Error

5. Suppose:

- forecast: $\hat{x}_{T+1} = 20$  
- actual: $x_{T+1} = 23$  

- Compute the forecast error.

---

### Interpretation

6. Suppose forecast errors are consistently positive.

- What does this imply about the model?

---

### Dynamic Forecasting

7. Suppose a dynamic forecast is used.

- Why might errors accumulate over time?

---

### Challenge

8. Suppose two models produce:

- Model A: very accurate one-step forecasts  
- Model B: better long-horizon forecasts  

- Which would you choose? Why?

9. You are forecasting monthly sales.

You estimate an AR(1) model and generate forecasts for the next 12 months.

- Why might the 1-month-ahead forecast be reliable?
- Why might the 12-month-ahead forecast be much less reliable?