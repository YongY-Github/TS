---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 16 — Evaluating Forecasts

In the previous chapter, we learned how forecasts are generated from time series models.

But an important question remains:

```{admonition} Central Question
How do we know whether one forecast is better than another?
```

Forecasting is not only about producing predictions.

It is also about evaluating forecast quality.

A model that fits historical data very well may still forecast poorly out of sample.

This chapter introduces the most commonly used measures of forecast accuracy and forecast quality.

We will use a real example based on Thai inflation forecasts to illustrate the ideas.

---

## Learning Objectives

By the end of this chapter, you should be able to:

* compute forecast errors
* distinguish bias from variability
* interpret MSE, RMSE, MAE, and MAPE
* understand the tradeoff between different forecast evaluation measures
* interpret Theil’s $U_1$ and $U_2$ statistics
* understand forecast error decomposition
* compare competing forecasting models

---

# 16.1 Forecast Errors

Forecast evaluation begins with the forecast error.

Suppose:

* actual value: $x_t$
* forecast value: $\hat{x}_t$

The forecast error is:

```{math}
:enumerated: false
e_t = x_t - \hat{x}_t
```

```{admonition} Interpretation
A forecast error measures the difference between what actually happened and what the model predicted.
```

If:

* $e_t > 0$: the model underpredicted
* $e_t < 0$: the model overpredicted

---

# 16.2 A Thai Inflation Forecast Example

Suppose we compare two competing forecasts of Thailand’s year-on-year CPI inflation.

The table below shows:

* actual inflation
* Forecast 1 - using AR(1)
* Forecast 2 - random walk model
* forecast errors

| Date     | Actual Inflation | Forecast 1 | Forecast 2 |
| -------- | ---------------- | ---------- | ---------- |
| Jan 2014 | 1.93             | 1.84       | 1.67       |
| Feb 2014 | 1.96             | 1.92       | 1.93       |
| Mar 2014 | 2.11             | 1.96       | 1.96       |
| Apr 2014 | 2.45             | 2.31       | 2.11       |
| May 2014 | 2.62             | 3.61       | 2.45       |
| Jun 2014 | 2.35             | 2.45       | 2.62       |
| Jul 2014 | 2.16             | 2.01       | 2.35       |
| Aug 2014 | 2.09             | 1.99       | 2.16       |
| Sep 2014 | 1.75             | 1.81       | 2.09       |
| Oct 2014 | 1.48             | 1.65       | 1.75       |
| Nov 2014 | 1.26             | 1.21       | 1.48       |
| Dec 2014 | 0.60             | 1.13       | 1.26       |

We now ask:

```{admonition} Main Goal
Which forecasting method performed better?
```

For data and computations in Excel see [LINK](figs\ch16\TSF_cpi_.xlsx).

---

# 16.3 Mean Forecast Error (Bias)

The simplest measure is the average forecast error.

```{math}
:enumerated: false
\text{Bias}
=
\frac{1}{T}
\sum_{t=1}^{T}
e_t
```

or equivalently:

```{math}
:enumerated: false
\text{Bias}
=
\frac{1}{T}
\sum_{t=1}^{T}
(x_t - \hat{x}_t)
```

```{admonition} Interpretation
Bias measures whether forecasts systematically overpredict or underpredict.
```

* positive bias $\rightarrow$ forecasts tend to be too low
* negative bias $\rightarrow$ forecasts tend to be too high

For the Thai inflation example:

| Measure | Forecast 1 | Forecast 2 |
| ------- | ---------- | ---------- |
| Bias    | 0.094      | 0.089      |

Both forecasts slightly underpredict inflation on average.

---

# 16.4 Mean Squared Error (MSE)

One problem with simple bias is that positive and negative errors can cancel out.

To avoid this, we square the errors.

```{math}
:enumerated: false
\text{MSE}
=
\frac{1}{T}
\sum_{t=1}^{T}
e_t^2
```

```{admonition} Key Idea
MSE penalizes large forecast errors heavily because errors are squared.
```

Large mistakes therefore matter disproportionately.

## Thai Inflation Example

| Measure | Forecast 1 | Forecast 2 |
| ------- | ---------- | ---------- |
| MSE     | 0.116      | 0.084      |

Forecast 2 has the smaller MSE.

So Forecast 2 performs better according to this criterion.

---

# 16.5 Root Mean Squared Error (RMSE)

MSE is useful mathematically, but its units are squared.

To restore the original units, we take the square root.

```{math}
:enumerated: false
\text{RMSE}
=
\sqrt{
\frac{1}{T}
\sum_{t=1}^{T}
e_t^2
}
```

```{admonition} Interpretation
RMSE measures the typical size of forecast errors in the original units of the variable.
```

For inflation forecasting, RMSE is measured in percentage points of inflation.

## Thai Inflation Example

| Measure | Forecast 1 | Forecast 2 |
| ------- | ---------- | ---------- |
| RMSE    | 0.341      | 0.290      |

Again, Forecast 2 performs better.

---

# 16.6 Why RMSE Is Popular

RMSE is one of the most widely used forecast evaluation measures.

Why?

Because:

* it penalizes large errors
* it is easy to interpret
* it preserves original units

```{admonition} Important
RMSE is especially useful when large forecast mistakes are particularly costly.
```

Examples include:

* inflation forecasting by central banks
* electricity demand forecasting
* financial risk forecasting

---

# 16.7 Mean Absolute Error (MAE)

Instead of squaring errors, we can use absolute values.

```{math}
:enumerated: false
\text{MAE}
=
\frac{1}{T}
\sum_{t=1}^{T}
|e_t|
```

```{admonition} Interpretation
MAE measures the average absolute size of forecast errors.
```

Unlike RMSE:

* MAE penalizes errors linearly
* extreme errors matter less

## Thai Inflation Example

| Measure | Forecast 1 | Forecast 2 |
| ------- | ---------- | ---------- |
| MAE     | 0.214      | 0.247      |

Now Forecast 1 performs better.

This illustrates an important idea:

```{admonition} Important Insight
Different evaluation measures may rank forecasts differently.
```

---

# 16.8 RMSE vs MAE

RMSE and MAE emphasize different aspects of forecast performance.

| Measure | Sensitive to Large Errors? |
| ------- | -------------------------- |
| MAE     | Less sensitive             |
| RMSE    | More sensitive             |

## Intuition

Suppose one model makes:

* many moderate errors
* but no extremely large mistakes

Another model makes:

* mostly small errors
* but occasionally a very large mistake

RMSE may strongly penalize the second model.

MAE may prefer it.

```{admonition} Practical Advice
There is no universally “best” forecast measure.

The preferred measure depends on the forecasting objective.
```

---

# 16.9 Percentage Errors

Sometimes variables have different scales.

In such cases, percentage-based measures may be useful.

The percentage forecast error is:

```{math}
:enumerated: false
\text{Percentage Error}_t
=
100
\times
\frac{e_t}{x_t}
```

---

# 16.10 Mean Absolute Percentage Error (MAPE)

A common percentage-based measure is:

```{math}
:enumerated: false
\text{MAPE}
=
\frac{100}{T}
\sum_{t=1}^{T}
\left|
\frac{e_t}{x_t}
\right|
```

```{admonition} Interpretation
MAPE measures forecast errors as percentages of the actual values.
```

## Thai Inflation Example

| Measure | Forecast 1 | Forecast 2 |
| ------- | ---------- | ---------- |
| MAPE    | 14.96      | 19.11      |

Forecast 1 performs better according to MAPE.

---

# 16.11 Problems with MAPE

MAPE has some important weaknesses.

## Problem 1: Division by Small Numbers

If actual values are close to zero:

```{math}
:enumerated: false
\frac{e_t}{x_t}
```

can become extremely large.

## Problem 2: Undefined for Zero Values

If:

```{math}
:enumerated: false
x_t = 0
```

then MAPE is undefined.

## Problem 3: Asymmetry

MAPE can penalize overpredictions and underpredictions differently.

```{admonition} Warning
MAPE can behave poorly when the series contains values near zero.
```

This is common in:

* inflation
* growth rates
* financial returns

---

# 16.12 Relationship Between MSE, RMSE, and Bias

Recall:

```{math}
:enumerated: false
\text{MSE}
=
\frac{1}{T}
\sum e_t^2
```

MSE can be decomposed into:

```{math}
:enumerated: false
\text{MSE}
=
\text{Variance of Errors}
+
(\text{Bias})^2
```

or approximately:

```{math}
:enumerated: false
\text{MSE}
=
SE^2 + \text{Bias}^2
```

where:

* $SE$ = standard forecast error
* Bias = mean forecast error

```{admonition} Key Insight
A forecast can perform poorly because:
- it is systematically biased,
- or because its errors are highly variable,
- or both.
```

---

# 16.13 Theil’s $U_1$ Statistic

Theil’s $U_1$ is a normalized measure of forecast accuracy.

One common version is:

```{math}
:enumerated: false
U_1
=
\frac{
\sqrt{
\frac{1}{T}
\sum
(\hat{x}_t - x_t)^2
}
}{
\sqrt{
\frac{1}{T}
\sum
\hat{x}_t^2
}
+
\sqrt{
\frac{1}{T}
\sum
x_t^2
}
}
```

```{admonition} Interpretation
Theil’s $U_1$ scales forecast errors relative to the magnitudes of the actual and forecast series.
```

## Properties

```{admonition} Properties of $U_1$
- $0 \le U_1 \le 1$
- $U_1 = 0$ implies a perfect forecast
- smaller values imply better forecasts
```

## Thai Inflation Example

| Measure       | Forecast 1 | Forecast 2 |
| ------------- | ---------- | ---------- |
| Theil’s $U_1$ | 0.084      | 0.071      |

Forecast 2 performs slightly better.

---

# 16.14 Theil’s $U_2$ Statistic

Theil’s $U_2$ compares a forecast against a benchmark forecast.

Usually the benchmark is a naive forecast:

```{math}
:enumerated: false
\hat{x}_t = x_{t-1}
```

For example:

* tomorrow equals today
* next month equals this month

## Definition

A common form is:

```{math}
:enumerated: false
U_2
=
\sqrt{
\frac{
\frac{1}{T}
\sum 
\left(
\frac{\hat{x}_t - x_t}{x_{t-1}}
\right)^2
}{
\frac{1}{T}
\sum 
\left(
\frac{x_t - x_{t-1}}{x_{t-1}}
\right)^2
}
}
```

## Interpretation

```{admonition} Key Interpretation
$U_2$ compares your forecasting model against a naive “no-change” forecast.
```

| Value     | Interpretation             |
| --------- | -------------------------- |
| $U_2 < 1$ | model beats naive forecast |
| $U_2 = 1$ | same as naive forecast     |
| $U_2 > 1$ | worse than naive forecast  |

## Thai Inflation Example

| Measure       | Forecast 1 | Forecast 2 |
| ------------- | ---------- | ---------- |
| Theil’s $U_2$ | 0.94       | 1.00       |

Forecast 1 slightly outperforms the naive benchmark.

Forecast 2 performs roughly the same as the naive forecast.

---

# 16.15 Forecast Error Decomposition

Theil also proposed decomposing forecast errors into components.

A common decomposition separates MSE into:

* bias proportion
* variance proportion
* covariance proportion

## Bias Proportion

Measures systematic differences between forecast mean and actual mean.

```{admonition} Interpretation
High bias proportion means the model systematically overpredicts or underpredicts.
```

## Variance Proportion

Measures differences in variability.

```{admonition} Interpretation
High variance proportion means the forecast fluctuates too much or too little relative to the actual series.
```

## Covariance Proportion

Measures unsystematic error.

```{admonition} Important
A high covariance proportion is generally desirable because it means most forecast errors are random rather than systematic.
```

## Thai Inflation Example

### Forecast 1

| Component             | Value |
| --------------------- | ----- |
| Bias proportion       | 0.076 |
| Variance proportion   | 0.040 |
| Covariance proportion | 0.885 |

Most errors are unsystematic.

This is generally a good sign.

### Forecast 2

| Component             | Value |
| --------------------- | ----- |
| Bias proportion       | 0.094 |
| Variance proportion   | 0.300 |
| Covariance proportion | 0.624 |

Forecast 2 has a larger variance component.

This suggests that the model may not adequately capture changes in volatility or variability.

---

# 16.16 Understanding the Sources of Forecast Error

An alternative way to understand forecast errors is to decompose MSE into three components:

```{math}
:enumerated: false
\text{MSE}
=
\text{BIAS}^2
+
(s_x - s_{\hat x})^2
+
2(1-r)s_x s_{\hat x}
```

where:

- $s_x$ is the standard deviation of the actual series
- $s_{\hat x}$ is the standard deviation of the forecast series
- $r$ is the correlation between actual and forecast values

```{admonition} Interpretation
Forecast errors can arise because:

1. the forecast is systematically biased,
2. the forecast has the wrong variability,
3. the forecast fails to move together with the actual series.
```

## Bias Component

The first term:

```{math}
:enumerated: false
\text{BIAS}^2
```

captures systematic overprediction or underprediction.

## Variability Component

The second term:

```{math}
:enumerated: false
(s_x - s_{\hat x})^2
```

measures whether the forecast is too volatile or too smooth relative to the actual data.

A forecast may correctly predict the average level while still failing to match the fluctuations of the series.

## Covariance Component

The final term:

```{math}
:enumerated: false
2(1-r)s_x s_{\hat x}
```

captures failure to track movements in the actual series.

If the correlation between forecasts and actual values is high, this component becomes small.

```{admonition} Key Insight
A good forecast should not only predict the correct average level.

It should also move together with the actual data over time.
```

---

# 16.17 Choosing Between Forecasts

Which forecast is “best”?

There is no universal answer.

Different measures emphasize different aspects of performance.

| Criterion     | Forecast 1 Better? | Forecast 2 Better? |
| ------------- | ------------------ | ------------------ |
| MSE           |                    | ✓                  |
| RMSE          |                    | ✓                  |
| MAE           | ✓                  |                    |
| MAPE          | ✓                  |                    |
| Theil’s $U_1$ |                    | ✓                  |
| Theil’s $U_2$ | ✓                  |                    |

```{admonition} Practical Lesson
Forecast evaluation always depends on the decision problem and loss function.
```

---

# 16.18 Why Forecast Evaluation Matters

Forecast evaluation is central in:

* monetary policy
* financial trading
* inventory management
* energy demand forecasting
* macroeconomic planning

A model that performs well historically may fail during:

* crises
* structural breaks
* regime changes
* periods of unusual volatility

```{admonition} Important
Forecasting is not only about models.

It is also about continuously evaluating and revising models.
```

---

# 16.19 Python Example: Comparing Forecast Accuracy

```{code-cell} python
import numpy as np
import pandas as pd

actual = np.array([2.11,2.45,2.62,2.35,2.16,2.09,1.75,1.48,1.26,0.60])

f1 = np.array([1.96,2.31,3.61,2.45,2.01,1.99,1.81,1.65,1.21,1.13])

f2 = np.array([1.96,2.11,2.45,2.62,2.35,2.16,2.09,1.75,1.48,1.26])

e1 = actual - f1
e2 = actual - f2

def forecast_stats(errors, actual):
    
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors/actual))*100
    bias = np.mean(errors)
    
    return pd.Series({
        "Bias": bias,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape
    })

results = pd.DataFrame({
    "Forecast 1": forecast_stats(e1, actual),
    "Forecast 2": forecast_stats(e2, actual)
})

print(results.round(3))
```

``` verbatim
      Forecast 1  Forecast 2
Bias      -0.126      -0.136
MSE        0.138       0.095
RMSE       0.372       0.309
MAE        0.244       0.268
MAPE      17.381      21.624
```

---

# 16.20 Gretl Example: Forecast Evaluation

GRETL provides several forecast evaluation tools.

After generating forecasts:

`Model window → Analysis → Forecast evaluation`

depending on the GRETL version.

GRETL may report:

* MSE
* RMSE
* MAE
* MAPE
* Theil statistics

---

```markdown
[GRETL Screenshot Placeholder: Forecast evaluation statistics]
```

---

## Comparing Competing Models

You can compare:

* AR models
* ARIMA models
* VAR forecasts
* naive forecasts

using the same forecast sample.

```{admonition} Important
Forecast comparisons should always use the same evaluation period.
```

---

# 16.21 Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Evaluating forecasts in-sample only**  
Good fit does not guarantee good forecasting.

**2. Using only one evaluation measure**  
Different measures capture different aspects of performance.

**3. Ignoring benchmark forecasts**  
Simple naive forecasts are often surprisingly difficult to beat.

**4. Ignoring structural breaks**  
Forecast performance may change over time.

**5. Using MAPE when data contain zeros or near-zeros**  
Percentage errors can become unstable.
```

---

# 16.22 Looking Ahead

So far, we have focused mainly on forecasting individual time series.

In the next part of the book, we move to relationships between time series.

We will study:

* spurious regression
* dynamic relationships
* Granger causality
* cointegration
* error correction models

# Key Takeaways

```{admonition} Summary
- Forecast evaluation compares predictions with actual outcomes.
- Forecast errors measure the gap between forecasts and realizations.
- MSE and RMSE penalize large errors heavily.
- MAE measures average absolute forecast errors.
- MAPE expresses errors in percentage terms.
- Theil’s $U_1$ measures normalized forecast accuracy.
- Theil’s $U_2$ compares forecasts against a naive benchmark.
- Forecast error decomposition separates systematic and unsystematic errors.
- Different evaluation measures may rank forecasts differently.
```

