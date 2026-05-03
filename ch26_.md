---
title: Part VIII Capstone — Forecasting Volatility and Financial Risk
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Part VIII Capstone — Forecasting Volatility and Financial Risk

In Part VIII, we studied models for time-varying volatility:

- ARCH models,
- GARCH models,
- volatility clustering,
- conditional variance,
- and volatility forecasting.

This capstone applies these ideas to daily financial returns.

```{admonition} Central Question
Can we model and forecast financial market risk?
```

We use SPY, an ETF tracking the S&P 500.

---

# Exercise 1 — Download Price Data

```{code-cell} python
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

spy = yf.download(
    "SPY",
    start="2015-01-01",
    auto_adjust=False
)

prices = spy["Adj Close"].squeeze()

prices.plot(figsize=(10,4))

plt.title("SPY Adjusted Closing Prices")
plt.ylabel("Price")
plt.xlabel("Date")

plt.savefig("figs/ch26_/SPY.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![SPY](figs/ch26_/SPY.png)

---

# Exercise 2 — Compute Log Returns

```{code-cell} python
returns = 100 * np.log(
    prices / prices.shift(1)
)

returns = returns.dropna()

returns.plot(figsize=(10,4))

plt.title("SPY Daily Log Returns")
plt.ylabel("Return (%)")
plt.xlabel("Date")

plt.savefig("figs/ch26_/rtn.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![SPY](figs/ch26_/rtn.png)


```{admonition} Observation
Large returns tend to cluster together.

This is the visual signature of volatility clustering.
```

---

# Exercise 3 — Rolling Volatility

```{code-cell} python
rolling_vol = returns.rolling(30).std()

rolling_vol.plot(figsize=(10,4))

plt.title("30-Day Rolling Volatility")
plt.ylabel("Volatility")
plt.xlabel("Date")

plt.savefig("figs/ch26_/rol_vol.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![SPY](figs/ch26_/rol_vol.png)

---

# Exercise 4 — Testing for ARCH Effects

```{code-cell} python
from statsmodels.stats.diagnostic import het_arch

arch_test = het_arch(
    returns,
    nlags=10
)

print("LM Statistic:", arch_test[0])
print("LM Test p-value:", arch_test[1])
print("F Statistic:", arch_test[2])
print("F Test p-value:", arch_test[3])
```

``` verbatim
LM Statistic: 848.0078628375993
LM Test p-value: 9.790554714049502e-176
F Statistic: 120.46872866846164
F Test p-value: 1.6667139182935798e-209
```

```{admonition} Interpretation
A small p-value suggests evidence of ARCH effects.

This means volatility is not constant through time.
```

---

# Exercise 5 — Install and Load the ARCH Package

````{admonition} Package Note
The `arch_model` function comes from the external Python package `arch`.

If needed, run the installation command below once.

```{code-cell} python
!pip install arch
```

````

---

# Exercise 6 — Estimate a GARCH(1,1) Model

```{code-cell} python
from arch import arch_model

garch_model = arch_model(
    returns,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    dist="normal"
)

garch_results = garch_model.fit(
    disp="off"
)

print(garch_results.summary())
```

``` verbatim
                     Constant Mean - GARCH Model Results                      
==============================================================================
Dep. Variable:                    SPY   R-squared:                       0.000
Mean Model:             Constant Mean   Adj. R-squared:                  0.000
Vol Model:                      GARCH   Log-Likelihood:               -3671.36
Distribution:                  Normal   AIC:                           7350.72
Method:            Maximum Likelihood   BIC:                           7374.53
                                        No. Observations:                 2848
Date:                Sun, May 03 2026   Df Residuals:                     2847
Time:                        23:25:08   Df Model:                            1
                                Mean Model                                
==========================================================================
                 coef    std err          t      P>|t|    95.0% Conf. Int.
--------------------------------------------------------------------------
mu             0.0851  1.401e-02      6.079  1.208e-09 [5.769e-02,  0.113]
                              Volatility Model                              
============================================================================
                 coef    std err          t      P>|t|      95.0% Conf. Int.
----------------------------------------------------------------------------
omega          0.0405  1.034e-02      3.918  8.926e-05 [2.026e-02,6.081e-02]
alpha[1]       0.1764  2.711e-02      6.508  7.629e-11     [  0.123,  0.230]
beta[1]        0.7907  2.803e-02     28.213 4.059e-175     [  0.736,  0.846]
============================================================================

Covariance estimator: robust
```

---

# Exercise 7 — Plot Conditional Volatility

```{code-cell} python
conditional_vol = garch_results.conditional_volatility

conditional_vol.plot(figsize=(10,4))

plt.title("Estimated GARCH Conditional Volatility")
plt.ylabel("Volatility")
plt.xlabel("Date")

plt.savefig("figs/ch26_/garch11.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Garch11](figs/ch26_/garch11.png)


```{admonition} Key Idea
GARCH models estimate volatility as a time-varying process.
```

---

# Exercise 8 — Compare Returns and Conditional Volatility

```{code-cell} python
fig, ax1 = plt.subplots(figsize=(10,5))

ax1.plot(
    returns,
    label="Returns",
    alpha=0.6
)

ax1.set_ylabel("Return (%)")

ax2 = ax1.twinx()

ax2.plot(
    conditional_vol,
    label="Conditional Volatility",
    linestyle="--"
)

ax2.set_ylabel("Volatility")

plt.title("Returns and GARCH Conditional Volatility")

plt.savefig("figs/ch26_/garch11_r.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![Garch11](figs/ch26_/garch11_r.png)

---

# Exercise 9 — Forecast Volatility

```{code-cell} python
forecast = garch_results.forecast(
    horizon=10
)

variance_forecast = forecast.variance.iloc[-1]

vol_forecast = np.sqrt(variance_forecast)

vol_forecast
```

``` verbatim
h.01    0.708746
h.02    0.725500
h.03    0.741344
h.04    0.756351
h.05    0.770588
h.06    0.784110
h.07    0.796971
h.08    0.809214
h.09    0.820882
h.10    0.832011
```

---

# Exercise 10 — Plotting Recent Volatility and the Forecast

We now compare:

- recent estimated volatility,
- and the 10-day ahead volatility forecast.

This helps visualize how GARCH models project volatility into the future.

---

```{code-cell} python
# ==========================================
# Extract recent volatility history
# ==========================================

recent_vol = conditional_vol.iloc[-100:]

# ==========================================
# Forecast future volatility
# ==========================================

forecast = garch_results.forecast(
    horizon=10
)

forecast_var = forecast.variance.iloc[-1]

forecast_vol = np.sqrt(forecast_var)

# ==========================================
# Construct forecast index
# ==========================================

forecast_index = range(
    len(recent_vol),
    len(recent_vol) + len(forecast_vol)
)

# ==========================================
# Plot
# ==========================================

plt.figure(figsize=(10,5))

# Historical volatility
plt.plot(
    range(len(recent_vol)),
    recent_vol,
    label="Historical Volatility",
    linewidth=2
)

# Forecast volatility
plt.plot(
    forecast_index,
    forecast_vol,
    label="Forecast Volatility",
    linewidth=2,
    linestyle="--"
)

plt.title("Recent and Forecast GARCH Volatility")

plt.ylabel("Volatility")

plt.xlabel("Time")

plt.legend()

plt.show()
```

---

```{admonition} Interpretation
Notice how the volatility forecast gradually evolves from recent volatility conditions.

GARCH models typically forecast volatility to revert gradually toward a longer-run level over time.
```

---

# Exercise 11 — Simple Value-at-Risk

We compute a simple one-day 5% Value-at-Risk.

```{code-cell} python
latest_vol = conditional_vol.iloc[-1]

var_5 = -1.645 * latest_vol

print("One-day 5% VaR:", var_5)
```

```
One-day 5% VaR: -1.2482842037087853
```

```{admonition} Interpretation
A one-day 5% VaR gives an estimate of a large negative return that may be exceeded with probability 5%.
```

---

# Exercise 12 — Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Confusing volatility forecasts with return forecasts**  
GARCH forecasts risk, not market direction.

**2. Ignoring volatility clustering**  
Financial risk changes through time.

**3. Treating GARCH results as guaranteed predictions**  
Volatility forecasts are model-based estimates.

**4. Ignoring fat tails**  
Extreme losses may occur more often than normal models predict.

**5. Ignoring structural breaks**  
Volatility dynamics may change during crises.
```

---

# Key Takeaways

```{admonition} Summary
- Financial returns often display volatility clustering.
- ARCH tests detect time-varying volatility.
- GARCH models estimate conditional volatility.
- Conditional volatility rises during market stress.
- GARCH models can forecast future volatility.
- Volatility forecasts are useful for risk management.
- VaR connects volatility modeling to financial risk.
```
````


