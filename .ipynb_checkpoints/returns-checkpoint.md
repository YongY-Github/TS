---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# All about returns

This section introduces simple returns and log returns, shows how to compute them in Python with pandas, and explains why log returns are so handy for compounding and time-additivity.

We’ll use numpy and pandas, and set a nice number format for tables.

```{code-cell} python
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.5f}'.format
```

Let's loook at a small, minimal example.

```{code-cell} python
price = [100, 80, 100]
year = [2020, 2021, 2022]
df = pd.DataFrame({"price": price}, index=year)
df
```

The price drops from 100 to 80 (−20%), then rebounds to 100 (+25%). 
Notice already that −20% and +25% do not cancel out to zero if you just average them arithmetically—this is the key motivation for geometric compounding and log returns.

## Simple (artithmetic) returns

For a price $P_t$, the **simple return** is

$$
R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1.
$$

```{code-cell} python
df["rtn"] = df["price"].pct_change()  # (p_t / p_{t-1}) - 1
df
```

In this series, 2021 has $r_{2021} = -0.20$ and 2022 has $r_{2022} = +0.25$.

## Log returns

The **log return** often denoted $r_t$ (a.k.a. continuously compounded return) is:

$$
log(1 + R_t) = \log\left(\frac{P_t}{P_{t-1}}\right)
$$

```{code-cell} python
df["log_rtn"] = np.log(df["price"] / df["price"].shift(1))
df[["rtn","log_rtn"]]
```

> **Why logs?** Logs convert multiplication into addition. That turns compounded growth over time into a simple **sum** of log returns.

## Why you can’t “just average” simple returns

If you naïvely sum simple returns over time, you don’t get the correct multi-period return. For two periods (1) and (2):

:::{math}
:enumerated: false
(1+R_1)(1+R_2) - 1 \neq R_1 + R_2 \quad \text{(in general)}
:::

```{code-cell} python
df[["rtn","log_rtn"]].cumsum()         # running sum (for illustration)
df[["rtn","log_rtn"]].mean()           # arithmetic mean (not time-additive)
```

Instead, use **geometric compounding**:

```{code-cell} python
(1 - 0.20) * (1 + 0.25) - 1   # equals 0.0
```

> **Key Takeaway:** A −20% then +25% sequence returns you exactly to the starting price (net 0%), but (-0.20 + 0.25 = 0.05) is **not** the correct multi-period return. Simple returns don’t “sum nicely.”

## Compounding and time-additivity with logs

For (T) periods, the compounded gross return is

$$
\prod_{t=1}^T (1 + R_t) = \frac{P_T}{P_0}
$$

Taking logs,

$$
\sum_{t=1}^T \log(1+R_t) = \log\left(\frac{P_T}{P_0}\right) = \log P_T - \log P_0
$$

> **Time-additivity:** This is the killer feature - **log returns add over time**! That’s why they’re convenient for analytics, modeling, and comparing horizons.

## A slightly longer example

Let’s extend the series and compute both return types.

```{code-cell} python
price = [100, 80, 100, 115, 125]
year = [2020, 2021, 2022, 2023, 2024]
df = pd.DataFrame({"price": price}, index=year)

df["rtn"] = df["price"].pct_change()
df["log_rtn"] = np.log(df["price"] / df["price"].shift(1))
df
```

### Cumulative (geometric) simple return

To get the total return from start to any time $t$, compound the simple returns:

```{code-cell} python
cum_gross = (df["rtn"] + 1).cumprod()
print(cum_gross)
# subtract 1 if you prefer the cumulative percentage return
```

### Cumulative log return

Cumulative log return should equal $\log P_t - \log P_0$:

```{code-cell} python
cum_log = np.log(df["rtn"] + 1).cumsum()
cum_log

# Check at the end:
np.log(df["price"].iloc[-1]) - np.log(df["price"].iloc[0])  # ≈ 0.22314
```

> **Consistency check:** The final cumulative log return is $\log(125) - \log(100) \approx 0.22314$. Exponentiating brings you back to the geometric gross return:

```{code-cell} python
np.exp(0.2231435513142097)   # ≈ 1.25  → +25%
```

## Small-return approximation

For small returns, $log(1+R) \approx R$. The code below shows how close this is:

```{code-cell} python
print(f"The log of .1% is {np.log(1.001)}")
print(f"The log of .5% is {np.log(1.005)}")
print(f"The log of 1% is {np.log(1.01)}")
print(f"The log of 5% is {np.log(1.05)}")
print(f"The log of 10% is {np.log(1.1)}")
print(f"The log of 20% is {np.log(1.2)}")
```

> **Rule of thumb:** Below ~5–10% per period, $\log(1+R)$ and $R$ are quite close. As returns get larger (or negative and large in magnitude), the approximation worsens—use logs exactly.

## Practical notes

* **Reporting vs modeling.** Analysts often model with log returns (sums, normal-ish behavior) but report simple percentage returns (more intuitive).
* **Multi-period performance.** Use **geometric compounding** of simple returns or, equivalently, **sum** log returns and then exponentiate:

$$
  \exp\Big(\sum_t r_t \Big) - 1.
$$
  
* **Arithmetic vs geometric means.** Over time, the geometric mean return reflects compounding. The arithmetic mean can overstate multi-period growth when volatility is present.
