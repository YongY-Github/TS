---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Part VI Capstone — Spurious Regression, Cointegration, and Dynamic Relationships

In Part VI, we studied how relationships between time series can become substantially more complicated when variables are:

- persistent,
- trending,
- and nonstationary.

We introduced:

- spurious regression,
- dynamic models,
- Granger causality,
- cointegration,
- and error correction models (ECMs).

We learned that standard regression tools may become misleading when applied directly to nonstationary time series.

At the same time, we also saw that some nonstationary variables may still share meaningful long-run equilibrium relationships.

This capstone integrates these ideas through two applied case studies:

- a macroeconomic example using U.S. and Mexico GDP,
- and a financial example using international ETFs.

```{admonition} Central Question
How can we distinguish between spurious relationships and genuine long-run equilibrium relationships in time series data?
```

The emphasis is practical and intuition-first.

We focus on:

- diagnosing nonstationarity,
- identifying spurious regression,
- testing for cointegration,
- estimating dynamic relationships,
- constructing spreads,
- and interpreting equilibrium adjustment.

---

# Learning Goals

By completing this capstone, you should be able to:

- recognize the dangers of spurious regression
- test for unit roots using the ADF test
- distinguish between stationary and nonstationary relationships
- perform Engle–Granger cointegration tests
- interpret long-run equilibrium relationships
- construct and analyze spreads
- estimate simple error correction models
- understand the logic of pairs trading
- distinguish short-run dynamics from long-run equilibrium adjustment
- interpret empirical time series relationships carefully

---

# Case A — Does U.S. GDP Help Explain Mexico GDP?

# Exercise 1 — Download Real GDP Data

We begin by downloading quarterly real GDP data for:

- the United States,
- and Mexico.

We use the FRED database through `pandas_datareader`.

````{admonition} Practical Note
You may need to install:

```bash
pip install pandas_datareader
```
````

---

# Downloading GDP Data from FRED

```{code-cell} python
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

usa_gdp = web.DataReader(
    "GDPC1",
    "fred",
    start="1995-01-01"
)

mex_gdp = web.DataReader(
    "NGDPRSAXDCMXQ",
    "fred",
    start="1995-01-01"
)

usa_gdp.columns = ["USA_GDP"]
mex_gdp.columns = ["MEXICO_GDP"]

gdp = pd.concat(
    [usa_gdp, mex_gdp],
    axis=1
).dropna()

gdp.head()
```

``` verbatum
| DATE       | USA_GDP   | MEXICO_GDP |
|------------|-----------|------------|
| 1995-01-01 | 11319.951 | 3519803.5  |
| 1995-04-01 | 11353.721 | 3306332.9  |
| 1995-07-01 | 11450.310 | 3377180.2  |
| 1995-10-01 | 11528.067 | 3454404.3  |
| 1996-01-01 | 11614.418 | 3534597.8  |
...
```

---

```{admonition} Observation
Both GDP series display strong upward trends through time.

This immediately raises an important econometric question:

Are these relationships genuine, or are they partly driven by common trends?
```

---

# Exercise 2 — Plot GDP Levels

```{code-cell} python
fig, ax1 = plt.subplots(figsize=(10,5))

# ==========================================
# USA GDP
# ==========================================

ax1.plot(
    gdp.index,
    gdp["USA_GDP"],
    linewidth=2,
    label="USA Real GDP"
)

ax1.set_ylabel("USA Real GDP")

# ==========================================
# Mexico GDP
# ==========================================

ax2 = ax1.twinx()

ax2.plot(
    gdp.index,
    gdp["MEXICO_GDP"],
    linewidth=2,
    linestyle="--",
    label="Mexico Real GDP"
)

ax2.set_ylabel("Mexico Real GDP")

# ==========================================
# Title
# ==========================================

plt.title("USA and Mexico Real GDP")

# ==========================================
# Combined legend
# ==========================================

lines1, labels1 = ax1.get_legend_handles_labels()

lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper left"
)

plt.savefig("figs/ch21_/USA_Mexico.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![USA Mexico real GDP](figs/ch21_/USA_Mexico.png)

---

# Questions

1. Do the two GDP series appear to move together?
2. Do both series appear nonstationary?
3. Could a regression in levels produce misleading results?

---

## Exercise 3 — A Naïve Levels Regression

Suppose we regress Mexico real GDP on U.S. real GDP.

```{code-cell} python
import statsmodels.api as sm

# ==========================================
# Regression variables
# ==========================================

y = gdp["MEXICO_GDP"]

X = gdp["USA_GDP"]

X = sm.add_constant(X)

# ==========================================
# Estimate regression
# ==========================================

model = sm.OLS(y, X).fit()

print(model.summary())
```

``` verbatim
                            OLS Regression Results                            
==============================================================================
Dep. Variable:             MEXICO_GDP   R-squared:                       0.947
Model:                            OLS   Adj. R-squared:                  0.946
Method:                 Least Squares   F-statistic:                     2169.
Date:                Sun, 03 May 2026   Prob (F-statistic):           1.50e-79
Time:                        19:52:53   Log-Likelihood:                -1681.8
No. Observations:                 124   AIC:                             3368.
Df Residuals:                     122   BIC:                             3373.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        1.07e+06   8.87e+04     12.067      0.000    8.95e+05    1.25e+06
USA_GDP      234.0011      5.024     46.572      0.000     224.055     243.948
==============================================================================
Omnibus:                        8.847   Durbin-Watson:                   0.243
Prob(Omnibus):                  0.012   Jarque-Bera (JB):                8.701
Skew:                          -0.572   Prob(JB):                       0.0129
Kurtosis:                       3.614   Cond. No.                     9.21e+04
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 9.21e+04. This might indicate that there are
strong multicollinearity or other numerical problems.
```

---

## Questions

1. Is the estimated relationship statistically significant?

2. Is the \(R^2\) large?

3. Does this necessarily imply a genuine economic relationship?

4. Why might trending variables create misleading regressions?

---

````{admonition} Important
Two unrelated trending variables can produce:
- high \(R^2\),
- significant t-statistics,
- and apparently strong relationships.

This phenomenon is called:

```text
spurious regression
```
````

---

## Visualizing the Fitted Relationship

```{code-cell} python
plt.figure(figsize=(8,5))

# Scatter plot
plt.scatter(
    gdp["USA_GDP"],
    gdp["MEXICO_GDP"],
    alpha=0.7
)

# Fitted regression line
plt.plot(
    gdp["USA_GDP"],
    model.fittedvalues,
    linewidth=2
)

plt.title("Mexico GDP vs USA GDP")

plt.xlabel("USA Real GDP")

plt.ylabel("Mexico Real GDP")

plt.savefig(
    "figs/ch21_/corr.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
```

![USA Mexico real GDP Correlation](figs/ch21_/corr.png)

---

```{admonition} Observation
Strong visual relationships between trending variables may still be statistically misleading.
```

---

# Exercise 4 — Testing for Unit Roots

We now investigate whether the GDP series are stationary.

This is crucial because regressions involving nonstationary variables may be misleading.

We use the:

```{admonition} Definition
Augmented Dickey-Fuller (ADF) test.
```

---

```{admonition} Important
The null hypothesis of the ADF test is:

The series contains a unit root.
```

A large p-value means we fail to reject nonstationarity.

---

## ADF Test for USA GDP

```{code-cell} python
from statsmodels.tsa.stattools import adfuller

adf_usa = adfuller(
    gdp["USA_GDP"]
)

print("ADF Statistic:", adf_usa[0])

print("p-value:", adf_usa[1])
```

``` verbatim
ADF Statistic: 0.32391792020851595
p-value: 0.9784156046782101
```

---

## ADF Test for Mexico GDP

```{code-cell} python
adf_mex = adfuller(
    gdp["MEXICO_GDP"]
)

print("ADF Statistic:", adf_mex[0])

print("p-value:", adf_mex[1])
```

``` verbatim
ADF Statistic: -1.5738505359705157
p-value: 0.49671453217914335
```

---

## Questions

1. Are the p-values small or large?

2. Do we reject the unit root null?

3. Do the GDP series appear stationary?

4. Why might macroeconomic variables often contain unit roots?

---

```{admonition} Observation
Macroeconomic variables such as GDP often display:
- persistence,
- trends,
- and nonstationarity.
```

---

# Exercise 5 — Differencing the Data

We now difference the GDP series.

```{code-cell} python
gdp_diff = gdp.diff().dropna()

gdp_diff.head()
```

``` verbatim
| DATE       | USA_GDP  | MEXICO_GDP  |
|------------|----------|-------------|
| 1995-04-01 | 33.770   | -213470.6   |
| 1995-07-01 | 96.589   | 70847.3     |
| 1995-10-01 | 77.757   | 77224.1     |
| 1996-01-01 | 86.351   | 80193.5     |
| 1996-04-01 | 193.722  | 37988.0     |
...
```

---

## Plotting GDP Differences

```{code-cell} python
fig, ax1 = plt.subplots(figsize=(10,5))

# ==========================================
# USA GDP Growth
# ==========================================

ax1.plot(
    gdp_diff.index,
    gdp_diff["USA_GDP"],
    linewidth=2,
    label="USA GDP Difference"
)

ax1.set_ylabel("USA GDP Difference")

# ==========================================
# Mexico GDP Growth
# ==========================================

ax2 = ax1.twinx()

ax2.plot(
    gdp_diff.index,
    gdp_diff["MEXICO_GDP"],
    linewidth=2,
    linestyle="--",
    label="Mexico GDP Difference"
)

ax2.set_ylabel("Mexico GDP Difference")

# ==========================================
# Title
# ==========================================

plt.title("Differenced GDP Series")

# ==========================================
# Combined Legend
# ==========================================

lines1, labels1 = ax1.get_legend_handles_labels()

lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper left"
)

plt.savefig("figs/ch21_/diff.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![USA Mexico real GDP Difference](figs/ch21_/diff.png)

---

```{admonition} Observation
Differencing removes much of the common trend visible in the original series.
```

---

# Exercise 6 — ADF Tests on Differenced GDP

We now test whether the differenced series are stationary.

---

## USA GDP Differences

```{code-cell} python
adf_usa_diff = adfuller(
    gdp_diff["USA_GDP"]
)

print("ADF Statistic:", adf_usa_diff[0])

print("p-value:", adf_usa_diff[1])
```

``` verbatim
ADF Statistic: -13.278665554999852
p-value: 7.772135998540502e-25
```

---

## Mexico GDP Differences

```{code-cell} python
adf_mex_diff = adfuller(
    gdp_diff["MEXICO_GDP"]
)

print("ADF Statistic:", adf_mex_diff[0])

print("p-value:", adf_mex_diff[1])
```

``` verbatim
ADF Statistic: -9.874336285173483
p-value: 3.9187214400492124e-17
```

---

## Questions

1. Are the differenced series more stationary?

2. How do the p-values compare with the level series?

3. Why does differencing often help remove unit roots?

---

````{admonition} Key Idea
Many macroeconomic variables are integrated of order one:

```{math}
:enumerated: false
I(1)
```

meaning they become stationary after first differencing.
````

---

# Exercise 7 — Regression in Differences

We now estimate a regression using differenced GDP.

```{code-cell} python
import statsmodels.api as sm

y_diff = gdp_diff["MEXICO_GDP"]

X_diff = gdp_diff["USA_GDP"]

X_diff = sm.add_constant(X_diff)

diff_model = sm.OLS(
    y_diff,
    X_diff
).fit()

print(diff_model.summary())
```

``` verbatim
                            OLS Regression Results                            
==============================================================================
Dep. Variable:             MEXICO_GDP   R-squared:                       0.760
Model:                            OLS   Adj. R-squared:                  0.758
Method:                 Least Squares   F-statistic:                     383.3
Date:                Sun, 03 May 2026   Prob (F-statistic):           2.58e-39
Time:                        19:52:54   Log-Likelihood:                -1539.9
No. Observations:                 123   AIC:                             3084.
Df Residuals:                     121   BIC:                             3089.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const      -3.059e+04   6627.436     -4.616      0.000   -4.37e+04   -1.75e+04
USA_GDP      524.0910     26.768     19.579      0.000     471.097     577.085
==============================================================================
Omnibus:                        8.204   Durbin-Watson:                   2.057
Prob(Omnibus):                  0.017   Jarque-Bera (JB):               10.558
Skew:                          -0.370   Prob(JB):                      0.00510
Kurtosis:                       4.230   Cond. No.                         273.
==============================================================================
```

---

## Questions

1. How does this regression differ from the levels regression?

2. Is the relationship weaker or stronger?

3. Why might differenced regressions be more reliable statistically?

---

```{admonition} Important
Differencing may reduce the risk of spurious regression by removing common trends.
```

---

# Exercise 8 — Dynamic Interpretation

Even if GDP levels are nonstationary, changes in U.S. GDP may still influence changes in Mexico GDP.

This creates a more meaningful interpretation:

```{admonition} Interpretation
Short-run changes in U.S. economic activity may affect short-run changes in Mexican economic activity.
```

Possible channels include:

- trade,
- manufacturing supply chains,
- exports,
- investment,
- tourism,
- and financial conditions.

---

## Looking Ahead

We now face an important question:

```{admonition} Central Question
Do the GDP series still share a meaningful long-run equilibrium relationship?
```

This leads naturally to:

- cointegration,
- error correction models,
- and dynamic adjustment.

We study these next.

---

# Case B — Cointegration and Pairs Trading

# Exercise 9 — Download ETF Price Data

We now examine two international equity ETFs:

- EWA — Australia ETF
- EWC — Canada ETF

These economies share several structural similarities:

- resource dependence,
- commodity exposure,
- sensitivity to global growth,
- and strong integration into global financial markets.

This makes them plausible candidates for long-run co-movement.

```{code-cell} python
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

ewa = yf.download(
    "EWA",
    start="2015-01-01",
    auto_adjust=False
)

ewc = yf.download(
    "EWC",
    start="2015-01-01",
    auto_adjust=False
)

ewa_prices = ewa["Adj Close"].squeeze()

ewc_prices = ewc["Adj Close"].squeeze()

etf = pd.concat(
    [ewa_prices, ewc_prices],
    axis=1
)

etf.columns = [
    "EWA",
    "EWC"
]

etf = etf.dropna()

etf.head()
```

```
| Date       | EWA       | EWC       |
|------------|-----------|-----------|
| 2015-01-02 | 13.892484 | 22.766466 |
| 2015-01-05 | 13.760471 | 22.155445 |
| 2015-01-06 | 13.703897 | 21.830097 |
| 2015-01-07 | 13.829621 | 21.853903 |
| 2015-01-08 | 14.011924 | 22.123701 |
...
```

## Exercise 10 — Plot the Two Price Series

```{code-cell} python
indexed = 100 * etf / etf.iloc[0]

indexed.plot(figsize=(10,5))

plt.title("EWA and EWC Indexed Prices")

plt.ylabel("Index (Start = 100)")

plt.savefig("figs/ch21_/ewa_ewc.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![EWA EWC](figs/ch21_/ewa_ewc.png)

# Exercise 11 — Testing ETF Prices for Unit Roots

Before testing for cointegration, we must first determine whether the ETF price series are nonstationary.

We use the:

```{admonition} Definition
Augmented Dickey-Fuller (ADF) test.
```

---

## ADF Test for EWA

```{code-cell} python
from statsmodels.tsa.stattools import adfuller

adf_ewa = adfuller(
    etf["EWA"]
)

print("ADF Statistic:", adf_ewa[0])

print("p-value:", adf_ewa[1])
```

``` verbatim
ADF Statistic: -0.2498282426693312
p-value: 0.9323156991762888
```

---

## ADF Test for EWC

```{code-cell} python
adf_ewc = adfuller(
    etf["EWC"]
)

print("ADF Statistic:", adf_ewc[0])

print("p-value:", adf_ewc[1])
```

``` verbatim
ADF Statistic: 1.8561991459231966
p-value: 0.9984542009070559
```

---

## Questions

1. Are the ETF price series stationary?

2. Are the p-values large or small?

3. Why are financial price levels often nonstationary?

---

```{admonition} Important
Cointegration analysis usually requires:
- nonstationary individual series,
- but stationary equilibrium relationships.
```

# Exercise 12 — Testing for Cointegration

Both ETF price series appear to be nonstationary.

We now ask whether they share a stable long-run relationship.

```{admonition} Central Question
Are EWA and EWC individually nonstationary but jointly tied together in the long run?
```

We use the Engle–Granger cointegration test.

---

```{code-cell} python
from statsmodels.tsa.stattools import coint

coint_stat, p_value, crit_values = coint(
    etf["EWA"],
    etf["EWC"]
)

print("Cointegration test statistic:", coint_stat)
print("p-value:", p_value)
print("Critical values:", crit_values)
```

``` verbatim
Cointegration test statistic: -2.6696501391297005
p-value: 0.21068363705581145
Critical values: [-3.9002896  -3.33827624 -3.04593952]
```

---

## Interpretation

```{admonition} Important
The null hypothesis of the Engle–Granger test is no cointegration.
```

So:

- small p-value → evidence of cointegration
- large p-value → little evidence of cointegration

---

````{admonition} Important
The Engle–Granger cointegration procedure is asymmetric.

In this example, we estimate:

```{math}
:enumerated: false
EWA_t
=
\alpha
+
\beta EWC_t
+
u_t
```

Reversing the regression direction may produce slightly different residual spreads and cointegration results.
````

---

## Questions

1. Is the p-value small?
2. Do we reject the null of no cointegration?
3. Does the result support the visual impression from the indexed price plot?
4. Why is cointegration important for pairs trading?

---

# Exercise 13 — Cointegration and Sample Periods

Financial relationships may change over time.

We now restrict the sample to:

```text
2015–2024
```

to investigate whether the apparent divergence after 2025 affects the cointegration result.

---

```{code-cell} python
etf_sub = etf.loc[
    :"2024-12-31"
]

etf_sub.head()
```

---

## Plotting the Restricted Sample

```{code-cell} python
indexed_sub = 100 * etf_sub / etf_sub.iloc[0]

indexed_sub.plot(figsize=(10,5))

plt.title("EWA and EWC Indexed Prices (2015–2024)")

plt.ylabel("Index (Start = 100)")

plt.savefig("figs/ch21_/ewa_ewc_.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![EWA EWC](figs/ch21_/ewa_ewc_.png)

---

## Repeating the Cointegration Test

```{code-cell} python
from statsmodels.tsa.stattools import coint

coint_stat, p_value, crit_values = coint(
    etf_sub["EWA"],
    etf_sub["EWC"]
)

print("Cointegration test statistic:", coint_stat)

print("p-value:", p_value)

print("Critical values:", crit_values)
```

``` verbatim
Cointegration test statistic: -3.6441869386563295
p-value: 0.021579983072396378
Critical values: [-3.90079993 -3.33856054 -3.04613679]
```

```{admonition} Interpretation
Using the 2015–2024 sample, the Engle–Granger test provides evidence of cointegration between EWA and EWC at the 5% significance level.

However, the result is sensitive to the sample period.

Including data beyond 2024 weakened the relationship substantially, suggesting possible structural change or divergence in the post-2024 period.
```


## Questions

1. Does the cointegration result change?

2. Why might structural breaks affect cointegration tests?

3. Why are financial relationships sometimes unstable through time?

---

```{admonition} Observation
Cointegration relationships may weaken or disappear when market structure changes.
```

----

# Exercise 14 — Estimating the Long-Run Relationship

Because the Engle–Granger test suggests evidence of cointegration, we now estimate the long-run equilibrium relationship between:

- EWA,
- and EWC.

We estimate:

```{math}
:enumerated: false
EWA_t
=
\alpha
+
\beta EWC_t
+
u_t
```

where:

- $u_t$ represents deviations from long-run equilibrium.

---

```{code-cell} python
import statsmodels.api as sm

# ==========================================
# Regression variables
# ==========================================

y = etf_sub["EWA"]

X = etf_sub["EWC"]

X = sm.add_constant(X)

# ==========================================
# Estimate long-run relationship
# ==========================================

longrun_model = sm.OLS(
    y,
    X
).fit()

print(longrun_model.summary())
```

``` verbatim
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    EWA   R-squared:                       0.961
Model:                            OLS   Adj. R-squared:                  0.961
Method:                 Least Squares   F-statistic:                 6.117e+04
Date:                Sun, 03 May 2026   Prob (F-statistic):               0.00
Time:                        20:51:39   Log-Likelihood:                -2630.8
No. Observations:                2516   AIC:                             5266.
Df Residuals:                    2514   BIC:                             5277.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.6735      0.062     43.093      0.000       2.552       2.795
EWC            0.5507      0.002    247.316      0.000       0.546       0.555
==============================================================================
Omnibus:                       16.078   Durbin-Watson:                   0.039
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               15.978
Skew:                          -0.179   Prob(JB):                     0.000339
Kurtosis:                       2.844   Cond. No.                         126.
==============================================================================
```

---

## Questions

1. Is the estimated relationship statistically significant?

2. What does the slope coefficient imply?

3. Why should we interpret this relationship cautiously despite evidence of cointegration?

---

```{admonition} Important
Cointegration does not imply perfect tracking.

It implies that deviations from equilibrium tend to be temporary rather than permanent.
```

---

# Exercise 15 — Constructing the Spread

The residuals from the long-run regression measure deviations from equilibrium.

We define:

```{math}
:enumerated: false
\hat u_t
=
EWA_t
-
\widehat{\alpha}
-
\widehat{\beta} EWC_t
```

This residual series is often called the:

```{admonition} Definition
Spread.
```

---

```{code-cell} python
spread = longrun_model.resid

spread.head()
```

``` verbatim
| Date       | Value     |
|------------|-----------|
| 2015-01-02 | -1.319356 |
| 2015-01-05 | -1.114858 |
| 2015-01-06 | -0.992248 |
| 2015-01-07 | -0.879634 |
| 2015-01-08 | -0.845926 |
```

---

## Plotting the Spread

```{code-cell} python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))

plt.plot(
    spread,
    linewidth=1.5
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.title("Cointegration Spread: EWA vs EWC")

plt.ylabel("Spread")

plt.savefig("figs/ch21_/spread.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![EWA EWC](figs/ch21_/spread.png)


---

## Questions

1. Does the spread appear mean-reverting?

2. Does the spread fluctuate around zero?

3. Why is mean reversion important for pairs trading?

---

```{admonition} Key Idea
Cointegration implies that the spread should be more stable than the individual price series.
```

---

# Exercise 16 — Testing the Spread for Stationarity

We now test whether the spread itself is stationary.

---

```{code-cell} python
from statsmodels.tsa.stattools import adfuller

spread_adf = adfuller(
    spread
)

print("ADF Statistic:", spread_adf[0])

print("p-value:", spread_adf[1])
```

```
ADF Statistic: -3.6414413662018656
p-value: 0.005016429882813343
```

---

## Questions

1. Is the spread stationary?

2. Why is spread stationarity central to cointegration?

3. Why might stationary spreads create trading opportunities?

---

```{admonition} Observation
A stationary spread suggests temporary deviations from equilibrium rather than permanent divergence.
```

---

# Exercise 17 — Standardizing the Spread

Pairs trading strategies often standardize the spread using a z-score.

---

```{code-cell} python
spread_mean = spread.mean()

spread_std = spread.std()

zscore = (
    spread - spread_mean
) / spread_std

zscore.head()
```

---

## Plotting the Z-Score

```{code-cell} python
plt.figure(figsize=(10,4))

plt.plot(
    zscore,
    linewidth=1.5
)

plt.axhline(
    2,
    linestyle="--",
    linewidth=1
)

plt.axhline(
    -2,
    linestyle="--",
    linewidth=1
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.title("Spread Z-Score")

plt.ylabel("Z-Score")

plt.show()
```

---

```{admonition} Interpretation
Large positive or negative z-scores may indicate unusually large deviations from equilibrium.
```

---

## Questions

1. When does the spread appear unusually high?

2. When does the spread appear unusually low?

3. Why might traders interpret extreme z-scores as temporary mispricing?

---

# Exercise 18 — A Simple Pairs Trading Rule

A very simple rule might be:

| Condition | Action |
|---|---|
| z-score > 2 | short spread |
| z-score < -2 | long spread |
| z-score near 0 | close position |

---

```{admonition} Important
This is only a simplified educational example.

Real statistical arbitrage strategies involve:
- transaction costs,
- risk management,
- leverage,
- execution constraints,
- and changing market conditions.
```

---

# Exercise 19 — Estimating an Error Correction Model (ECM)

We now model short-run changes together with long-run disequilibrium.

---

## Constructing Differences

```{code-cell} python
etf_diff = etf_sub.diff().dropna()

etf_diff.head()
```

``` verbatim
| Date       | EWA      | EWC      |
|------------|----------|----------|
| 2015-01-05 | -0.132009 | -0.611019 |
| 2015-01-06 | -0.056577 | -0.325348 |
| 2015-01-07 |  0.125723 |  0.023808 |
| 2015-01-08 |  0.182301 |  0.269798 |
| 2015-01-09 |  0.132012 | -0.190449 |
...
```

---

# ECM Estimation

We estimate:

```{math}
:enumerated: false
\Delta EWA_t
=
\alpha
+
\beta \Delta EWC_t
+
\lambda \hat u_{t-1}
+
\varepsilon_t
```

where:

- $\hat u_{t-1}$ is the lagged spread.

---

```{code-cell} python
ecm_data = etf_diff.copy()

ecm_data["spread_lag"] = spread.shift(1)

ecm_data = ecm_data.dropna()

y_ecm = ecm_data["EWA"]

X_ecm = ecm_data[
    ["EWC", "spread_lag"]
]

X_ecm = sm.add_constant(X_ecm)

ecm_model = sm.OLS(
    y_ecm,
    X_ecm
).fit()

print(ecm_model.summary())
```

``` verbatim
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    EWA   R-squared:                       0.681
Model:                            OLS   Adj. R-squared:                  0.681
Method:                 Least Squares   F-statistic:                     2680.
Date:                Sun, 03 May 2026   Prob (F-statistic):               0.00
Time:                        22:13:47   Log-Likelihood:                 1508.4
No. Observations:                2515   AIC:                            -3011.
Df Residuals:                    2512   BIC:                            -2993.
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         -0.0005      0.003     -0.200      0.841      -0.006       0.005
EWC            0.6254      0.009     73.139      0.000       0.609       0.642
spread_lag    -0.0201      0.004     -5.217      0.000      -0.028      -0.013
==============================================================================
Omnibus:                      269.386   Durbin-Watson:                   2.328
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2300.948
Skew:                          -0.022   Prob(JB):                         0.00
Kurtosis:                       7.686   Cond. No.                         3.23
==============================================================================
```

---

## Questions

1. Is the error correction coefficient statistically significant?

2. Is the coefficient negative?

3. Why should the error correction term usually be negative?

---

```{admonition} Key Idea
A negative error correction coefficient implies that deviations from equilibrium tend to be corrected over time.
```

---

# Exercise 20 — Economic Interpretation

The ECM combines:

- short-run dynamics,
- and long-run equilibrium adjustment.

This provides a richer interpretation than:

- simple correlation,
- or static regression.

---

## Questions

1. Why is ECM more appropriate than simple regression for cointegrated series?

2. Why is cointegration essential before estimating an ECM?

3. How does the ECM connect finance and time series econometrics?

---

# Synthesis

We now contrast the two cases in this capstone.

| Case | Main Lesson |
|---|---|
| USA–Mexico GDP | trending variables may produce spurious regression |
| EWA–EWC ETFs | nonstationary variables may still share equilibrium relationships |

---

```{admonition} Summary
Not all trending variables are cointegrated.

Cointegration requires:
- nonstationary individual series,
- but stable long-run equilibrium relationships.
```
---

# Synthesis Questions
1. Why can GDP levels produce spurious regression?
2. Why does cointegration change the interpretation?
3. Why is Granger causality not the same as true causality?
4. Why is an ECM appropriate only when cointegration exists?
5. How does pairs trading rely on mean reversion?

---

# Common Mistakes
- Treating high R² in levels as evidence of a real relationship
- Ignoring unit roots
- Using ECM without cointegration
- Treating Granger causality as structural causality
- Backtesting pairs trading without transaction costs

---

# Key Takeaways
- Relationships between time series require careful diagnosis.
- Trending variables can create spurious regression.
- Cointegration allows meaningful long-run relationships among nonstationary variables.
- Dynamic models capture short-run transmission.
- ECMs combine short-run changes with long-run adjustment.

---

# Part C — Pairs Trading with Bollinger Bands on the Spread

In the previous section, we estimated the long-run cointegration relationship:

```{math}
:enumerated: false
EWA_t = \alpha + \beta EWC_t + u_t
```

The residual from this equation is the **spread**:

```{math}
:enumerated: false
u_t = EWA_t - \alpha - \beta EWC_t
```

If the spread is stationary and mean-reverting, unusually large deviations may eventually move back toward equilibrium.

```{admonition} Central Idea
Pairs trading attempts to profit from temporary deviations from the long-run equilibrium relationship.
```

```{admonition} Key Insight
The coefficient $\beta$ from the cointegration equation is used as the hedge ratio in the pairs trading strategy.
```

---

# Exercise 21 — Extracting the Hedge Ratio

We first extract the estimated intercept and hedge ratio from the long-run regression.

```{code-cell} python
alpha = longrun_model.params["const"]

hedge_ratio = longrun_model.params["EWC"]

print("Alpha:", alpha)

print("Hedge ratio:", hedge_ratio)
```

``` verbatim
Alpha: 2.673501122389125
Hedge ratio: 0.5507372302051009
```

---

# Exercise 22 — Constructing the Cointegration Spread

The spread is:

```{math}
:enumerated: false
Spread_t = EWA_t - \alpha - \beta EWC_t
```

```{code-cell} python
spread = (
    etf_sub["EWA"]
    - alpha
    - hedge_ratio * etf_sub["EWC"]
)

spread.plot(figsize=(10,4))

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.title("Cointegration Spread: EWA vs EWC")

plt.ylabel("Spread")

plt.savefig(
    "figs/ch21_/spread_.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig("figs/ch21_/spread__.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![EWA EWC](figs/ch21_/spread__.png)

---

```{admonition} Interpretation
When the spread is above zero, EWA is high relative to its long-run relationship with EWC.

When the spread is below zero, EWA is low relative to its long-run relationship with EWC.
```

---

# Exercise 23 — Bollinger Bands and Entry Signals on the Spread

We now apply Bollinger Bands directly to the cointegration spread.

The bands help identify when the spread is unusually far from its recent average.

```{code-cell} python
window = 20

spread_mean = spread.rolling(window).mean()

spread_std = spread.rolling(window).std()

upper_band = spread_mean + 2 * spread_std

lower_band = spread_mean - 2 * spread_std
```

---

## Trading Rule

| Spread condition | Interpretation | Position |
|---|---|---|
| spread < lower band | EWA is relatively cheap | Long spread |
| spread > upper band | EWA is relatively expensive | Short spread |
| spread returns near mean | equilibrium restored | Close position |

---

## Long Spread

If:

```{math}
:enumerated: false
Spread_t < LowerBand_t
```

then EWA is relatively cheap.

```{admonition} Long Spread
Buy 1 share of EWA.

Sell $\beta$ shares of EWC.
```

---

## Short Spread

If:

```{math}
:enumerated: false
Spread_t > UpperBand_t
```

then EWA is relatively expensive.

```{admonition} Short Spread
Sell 1 share of EWA.

Buy $\beta$ shares of EWC.
```

---

## Generating Entry Signals

```{code-cell} python
signals = pd.DataFrame(index=spread.index)

signals["spread"] = spread

signals["upper_band"] = upper_band

signals["lower_band"] = lower_band

signals["position"] = 0

# Long spread: buy EWA, short hedge_ratio * EWC
signals.loc[
    signals["spread"] < signals["lower_band"],
    "position"
] = 1

# Short spread: short EWA, buy hedge_ratio * EWC
signals.loc[
    signals["spread"] > signals["upper_band"],
    "position"
] = -1

signals.head()
```

---

```{admonition} Interpretation
`position = 1` means long spread: buy 1 share of EWA and sell $\beta$ shares of EWC.

`position = -1` means short spread: sell 1 share of EWA and buy $\beta$ shares of EWC.
```

---

## Plotting the Bands and Entry Signals

```{code-cell} python
plt.figure(figsize=(12,5))

plt.plot(
    spread,
    label="Spread",
    linewidth=1.5
)

plt.plot(
    spread_mean,
    label="Rolling Mean",
    linestyle="--"
)

plt.plot(
    upper_band,
    label="Upper Band",
    linestyle=":"
)

plt.plot(
    lower_band,
    label="Lower Band",
    linestyle=":"
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

long_entries = signals[signals["position"] == 1]

short_entries = signals[signals["position"] == -1]

plt.scatter(
    long_entries.index,
    long_entries["spread"],
    marker="^",
    s=70,
    label="Long Spread Entry"
)

plt.scatter(
    short_entries.index,
    short_entries["spread"],
    marker="v",
    s=70,
    label="Short Spread Entry"
)

plt.legend()

plt.title("Bollinger Bands and Pairs Trading Signals")

plt.ylabel("Spread")

plt.savefig(
    "figs/ch21_/BBspread_signal.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig("figs/ch21_/BBspread_signal.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![EWA EWC](figs/ch21_/BBspread_signal.png)

---

```{admonition} Important
The Bollinger Bands are constructed from the cointegration spread, not from the individual ETF prices.

This means the trading signals are based on deviations from the estimated long-run equilibrium relationship.
```

# Exercise 24 — Constructing Hedge-Ratio Portfolio Returns

We now compute the approximate return from the hedge-ratio pairs strategy.

Recall that the long-run relationship is:

```{math}
:enumerated: false
EWA_t = \alpha + \beta EWC_t + u_t
```

The hedge ratio is:

```{math}
:enumerated: false
\beta
```

So the spread portfolio is:

```{math}
:enumerated: false
EWA_t - \beta EWC_t
```

---

## Strategy Return

For a long spread position:

```{math}
:enumerated: false
R_t^{long}
=
R_{EWA,t}
-
\beta R_{EWC,t}
```

For a short spread position:

```{math}
:enumerated: false
R_t^{short}
=
-
R_{EWA,t}
+
\beta R_{EWC,t}
```

This can be written compactly as:

```{math}
:enumerated: false
R_t^{strategy}
=
Position_{t-1}
\left(
R_{EWA,t}
-
\beta R_{EWC,t}
\right)
```

where:

- $Position_{t-1}=1$ means long spread,
- $Position_{t-1}=-1$ means short spread,
- $Position_{t-1}=0$ means no position.

---

```{admonition} Important
We use yesterday's position, $Position_{t-1}$, to compute today's strategy return.

This avoids using information from the same day to trade retroactively.
```

---

## Computing Strategy Returns

```{code-cell} python
ewa_returns = etf_sub["EWA"].pct_change()

ewc_returns = etf_sub["EWC"].pct_change()

spread_portfolio_returns = (
    ewa_returns
    - hedge_ratio * ewc_returns
)

strategy_position = signals["position"].shift(1)

strategy_returns = (
    strategy_position
    * spread_portfolio_returns
)

strategy_returns = strategy_returns.dropna()

strategy_returns.head()
```

---

```{admonition} Practical Note
This is a simplified educational backtest.

A full trading system would need to account for:
- transaction costs,
- bid-ask spreads,
- leverage,
- short-selling constraints,
- financing costs,
- and position sizing.
```

---

## Questions

1. Why do we use the lagged position rather than the current position?

2. What does a positive strategy return mean in this context?

3. Why does the hedge ratio matter for constructing the spread portfolio?

4. What practical trading costs are ignored in this simple calculation?

---

# Exercise 25 — Backtesting the Pairs Trading Strategy

We now evaluate the cumulative performance of the pairs trading strategy.

The goal is to examine whether the strategy was able to profit from mean reversion in the spread.

---

## Cumulative Strategy Performance

```{code-cell} python
cumulative_strategy = (
    1 + strategy_returns
).cumprod()

cumulative_strategy.plot(figsize=(10,5))

plt.title("Pairs Trading Strategy Performance")

plt.ylabel("Cumulative Growth")

plt.xlabel("Date")

plt.savefig(
    "figs/ch21_/pairs_backtest.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
```

![Pairs Trading Backtest](figs/ch21_/pairs_backtest.png)

---

```{admonition} Interpretation
The cumulative return series shows how the value of the strategy evolves over time.

Periods of rising performance suggest successful mean reversion trades.

Periods of falling performance suggest that the spread continued to diverge rather than revert.
```

---

## Comparing with Buy-and-Hold

We now compare the pairs trading strategy with a simple passive portfolio holding:

- 50% EWA
- 50% EWC

This helps illustrate the difference between:

- directional investing,
- and relative-value investing.

---

```{code-cell} python
buyhold_returns = (
    0.5 * ewa_returns
    +
    0.5 * ewc_returns
)

buyhold = (
    1 + buyhold_returns
).cumprod()

comparison = pd.concat(
    [
        cumulative_strategy.rename("Pairs Strategy"),
        buyhold.rename("50-50 Buy-and-Hold")
    ],
    axis=1
)

comparison = comparison.dropna()

comparison.plot(figsize=(10,5))

plt.title("Pairs Trading vs 50-50 Buy-and-Hold")

plt.ylabel("Cumulative Growth")

plt.xlabel("Date")

plt.savefig(
    "figs/ch21_/pairs_vs_buyhold.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
```

![Pairs vs Buy and Hold](figs/ch21_/pairs_vs_buyhold.png)

---

```{admonition} Important
Buy-and-hold strategies profit mainly from broad market appreciation.

Pairs trading attempts to profit from temporary deviations between related assets while reducing overall market exposure.
```

---

## Questions

1. Which strategy appears more stable?

2. Which strategy experiences larger drawdowns?

3. Why might a market-neutral strategy behave differently from buy-and-hold investing?

4. Why might the pairs strategy perform poorly during structural market change?

5. Does the strategy appear sensitive to the sample period?

---

# Exercise 26 — Evaluating Strategy Risk and Performance

Raw returns alone do not fully describe a trading strategy.

We also care about:

- volatility,
- stability,
- drawdowns,
- and risk-adjusted performance.

A strategy with high returns but extremely high risk may not be attractive to investors.

---

## Average Daily Return and Volatility

We first compute the average daily return and daily volatility of the pairs trading strategy.

```{code-cell} python
mean_return = strategy_returns.mean()

volatility = strategy_returns.std()

print("Average Daily Return:", mean_return)

print("Daily Volatility:", volatility)
```

``` verbatim
Average Daily Return: 0.00019655549078299717
Daily Volatility: 0.003472953960432861
```

---

```{admonition} Observation
The strategy generates relatively small daily returns, but also relatively low daily volatility.
```

---

## Annualized Performance Measures

Because the data are daily, annualized statistics are often easier to interpret and compare.

If:

- $\bar R_{daily}$ is the average daily return,
- and $\sigma_{daily}$ is the daily volatility,

then the approximate annualized return is:

```{math}
:enumerated: false
R_{annual} \approx 252 \times \bar R_{daily}
```

and the approximate annualized volatility is:

```{math}
:enumerated: false
\sigma_{annual} \approx \sqrt{252}\sigma_{daily}
```

where:

- 252 approximates the number of trading days in a year.

---

```{code-cell} python
annual_return = mean_return * 252

annual_volatility = volatility * (252**0.5)

print("Annualized Return:", annual_return)

print("Annualized Volatility:", annual_volatility)
```

``` verbatim
nnualized Return: 0.04953198367731529
Annualized Volatility: 0.055131434964493
```

---

## Sharpe Ratio

A common measure of risk-adjusted performance is the:

```{admonition} Definition
Sharpe Ratio
```

which compares:

- average return,
- relative to volatility.

A simple version is:

```{math}
:enumerated: false
Sharpe
=
\frac{\bar R}{\sigma_R}
```

where:

- $\bar R$ = average return,
- $\sigma_R$ = standard deviation of returns.

---

```{admonition} Interpretation
Higher Sharpe ratios indicate greater return per unit of risk.
```
---

## Daily Sharpe Ratio

```{code-cell} python
sharpe = mean_return / volatility

print("Daily Sharpe Ratio:", sharpe)
```

``` verbatim
Daily Sharpe Ratio: 0.056596054258807094
```

---

## Annualized Sharpe Ratio

Because volatility scales with the square root of time, the annualized Sharpe ratio is approximately:

```{math}
:enumerated: false
Sharpe_{annual}
\approx
\sqrt{252}
\times
Sharpe_{daily}
```

where:

- 252 approximates the number of trading days in a year.

---

```{code-cell} python
annual_sharpe = (252**0.5) * sharpe

print("Annualized Sharpe Ratio:", annual_sharpe)
```

``` verbatim
Annualized Sharpe Ratio: 0.8984345085379295
```

---

```{admonition} Interpretation
Higher Sharpe ratios indicate greater return per unit of risk.
```

---

# Comparing with Buy-and-Hold

We now compare the volatility of:

- the pairs strategy,
- and the passive 50-50 buy-and-hold portfolio.

```{code-cell} python
buyhold_volatility = buyhold_returns.std()

buyhold_annual_volatility = (
    buyhold_volatility
    * (252**0.5)
)

print("Pairs Strategy Annualized Volatility:",
      annual_volatility)

print("Buy-and-Hold Annualized Volatility:",
      buyhold_annual_volatility)
```

``` verbatim
Pairs Strategy Annualized Volatility: 0.05513143496449323
Buy-and-Hold Annualized Volatility: 0.20316504936587046
```

---

```{admonition} Observation
The pairs strategy appears substantially less volatile than the buy-and-hold portfolio.

This is consistent with the idea that pairs trading attempts to reduce broad market exposure by focusing on relative mispricing.
```

---

## Questions

1. Does the pairs strategy appear less volatile than buy-and-hold?

2. Why might market-neutral strategies exhibit different risk characteristics?

3. Is higher return always preferable?

4. Why are risk-adjusted measures important in finance?

5. Why might a strategy with lower volatility still be attractive even if raw returns are smaller?

---

```{admonition} Key Insight
A successful strategy is not defined solely by high returns.

Risk, stability, and drawdowns also matter.
```

---

# Exercise 27 — Drawdowns and Strategy Stability

Average returns and volatility are useful, but they do not show how painful losses can become during bad periods.

A common measure of downside risk is the:

```{admonition} Definition
Drawdown
```

A drawdown measures the percentage decline from a previous peak in cumulative performance.

---

## Computing Drawdowns

```{code-cell} python
running_peak = cumulative_strategy.cummax()

drawdown = (
    cumulative_strategy
    / running_peak
    - 1
)

drawdown.plot(figsize=(10,4))

plt.title("Drawdown of Pairs Trading Strategy")

plt.ylabel("Drawdown")

plt.xlabel("Date")

plt.savefig(
    "figs/ch21_/pairs_drawdown.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
```

![Pairs Trading Drawdown](figs/ch21_/pairs_drawdown.png)

---

## Maximum Drawdown

```{code-cell} python
max_drawdown = drawdown.min()

print("Maximum Drawdown:", max_drawdown)
```

``` verbatim
Maximum Drawdown: -0.10007716848807668
```

---

```{admonition} Interpretation
Maximum drawdown shows the largest peak-to-trough loss experienced by the strategy.
```

---

## Questions

1. When does the largest drawdown occur?

2. Does the strategy recover quickly from losses?

3. Why might drawdowns matter more to investors than average returns?

4. How would transaction costs affect drawdowns?

---

```{admonition} Key Insight
A strategy may have attractive average returns but still experience painful losses during periods when the spread fails to mean-revert.
```

---

# Exercise 28 — Structural Breakdown and Strategy Instability

Earlier, we found that the EWA–EWC cointegration relationship appeared stronger in the 2015–2024 sample, but weaker when later observations were included.

This is a crucial practical lesson.

```{admonition} Key Insight
Pairs trading strategies depend on the stability of the long-run relationship.

If the relationship breaks down, the spread may stop mean-reverting.
```

---

## Extending the Sample

We now compare the spread behavior when the sample is extended.

```{code-cell} python
:tags: [hide-input]

etf_full = etf.copy()

spread_full = (
    etf_full["EWA"]
    - alpha
    - hedge_ratio * etf_full["EWC"]
)

spread_full.plot(figsize=(10,4))

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.title("Cointegration Spread Using Extended Sample")

plt.ylabel("Spread")

plt.xlabel("Date")

plt.savefig(
    "figs/ch21_/spread_full_sample.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
```
![Extended Spread](figs/ch21_/spread_full_sample.png)

---

```{admonition} Observation
If the spread drifts persistently away from zero, this suggests that the earlier equilibrium relationship may have weakened.
```

---

## Testing Cointegration in the Extended Sample

```{code-cell} python
from statsmodels.tsa.stattools import coint

coint_stat_full, p_value_full, crit_values_full = coint(
    etf_full["EWA"],
    etf_full["EWC"]
)

print("Cointegration test statistic:", coint_stat_full)

print("p-value:", p_value_full)

print("Critical values:", crit_values_full)
```

``` erbatim
Cointegration test statistic: -2.6696490787249836
p-value: 0.21068403165318256
Critical values: [-3.9002896  -3.33827624 -3.04593952]
```

---

## Questions

1. Does the cointegration result change when the sample is extended?

2. Does the spread still appear mean-reverting?

3. Why might financial relationships break down over time?

4. What would happen to a pairs trading strategy if the spread stopped reverting?

---

```{admonition} Important
Cointegration is an empirical relationship, not a law of nature.

A relationship that appeared stable historically may weaken or disappear when market conditions change.
```

---

# Practical Lessons

Possible reasons for structural breakdown include:

- changes in commodity exposure,
- shifts in monetary policy,
- exchange-rate movements,
- sector composition changes within ETFs,
- changes in global investor behavior,
- crisis periods,
- and post-sample divergence.

```{admonition} Final Lesson
Backtesting is useful, but strategy robustness matters.

A trading rule should be tested across different periods, regimes, and market conditions.
```

---

# Final Reflection — Equilibrium, Instability, and Financial Markets

This capstone illustrates one of the deepest lessons in time series analysis:

```{admonition} Central Insight
Not all co-movement reflects a genuine long-run relationship.
```

In the first case study, U.S. and Mexico GDP appeared strongly related in levels.

The regression produced:

- high \(R^2\),
- significant coefficients,
- and convincing visual relationships.

Yet nonstationarity created the danger of:

```{admonition} Definition
Spurious regression
```

where unrelated trending variables may appear statistically connected.

---

In the second case study, the EWA and EWC ETFs also displayed strong co-movement.

However, unlike the GDP example, the ETF pair showed evidence of:

```{admonition} Definition
Cointegration
```

meaning that the series appeared linked through a long-run equilibrium relationship.

This allowed us to construct:

- spreads,
- error correction models,
- and pairs trading strategies.

---

At the same time, the capstone also revealed an important practical reality:

```{admonition} Important
Equilibrium relationships in financial markets may weaken or break down over time.
```

The cointegration relationship became less convincing when the sample period was extended beyond 2024.

This illustrates the importance of:

- structural change,
- regime shifts,
- and model instability.

---

# Broader Lessons

This capstone highlights several broader themes in applied time series analysis.

---

## 1. Statistical Significance Is Not Enough

High \(R^2\) and significant coefficients do not automatically imply meaningful economic relationships.

Understanding:

- trends,
- persistence,
- and nonstationarity

is essential.

---

## 2. Dynamic Relationships Matter

Many economic and financial variables evolve over time through:

- adjustment,
- feedback,
- and equilibrium correction.

Static regression models may miss these dynamics entirely.

---

## 3. Financial Markets Are Adaptive

Trading relationships that appear profitable historically may weaken or disappear.

This is especially important in:

- algorithmic trading,
- statistical arbitrage,
- and machine-learning finance.

---

## 4. Models Are Approximations

No model fully captures financial reality.

Time series models should therefore be viewed as:

- tools for understanding,
- simplifications of complex systems,
- and frameworks for disciplined thinking.

---

```{admonition} Final Insight
Time series analysis is not only about forecasting.

It is also about understanding persistence, equilibrium, instability, and dynamic adjustment in economic systems.
```