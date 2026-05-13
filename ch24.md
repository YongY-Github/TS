---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 24 — Vector Error Correction Models (VECM)

In earlier chapters, we studied:

- nonstationary time series,
- cointegration,
- and error correction models (ECMs).

We also introduced:

- VAR models,
- multivariate dynamics,
- and impulse response analysis.

VAR models allow economic variables to interact dynamically through time.

But an important problem arises when variables are:

```{admonition} Important
Nonstationary but cointegrated.
````

Many macroeconomic and financial variables display precisely this behavior.

Examples include:

* money supply and prices,
* GDP and credit,
* exchange rates and inflation,
* stock indices across markets.

These variables may drift individually through time, yet still maintain stable long-run relationships.

This raises an important question.

```{admonition} Central Question
How can we model short-run dynamics while preserving long-run equilibrium relationships?
```

The answer is the:

```{admonition} Definition
Vector Error Correction Model (VECM)
```

VECMs combine:

* multivariate dynamics,
* cointegration,
* and equilibrium adjustment

within a unified framework.

```{admonition} Key Idea
A VECM allows variables to drift temporarily apart, but not indefinitely.
```

Throughout this chapter, we use Thai macroeconomic data as a running example.

---

## Learning Objectives

By the end of this chapter, you should be able to:

* explain the intuition behind VECMs
* distinguish VARs from VECMs
* understand equilibrium correction
* interpret error correction terms
* understand short-run versus long-run dynamics
* estimate VECMs
* understand cointegration rank
* interpret adjustment coefficients
* perform Johansen cointegration tests
* interpret VECMs economically

---

# 24.1 Why Standard VAR Models Become Problematic

Standard VAR models usually require stationary variables.

But many macroeconomic variables are:

* trending,
* persistent,
* and integrated of order one.

Examples include:

* CPI,
* money supply,
* nominal GDP,
* and price levels.

---

## The Problem

Suppose we estimate a VAR using nonstationary variables.

Several problems may arise:

* spurious relationships,
* unreliable statistical inference,
* unstable long-run behavior.

One common solution is:

```{admonition} Observation
Difference the data.
```

Differencing often restores stationarity.

But differencing introduces another important problem.

```{admonition} Important
Differencing may remove economically meaningful long-run relationships.
```

For example:

* inflation and money supply may move together over decades,
* GDP and credit may share common long-run trends,
* exchange rates and prices may adjust toward purchasing power parity.

Pure differencing may destroy this information.

---

# 24.2 Cointegration and Long-Run Equilibrium

Suppose two variables:

* drift through time individually,
* but maintain a stable long-run relationship.

Then they may be:

```{admonition} Definition
Cointegrated.
```

Cointegration implies that long-run equilibrium forces exist even when short-run fluctuations are substantial.

```{admonition} Key Idea
Cointegration combines nonstationarity with long-run equilibrium.
```

---

## Intuition

Cointegrated variables may temporarily drift apart.

But equilibrium forces gradually pull them back together.

This creates a distinction between:

* short-run deviations,
* and long-run equilibrium restoration.

---

# 24.3 Rubber-Band Analogy

A useful analogy is:

```{admonition} Analogy
Imagine two variables connected by a rubber band.
```

Short-run shocks may pull the variables apart temporarily.

But the rubber band creates pressure toward equilibrium.

This is precisely the role of the:

```{admonition} Definition
Error correction mechanism.
```

---

# 24.4 From ECM to VECM

Earlier in the book, we studied single-equation ECMs.

For example:

```{math}
:enumerated: false
\Delta y_t
=
\alpha
+
\beta \Delta x_t
+
\lambda (y_{t-1}-\gamma x_{t-1})
+
u_t
```

The term:

```{math}
:enumerated: false
(y_{t-1}-\gamma x_{t-1})
```

measures deviation from long-run equilibrium.

---

## Extending to Multiple Variables

A VECM generalizes this idea to:

* several variables,
* several equations,
* and multiple equilibrium relationships.

```{admonition} Observation
A VECM is essentially a VAR designed for cointegrated systems.
```

---

# 24.5 Short-Run versus Long-Run Dynamics

One of the most important features of a VECM is the separation between:

* short-run movements,
* and long-run equilibrium adjustment.

---

## Short-Run Dynamics

Short-run fluctuations capture:

* temporary shocks,
* cyclical movements,
* and immediate reactions.

These effects may generate temporary deviations from equilibrium.

---

## Long-Run Dynamics

Long-run dynamics capture:

* equilibrium restoration,
* persistent relationships,
* and gradual adjustment forces.

```{admonition} Key Idea
VECMs simultaneously model temporary fluctuations and long-run equilibrium adjustment.
```

---

# 24.6 Thai Macroeconomic Example

We now examine Thai macroeconomic variables.

Our dataset contains:

* CPI,
* broad money supply (BM),
* and real GDP.

These variables are useful because:

* they display strong trends,
* they may be nonstationary,
* and economic theory suggests possible long-run relationships.

---

# 24.7 Loading the Data

```{code-cell} python
import pandas as pd
from io import StringIO

data_text = """
year,cpi,BM_,gdp_r
1991,50.70,204654.4084,38.23977231
1992,52.80,237852.2993,41.58407966
1993,54.50,272926.6351,43.40940123
1994,57.30,302096.2673,46.88149708
1995,60.60,355694.6742,50.68815184
1996,64.10,393470.4887,53.55329227
1997,67.70,470396.7099,52.07872106
1998,73.10,517768.2452,48.10357948
1999,73.30,537431.8537,50.30235946
2000,74.50,563808.6622,52.54366163
2001,75.70,594569.9039,54.3535078
2002,76.20,617075.0830,57.69600000
2003,77.60,707867.6497,61.84370042
2004,79.80,747291.5105,65.73352411
2005,83.40,792795.6350,68.48596905
2006,87.30,857454.6312,71.88876509
2007,89.20,911063.5572,75.79552154
2008,94.10,994546.5100,77.10330582
2009,93.30,1061828.8660,76.53419316
2010,96.33,1178006.9540,82.27951477
2011,100.00,1356089.1230,82.96559013
2012,103.02,1496760.9480,88.96449269
2013,105.27,1606334.2890,91.36862416
2014,107.26,1681035.3120,92.11543151
"""

thai = pd.read_csv(
    StringIO(data_text)
)

thai.head()
```

---

# 24.8 Plotting the Variables

```{code-cell} python
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(10,5))

# Left axis: CPI
ax1.plot(
    thai["year"],
    thai["cpi"],
    linewidth=2,
    label="CPI"
)

ax1.set_xlabel("Year")
ax1.set_ylabel("CPI")

# Right axis: Broad Money
ax2 = ax1.twinx()

ax2.plot(
    thai["year"],
    thai["BM_"],
    linewidth=2,
    linestyle="--",
    label="Broad Money"
)

ax2.set_ylabel("Broad Money")

plt.title("Thailand: CPI and Broad Money")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper left"
)

plt.savefig(
    "figs/ch24/cpiBM.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
```

![CPI BM](figs/ch24/cpiBM.png)

```{admonition} Observation
Both CPI and broad money display strong upward trends through time.
```

This immediately raises important questions:

* Are the variables nonstationary?
* Do they share a long-run equilibrium relationship?
* Could they be cointegrated?

---

# 24.9 A Simple VECM Representation

A simple VECM may be written as:

```{math}
:enumerated: false
\Delta y_t
=
\alpha
+
\beta_1 \Delta y_{t-1}
+
\beta_2 \Delta x_{t-1}
+
\lambda (y_{t-1} - \gamma x_{t-1})
+
u_t
````

This model combines:

* short-run dynamics,
* and long-run equilibrium correction.

```{admonition} Key Idea
VECMs allow temporary deviations from equilibrium, but include forces that gradually restore long-run balance.
```

---

## Short-Run Dynamics

The differenced variables:

```{math}
:enumerated: false
\Delta y_{t-1}
\quad \text{and} \quad
\Delta x_{t-1}
```

capture:

* short-run fluctuations,
* temporary shocks,
* and immediate dynamic interactions.

These effects describe how variables move from period to period.

---

## Long-Run Equilibrium Correction

The term:

```{math}
:enumerated: false
(y_{t-1} - \gamma x_{t-1})
```

measures deviation from long-run equilibrium.

If the variables drift too far apart, the system gradually adjusts.

```{admonition} Intuition
The error correction term acts like a restoring force pulling variables back toward equilibrium.
```

---

## Adjustment Speed

The coefficient:

```{math}
:enumerated: false
\lambda
```

measures how strongly the system reacts to disequilibrium.

* large values imply faster adjustment,
* smaller values imply slower correction.

```{admonition} Observation
Some economic systems return toward equilibrium quickly, while others adjust only gradually.
```

---

# 24.10 Extending to Multivariate Systems

In larger systems involving several variables, the same ideas continue to apply.

A multivariate VECM contains:

* short-run dynamics,
* long-run equilibrium relationships,
* and adjustment mechanisms.

Economists often write these systems compactly using matrix notation.

```{math}
:enumerated: false
\Delta Y_t
=
\Pi Y_{t-1}
+
\Gamma_1 \Delta Y_{t-1}
+
\cdots
+
u_t
```

You do not need to focus heavily on the matrix algebra.

Conceptually, the interpretation remains the same:

* differenced terms capture short-run movements,
* while the equilibrium term captures long-run correction forces.

```{admonition} Important
The key economic intuition is more important than the matrix notation itself.
```

---

# 24.11 Cointegration and Adjustment

The long-run equilibrium structure of a VECM is often summarized using:

```{math}
:enumerated: false
\Pi = \alpha \beta'
```

Conceptually:

* $\beta$ describes the long-run equilibrium relationships,
* while $\alpha$ measures how strongly variables adjust when equilibrium is disturbed.

```{admonition} Key Idea
$\beta$ describes equilibrium relationships, while $\alpha$ measures adjustment speed.
```

---

## Intuition

Suppose prices rise much faster than money supply.

The system may become temporarily unbalanced.

Adjustment mechanisms may then generate:

* slower price growth,
* faster money growth,
* or both.

This gradual return toward equilibrium is the essence of error correction dynamics.

---

# 24.12 Cointegration Rank

An important concept in VECMs is the:

```{admonition} Definition
Cointegration rank.
```

The rank determines how many long-run equilibrium relationships exist in the system.

| Rank     | Interpretation                    |
| -------- | --------------------------------- |
| 0        | no cointegration                  |
| 1        | one equilibrium relationship      |
| multiple | several equilibrium relationships |

```{admonition} Important
The cointegration rank determines the long-run structure of the system.
```

---

# 24.13 Johansen Cointegration Test

The Johansen procedure is the standard approach for testing cointegration in multivariate systems.

Unlike the Engle–Granger approach, Johansen testing allows:

* multiple variables,
* and multiple cointegrating relationships.

```{admonition} Definition
The Johansen test estimates the cointegration rank of a multivariate system.
```

---

# 24.14 Trace and Maximum Eigenvalue Tests

Johansen procedures commonly report:

* trace statistics,
* and maximum eigenvalue statistics.

These help answer the question:

```{admonition} Central Question
How many cointegrating relationships exist?
```

---

# 24.15 Johansen Test in Python

```{code-cell} python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

data = thai[["cpi","BM_"]].dropna()

johansen_test = coint_johansen(
    data,
    det_order=0,
    k_ar_diff=2
)

print(johansen_test.lr1)
```

```verbatim
[7.30400737e+00 1.97612579e-03]
```

```{admonition} Interpretation
Large test statistics suggest evidence of cointegration.
```

---

# 24.16 Estimating a VECM in Python

```{code-cell} python
from statsmodels.tsa.vector_ar.vecm import VECM

model = VECM(
    data,
    k_ar_diff=2,
    coint_rank=1
)

results = model.fit()

print(results.summary())
```

---

# 24.17 Interpreting the VECM Results

The VECM combines:

* differenced short-run dynamics,
* and long-run equilibrium correction.

A crucial component is the:

```{admonition} Definition
Error correction term.
```

This measures deviation from long-run equilibrium.

---

## Example

Suppose money supply rises much faster than prices.

The VECM captures pressure for future adjustment.

Possible responses include:

* inflation increasing,
* money growth slowing,
* or both.

```{admonition} Key Idea
The error correction term pulls the system back toward equilibrium.
```

---

# 24.18 Adjustment Speeds

Adjustment coefficients measure:

```{admonition} Central Question
How quickly do variables return toward equilibrium?
```

---

## Large Adjustment Coefficients

Large coefficients suggest:

* rapid correction,
* strong equilibrium restoration,
* and faster adjustment.

---

## Small Adjustment Coefficients

Small coefficients suggest:

* slow adjustment,
* persistent disequilibrium,
* and weaker correction forces.

---

# 24.19 VECMs versus VARs in Differences

A differenced VAR removes long-run equilibrium information.

A VECM preserves it.

| Feature                | VAR in Differences | VECM         |
| ---------------------- | ------------------ | ------------ |
| stationary dynamics    | ✓                  | ✓            |
| long-run equilibrium   | ✗                  | ✓            |
| cointegration          | ignored            | incorporated |
| equilibrium adjustment | ✗                  | ✓            |

```{admonition} Key Insight
A VECM preserves long-run equilibrium relationships that differenced VARs discard.
```

---

# 24.20 Forecasting with VECMs

VECMs are often superior to differenced VARs when cointegration exists.

Why?

Because they preserve:

* equilibrium relationships,
* long-run information,
* and adjustment dynamics.

```{admonition} Important
Ignoring cointegration may reduce forecasting performance.
```

---

# 24.22 Financial Applications of VECMs

VECMs are widely used in finance.

Examples include:

* pairs trading,
* stock market integration,
* exchange-rate systems,
* and interest-rate term structure.

---

## Example: Pairs Trading

If two stock prices are cointegrated:

* temporary deviations may create trading opportunities.

This idea underlies many statistical arbitrage strategies.

---

# 24.22 Macroeconomic Applications

VECMs are also widely used in macroeconomics.

Examples include:

* money demand,
* purchasing power parity,
* inflation dynamics,
* and monetary policy transmission.

---

# 24.23 Gretl Example: Johansen Test and VECM

Gretl provides built-in tools for cointegration testing and VECM estimation.

---

## Step 1

Load multiple nonstationary variables.

---

## Step 2

Menu:

```text
Model → Time Series → VECM
```

---

## Step 3

Choose:

* lag length,
* deterministic terms,
* and cointegration rank.

---

```markdown
[GRETL Screenshot Placeholder: Johansen test output]
```

---

## Step 4

Estimate the VECM.

GRETL reports:

* cointegration vectors,
* adjustment coefficients,
* and short-run dynamics.

---

```markdown
[GRETL Screenshot Placeholder: VECM estimation output]
```

---

# 24.24 Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Using VECMs without testing for cointegration**  
Cointegration testing is essential.

**2. Differencing away equilibrium information**  
Long-run relationships may be economically important.

**3. Misinterpreting adjustment coefficients**  
Adjustment measures equilibrium correction speed.

**4. Confusing VARs and VECMs**  
VECMs are specifically designed for cointegrated systems.

**5. Ignoring lag selection**  
Poor lag choices may distort inference.
```

---

# 24.25 Looking Ahead

This concludes our introduction to multivariate time series systems.

We have now studied:

* VAR models,
* impulse responses,
* and VECMs.

The next part of the book turns toward:

* volatility,
* ARCH models,
* and GARCH models.

We shift from modeling:

```{admonition} Observation
The conditional mean.
```

toward modeling:

```{admonition} Observation
The conditional variance.
```

of financial time series.

---

# Key Takeaways

```{admonition} Summary
- VECMs combine short-run dynamics with long-run equilibrium adjustment.
- Cointegration implies stable long-run relationships among nonstationary variables.
- The Johansen test estimates cointegration rank.
- Error correction terms measure disequilibrium.
- Adjustment coefficients determine equilibrium restoration speed.
- VECMs preserve long-run information that differenced VARs lose.
- VECMs are widely used in macroeconomics and finance.
```

# Concept Check

### Basic

1. What is a Vector Error Correction Model (VECM)?

2. How does a VECM differ from a standard VAR model?

3. When should a VECM be used instead of a VAR?

---

### Intuition

4. Why is differencing alone not sufficient when variables are cointegrated?

5. What is the economic meaning of cointegration in a multivariate system?

6. Explain the “rubber band” analogy in the context of VECM.

---

### Structure

7. What are the two main components of a VECM?

8. What does the term $\Pi Y_{t-1}$ represent?

9. What do the $\Gamma_i$ terms capture?

---

### α and β

10. What does the matrix $\beta$ represent?

11. What does the matrix $\alpha$ represent?

12. Why is the decomposition $\Pi = \alpha \beta'$ important?

---

### Challenge

13. Why is it not enough to estimate a VAR in differences when variables are cointegrated?

---

# Interpretation & Practice

1. A system shows strong cointegration.

- What does this imply about long-run relationships?

---

2. The cointegration rank is zero.

- What does this imply?

---

3. The cointegration rank is one.

- What does this imply?

---

4. Adjustment coefficients are large in magnitude.

- What does this suggest?

---

5. Adjustment coefficients are close to zero.

- What does this imply?

---

### Error Correction

6. The error correction term is significant in one equation but not the other.

- What does this imply?

---

7. A variable does not respond to disequilibrium.

- What might this suggest?

---

### Economic Interpretation

8. CPI and money supply are cointegrated.

- What does this imply about long-run behavior?

9. You estimate a system with:

- CPI  
- money supply  
- GDP  

You find:

- cointegration rank = 1  
- CPI adjusts strongly  
- money supply adjusts weakly  

---

- What does this suggest about economic dynamics?
- Which variable leads the system?
- Which variable follows?

---

### Challenge

9. Why is VECM considered a “restricted VAR”?

---

# Numerical Practice

### Cointegration Logic

1. Suppose:

- $x_t \sim I(1)$  
- $y_t \sim I(1)$  
- one cointegrating vector exists  

---

- What does this imply?

---

### Rank Interpretation

2. Suppose a system of 3 variables has:

- cointegration rank = 2  

---

- How many long-run relationships exist?

---

### Adjustment Coefficients

3. Suppose:

```{math}
:enumerated: false
\alpha =
\begin{pmatrix}
-0.3 \\
0.0
\end{pmatrix}
```

---

- Which variable adjusts to equilibrium?
- Which does not?

---

### Interpretation

4. Suppose:

```{math}
:enumerated: false
\beta' Y_{t-1} = y_{t-1} - 2x_{t-1}
```
---

- What does this represent?

---

### Short vs Long Run

5. Why is it important to include both:

- $\Delta Y_t$ terms  
- and $\Pi Y_{t-1}$ terms?

---

### Diagnostics

6. Suppose cointegration is ignored and a VAR in differences is estimated.

- What information is lost?

---

### Challenge

7. Suppose cointegration rank is incorrectly specified.

- What problems might arise?

---

# Johansen Test Interpretation

1. What does the Johansen test estimate?

2. What is the difference between:

- trace test  
- maximum eigenvalue test  

---

### Interpretation

3. Suppose the test suggests rank = 1.

- What does this imply?

---

4. Suppose test statistics are small.

- What does this suggest?

---

### Challenge

5. Why is determining the correct cointegration rank important?

---

# IRF & Forecasting in VECM

1. How do impulse responses differ in VECM vs VAR?

2. Why do long-run relationships affect IRFs?

---

### Interpretation

3. A shock causes variables to deviate, then gradually return.

- What does this reflect?

---

4. Why might VECM forecasts outperform differenced VAR forecasts?

---

### Challenge

5. Why is long-run information valuable in forecasting?


# Appendix 24A — Relationship Between VAR and VECM

A VECM can be derived algebraically from a VAR expressed in levels.

Suppose:

```{math}
:enumerated: false
Y_t
=
A_1 Y_{t-1}
+
\cdots
+
A_p Y_{t-p}
+
u_t
```

Rewriting the system in differences produces:

- short-run difference terms,
- and a long-run equilibrium term.

This decomposition leads directly to the VECM representation.

---

```{admonition} Observation
The VECM is not a completely different model.

It is a restricted VAR designed for cointegrated systems.
```

---

# Appendix 24B — Why Cointegration Matters Economically

Cointegration matters because many economic variables are tied together by long-run equilibrium forces.

Examples include:

- money and prices,
- income and consumption,
- exchange rates and inflation.

Without equilibrium adjustment:

- economic systems could drift apart indefinitely.

Cointegration formalizes the idea that:

```{admonition} Key Insight
Economic forces create long-run constraints even when short-run fluctuations are substantial.
```
