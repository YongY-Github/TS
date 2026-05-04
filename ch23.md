---
kernelspec:
  name: jb2-env
  display_name: Python (jb2-env)
---

# Chapter 23 — Impulse Response Functions

In the previous chapter, we introduced VAR models as systems of interacting time series.

But VAR coefficients themselves are often difficult to interpret directly.

Suppose we estimate a VAR involving:

- inflation,
- interest rates,
- and output.

A natural question immediately arises:

```{admonition} Central Question
What happens dynamically after an economic shock?
```

For example:

- What happens to inflation after a monetary policy shock?
- What happens to GDP after an oil-price shock?
- What happens to exchange rates after an interest-rate increase?

Impulse response functions (IRFs) were developed precisely to answer such questions.

This chapter introduces:

- economic shocks,
- dynamic propagation,
- impulse response functions,
- orthogonalization,
- identification,
- confidence intervals,
- and economic interpretation.

The emphasis is intuition-first and applications-oriented.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- explain the intuition behind impulse response functions
- interpret dynamic responses to shocks
- distinguish contemporaneous and lagged effects
- understand orthogonalized shocks
- interpret confidence bands
- generate IRFs in Python and GRETL
- understand identification issues in VAR analysis
- interpret IRFs economically

---

# 23.1 Why Impulse Responses Matter

Economic systems are dynamic.

A shock today may influence variables:

- immediately,
- gradually,
- or persistently.

For example:

- a monetary tightening may reduce inflation only slowly,
- an oil-price shock may affect output for many quarters,
- a financial crisis may generate persistent volatility.

```{admonition} Key Idea
Impulse responses describe how shocks propagate through time.
```

---

# 23.2 What Is an Impulse Response Function?

```{admonition} Definition
An impulse response function traces the effect of a one-time shock on variables in a dynamic system.
```

The response is tracked period by period.

---

## Example

Suppose interest rates suddenly increase unexpectedly.

We may ask:

- How does inflation respond?
- How long do effects persist?
- Do effects eventually disappear?

IRFs attempt to answer these questions visually and dynamically.

---

# 23.3 Intuition: Dropping a Stone into Water

A useful analogy is:

```text
Dropping a stone into water.
```

The initial splash creates waves.

The waves then:

- spread,
- interact,
- and gradually disappear.

Economic shocks behave similarly.

---

```{admonition} Intuition
A shock today may generate ripple effects across variables and across time.
```

---

# 23.4 Impulse Responses in a VAR

Consider a VAR involving:

- inflation,
- interest rate.

Suppose an unexpected interest-rate shock occurs today.

The VAR traces:

- inflation tomorrow,
- inflation next month,
- inflation next year,
- and so on.

---

## Dynamic Feedback

Because variables interact:

- inflation affects interest rates,
- interest rates affect inflation,
- and responses feed back through time.

This creates rich dynamic behavior.

---

# 23.5 A Simple Example

Suppose central banks unexpectedly raise interest rates.

Possible dynamic effects include:

| Period | Possible Effect |
|---|---|
| immediate | borrowing becomes more expensive |
| short run | investment falls |
| medium run | output slows |
| longer run | inflation declines |

---

```{admonition} Observation
Many economic effects emerge gradually rather than instantly.
```

---

# 23.6 Shape of Impulse Responses

Impulse responses may display different patterns.

---

## Rapid Decay

Effects disappear quickly.

---

## Persistent Effects

Responses decay slowly over time.

---

## Oscillating Responses

Variables may overshoot and fluctuate before stabilizing.

---

```{admonition} Important
The shape of the impulse response contains important economic information.
```

---

# 23.7 Positive and Negative Responses

Impulse responses may be:

- positive,
- negative,
- or mixed over time.

For example:

- a monetary tightening may reduce inflation,
- an oil-price shock may initially raise inflation,
- stock markets may react positively or negatively depending on expectations.

---

# 23.8 Orthogonalized Shocks

A major complication arises because VAR residuals are often correlated.

For example:

- inflation shocks,
- and interest-rate shocks

may occur simultaneously.

---

```{admonition} Problem
How can we isolate a pure shock to one variable?
```

---

# 23.9 Cholesky Decomposition

One common solution is:

```text
Cholesky decomposition
```

This transforms correlated shocks into orthogonal shocks.

---

```{admonition} Definition
Orthogonal shocks are shocks that are statistically uncorrelated with one another.
```

---

# 23.10 Ordering Matters

With Cholesky decomposition, variable ordering becomes important.

Example ordering:

1. GDP
2. Inflation
3. Interest rate

This ordering implicitly assumes:

- GDP affects inflation contemporaneously,
- inflation affects interest rates contemporaneously,
- but not vice versa within the same period.

---

```{admonition} Important
Impulse responses may depend heavily on variable ordering.
```

---

# 23.11 Identification Problems

VAR models describe correlations and dynamics.

But economists often want structural interpretation.

This requires identifying assumptions.

---

## Example

Question:

```text
What is a monetary policy shock?
```

Answering this requires economic theory.

---

```{admonition} Key Insight
Impulse responses are not automatically causal.
```

Structural interpretation requires assumptions.

---

# 23.12 Confidence Intervals

Impulse responses are estimated statistically.

Therefore they contain uncertainty.

Confidence intervals help assess statistical precision.

---

## Example

Wide confidence bands imply:

- substantial uncertainty,
- less reliable inference.

---

```{admonition} Important
Impulse responses should always be interpreted together with confidence intervals.
```

---

# 23.13 Estimating IRFs in Python

We now estimate impulse responses using Python.

```{code-cell} python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR

# Download data
spy = yf.download(
    "SPY",
    start="2015-01-01",
    auto_adjust=False
)

# Compute returns
returns = 100 * np.log(
    spy["Adj Close"] /
    spy["Adj Close"].shift(1)
)

returns = returns.dropna()

# Volatility proxy
volatility = returns.rolling(20).std()

# Combine variables
data = pd.concat(
    [returns, volatility],
    axis=1
)

data.columns = [
    "Returns",
    "Volatility"
]

data = data.dropna()

# Estimate VAR
model = VAR(data)

results = model.fit(2)

# Impulse responses
irf = results.irf(12)

irf.plot()

plt.savefig("figs/ch23/irf.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![IRF](figs/ch23/irf.png)

```{admonition} Observation
Impulse responses typically decay gradually through time.
```

---

# 23.14 Interpreting an IRF Plot

Suppose we observe:

- positive initial response,
- gradual decay,
- eventual convergence toward zero.

This suggests:

- shocks matter temporarily,
- but long-run stability eventually returns.

---

## Persistent Responses

If responses decay very slowly:

- shocks may have long-lasting effects.

This is common in:

- volatility,
- inflation,
- exchange rates.

---

# 23.15 Cumulative Impulse Responses

Sometimes we are interested in total accumulated effects.

This leads to cumulative impulse responses.

---

## Example

Question:

```text
How much total inflation is generated by an oil shock over two years?
```

Cumulative responses help answer this.

---

# 23.16 Forecast Error Variance Decomposition

Impulse responses are closely related to:

```text
Forecast Error Variance Decomposition (FEVD)
```

FEVD measures how much forecast uncertainty comes from different shocks.

---

```{admonition} Definition
Variance decomposition measures the relative importance of shocks in explaining forecast uncertainty.
```

---

# 23.17 Impulse Responses and Economic Theory

Impulse responses are especially useful because they connect:

- data,
- dynamics,
- and economic theory.

Examples include:

- monetary transmission,
- fiscal policy effects,
- exchange-rate dynamics,
- financial contagion.

---

```{admonition} Key Idea
IRFs translate abstract VAR coefficients into economically interpretable dynamic stories.
```

---

# 23.18 Example: Monetary Policy Shock

Suppose central banks unexpectedly raise interest rates.

Possible impulse responses:

| Variable | Possible Response |
|---|---|
| inflation | declines gradually |
| GDP | slows |
| unemployment | rises |
| exchange rate | appreciates |

---

## Delayed Effects

Many macroeconomic responses occur only after several periods.

This is sometimes called:

```text
policy transmission lag
```

---

# 23.19 Impulse Responses in Finance

IRFs are also widely used in finance.

Examples include:

- volatility spillovers,
- stock market contagion,
- oil-price shocks,
- exchange-rate transmission.

---

## Example

Question:

```text
How does a U.S. market shock affect Asian stock markets?
```

VAR and IRF methods are commonly used to study such problems.

---

# 23.20 Generalized Impulse Responses

Standard orthogonalized IRFs depend on ordering assumptions.

An alternative is:

```text
Generalized Impulse Responses
```

These reduce sensitivity to ordering.

---

```{admonition} Observation
Different identification approaches may produce different impulse responses.
```

---

# 23.21 IRFs and Stability

In stable systems:

- impulse responses eventually converge toward zero.

In unstable systems:

- responses may explode.

---

```{admonition} Important
Stable VAR systems generally produce impulse responses that die out over time.
```

---

# 23.22 GRETL Example: Impulse Responses

GRETL provides built-in tools for IRF analysis.

---

## Step 1

Estimate a VAR model.

Menu:

```text
Model → Time Series → VAR
```

---

## Step 2

From the VAR output window:

```text
Analysis → Impulse responses
```

---

## Step 3

Choose:

- shock variable,
- response variable,
- forecast horizon,
- confidence intervals.

---

```markdown
[GRETL Screenshot Placeholder: IRF settings]
```

---

# 23.23 Reading IRF Graphs

Typical IRF graphs show:

- horizontal axis = time horizon,
- vertical axis = response magnitude.

The zero line is especially important.

---

```{admonition} Interpretation
Responses close to zero imply weak dynamic effects.
```

---

# 23.24 Common Mistakes

```{admonition} Common Mistakes
:class: warning

**1. Treating IRFs as automatically causal**  
Structural interpretation requires identifying assumptions.

**2. Ignoring confidence intervals**  
Estimated responses contain uncertainty.

**3. Forgetting ordering sensitivity**  
Orthogonalized IRFs depend on variable ordering.

**4. Overinterpreting small responses**  
Minor fluctuations may not be economically meaningful.

**5. Ignoring stability conditions**  
Unstable systems may produce misleading impulse responses.
```

---

# 23.25 Looking Ahead

Impulse responses provide powerful tools for analyzing dynamic systems.

The next chapter introduces:

- Vector Error Correction Models (VECMs),
- cointegrated systems,
- and long-run equilibrium relationships.

We will combine:

- short-run dynamics,
- and long-run adjustment

within a unified multivariate framework.

---

# Key Takeaways

```{admonition} Summary
- Impulse responses trace the dynamic effects of shocks.
- Economic shocks often propagate gradually through time.
- IRFs help interpret VAR dynamics economically.
- Orthogonalization is needed because VAR shocks are often correlated.
- Cholesky decomposition introduces ordering assumptions.
- Confidence intervals are crucial for interpretation.
- IRFs are widely used in macroeconomics and finance.
- Structural interpretation requires identifying assumptions.
```

# Concept Check

## Basic

1. What is an impulse response function (IRF)?

2. What does a “shock” represent in a VAR model?

3. Why are IRFs used instead of directly interpreting VAR coefficients?

---

## Intuition

4. Explain the “stone in water” analogy for impulse responses.

5. Why do economic shocks often have effects over multiple periods?

6. What does it mean for a shock to “propagate” through a system?

---

## Dynamics

7. What is the difference between:

   - contemporaneous effects  
   - lagged effects  

8. Why do impulse responses often decay over time?

---

## Challenge

9. What does it mean if an impulse response does not decay?

---

# Interpretation & Practice

1. An IRF shows:

- a positive initial response  
- gradual decay to zero  

   - What does this imply?

2. An IRF shows a negative response after a positive shock.

   - What does this suggest about the relationship?

3. An IRF oscillates before stabilizing.

   - What might this indicate?

4. An IRF is close to zero at all horizons.

   - What does this imply?

---

## Persistence

5. A shock has effects that persist for many periods.

   - What does this suggest about the system?

6. A shock disappears quickly.

   - What does this imply?

---

## Confidence Bands

7. Confidence intervals are wide.

   - What does this imply?

8. Confidence bands include zero.

   - What does this suggest?

---

## Challenge

9. Why should IRFs always be interpreted together with confidence intervals?

---

# Orthogonalization & Identification

1. Why are VAR shocks often correlated?

2. What is orthogonalization?

3. What does Cholesky decomposition do?

---

## Ordering

4. Why does variable ordering matter?

5. What assumption is made when a variable is ordered first?

6. What assumption is made when a variable is ordered last?

---

## Interpretation

7. Why are IRFs not automatically causal?

---

## Challenge

8. How can different ordering assumptions change conclusions?

---

# Numerical & Graph Interpretation

## Interpreting an Impulse Response Function

1. Consider the following impulse response function:

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

horizon = np.arange(10)

# stylized IRF: rise → peak → decay
irf = [0.0, 0.5, 0.9, 1.2, 1.0, 0.7, 0.4, 0.2, 0.1, 0.0]

plt.figure(figsize=(8,4))
plt.plot(horizon, irf, marker='o')

plt.axhline(0, linestyle='--', linewidth=1)

plt.title("Stylized Impulse Response Function")
plt.xlabel("Horizon")
plt.ylabel("Response")

plt.tight_layout()

plt.savefig("figs/ch23/irf_Q.png", dpi=300, bbox_inches="tight")
plt.close()   # replace with plt.show()
```

![IRF](figs/ch23/irf_Q.png)


---

- Describe the pattern of the impulse response.
- At what period does the response peak?
- Does the effect persist or disappear quickly?
- What does this suggest about the dynamic impact of the shock?

---

```{admonition} Hint
Focus on:
- timing of the peak  
- persistence  
- speed of decay  
```

---

## Reading an IRF

2. Suppose an IRF shows:

- initial increase  
- peak after 3 periods  
- slow decline  

---

- What does this suggest?

---

## Cumulative Effects

3. Why might we use cumulative impulse responses?

4. Suppose cumulative IRF increases steadily.

- What does this imply?

---

## Stability

5. Suppose IRFs grow larger over time.

- What does this imply?

---

## Comparison

6. Two IRFs:

- one decays quickly  
- one decays slowly  

---

- Which system is more persistent?

---

## Challenge

7. Suppose a shock produces different responses depending on ordering.

- What does this suggest?

---

# Appendix 23A — Orthogonalized vs Generalized IRFs

Orthogonalized IRFs:

- rely on Cholesky decomposition,
- depend on variable ordering.

Generalized IRFs:

- reduce ordering sensitivity,
- but may be harder to interpret structurally.

Different approaches may produce somewhat different results.

---

```{admonition} Important
Impulse responses are partly statistical objects and partly economic interpretations.
```

---

# Appendix 23B — Why Impulse Responses Became Popular

VAR coefficient matrices are often difficult to interpret directly.

Impulse responses became popular because they:

- summarize complex dynamics visually,
- describe propagation mechanisms,
- connect statistical models with economic narratives.

They transformed VAR analysis from:

```text
large coefficient tables
```

into dynamic economic interpretation.