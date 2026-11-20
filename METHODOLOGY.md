# Quantitative Methodology & Model Limitations

This document details the mathematical formulations, statistical assumptions, validation tests, and known failure modes of the **Multi-Currency Equity Portfolio Risk & Attribution Engine**.

---

## 1. Multi-Currency Data Pipeline & Return Construction

### 1.1 Base-Currency Price Translation
For a portfolio holding $N$ equities across multiple trading currencies, every local adjusted close price series $P_{i,t}^{\text{local}}$ (adjusted for splits and dividends via `yfinance` `auto_adjust=True`) is converted into the configured base reporting currency $B$ (default: `EUR`) using daily spot exchange rates $X_{B/c, t}$ (units of local currency $c$ per $1$ unit of base currency $B$):

$$P_{i,t}^{B} = \frac{P_{i,t}^{\text{local}}}{X_{B/c, t}}$$

Converting daily prices into base currency *prior* to computing returns ensures that continuously compounded base-currency log returns naturally embed both local equity volatility and foreign exchange (FX) translation risk, as well as the empirical covariance between local equity prices and the exchange rate:

$$r_{i,t}^{B} = \ln\left(\frac{P_{i,t}^{B}}{P_{i,t-1}^{B}}\right) = \ln\left(\frac{P_{i,t}^{\text{local}}}{P_{i,t-1}^{\text{local}}}\right) - \ln\left(\frac{X_{B/c, t}}{X_{B/c, t-1}}\right)$$

### 1.2 Calendar Alignment & Missing Data
International exchanges (e.g., NYSE/NASDAQ, Euronext Amsterdam/Paris/Brussels, XETRA, BME) observe different national holidays. A capped forward-fill (`ffill(limit=5)`) bridges short 1–2 day bank holiday mismatches by carrying forward the prior closing price (zero return on closed exchange days), while avoiding silent flatlining of delisted or suspended securities.

---

## 2. Value at Risk (VaR) & Expected Shortfall (CVaR) Models

Let $R_{p, h}$ denote the simple portfolio return over horizon $h$ trading days, where negative values represent losses. At confidence level $\alpha \in \{0.95, 0.99\}$:

* **Value at Risk ($\text{VaR}_\alpha$):** The loss threshold exceeded with probability $1 - \alpha$:
  $$\mathbb{P}(R_{p,h} \le -\text{VaR}_\alpha) = 1 - \alpha$$
* **Expected Shortfall / Conditional VaR ($\text{CVaR}_\alpha$):** The conditional expected loss given that the loss breaches $\text{VaR}_\alpha$:
  $$\text{CVaR}_\alpha = -\mathbb{E}\left[R_{p,h} \mid R_{p,h} \le -\text{VaR}_\alpha\right]$$

*(Note on Regulatory Standards: Basel III's Fundamental Review of the Trading Book [FRTB, MAR33] specifies Expected Shortfall at the **97.5%** confidence level over a liquidity-adjusted horizon. This framework evaluates **95% and 99%** VaR and CVaR to allow direct quantile-for-quantile comparison across models.)*

### 2.1 Method 1: Historical Simulation (Non-Parametric)
Applies current portfolio weights $w = (w_1, \dots, w_N)^T$ retroactively to realized historical simple return vectors $R_{i,t} = \exp(r_{i,t}^{B}) - 1$. Multi-day $h$-period returns are computed via rolling $h$-day compounded returns.
* **Assumptions:** Stationarity of the empirical return distribution; future joint asset behavior is fully represented by the historical lookback window.
* **Failure Modes:** Blind to structural breaks or tail events absent from the sample window; exhibits "ghost effects" when extreme historical dates roll out of the estimation window.

### 2.2 Method 2: Parametric Variance-Covariance (Delta-Normal)
Assumes asset log returns follow a multivariate normal distribution $\mathcal{N}(\mu_h, \Sigma_h)$ with $\Sigma_h = h \Sigma_{\text{daily}}$ and $\mu_h = 0$ over short risk horizons ($h \in \{1, 10\}$):

$$\sigma_{p,h} = \sqrt{w^T \Sigma_h w}, \qquad \text{VaR}_\alpha = z_\alpha \sigma_{p,h}, \qquad \text{CVaR}_\alpha = \sigma_{p,h} \frac{\phi(z_\alpha)}{1 - \alpha}$$

where $z_\alpha = \Phi^{-1}(\alpha)$ and $\phi(\cdot)$ is the standard normal probability density function.
* **Assumptions:** Joint normality, linear portfolio aggregation, and constant variance scaling ($\sqrt{h}$).
* **Failure Modes:** Financial returns exhibit excess kurtosis (fat tails) and negative skewness, causing Gaussian models to systematically underestimate 99% VaR and 99% CVaR.

### 2.3 Method 3: Gaussian Cholesky Monte Carlo (Multivariate GBM)
Factorizes the horizon covariance matrix via Cholesky decomposition $\Sigma_h = L L^T$, draws independent standard normal vectors $Z \sim \mathcal{N}(0, I_N)$, and simulates correlated asset log returns with Itô's Lemma convexity drift correction:

$$r_{\text{sim}, i} = (Z L^T)_i + \left(\mu_i - \frac{1}{2}\sigma_i^2\right)h, \qquad R_{p,\text{sim}} = \sum_{i=1}^{N} w_i \left(\exp(r_{\text{sim}, i}) - 1\right)$$

* **Assumptions:** Multivariate log-normality of asset prices, constant covariance $\Sigma$, and positive-definite covariance matrix.
* **Failure Modes:** Underestimates extreme joint crashes because Gaussian copulas have zero asymptotic tail dependence ($\lambda_L = 0$).

### 2.4 Method 4: Variance-Matched Multivariate Student-$t$ Monte Carlo
Estimates degrees of freedom $\nu$ via Maximum Likelihood Estimation (MLE) on historical portfolio returns (bounded within $\nu \in [3.0, 15.0]$ for finite variance) and scales correlated Gaussian shocks by a shared radial Chi-Square variable $W \sim \chi^2_\nu$:

$$\epsilon_t = (Z L^T) \cdot \sqrt{\frac{\nu}{W}} \cdot \sqrt{\frac{\nu - 2}{\nu}}$$

* **Why Variance Matching Matters:** Multiplying by $\sqrt{(\nu - 2)/\nu}$ forces $\text{Cov}(\epsilon_t) = \Sigma_h$ to match the Gaussian Monte Carlo exactly. This isolates pure **tail kurtosis** and **tail dependence** (because all $N$ assets share the same radial shock $W$ on a given simulation draw, extreme crashes hit all holdings simultaneously).
* **Failure Modes:** Elliptical symmetry assumes upside and downside tails have equal heaviness rather than modeling negative crash skewness or GARCH volatility clustering.

---

## 3. Euler Risk Attribution & Diversification Metrics

Because Parametric portfolio volatility $\sigma_p(w) = \sqrt{w^T \Sigma w}$ is homogeneous of degree $1$ in weights $w$, **Euler's Homogeneous Function Theorem** partitions 100% of portfolio VaR additively across positions:

$$\text{Marginal VaR}_i = \frac{\partial \text{VaR}_p}{\partial w_i} = z_\alpha \frac{(\Sigma w)_i}{\sigma_p}, \qquad \text{Component VaR}_i = w_i \cdot \text{Marginal VaR}_i, \qquad \sum_{i=1}^{N} \text{Component VaR}_i = \text{VaR}_p$$

* **Risk / Weight Ratio (Internal Portfolio Beta $\beta_{i,p}$):**
  $$\text{Risk / Weight Ratio}_i = \frac{\text{Component VaR}_i / \text{VaR}_p}{w_i} = \frac{\text{Cov}(r_i, r_p)}{\text{Var}(r_p)} = \beta_{i,p}$$
  Positions with $\beta_{i,p} > 1.25$ act as **Risk Amplifiers**, while positions with $\beta_{i,p} < 0.80$ act as **Diversifiers**.
* **Choueifaty Diversification Ratio ($\text{DR}$):** $\text{DR} = \frac{\sum_{i=1}^{N} w_i \sigma_i}{\sigma_p} \ge 1.0$.
* **Effective Number of Bets ($N_{\text{eff}}$):** $N_{\text{eff}}^{\text{cap}} = \frac{1}{\sum w_i^2}$ (capital breadth) and $N_{\text{eff}}^{\text{risk}} = \frac{1}{\sum \text{RC}_i^2}$ (risk breadth).

---

## 4. Out-of-Sample Backtesting & Crisis Stress Testing

### 4.1 Rolling Window & Kupiec (1995) POF Likelihood-Ratio Test
Over a rolling estimation window of $W = 252$ trading days, 1-day $\alpha$-VaR is forecast strictly out-of-sample for day $t+1$ using only observations $[t-W+1, t]$. Let $x$ be the number of realized breaches ($R_{p, t+1} < -\text{VaR}_{t+1}$) over $T$ out-of-sample days, $p = 1 - \alpha$, and $\hat{p} = x / T$. The Kupiec Proportion of Failures statistic tests $H_0: \hat{p} = p$:

$$LR_{\text{POF}} = -2 \ln \left[ \frac{(1 - p)^{T - x} p^x}{(1 - \hat{p})^{T - x} \hat{p}^x} \right] \stackrel{d}{\longrightarrow} \chi^2_1$$

If $p\text{-value} = 1 - F_{\chi^2_1}(LR_{\text{POF}}) < 0.05$ ($LR_{\text{POF}} > 3.841$), unconditional coverage is rejected. Note that Kupiec's test checks total breach frequency but does not test temporal independence (Christoffersen conditional coverage), which is why the advisory engine also checks for peak-year breach clustering.

### 4.2 Historical Crisis Replays & Pre-IPO Proxies
Replays the **2008–09 Global Financial Crisis** (`2008-09-15` to `2009-03-09`), **COVID-19 Liquidity Crash** (`2020-02-19` to `2020-03-23`), and **2022 Inflation & Rate Tightening** (`2022-01-03` to `2022-10-14`). Holdings lacking public trading history during a historical crisis window are automatically mapped to liquid sector ETFs configured in `sector_proxy_map`, and pre-inception benchmark windows fall back to `benchmark_crisis_proxy` (with exact data coverage % and proxy dates disclosed in the report).