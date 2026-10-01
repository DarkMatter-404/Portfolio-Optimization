# Portfolio Optimization as a Linear Regression

A machine-learning group project that uses portfolio construction to study **overfitting**: we rewrite the
minimum-variance portfolio as a linear regression, show that it overfits, and test how far LASSO, Ridge and a few
reasoned constraints can close the gap to a simple equally weighted portfolio.

**NUS MSc in Business Analytics · DBA5106 Foundations of Business Analytics · Group 20 · September 2026**

> **Main finding:** The minimum-variance portfolio's daily risk was **0.18% inside its training windows but 0.88% on
> the following days**, almost five times larger: a direct picture of overfitting. Regularization and constraints
> closed most of that gap. Our selection rule, fixed in advance, chose **Ridge with one weight per company-size group**
> (daily Sharpe ratio **0.100** over 2025–2026 against **0.062** for equal weights), but its lead is smaller than
> the statistical noise, and we say so.

[Read the final report](DBA5106_Group20_Project_Report.pdf) · [Open the notebook](DBA5106_Group20_Portfolio_Optimization.ipynb)

## The assignment

From the assignment brief:

> *"This project does not apply machine learning to portfolio optimization. Instead, it uses a portfolio
> optimization framework to examine the critical issue of overfitting within traditional machine learning models."*

Every trading day, money is split across **98 assets** (US stock portfolios sorted by company size and
book-to-market). Two classic rules are compared:

- **Equally weighted (EW):** 1/98 of the money in each asset.
- **Minimum variance (MinVar):** the weights whose return had the lowest variance over the previous **126 trading
  days**, re-estimated every day on a rolling window.

On the validation period (2026-01-02 to 2026-07-31), the "optimized" MinVar portfolio reaches a daily Sharpe ratio of
only **0.021**, against **0.111** for EW. This echoes DeMiguel, Garlappi and Uppal (2009), where none of 14
sophisticated models consistently beats the 1/N rule.

The brief then turns MinVar into a regression. The weights are written as $w = w_{EW} - N\beta$, where $N$ is
the 97 × 97 identity matrix with a row of −1s added below it, so the weights always sum to one. With $R$ the matrix
of past returns, MinVar becomes an ordinary least-squares problem:

$$\min_{\alpha,\beta}\ \lVert y - (\alpha + X\beta) \rVert_2^2, \qquad y = R w_{EW}, \qquad X = RN .$$

With 97 coefficients estimated from 126 days, overfitting is expected. What we were asked to do:

- **Required:** build LASSO and Ridge portfolios, with the penalty chosen by cross-validation, and report their
  out-of-sample performance (daily Sharpe ratio) on the validation period.
- **The open part:** *"Think about ways to improve! Use the data + loss + structure + constraint framework!"*
  A perfect LASSO/Ridge implementation was worth only 70% of the grade; the brief warns that *"Playing safe will not
  be enough!"* and that *"The way you think is more important than the performance."*
- **Rules:** stay within the regression framework (solving the minimum-variance problem directly scores low), keep
  the August 2026 test month untouched (the instructor holds it), and report in six pages.

## What we did

1. **Rebuilt the set-up** and reproduced the brief's reference numbers exactly (EW 0.111, MinVar 0.021, plus three
   debugging values).
2. **Ran the baseline** (EW, MinVar, LASSO, Ridge) and measured overfitting directly: the SD of each portfolio's
   return inside its training window against the SD on the days that followed.
3. **Planned four explorations before running them**, one per part of the course framework, each applied to both
   LASSO and Ridge. We kept the list short on purpose, to limit the *optimizer's curse* (picking the luckiest of many
   versions judged on the same days). After taking stock of the results, we added a fifth idea from our plan: no
   negative weights.
4. **Fixed the selection rule in advance:** among the LASSO and Ridge versions, choose the highest daily Sharpe ratio
   over 2025 and 2026 together (395 days). We added 2025 as an extra check because 145 days is short.

### Baseline: overfitting made visible (2026 validation period)

| Portfolio | Sharpe ratio | SD on new days (%) | SD inside training window (%) | Sum of absolute weights |
| --- | ---: | ---: | ---: | ---: |
| EW (benchmark) | **0.111** | 0.958 | 1.021 | 1.0 |
| MinVar (OLS) | 0.021 | 0.877 | 0.183 | 12.2 |
| LASSO | 0.061 | 0.555 | 0.351 | 3.2 |
| Ridge | −0.004 | 0.581 | 0.309 | 4.2 |

MinVar looks calm in training and is not on new days, and it uses extreme long/short weights. LASSO and Ridge pull
the weights toward EW and give the lowest risk of the four. In 2026 they still trail EW, because minimizing risk
moved the money into large companies, which rose the least that year. In 2025 EW came last of the four.

### Five explorations

| Part of the framework | Idea | Why we tried it | What we found |
| --- | --- | --- | --- |
| Data | Longer window (1008 days) | ~10 observations per coefficient needs ~970 days | Overfitting gap closed; Sharpe ratio mixed (Ridge up, LASSO flat) |
| Loss | Choose λ by the Sharpe ratio, not MSE | Match the tuning loss to the goal | Worse: a Sharpe ratio from ~25-day folds is mostly noise |
| Structure | One weight per company-size group | Similar-size assets move together; 9 coefficients instead of 97 | Best in 2025, weak in 2026: its tilt between size groups reversed |
| Constraints | Larger λ (one-standard-error rule) | The CV-optimal λ is itself noisy | Helped both models in 2026 without hurting 2025 |
| Constraints | No negative weights | "Wrong" constraints can still cut variance | Only versions above EW in both years; money in very few assets |

| Version | Sharpe 2026 | Sharpe 2025 | Sharpe, both periods |
| --- | ---: | ---: | ---: |
| EW (benchmark) | 0.111 | 0.043 | 0.062 |
| LASSO / Ridge (baseline) | 0.061 / −0.004 | 0.052 / 0.051 | 0.053 / 0.035 |
| 1008-day window | 0.056 / 0.051 | 0.051 / 0.053 | 0.052 / 0.052 |
| λ chosen by Sharpe | 0.018 / −0.012 | 0.004 / 0.053 | 0.007 / 0.034 |
| Size groups | 0.029 / **0.025** | 0.120 / **0.132** | 0.093 / **0.100** |
| One-standard-error λ | 0.095 / 0.034 | 0.052 / 0.056 | 0.061 / 0.049 |
| No negative weights | 0.130 / 0.134 | 0.063 / 0.067 | 0.078 / 0.081 |

*Each cell shows LASSO / Ridge. Bold: our final choice, Ridge with size groups. Source: report, Table 3.*

### Final choice and how confident we are

By the rule fixed in advance, the final procedure is **Ridge with one weight per company-size group**. We are open
about its weakness. Its lead over EW comes from 2025; in 2026 it was below EW. A daily Sharpe ratio over 395 days has
a standard error of about 0.05, which is larger than the lead. We kept the rule anyway, because switching after seeing
the results would reintroduce the optimizer's curse.

### What we learned

- Overfitting can be measured directly, by comparing in-sample risk with risk on the next days.
- Every way of limiting the freedom of β narrowed that gap: more data, fewer coefficients, a larger penalty, no
  negative weights. LASSO and Ridge are one kind of constraint among several.
- Lower risk does not guarantee a higher Sharpe ratio, because average returns are very hard to predict.
- A loss must be estimable as well as relevant: the Sharpe ratio is the goal, but 25 days are too few to estimate it.

## Repository contents

| File | What it contains |
| --- | --- |
| [DBA5106_Group20_Project_Report.pdf](DBA5106_Group20_Project_Report.pdf) | Final group report (cover, six pages, one-page appendix): method, every exploration with its reasoning, results, final choice and limitations. |
| [DBA5106_Group20_Portfolio_Optimization.ipynb](DBA5106_Group20_Portfolio_Optimization.ipynb) | The complete analysis: data checks, MinVar as a regression, LASSO and Ridge, the five explorations, the rolling-window backtest, and all tables and figures, with saved outputs. |

Start with the report for the argument, then open the notebook. Its saved outputs show every result without running
any code.

## Data

We used the daily **100 Portfolios Formed on Size and Book-to-Market (10 × 10)** file from
[Kenneth French's Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). We kept
only its second table (*Average Equal Weighted Returns, Daily*), with returns in percent. We removed `ME9 BM10` and
`BIG HiBM`, which have missing data, leaving 98 assets. Our copy runs from 1926-07-01 to 2026-07-31. The data are not
included in this repository.

## Run the notebook

```bash
git clone https://github.com/DarkMatter-404/Portfolio-Optimization.git
cd Portfolio-Optimization
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.5.3 pandas==3.0.6 scikit-learn==1.9.1 cvxpy==1.9.3 clarabel==0.11.1 matplotlib==3.11.2 jupyterlab
python -m jupyter lab
```

Before running, download the daily file from the library above. Keep only the second table (its one title line,
then the header and data), save it as `100_Portfolios_10x10_Daily.csv` next to the notebook, and run all cells. The
notebook writes its daily results and figures to `results/`. Without saved results, the full rolling-window backtest
for both periods takes about three hours on a laptop. French's library revises past data over time, so a newer
download can give slightly different numbers.

**Publication checks:** the notebook is the submitted file, byte for byte, renamed from `portfolio_project1.ipynb`.
The report is the submitted PDF with one change on the cover: student IDs were removed and the code file name updated
to match this repository. Pages 2 to 8 are unchanged. A fresh end-to-end run in a new environment has not been
repeated for this portfolio edition.

## Team and provenance

This is a **group project**, documented on Aman Raj Mohta's portfolio. Team members:

- Aman Raj Mohta
- Subham Kundu
- Zhu DingCheng
- Maksim Silchenko

The report and notebook are the team's final submission. The project was developed in a private team repository;
this portfolio repository presents the completed work and does not claim sole authorship of it. As the course
encouraged, we used Claude (Anthropic) to help write and check code and to draft the report. The report's AI
declaration gives the details.

The assignment objective follows the course brief *Portfolio Optimization* by Dr. Long Zhao (NUS, 2026). The data
come from Kenneth French's Data Library and are not redistributed here. No open-source licence is asserted for the
jointly authored submission.
