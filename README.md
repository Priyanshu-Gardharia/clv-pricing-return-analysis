# Customer Lifetime Value with Pricing & Return Behavior Analysis

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Project Summary

Standard CLV models rank customers by how much they spend.
This project ranks customers by how much they are worth —
after accounting for what they return.

Analysis of 4,338 customers across 541,909 transactions
revealed £9,889,403 in hidden return losses that standard
CLV models completely ignore — 85.1% caused by just 69
Whale Buyers who represent only 1.6% of all customers.

---

## The Problem

Every e-commerce business tracks Customer Lifetime Value.
But standard CLV ignores returns completely.

Example from this dataset:
CustomerID 12346 spent £77,183 in one order and returned
£77,183. Standard CLV = £77,183. Adjusted CLV = £0.
Zero contribution to the business — yet a standard model
would treat this customer as a top-priority VIP.

This project builds Adjusted CLV — true revenue after
subtracting return losses — and proves that return behaviour
is a genuine mathematical driver of customer value.

---

## Dataset

Source        : UK Retail Dataset — Kaggle
Raw rows      : 541,909 transactions
Customers     : 4,338 unique customers
Period        : December 2010 to December 2011
Link          : kaggle.com/datasets/carrie1/ecommerce-data

---

## Tech Stack

Python 3.12      — Core analysis
Pandas           — Data cleaning and feature engineering
NumPy            — Mathematical transformations
Matplotlib       — Charts and visualisations
Seaborn          — Statistical charts and heatmaps
Scikit-learn     — Linear Regression and evaluation metrics
XGBoost          — CLV prediction model
KMeans           — Customer segmentation
Google Colab     — Development environment
Power BI         — Interactive dashboard

---

## Key Innovation

Most CLV projects do this on Day 1 and never look back:

    df = df[df['Quantity'] > 0]

This project does this instead:

    df_returns = df[df['InvoiceNo'].str.startswith('C')].copy()
    df_returns['ReturnValue'] = df_returns['Quantity'].abs() * df_returns['UnitPrice']

Saving cancellations BEFORE cleaning enables return rate
calculation per customer, Adjusted CLV, and identification
of deceptive customers whose apparent value is destroyed
by return losses.

---

## Project Structure

Day 1  — Setup and data exploration
Day 2  — Data cleaning — two dataframes (purchases + returns)
Day 3  — RFM calculation and return rate per customer
Day 4  — Pricing behaviour analysis
Day 5  — Return behaviour deep dive
Day 6  — Feature engineering and Adjusted CLV
Day 7  — KMeans clustering — 4 customer personas
Day 8  — XGBoost CLV prediction model

---

## Results

CUSTOMER PERSONAS

Persona            Customers    Pct    Avg CLV True    Total Return Loss
Regular Buyers     3,959        91.3%  £4,458          £431,603
Champion Buyers    283          6.5%   £106,677         £951,913
Whale Buyers       69           1.6%   £2,111,199       £8,418,482
Extreme Returners  27           0.6%   £378             £87,405

REVENUE IMPACT

Gross Revenue                  £8,911,408
Net Revenue after returns      £8,313,295
Total Return Loss              £9,889,403
Whale Buyer Return Loss        £8,418,482  (85.1% of total)

MACHINE LEARNING PERFORMANCE

Model                   R2        RMSE
Linear Regression       0.9390    0.5218
XGBoost                 0.9990    0.0660
Improvement             +6.4%     8x lower error

FEATURE IMPORTANCE — TOP 5 DRIVERS OF TRUE CLV

Rank 1   M_Score              0.4330   RFM feature
Rank 2   Frequency            0.2219   RFM feature
Rank 3   Monetary             0.2133   RFM feature
Rank 4   Return_Value_Rate    top 36%  Return behaviour
Rank 5   Return_Damage        top 43%  Return behaviour

Return behaviour features ranked in the top half of all
14 features — mathematical proof that return behaviour
drives true customer value.

---

## Key Findings

FINDING 1 — THE £9.89M HIDDEN GAP
£9,889,403 separates gross CLV from true CLV across all
customers. Completely invisible to standard CLV models
that delete cancelled orders without analysing them.

FINDING 2 — THE 1.6% PROBLEM
69 Whale Buyers are 1.6% of customers but cause £8,418,482
which is 85.1% of all return losses. They appear to be the
most valuable customers by spend but are the most damaging
by return impact.

FINDING 3 — PRICE DOES NOT PREDICT RETURNS
Correlation between average unit price and return rate
is 0.019 — near zero. Buying expensive items does not
make a customer more likely to return them. This contradicts
common retail assumptions.

FINDING 4 — HIGH VALUE BUYERS RETURN 350% MORE VALUE
Customers in the top 25% by unit price have Return_Value_Rate
of 0.0952 versus 0.0211 for regular buyers. That is 350.6%
higher proportional return value despite price having near
zero correlation with return frequency.

FINDING 5 — EXTREME RETURNERS COST MORE THAN THEY EARN
27 customers returned more value than they spent. Their
combined true CLV is only £10,228 despite £90,555 in gross
revenue. Return_Damage score above 1.0 for all 27.

---

## Business Recommendations

CHAMPION BUYERS — 283 customers
Enroll in VIP loyalty programme immediately.
Offer early access to new products before public launch.
Personalised outreach from senior team.
Goal: Retain at all costs. Lowest return damage at 2.98%.

REGULAR BUYERS — 3,959 customers
Frequency campaigns to trigger repeat purchases.
Bundle deals to increase average order value.
Personalised product recommendations based on history.
Goal: Grow frequency from 3 to 5 orders — doubles CLV.

WHALE BUYERS — 69 customers
URGENT: Assign dedicated account manager to each customer.
Implement return friction — require return reason codes.
Investigate return causes — quality issues, sizing, impulse?
Consider B2B contracts with return rate clauses.
Goal: Reducing return rate by 20% saves £1,683,696.

EXTREME RETURNERS — 27 customers
Flag all 27 accounts for immediate manual review.
Suspend unrestricted return privileges pending investigation.
Escalate suspected fraud cases to fraud team.
Goal: True CLV of £10,228 total does not justify continued
unrestricted return access.

---

## How to Run

Step 1 — Download data
Go to kaggle.com/datasets/carrie1/ecommerce-data
Download data.csv

Step 2 — Open Google Colab
Go to colab.research.google.com
Upload data.csv to session storage

Step 3 — Run the notebook
Open clv_analysis.ipynb
Click Runtime — Run All

All cells include detailed comments explaining what each
step does and why it is done that way.

---

## Files in This Repository

clv_analysis.ipynb        Main notebook — all 8 days of work
clv_master_final.csv      Master table — 4,338 customers, 28 features
clv_persona_summary.csv   Persona level summary table
clv_segment_summary.csv   RFM segment summary table
clv_bucket_summary.csv    Price bucket summary table

---

## What I Learned

Separating cancelled orders before cleaning is the single
most important decision in this project. It enables return
behaviour analysis that standard CLV projects never attempt.

KMeans clustering on raw data with extreme outliers gives
meaningless results — 99.8% of customers end up in one
cluster. Outlier separation before clustering is essential.

Log transformation of skewed CLV values is required for
XGBoost to learn effectively across all customer types.

Return behaviour features rank in the top half of ML
feature importance — proving they are genuine mathematical
drivers of true customer value, not just business intuition.

The difference between gross CLV and true CLV is not a
small rounding error. It is a £9,889,403 business problem
that most analytics teams are currently not measuring.

---

## Connect

Name     : Priyanshu Gardharia
LinkedIn : https://www.linkedin.com/in/priyanshu-gardharia-3aa791326/
Email    : priyanshugardharia@gmail.com

---

Built as a 10-day data science portfolio project.
Dataset  : UK Retail — Kaggle
Tools    : Python — XGBoost — KMeans — Power BI
