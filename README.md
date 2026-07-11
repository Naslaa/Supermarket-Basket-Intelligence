# Supermarket-Basket-Intelligence

Market Basket Analysis and Customer Segmentation

Overview

This project applies a full data science pipeline to retail transaction data, covering data wrangling, descriptive and inferential statistics, unsupervised learning (association rule mining and customer segmentation), supervised learning (high-value customer prediction), and a discussion of ethical and deployment considerations. It was originally developed as a personal project and extended to serve as applied evidence supporting an RPL (Recognition of Prior Learning) application.

Dataset

The dataset follows the schema of the well-known UCI Online Retail dataset (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country) — transactional records from a UK-based online retailer.

Note: the original analysis was developed on the full UCI dataset (541,909 transactions). This copy of the notebook runs on a representative sample dataset (~7,000 transactions, ~350 customers) generated to match the same schema and realistic data quirks (cancelled orders, missing customer IDs, product co-purchase patterns), so that every cell in the notebook executes end-to-end and all outputs shown are genuinely computed rather than illustrative. See the "Note on data used" cell in the notebook for detail.

Objectives


Clean and prepare raw transactional data for analysis.
Explore sales trends by product, country, and basket size.
Test a specific business hypothesis using inferential statistics.
Identify frequently co-purchased products using association rule mining (market basket analysis).
Segment customers by purchasing behaviour using RFM features and clustering.
Predict high-value customers using a supervised classification model.
Reflect on the ethical implications of the analysis and how it would be deployed in practice.

Key Findings


Association rules: several product pairs/groups (e.g. tea-set items, Christmas décor items) show lift > 1, indicating genuine cross-sell opportunities.
Customer segments: K-Means (k=4, chosen via the elbow method) on RFM features identifies distinct groups, including a high-value, low-recency "champion" segment and a lapsing "at-risk" segment — each with different average recency, frequency, and spend.
Inferential test: no statistically significant difference in average transaction spend between UK and non-UK customers (t = -0.79, p = 0.43 at α = 0.05).
Predictive model: a logistic regression classifier using only Recency and Frequency predicts high-value customers with 95% test accuracy, evaluated with a full precision/recall/F1 breakdown and confusion matrix.


Business Recommendations


Target the "at-risk" segment with win-back campaigns before they fully lapse.
Use top association rules to design bundle promotions and checkout cross-sell prompts.
Use the classifier as an early-warning signal in the CRM to flag emerging high-value customers.
Re-run the pipeline on a recurring schedule as new transaction data arrives.

Tools & Libraries

Python, pandas, NumPy, Matplotlib, Seaborn, SciPy (hypothesis testing), mlxtend (Apriori / association rules), scikit-learn (K-Means, logistic regression, preprocessing).