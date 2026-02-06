# Databricks notebook source
# MAGIC %md
# MAGIC # 04: Compliance Reporting
# MAGIC
# MAGIC ## Automated SAR Narratives & Executive Dashboards
# MAGIC
# MAGIC This notebook demonstrates how Claude can assist with:
# MAGIC - Generating Suspicious Activity Report (SAR) narratives
# MAGIC - Creating executive-level risk dashboards
# MAGIC - Producing audit-ready documentation
# MAGIC
# MAGIC **Target Audience:** Compliance Officers, Risk Managers, Executives

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install anthropic plotly kaleido

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import anthropic
import json
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pyspark.sql import functions as F
from pyspark.sql.types import *

# Configuration
ANTHROPIC_API_KEY = dbutils.secrets.get(scope="anthropic", key="api_key")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

print("✅ Environment ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Scored Transactions

# COMMAND ----------

# Load gold layer data
try:
    df_scored = spark.table("gold_transactions_scored")
    print(f"Loaded {df_scored.count()} scored transactions")
except:
    # Fallback to silver layer if gold not available
    df_scored = spark.table("silver_transactions")
    print(f"Loaded {df_scored.count()} transactions from silver layer")
    print("Note: Run notebook 03 first for full analysis results")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. SAR Narrative Generation
# MAGIC
# MAGIC When a transaction or pattern warrants a Suspicious Activity Report, Claude can draft
# MAGIC the narrative section that compliance officers typically spend hours writing.

# COMMAND ----------

# Select high-risk cases for SAR generation
df_high_risk = df_scored.filter(
    (F.col("fraud_label") == 1) |
    (F.col("risk_level").isin("HIGH", "CRITICAL") if "risk_level" in df_scored.columns else F.lit(False))
)

# Get a case study - structuring pattern
structuring_cases = df_scored.filter(F.col("fraud_type") == "STRUCTURING")

if structuring_cases.count() > 0:
    # Get all transactions for one account with structuring
    sample_account = structuring_cases.select("account_id").first()['account_id']
    case_transactions = df_scored.filter(F.col("account_id") == sample_account).orderBy("transaction_date").toPandas()
    print(f"Selected case: Account {sample_account} with {len(case_transactions)} transactions")
else:
    # Fallback
    case_transactions = df_scored.limit(5).toPandas()
    sample_account = case_transactions.iloc[0]['account_id']

# COMMAND ----------

# Format case data for SAR narrative
def format_case_for_sar(account_id: str, transactions: pd.DataFrame) -> str:
    """Format transaction data for SAR narrative generation."""

    txn_summary = ""
    total_amount = 0

    for idx, txn in transactions.iterrows():
        txn_summary += f"""
- {txn['transaction_date']}: {txn['transaction_type']} of ${txn['amount']:,.2f}
  From: {txn['originator_country']} → To: {txn['beneficiary_country']}
  Description: {txn['transaction_description']}
"""
        total_amount += txn['amount']

    case_summary = f"""
SUBJECT INFORMATION:
- Account ID: {account_id}
- Account Age: {transactions.iloc[0]['account_age_days']} days at time of first suspicious activity
- Review Period: {transactions['transaction_date'].min()} to {transactions['transaction_date'].max()}

TRANSACTION ACTIVITY:
- Total Transactions: {len(transactions)}
- Total Amount: ${total_amount:,.2f}
- Suspicious Pattern: {transactions.iloc[0].get('fraud_type', 'SUSPICIOUS_ACTIVITY')}

TRANSACTION DETAILS:
{txn_summary}

RISK INDICATORS IDENTIFIED:
"""

    # Add risk indicators based on data
    if transactions['amount'].between(9000, 10000).any():
        case_summary += "- Multiple cash transactions just below $10,000 CTR threshold\n"
    if (transactions['beneficiary_country'].isin(['CY', 'PA', 'VG', 'KY', 'BZ', 'MT'])).any():
        case_summary += "- Transactions involving high-risk jurisdictions\n"
    if transactions['is_new_beneficiary'].any():
        case_summary += "- Transfers to new/unknown beneficiaries\n"
    if transactions.iloc[0]['account_age_days'] < 90:
        case_summary += "- Recently opened account with unusual activity volume\n"

    return case_summary

case_data = format_case_for_sar(sample_account, case_transactions)
print(case_data)

# COMMAND ----------

# Generate SAR narrative with Claude
SAR_NARRATIVE_PROMPT = """You are a senior BSA/AML compliance officer at a major financial institution. Generate a professional Suspicious Activity Report (SAR) narrative for the following case.

CASE DATA:
{case_data}

Generate a complete SAR narrative following FinCEN guidelines. The narrative should:
1. Describe the suspicious activity clearly and concisely
2. Include all relevant dates, amounts, and parties
3. Explain why the activity is suspicious
4. Describe any patterns identified
5. Note any additional investigative steps taken or recommended
6. Be written in professional, objective language suitable for regulatory filing

Format the narrative with these sections:
- SUMMARY OF SUSPICIOUS ACTIVITY
- DETAILED DESCRIPTION OF ACTIVITY
- INDICATORS OF SUSPICIOUS ACTIVITY
- SUBJECT INFORMATION
- ADDITIONAL INFORMATION / RECOMMENDATIONS

The narrative should be 400-600 words."""

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    messages=[
        {
            "role": "user",
            "content": SAR_NARRATIVE_PROMPT.format(case_data=case_data)
        }
    ]
)

sar_narrative = response.content[0].text

print("=" * 70)
print("GENERATED SAR NARRATIVE")
print("=" * 70)
print(sar_narrative)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Batch SAR Generation
# MAGIC
# MAGIC Generate narratives for multiple cases efficiently.

# COMMAND ----------

def generate_sar_narrative(account_id: str, transactions_pdf: pd.DataFrame) -> dict:
    """Generate SAR narrative for a single case."""
    case_data = format_case_for_sar(account_id, transactions_pdf)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": SAR_NARRATIVE_PROMPT.format(case_data=case_data)
            }
        ]
    )

    return {
        'account_id': account_id,
        'narrative': response.content[0].text,
        'generated_at': datetime.now().isoformat(),
        'transaction_count': len(transactions_pdf),
        'total_amount': transactions_pdf['amount'].sum()
    }

# Generate SARs for top suspicious accounts
suspicious_accounts = df_scored.filter(F.col("fraud_label") == 1) \
    .groupBy("account_id") \
    .agg(F.count("*").alias("txn_count"), F.sum("amount").alias("total_amount")) \
    .orderBy(F.desc("total_amount")) \
    .limit(3) \
    .collect()

print(f"Generating SARs for {len(suspicious_accounts)} accounts...")

sar_reports = []
for row in suspicious_accounts:
    account_txns = df_scored.filter(F.col("account_id") == row['account_id']).toPandas()
    sar = generate_sar_narrative(row['account_id'], account_txns)
    sar_reports.append(sar)
    print(f"  ✓ Generated SAR for {row['account_id']}")

# COMMAND ----------

# Display generated SARs
for i, sar in enumerate(sar_reports):
    print(f"\n{'='*70}")
    print(f"SAR #{i+1} - Account: {sar['account_id']}")
    print(f"Transactions: {sar['transaction_count']}, Total: ${sar['total_amount']:,.2f}")
    print(f"{'='*70}")
    print(sar['narrative'][:1500] + "..." if len(sar['narrative']) > 1500 else sar['narrative'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Executive Dashboard Data Preparation

# COMMAND ----------

# Prepare dashboard metrics
pdf_scored = df_scored.toPandas()

# Summary metrics
total_transactions = len(pdf_scored)
total_amount = pdf_scored['amount'].sum()
suspicious_count = pdf_scored[pdf_scored['fraud_label'] == 1].shape[0]
suspicious_amount = pdf_scored[pdf_scored['fraud_label'] == 1]['amount'].sum()

print("=" * 50)
print("EXECUTIVE SUMMARY METRICS")
print("=" * 50)
print(f"Total Transactions Analyzed: {total_transactions:,}")
print(f"Total Transaction Value: ${total_amount:,.2f}")
print(f"Suspicious Transactions: {suspicious_count:,} ({suspicious_count/total_transactions*100:.1f}%)")
print(f"Suspicious Value: ${suspicious_amount:,.2f} ({suspicious_amount/total_amount*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Interactive Dashboards

# COMMAND ----------

# Risk Distribution Chart
if 'risk_level' in pdf_scored.columns and pdf_scored['risk_level'].notna().any():
    risk_dist = pdf_scored.groupby('risk_level').agg({
        'transaction_id': 'count',
        'amount': 'sum'
    }).reset_index()
    risk_dist.columns = ['Risk Level', 'Transaction Count', 'Total Amount']

    fig1 = px.bar(
        risk_dist,
        x='Risk Level',
        y='Transaction Count',
        color='Risk Level',
        color_discrete_map={
            'LOW': '#2ecc71',
            'MEDIUM': '#f39c12',
            'HIGH': '#e74c3c',
            'CRITICAL': '#8e44ad'
        },
        title='Transaction Risk Distribution'
    )
    fig1.update_layout(showlegend=False)
    fig1.show()

# COMMAND ----------

# Fraud Type Breakdown
fraud_breakdown = pdf_scored[pdf_scored['fraud_label'] == 1].groupby('fraud_type').agg({
    'transaction_id': 'count',
    'amount': 'sum'
}).reset_index()
fraud_breakdown.columns = ['Fraud Type', 'Count', 'Total Amount']

fig2 = px.pie(
    fraud_breakdown,
    values='Count',
    names='Fraud Type',
    title='Suspicious Activity by Type',
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig2.show()

# COMMAND ----------

# Geographic Risk Analysis
country_risk = pdf_scored.groupby('beneficiary_country').agg({
    'transaction_id': 'count',
    'amount': 'sum',
    'fraud_label': 'sum'
}).reset_index()
country_risk.columns = ['Country', 'Transactions', 'Total Amount', 'Suspicious']
country_risk['Suspicious Rate'] = country_risk['Suspicious'] / country_risk['Transactions'] * 100

fig3 = px.scatter(
    country_risk,
    x='Transactions',
    y='Suspicious Rate',
    size='Total Amount',
    color='Suspicious Rate',
    hover_name='Country',
    color_continuous_scale='RdYlGn_r',
    title='Geographic Risk Profile'
)
fig3.update_layout(
    xaxis_title='Number of Transactions',
    yaxis_title='Suspicious Rate (%)'
)
fig3.show()

# COMMAND ----------

# Transaction Amount Distribution by Risk
fig4 = px.histogram(
    pdf_scored,
    x='amount',
    color='fraud_label',
    nbins=50,
    title='Transaction Amount Distribution',
    labels={'fraud_label': 'Suspicious', 'amount': 'Amount ($)'},
    color_discrete_map={0: '#3498db', 1: '#e74c3c'}
)
fig4.update_layout(barmode='overlay')
fig4.update_traces(opacity=0.7)
fig4.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Generate Executive Summary Report

# COMMAND ----------

# Prepare executive summary data
summary_data = f"""
TRANSACTION MONITORING SUMMARY
Period: Last 30 Days
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

KEY METRICS:
- Total Transactions Analyzed: {total_transactions:,}
- Total Transaction Value: ${total_amount:,.2f}
- Suspicious Transactions Identified: {suspicious_count:,} ({suspicious_count/total_transactions*100:.1f}%)
- Suspicious Transaction Value: ${suspicious_amount:,.2f}

SUSPICIOUS ACTIVITY BREAKDOWN:
{fraud_breakdown.to_string(index=False)}

TOP RISK COUNTRIES:
{country_risk.nlargest(5, 'Suspicious Rate')[['Country', 'Transactions', 'Suspicious Rate']].to_string(index=False)}

NOTABLE PATTERNS:
- Structuring activity detected in {pdf_scored[pdf_scored['fraud_type'] == 'STRUCTURING']['account_id'].nunique()} accounts
- International wire transfers to high-risk jurisdictions: {len(pdf_scored[(pdf_scored['transaction_type'] == 'WIRE_INTERNATIONAL') & (pdf_scored['fraud_label'] == 1)])} transactions
- New accounts (<30 days) with suspicious activity: {len(pdf_scored[(pdf_scored['account_age_days'] < 30) & (pdf_scored['fraud_label'] == 1)])} cases
"""

# Generate executive insights with Claude
EXECUTIVE_PROMPT = """You are a Chief Compliance Officer preparing a board-level summary of AML/fraud monitoring results.

DATA SUMMARY:
{summary_data}

Generate a concise executive summary (300-400 words) that:
1. Highlights key findings and trends
2. Identifies areas of concern
3. Provides context for the metrics
4. Recommends strategic actions
5. Uses professional, board-appropriate language

Focus on business impact and risk implications, not technical details."""

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": EXECUTIVE_PROMPT.format(summary_data=summary_data)
        }
    ]
)

executive_summary = response.content[0].text

print("=" * 70)
print("EXECUTIVE SUMMARY - BOARD REPORT")
print("=" * 70)
print(executive_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Export Reports

# COMMAND ----------

# Save SAR narratives to Delta
sar_df = spark.createDataFrame(pd.DataFrame(sar_reports))
sar_df.write.format("delta") \
    .mode("overwrite") \
    .save("/FileStore/demo/claude_fsi/gold/sar_narratives")

print("✅ SAR narratives saved to Delta table")

# COMMAND ----------

# Create comprehensive report document
report_content = f"""
================================================================================
                    TRANSACTION MONITORING COMPLIANCE REPORT
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

SECTION 1: EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
{executive_summary}

SECTION 2: KEY METRICS
--------------------------------------------------------------------------------
Total Transactions Analyzed:     {total_transactions:>15,}
Total Transaction Value:         ${total_amount:>14,.2f}
Suspicious Transactions:         {suspicious_count:>15,} ({suspicious_count/total_transactions*100:.1f}%)
Suspicious Value:                ${suspicious_amount:>14,.2f}

SECTION 3: SUSPICIOUS ACTIVITY BREAKDOWN
--------------------------------------------------------------------------------
{fraud_breakdown.to_string(index=False)}

SECTION 4: GEOGRAPHIC RISK ANALYSIS
--------------------------------------------------------------------------------
{country_risk.nlargest(10, 'Suspicious Rate').to_string(index=False)}

SECTION 5: SAR FILINGS PREPARED
--------------------------------------------------------------------------------
Total SARs Generated: {len(sar_reports)}

"""

for i, sar in enumerate(sar_reports):
    report_content += f"""
SAR #{i+1}
Account: {sar['account_id']}
Transactions: {sar['transaction_count']} | Total Amount: ${sar['total_amount']:,.2f}
Generated: {sar['generated_at']}

{sar['narrative']}

{'='*80}
"""

# Save report
report_path = "/FileStore/demo/claude_fsi/reports"
dbutils.fs.mkdirs(report_path)

# Write to DBFS
dbutils.fs.put(
    f"{report_path}/compliance_report_{datetime.now().strftime('%Y%m%d')}.txt",
    report_content,
    overwrite=True
)

print(f"✅ Report saved to {report_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary & Key Benefits
# MAGIC
# MAGIC ### What Claude Delivers for Compliance:
# MAGIC
# MAGIC | Traditional Approach | With Claude |
# MAGIC |---------------------|-------------|
# MAGIC | 2-4 hours per SAR narrative | Minutes per narrative |
# MAGIC | Inconsistent quality across analysts | Consistent, comprehensive coverage |
# MAGIC | Manual pattern documentation | Automated pattern explanation |
# MAGIC | Periodic manual reviews | Continuous intelligent monitoring |
# MAGIC
# MAGIC ### Compliance Officer Benefits:
# MAGIC 1. **Time Savings**: Reduce SAR drafting time by 80%+
# MAGIC 2. **Consistency**: Every narrative follows regulatory guidelines
# MAGIC 3. **Coverage**: Analyze 100% of transactions, not just samples
# MAGIC 4. **Documentation**: Complete audit trail for every decision
# MAGIC
# MAGIC ### Executive Benefits:
# MAGIC 1. **Real-time Visibility**: Dashboards updated with each analysis run
# MAGIC 2. **Risk Quantification**: Clear metrics on exposure and trends
# MAGIC 3. **Regulatory Readiness**: Always prepared for examinations
# MAGIC 4. **Cost Efficiency**: Reduce manual review costs significantly

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC *Demo notebook - Claude + Databricks Financial Services Risk Analysis*
