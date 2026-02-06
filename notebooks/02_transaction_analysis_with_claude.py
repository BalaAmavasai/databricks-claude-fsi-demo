# Databricks notebook source
# MAGIC %md
# MAGIC # 02: Transaction Analysis with Claude
# MAGIC
# MAGIC ## Intelligent Risk Assessment Using Claude API
# MAGIC
# MAGIC This notebook demonstrates how Claude can analyze individual transactions and provide
# MAGIC human-readable risk assessments with detailed explanations.
# MAGIC
# MAGIC **What you'll learn:**
# MAGIC - Crafting effective prompts for transaction analysis
# MAGIC - Parsing structured responses from Claude
# MAGIC - Understanding Claude's reasoning capabilities for compliance

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install anthropic

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import anthropic
import json
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Load API key
ANTHROPIC_API_KEY = dbutils.secrets.get(scope="anthropic", key="api_key")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

print("✅ Claude client initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Enriched Transaction Data

# COMMAND ----------

# Load silver layer data
df_transactions = spark.table("silver_transactions")

# Get sample transactions for analysis
sample_normal = df_transactions.filter(F.col("fraud_label") == 0).limit(3).toPandas()
sample_suspicious = df_transactions.filter(F.col("fraud_label") == 1).limit(5).toPandas()

print(f"Loaded {len(sample_normal)} normal and {len(sample_suspicious)} suspicious transactions for analysis")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Single Transaction Analysis
# MAGIC
# MAGIC Let's start by analyzing a single suspicious transaction to understand Claude's capabilities.

# COMMAND ----------

# Select a suspicious transaction
txn = sample_suspicious.iloc[0].to_dict()

# Format transaction details for the prompt
txn_details = f"""
Transaction ID: {txn['transaction_id']}
Account ID: {txn['account_id']}
Date/Time: {txn['transaction_date']}
Type: {txn['transaction_type']}
Amount: ${txn['amount']:,.2f} {txn['currency']}
Originator Country: {txn['originator_country']}
Beneficiary Country: {txn['beneficiary_country']}
Merchant Category: {txn['merchant_category'] or 'N/A'}
Account Age: {txn['account_age_days']} days
New Beneficiary: {txn['is_new_beneficiary']}
Description: {txn['transaction_description']}
"""

print("Transaction to analyze:")
print(txn_details)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Analysis Prompt
# MAGIC
# MAGIC We'll ask Claude to provide a structured risk assessment.

# COMMAND ----------

ANALYSIS_PROMPT = """You are an expert financial crimes analyst specializing in AML (Anti-Money Laundering) and fraud detection. Analyze the following transaction for potential risk.

TRANSACTION DETAILS:
{transaction_details}

Provide your analysis in the following JSON format:
{{
    "risk_score": <integer 0-100>,
    "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>",
    "confidence": <float 0.0-1.0>,
    "risk_factors": [
        "<list of specific risk factors identified>"
    ],
    "explanation": "<2-3 sentence explanation suitable for a compliance report>",
    "recommended_action": "<APPROVE|REVIEW|ESCALATE|BLOCK>",
    "regulatory_flags": [
        "<any specific BSA/AML regulatory concerns>"
    ]
}}

Consider these risk indicators:
- Transaction structuring (amounts near $10,000 reporting threshold)
- High-risk jurisdictions (offshore financial centers, sanctioned countries)
- Rapid movement of funds (deposits quickly followed by international transfers)
- New accounts with unusual activity
- Inconsistent transaction patterns
- High-risk merchant categories (casinos, crypto exchanges, money services)

Respond ONLY with the JSON object, no additional text."""

# COMMAND ----------

# Analyze the transaction with Claude
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": ANALYSIS_PROMPT.format(transaction_details=txn_details)
        }
    ]
)

# Parse the response
analysis_text = response.content[0].text
print("Claude's Analysis:")
print(analysis_text)

# COMMAND ----------

# Parse JSON response
try:
    analysis = json.loads(analysis_text)
    print("\n" + "=" * 60)
    print("PARSED RISK ASSESSMENT")
    print("=" * 60)
    print(f"\n🎯 Risk Score: {analysis['risk_score']}/100")
    print(f"📊 Risk Level: {analysis['risk_level']}")
    print(f"🔒 Confidence: {analysis['confidence']*100:.0f}%")
    print(f"\n⚠️  Risk Factors:")
    for factor in analysis['risk_factors']:
        print(f"   • {factor}")
    print(f"\n📝 Explanation:\n   {analysis['explanation']}")
    print(f"\n🚦 Recommended Action: {analysis['recommended_action']}")
    if analysis.get('regulatory_flags'):
        print(f"\n🏛️  Regulatory Flags:")
        for flag in analysis['regulatory_flags']:
            print(f"   • {flag}")
except json.JSONDecodeError as e:
    print(f"Error parsing response: {e}")
    print("Raw response:", analysis_text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Analyze Multiple Transactions
# MAGIC
# MAGIC Now let's analyze several transactions and compare Claude's assessments.

# COMMAND ----------

def analyze_transaction(txn_dict):
    """Analyze a single transaction using Claude."""
    txn_details = f"""
Transaction ID: {txn_dict['transaction_id']}
Account ID: {txn_dict['account_id']}
Date/Time: {txn_dict['transaction_date']}
Type: {txn_dict['transaction_type']}
Amount: ${txn_dict['amount']:,.2f} {txn_dict['currency']}
Originator Country: {txn_dict['originator_country']}
Beneficiary Country: {txn_dict['beneficiary_country']}
Merchant Category: {txn_dict['merchant_category'] or 'N/A'}
Account Age: {txn_dict['account_age_days']} days
New Beneficiary: {txn_dict['is_new_beneficiary']}
Description: {txn_dict['transaction_description']}
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": ANALYSIS_PROMPT.format(transaction_details=txn_details)
            }
        ]
    )

    try:
        result = json.loads(response.content[0].text)
        result['transaction_id'] = txn_dict['transaction_id']
        result['actual_fraud_label'] = txn_dict['fraud_label']
        result['actual_fraud_type'] = txn_dict['fraud_type']
        return result
    except Exception as e:
        return {
            'transaction_id': txn_dict['transaction_id'],
            'error': str(e),
            'raw_response': response.content[0].text
        }

# COMMAND ----------

# Analyze all sample transactions
import time

all_results = []

# Analyze suspicious transactions
print("Analyzing suspicious transactions...")
for idx, row in sample_suspicious.iterrows():
    result = analyze_transaction(row.to_dict())
    all_results.append(result)
    print(f"  ✓ {result.get('transaction_id')}: Score={result.get('risk_score', 'N/A')}, Level={result.get('risk_level', 'N/A')}")
    time.sleep(0.5)  # Rate limiting

# Analyze normal transactions
print("\nAnalyzing normal transactions...")
for idx, row in sample_normal.iterrows():
    result = analyze_transaction(row.to_dict())
    all_results.append(result)
    print(f"  ✓ {result.get('transaction_id')}: Score={result.get('risk_score', 'N/A')}, Level={result.get('risk_level', 'N/A')}")
    time.sleep(0.5)

print(f"\n✅ Analyzed {len(all_results)} transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Compare Claude's Assessment vs Actual Labels

# COMMAND ----------

import pandas as pd

# Create comparison dataframe
comparison_data = []
for r in all_results:
    if 'error' not in r:
        comparison_data.append({
            'transaction_id': r['transaction_id'],
            'claude_risk_score': r['risk_score'],
            'claude_risk_level': r['risk_level'],
            'claude_action': r['recommended_action'],
            'actual_label': 'SUSPICIOUS' if r['actual_fraud_label'] == 1 else 'NORMAL',
            'actual_type': r['actual_fraud_type'] or 'N/A',
            'explanation': r['explanation'][:100] + '...' if len(r['explanation']) > 100 else r['explanation']
        })

df_comparison = pd.DataFrame(comparison_data)
display(df_comparison)

# COMMAND ----------

# Accuracy summary
print("=" * 60)
print("CLAUDE DETECTION ACCURACY")
print("=" * 60)

# Define what Claude considers suspicious
df_comparison['claude_flagged'] = df_comparison['claude_risk_level'].isin(['HIGH', 'CRITICAL'])
df_comparison['is_actually_suspicious'] = df_comparison['actual_label'] == 'SUSPICIOUS'

# Calculate metrics
true_positives = len(df_comparison[(df_comparison['claude_flagged']) & (df_comparison['is_actually_suspicious'])])
false_positives = len(df_comparison[(df_comparison['claude_flagged']) & (~df_comparison['is_actually_suspicious'])])
true_negatives = len(df_comparison[(~df_comparison['claude_flagged']) & (~df_comparison['is_actually_suspicious'])])
false_negatives = len(df_comparison[(~df_comparison['claude_flagged']) & (df_comparison['is_actually_suspicious'])])

total = len(df_comparison)
accuracy = (true_positives + true_negatives) / total if total > 0 else 0
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

print(f"\nSample Size: {total} transactions")
print(f"\nConfusion Matrix:")
print(f"  True Positives:  {true_positives}")
print(f"  False Positives: {false_positives}")
print(f"  True Negatives:  {true_negatives}")
print(f"  False Negatives: {false_negatives}")
print(f"\nMetrics:")
print(f"  Accuracy:  {accuracy*100:.1f}%")
print(f"  Precision: {precision*100:.1f}%")
print(f"  Recall:    {recall*100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Detailed Explanation Analysis
# MAGIC
# MAGIC One of Claude's key advantages is providing clear, auditable explanations.

# COMMAND ----------

# Display full explanations for suspicious transactions
print("=" * 60)
print("CLAUDE'S EXPLANATIONS FOR SUSPICIOUS TRANSACTIONS")
print("=" * 60)

for r in all_results:
    if r.get('actual_fraud_label') == 1 and 'error' not in r:
        print(f"\n📋 Transaction: {r['transaction_id']}")
        print(f"   Actual Type: {r['actual_fraud_type']}")
        print(f"   Claude Score: {r['risk_score']}/100 ({r['risk_level']})")
        print(f"\n   Risk Factors:")
        for factor in r.get('risk_factors', []):
            print(f"   • {factor}")
        print(f"\n   Explanation:")
        print(f"   {r['explanation']}")
        print("-" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Pattern Analysis Across Related Transactions
# MAGIC
# MAGIC Claude can also analyze patterns across multiple related transactions.

# COMMAND ----------

# Get transactions from a single account with suspicious activity
structuring_account = df_transactions.filter(
    (F.col("fraud_type") == "STRUCTURING")
).select("account_id").first()

if structuring_account:
    account_txns = df_transactions.filter(
        F.col("account_id") == structuring_account['account_id']
    ).orderBy("transaction_date").toPandas()

    # Format all transactions for pattern analysis
    txn_list = ""
    for idx, row in account_txns.iterrows():
        txn_list += f"""
Transaction {idx+1}:
  Date: {row['transaction_date']}
  Type: {row['transaction_type']}
  Amount: ${row['amount']:,.2f}
  To/From: {row['beneficiary_country']}
"""

    print(f"Account {structuring_account['account_id']} - {len(account_txns)} transactions")

# COMMAND ----------

# Pattern analysis prompt
PATTERN_PROMPT = """You are an expert financial crimes analyst. Analyze the following set of transactions from a single account for suspicious patterns.

ACCOUNT TRANSACTIONS:
{transactions}

Identify any concerning patterns and provide your analysis in JSON format:
{{
    "pattern_identified": "<name of pattern, e.g., STRUCTURING, LAYERING, RAPID_MOVEMENT>",
    "pattern_confidence": <float 0.0-1.0>,
    "pattern_description": "<detailed description of the suspicious pattern>",
    "key_indicators": [
        "<specific indicators that led to this conclusion>"
    ],
    "timeline_analysis": "<analysis of the timing and sequence of transactions>",
    "risk_assessment": "<overall risk assessment for this account>",
    "recommended_action": "<MONITOR|INVESTIGATE|FILE_SAR|CLOSE_ACCOUNT>",
    "sar_narrative_draft": "<draft narrative suitable for a Suspicious Activity Report>"
}}

Respond ONLY with the JSON object."""

if structuring_account:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": PATTERN_PROMPT.format(transactions=txn_list)
            }
        ]
    )

    pattern_analysis = json.loads(response.content[0].text)
    print("\n" + "=" * 60)
    print("PATTERN ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\n🔍 Pattern Identified: {pattern_analysis['pattern_identified']}")
    print(f"📊 Confidence: {pattern_analysis['pattern_confidence']*100:.0f}%")
    print(f"\n📝 Description:\n{pattern_analysis['pattern_description']}")
    print(f"\n⏱️  Timeline Analysis:\n{pattern_analysis['timeline_analysis']}")
    print(f"\n🚨 Risk Assessment:\n{pattern_analysis['risk_assessment']}")
    print(f"\n🚦 Recommended Action: {pattern_analysis['recommended_action']}")
    print(f"\n📄 Draft SAR Narrative:\n{pattern_analysis['sar_narrative_draft']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Key Takeaways
# MAGIC
# MAGIC ### Claude's Strengths for Transaction Analysis:
# MAGIC
# MAGIC 1. **Contextual Understanding**: Claude considers multiple factors together, not just individual thresholds
# MAGIC
# MAGIC 2. **Human-Readable Explanations**: Every assessment includes clear rationale suitable for audit trails
# MAGIC
# MAGIC 3. **Pattern Recognition**: Can identify sophisticated schemes spanning multiple transactions
# MAGIC
# MAGIC 4. **Regulatory Awareness**: Understands BSA/AML requirements and can draft SAR narratives
# MAGIC
# MAGIC 5. **Consistent Format**: Structured JSON output enables integration with existing workflows
# MAGIC
# MAGIC ### Next Steps:
# MAGIC - **Notebook 03**: Scale this to batch processing with Spark UDFs
# MAGIC - **Notebook 04**: Generate compliance reports and dashboards

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC *Demo notebook - Claude + Databricks Financial Services Risk Analysis*
