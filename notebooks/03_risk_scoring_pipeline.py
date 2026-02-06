# Databricks notebook source
# MAGIC %md
# MAGIC # 03: Risk Scoring Pipeline
# MAGIC
# MAGIC ## Scalable Batch Processing with Claude
# MAGIC
# MAGIC This notebook demonstrates how to process transactions at scale using:
# MAGIC - Pandas UDFs for distributed Claude API calls
# MAGIC - Rate limiting and error handling
# MAGIC - MLflow for experiment tracking
# MAGIC - Caching strategies for cost optimization
# MAGIC
# MAGIC **Target Audience:** Data Engineers, ML Engineers

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install anthropic mlflow tenacity

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import anthropic
import json
import time
import hashlib
from datetime import datetime
from typing import Iterator

import pandas as pd
import numpy as np
import mlflow
from tenacity import retry, stop_after_attempt, wait_exponential

from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType

# Configuration
ANTHROPIC_API_KEY = dbutils.secrets.get(scope="anthropic", key="api_key")
MODEL_NAME = "claude-sonnet-4-20250514"
MAX_RETRIES = 3
BATCH_SIZE = 10
RATE_LIMIT_DELAY = 0.5  # seconds between API calls

print("✅ Environment configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define Paths and Load Data

# COMMAND ----------

# Paths
DEMO_PATH = "/FileStore/demo/claude_fsi"
SILVER_PATH = f"{DEMO_PATH}/silver"
GOLD_PATH = f"{DEMO_PATH}/gold"
CACHE_PATH = f"{DEMO_PATH}/cache"

# Create cache directory
dbutils.fs.mkdirs(CACHE_PATH)

# Load enriched transactions
df_transactions = spark.table("silver_transactions")
print(f"Loaded {df_transactions.count()} transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Response Caching Strategy
# MAGIC
# MAGIC To optimize costs and API usage, we cache Claude's responses for similar transactions.

# COMMAND ----------

def create_transaction_hash(txn: dict) -> str:
    """Create a hash for caching based on transaction characteristics."""
    # Create a signature based on key risk-relevant features
    signature = f"{txn['transaction_type']}|{txn['amount']:.0f}|{txn['originator_country']}|{txn['beneficiary_country']}|{txn['account_age_days']//30}|{txn['is_new_beneficiary']}|{txn.get('merchant_category', 'N/A')}"
    return hashlib.md5(signature.encode()).hexdigest()

# Example
sample_txn = df_transactions.limit(1).toPandas().iloc[0].to_dict()
print(f"Sample hash: {create_transaction_hash(sample_txn)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Batch Analysis Function with Retry Logic

# COMMAND ----------

# Analysis prompt template
BATCH_ANALYSIS_PROMPT = """Analyze this financial transaction for AML/fraud risk.

Transaction:
- Type: {transaction_type}
- Amount: ${amount:,.2f}
- From: {originator_country} → To: {beneficiary_country}
- Account Age: {account_age_days} days
- New Beneficiary: {is_new_beneficiary}
- Merchant: {merchant_category}

Respond with JSON only:
{{"risk_score": <0-100>, "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>", "confidence": <0.0-1.0>, "explanation": "<brief explanation>", "action": "<APPROVE|REVIEW|ESCALATE|BLOCK>"}}"""

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=10))
def analyze_single_transaction(client, txn: dict) -> dict:
    """Analyze a single transaction with retry logic."""
    prompt = BATCH_ANALYSIS_PROMPT.format(
        transaction_type=txn['transaction_type'],
        amount=txn['amount'],
        originator_country=txn['originator_country'],
        beneficiary_country=txn['beneficiary_country'],
        account_age_days=txn['account_age_days'],
        is_new_beneficiary=txn['is_new_beneficiary'],
        merchant_category=txn.get('merchant_category') or 'N/A'
    )

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(response.content[0].text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Define Pandas UDF for Distributed Processing

# COMMAND ----------

# Define output schema for the UDF
analysis_schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("risk_score", IntegerType(), True),
    StructField("risk_level", StringType(), True),
    StructField("confidence", DoubleType(), True),
    StructField("explanation", StringType(), True),
    StructField("recommended_action", StringType(), True),
    StructField("analysis_timestamp", TimestampType(), True),
    StructField("model_version", StringType(), True),
    StructField("error", StringType(), True)
])

# Broadcast API key to workers
api_key_broadcast = spark.sparkContext.broadcast(ANTHROPIC_API_KEY)
model_broadcast = spark.sparkContext.broadcast(MODEL_NAME)

@pandas_udf(analysis_schema, PandasUDFType.GROUPED_MAP)
def analyze_transactions_batch(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas UDF to analyze transactions in batches.
    Uses rate limiting and caching for efficiency.
    """
    import anthropic
    import json
    import time
    from datetime import datetime

    # Initialize client (per partition)
    client = anthropic.Anthropic(api_key=api_key_broadcast.value)

    results = []
    cache = {}  # Simple in-memory cache per partition

    for idx, row in pdf.iterrows():
        txn = row.to_dict()

        try:
            # Check cache
            cache_key = f"{txn['transaction_type']}|{int(txn['amount']//1000)}|{txn['beneficiary_country']}"

            if cache_key in cache:
                analysis = cache[cache_key].copy()
            else:
                # Make API call
                prompt = f"""Analyze for AML risk:
Type: {txn['transaction_type']}, Amount: ${txn['amount']:,.2f}
From: {txn['originator_country']} → To: {txn['beneficiary_country']}
Account Age: {txn['account_age_days']} days, New Beneficiary: {txn['is_new_beneficiary']}
Merchant: {txn.get('merchant_category') or 'N/A'}

JSON response only:
{{"risk_score": <0-100>, "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>", "confidence": <0.0-1.0>, "explanation": "<brief>", "action": "<APPROVE|REVIEW|ESCALATE|BLOCK>"}}"""

                response = client.messages.create(
                    model=model_broadcast.value,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}]
                )

                analysis = json.loads(response.content[0].text)
                cache[cache_key] = analysis
                time.sleep(0.3)  # Rate limiting

            results.append({
                'transaction_id': txn['transaction_id'],
                'risk_score': analysis.get('risk_score'),
                'risk_level': analysis.get('risk_level'),
                'confidence': analysis.get('confidence'),
                'explanation': analysis.get('explanation'),
                'recommended_action': analysis.get('action'),
                'analysis_timestamp': datetime.now(),
                'model_version': model_broadcast.value,
                'error': None
            })

        except Exception as e:
            results.append({
                'transaction_id': txn['transaction_id'],
                'risk_score': None,
                'risk_level': None,
                'confidence': None,
                'explanation': None,
                'recommended_action': None,
                'analysis_timestamp': datetime.now(),
                'model_version': model_broadcast.value,
                'error': str(e)
            })

    return pd.DataFrame(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. MLflow Experiment Setup

# COMMAND ----------

# Set up MLflow experiment
experiment_name = "/Shared/claude_transaction_analysis"
mlflow.set_experiment(experiment_name)

print(f"MLflow experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Run Batch Analysis Pipeline

# COMMAND ----------

# For demo, analyze a subset of transactions
# In production, remove the limit
DEMO_LIMIT = 50  # Set to None for full dataset

df_to_analyze = df_transactions.select(
    "transaction_id",
    "account_id",
    "transaction_type",
    "amount",
    "originator_country",
    "beneficiary_country",
    "merchant_category",
    "account_age_days",
    "is_new_beneficiary",
    "fraud_label",
    "fraud_type"
)

if DEMO_LIMIT:
    df_to_analyze = df_to_analyze.limit(DEMO_LIMIT)

print(f"Analyzing {df_to_analyze.count()} transactions...")

# COMMAND ----------

# Start MLflow run
with mlflow.start_run(run_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

    # Log parameters
    mlflow.log_param("model", MODEL_NAME)
    mlflow.log_param("num_transactions", df_to_analyze.count())
    mlflow.log_param("batch_size", BATCH_SIZE)

    start_time = time.time()

    # Run batch analysis using groupBy to trigger the UDF
    # Group by a computed column to create batches
    df_with_batch = df_to_analyze.withColumn(
        "batch_id",
        (F.monotonically_increasing_id() % 10).cast("int")  # 10 batches
    )

    df_analyzed = df_with_batch.groupBy("batch_id").apply(analyze_transactions_batch)

    # Cache results
    df_analyzed = df_analyzed.cache()
    result_count = df_analyzed.count()

    elapsed_time = time.time() - start_time

    # Log metrics
    mlflow.log_metric("processing_time_seconds", elapsed_time)
    mlflow.log_metric("transactions_processed", result_count)
    mlflow.log_metric("transactions_per_second", result_count / elapsed_time if elapsed_time > 0 else 0)

    # Calculate accuracy metrics
    df_with_labels = df_analyzed.join(
        df_to_analyze.select("transaction_id", "fraud_label"),
        "transaction_id"
    )

    # Count predictions
    high_risk_count = df_with_labels.filter(
        F.col("risk_level").isin("HIGH", "CRITICAL")
    ).count()

    actual_fraud_count = df_with_labels.filter(F.col("fraud_label") == 1).count()

    mlflow.log_metric("high_risk_flagged", high_risk_count)
    mlflow.log_metric("actual_fraud_count", actual_fraud_count)

    print(f"✅ Analysis complete!")
    print(f"   Processed: {result_count} transactions")
    print(f"   Time: {elapsed_time:.1f} seconds")
    print(f"   Rate: {result_count/elapsed_time:.1f} txn/sec")

# COMMAND ----------

# View results
display(df_analyzed.orderBy(F.desc("risk_score")).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Create Gold Layer (Risk-Scored Transactions)

# COMMAND ----------

# Join analysis results back to original data
df_gold = df_transactions.join(
    df_analyzed.select(
        "transaction_id",
        "risk_score",
        "risk_level",
        "confidence",
        "explanation",
        "recommended_action",
        "analysis_timestamp",
        "model_version"
    ),
    "transaction_id",
    "left"
)

# Add processing metadata
df_gold = df_gold.withColumn("_processed_at", F.current_timestamp())

# Write to Gold layer
df_gold.write.format("delta") \
    .mode("overwrite") \
    .save(f"{GOLD_PATH}/transactions_scored")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS gold_transactions_scored
    USING DELTA
    LOCATION '{GOLD_PATH}/transactions_scored'
""")

print(f"✅ Gold layer created with {df_gold.count()} scored transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Performance Analysis

# COMMAND ----------

# Risk score distribution
print("Risk Score Distribution:")
df_analyzed.groupBy("risk_level") \
    .agg(
        F.count("*").alias("count"),
        F.avg("risk_score").alias("avg_score"),
        F.avg("confidence").alias("avg_confidence")
    ) \
    .orderBy("risk_level") \
    .display()

# COMMAND ----------

# Accuracy analysis
df_accuracy = df_analyzed.join(
    df_to_analyze.select("transaction_id", "fraud_label", "fraud_type"),
    "transaction_id"
)

# Confusion matrix
df_accuracy = df_accuracy.withColumn(
    "predicted_suspicious",
    F.when(F.col("risk_level").isin("HIGH", "CRITICAL"), 1).otherwise(0)
)

confusion = df_accuracy.groupBy("fraud_label", "predicted_suspicious") \
    .count() \
    .orderBy("fraud_label", "predicted_suspicious")

print("Confusion Matrix (fraud_label vs predicted_suspicious):")
display(confusion)

# COMMAND ----------

# Calculate metrics
tp = df_accuracy.filter((F.col("fraud_label") == 1) & (F.col("predicted_suspicious") == 1)).count()
fp = df_accuracy.filter((F.col("fraud_label") == 0) & (F.col("predicted_suspicious") == 1)).count()
tn = df_accuracy.filter((F.col("fraud_label") == 0) & (F.col("predicted_suspicious") == 0)).count()
fn = df_accuracy.filter((F.col("fraud_label") == 1) & (F.col("predicted_suspicious") == 0)).count()

total = tp + fp + tn + fn
accuracy = (tp + tn) / total if total > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("=" * 50)
print("MODEL PERFORMANCE METRICS")
print("=" * 50)
print(f"Accuracy:  {accuracy*100:.1f}%")
print(f"Precision: {precision*100:.1f}%")
print(f"Recall:    {recall*100:.1f}%")
print(f"F1 Score:  {f1*100:.1f}%")

# Log to MLflow
with mlflow.start_run(run_name="metrics_analysis"):
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Cost Estimation

# COMMAND ----------

# Estimate costs based on token usage
# Claude Sonnet: ~$3 per 1M input tokens, ~$15 per 1M output tokens

AVG_INPUT_TOKENS = 150  # per transaction
AVG_OUTPUT_TOKENS = 100  # per transaction

INPUT_COST_PER_1M = 3.0
OUTPUT_COST_PER_1M = 15.0

def estimate_costs(num_transactions, cache_hit_rate=0.3):
    """Estimate API costs for transaction analysis."""
    effective_transactions = num_transactions * (1 - cache_hit_rate)

    input_tokens = effective_transactions * AVG_INPUT_TOKENS
    output_tokens = effective_transactions * AVG_OUTPUT_TOKENS

    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_1M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_1M

    return {
        'transactions': num_transactions,
        'cache_hit_rate': cache_hit_rate,
        'effective_api_calls': effective_transactions,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': input_cost + output_cost
    }

# Cost projections
print("=" * 60)
print("COST PROJECTIONS")
print("=" * 60)

for volume, cache_rate in [(10_000, 0.2), (100_000, 0.3), (1_000_000, 0.4)]:
    costs = estimate_costs(volume, cache_rate)
    print(f"\n{volume:,} transactions (cache hit rate: {cache_rate*100:.0f}%):")
    print(f"  Effective API calls: {costs['effective_api_calls']:,.0f}")
    print(f"  Estimated cost: ${costs['total_cost']:,.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Summary
# MAGIC
# MAGIC ### Pipeline Features:
# MAGIC 1. **Distributed Processing**: Pandas UDFs enable parallel API calls across Spark workers
# MAGIC 2. **Rate Limiting**: Built-in delays prevent API throttling
# MAGIC 3. **Caching**: Similar transactions share cached responses
# MAGIC 4. **Error Handling**: Retry logic handles transient failures
# MAGIC 5. **MLflow Tracking**: All runs are logged for reproducibility
# MAGIC
# MAGIC ### Production Considerations:
# MAGIC - Increase batch sizes for higher throughput
# MAGIC - Implement persistent caching (Delta table or Redis)
# MAGIC - Add circuit breakers for API failures
# MAGIC - Monitor token usage and costs
# MAGIC - Consider async API calls for better parallelism
# MAGIC
# MAGIC ### Next: Notebook 04 - Compliance Reporting

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC *Demo notebook - Claude + Databricks Financial Services Risk Analysis*
