# Databricks notebook source
# MAGIC %md
# MAGIC # 01: Setup and Data Preparation
# MAGIC
# MAGIC ## Claude + Databricks Transaction Risk Analysis Demo
# MAGIC
# MAGIC This notebook sets up the environment and prepares transaction data for analysis.
# MAGIC
# MAGIC **What you'll learn:**
# MAGIC - How to configure the Anthropic API in Databricks
# MAGIC - Loading and exploring transaction data
# MAGIC - Creating Delta tables for the medallion architecture

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Setup

# COMMAND ----------

# Install required packages
%pip install anthropic pandas matplotlib seaborn

# COMMAND ----------

# Restart Python to use newly installed packages
dbutils.library.restartPython()

# COMMAND ----------

# Import libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Display settings
pd.set_option('display.max_columns', None)
spark.conf.set("spark.sql.shuffle.partitions", "8")

print("✅ Libraries imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configure Anthropic API
# MAGIC
# MAGIC Store your API key securely using Databricks Secrets.
# MAGIC
# MAGIC **First-time setup** (run in a separate cell or terminal):
# MAGIC ```python
# MAGIC # Create a secret scope (if not exists)
# MAGIC databricks secrets create-scope --scope anthropic
# MAGIC
# MAGIC # Add your API key
# MAGIC databricks secrets put --scope anthropic --key api_key
# MAGIC ```

# COMMAND ----------

# Retrieve API key from secrets
try:
    ANTHROPIC_API_KEY = dbutils.secrets.get(scope="anthropic", key="api_key")
    print("✅ Anthropic API key loaded from secrets")
except Exception as e:
    print("⚠️  Anthropic API key not found in secrets.")
    print("   For demo purposes, you can set it manually (not recommended for production):")
    print("   ANTHROPIC_API_KEY = 'your-api-key-here'")
    ANTHROPIC_API_KEY = None

# COMMAND ----------

# Test API connection
if ANTHROPIC_API_KEY:
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Quick test
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say 'API connection successful!' in exactly those words."}]
    )
    print(f"✅ {response.content[0].text}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Transaction Data
# MAGIC
# MAGIC Upload the `transactions.csv` file to DBFS or use Unity Catalog volumes.

# COMMAND ----------

# Define paths
DEMO_PATH = "/FileStore/demo/claude_fsi"
DATA_PATH = f"{DEMO_PATH}/data"
BRONZE_PATH = f"{DEMO_PATH}/bronze"
SILVER_PATH = f"{DEMO_PATH}/silver"
GOLD_PATH = f"{DEMO_PATH}/gold"

# Create directories
dbutils.fs.mkdirs(DATA_PATH)
dbutils.fs.mkdirs(BRONZE_PATH)
dbutils.fs.mkdirs(SILVER_PATH)
dbutils.fs.mkdirs(GOLD_PATH)

print("✅ Directory structure created")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upload Transaction Data
# MAGIC
# MAGIC **Option 1:** Upload `transactions.csv` via the Databricks UI to `/FileStore/demo/claude_fsi/data/`
# MAGIC
# MAGIC **Option 2:** If running locally, use the widget below to upload

# COMMAND ----------

# Check if data exists, provide sample if not
try:
    df_check = spark.read.csv(f"{DATA_PATH}/transactions.csv", header=True)
    print(f"✅ Found transaction data: {df_check.count()} rows")
except:
    print("⚠️  Transaction data not found. Creating sample data for demo...")

    # Generate minimal sample data for demo
    sample_data = [
        ("TXN-2024-001", "ACCT-100001", "2024-11-10 09:15:00", "WIRE_DOMESTIC", 5000.00, "USD", "US", "US", None, 1500, False, "Payroll deposit", 0, None),
        ("TXN-2024-002", "ACCT-100001", "2024-11-10 14:30:00", "CARD_PURCHASE", 125.50, "USD", "US", "US", "RETAIL", 1500, False, "Amazon purchase", 0, None),
        ("TXN-2024-003", "ACCT-100002", "2024-11-11 08:00:00", "CASH_DEPOSIT", 9500.00, "USD", "US", "US", None, 45, False, "Cash deposit branch 101", 1, "STRUCTURING"),
        ("TXN-2024-004", "ACCT-100002", "2024-11-11 15:00:00", "CASH_DEPOSIT", 9400.00, "USD", "US", "US", None, 45, False, "Cash deposit branch 102", 1, "STRUCTURING"),
        ("TXN-2024-005", "ACCT-100002", "2024-11-12 10:00:00", "CASH_DEPOSIT", 9600.00, "USD", "US", "US", None, 46, False, "Cash deposit branch 103", 1, "STRUCTURING"),
        ("TXN-2024-006", "ACCT-100003", "2024-11-12 11:00:00", "WIRE_INTERNATIONAL", 150000.00, "USD", "CY", "US", None, 14, True, "Incoming wire", 1, "LAYERING"),
        ("TXN-2024-007", "ACCT-100003", "2024-11-12 16:00:00", "WIRE_INTERNATIONAL", 45000.00, "USD", "US", "PA", None, 14, True, "Outgoing wire", 1, "LAYERING"),
        ("TXN-2024-008", "ACCT-100003", "2024-11-13 09:00:00", "WIRE_INTERNATIONAL", 52000.00, "USD", "US", "VG", None, 15, True, "Outgoing wire", 1, "LAYERING"),
        ("TXN-2024-009", "ACCT-100004", "2024-11-13 10:00:00", "ACH", 2500.00, "USD", "US", "US", None, 2000, False, "Monthly rent", 0, None),
        ("TXN-2024-010", "ACCT-100005", "2024-11-14 14:00:00", "CARD_PURCHASE", 500.00, "USD", "US", "US", "CRYPTO_EXCHANGE", 60, False, "Crypto purchase", 1, "MSB_SUSPICIOUS"),
    ]

    schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("account_id", StringType(), True),
        StructField("transaction_date", StringType(), True),
        StructField("transaction_type", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("currency", StringType(), True),
        StructField("originator_country", StringType(), True),
        StructField("beneficiary_country", StringType(), True),
        StructField("merchant_category", StringType(), True),
        StructField("account_age_days", IntegerType(), True),
        StructField("is_new_beneficiary", BooleanType(), True),
        StructField("transaction_description", StringType(), True),
        StructField("fraud_label", IntegerType(), True),
        StructField("fraud_type", StringType(), True)
    ])

    df_sample = spark.createDataFrame(sample_data, schema)
    df_sample.write.mode("overwrite").option("header", True).csv(f"{DATA_PATH}/transactions.csv")
    print("✅ Sample data created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Bronze Layer (Raw Data)

# COMMAND ----------

# Define schema for transaction data
transaction_schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("account_id", StringType(), True),
    StructField("transaction_date", StringType(), True),
    StructField("transaction_type", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("currency", StringType(), True),
    StructField("originator_country", StringType(), True),
    StructField("beneficiary_country", StringType(), True),
    StructField("merchant_category", StringType(), True),
    StructField("account_age_days", IntegerType(), True),
    StructField("is_new_beneficiary", BooleanType(), True),
    StructField("transaction_description", StringType(), True),
    StructField("fraud_label", IntegerType(), True),
    StructField("fraud_type", StringType(), True)
])

# Load raw data
df_bronze = spark.read.csv(
    f"{DATA_PATH}/transactions.csv",
    header=True,
    schema=transaction_schema
)

# Add ingestion metadata
df_bronze = df_bronze.withColumn("_ingestion_timestamp", F.current_timestamp()) \
                     .withColumn("_source_file", F.input_file_name())

# Write to Bronze Delta table
df_bronze.write.format("delta") \
    .mode("overwrite") \
    .save(f"{BRONZE_PATH}/transactions")

# Create table reference
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS bronze_transactions
    USING DELTA
    LOCATION '{BRONZE_PATH}/transactions'
""")

print(f"✅ Bronze layer created: {df_bronze.count()} transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Exploratory Data Analysis

# COMMAND ----------

# Load bronze data
df = spark.table("bronze_transactions")
display(df.limit(10))

# COMMAND ----------

# Transaction summary statistics
print("=" * 60)
print("TRANSACTION SUMMARY")
print("=" * 60)

# Basic counts
total_txns = df.count()
fraud_txns = df.filter(F.col("fraud_label") == 1).count()
print(f"\nTotal Transactions: {total_txns:,}")
print(f"Suspicious Transactions: {fraud_txns:,} ({fraud_txns/total_txns*100:.1f}%)")

# Amount statistics
amount_stats = df.select(
    F.min("amount").alias("min"),
    F.max("amount").alias("max"),
    F.avg("amount").alias("avg"),
    F.sum("amount").alias("total")
).collect()[0]

print(f"\nAmount Statistics:")
print(f"  Min: ${amount_stats['min']:,.2f}")
print(f"  Max: ${amount_stats['max']:,.2f}")
print(f"  Avg: ${amount_stats['avg']:,.2f}")
print(f"  Total: ${amount_stats['total']:,.2f}")

# COMMAND ----------

# Transaction type breakdown
print("\nTransactions by Type:")
df.groupBy("transaction_type") \
  .agg(
      F.count("*").alias("count"),
      F.sum("amount").alias("total_amount"),
      F.sum(F.when(F.col("fraud_label") == 1, 1).otherwise(0)).alias("suspicious")
  ) \
  .orderBy(F.desc("count")) \
  .display()

# COMMAND ----------

# Fraud type breakdown
print("\nSuspicious Transactions by Type:")
df.filter(F.col("fraud_label") == 1) \
  .groupBy("fraud_type") \
  .agg(
      F.count("*").alias("count"),
      F.sum("amount").alias("total_amount"),
      F.avg("amount").alias("avg_amount")
  ) \
  .orderBy(F.desc("count")) \
  .display()

# COMMAND ----------

# Country risk analysis
print("\nTransactions by Beneficiary Country (Top 10):")
df.groupBy("beneficiary_country") \
  .agg(
      F.count("*").alias("txn_count"),
      F.sum("amount").alias("total_amount"),
      F.sum(F.when(F.col("fraud_label") == 1, 1).otherwise(0)).alias("suspicious_count")
  ) \
  .withColumn("suspicious_rate", F.round(F.col("suspicious_count") / F.col("txn_count") * 100, 1)) \
  .orderBy(F.desc("total_amount")) \
  .limit(10) \
  .display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Create Silver Layer (Enriched Data)

# COMMAND ----------

# Add derived features for risk analysis
df_silver = df.withColumn(
    "transaction_timestamp",
    F.to_timestamp("transaction_date", "yyyy-MM-dd HH:mm:ss")
).withColumn(
    "transaction_hour",
    F.hour("transaction_timestamp")
).withColumn(
    "transaction_day_of_week",
    F.dayofweek("transaction_timestamp")
).withColumn(
    "is_high_risk_country",
    F.when(F.col("beneficiary_country").isin("CY", "MT", "PA", "VG", "KY", "BZ"), True).otherwise(False)
).withColumn(
    "is_round_amount",
    F.when((F.col("amount") % 1000 == 0) & (F.col("amount") >= 5000), True).otherwise(False)
).withColumn(
    "is_near_reporting_threshold",
    F.when((F.col("amount") >= 9000) & (F.col("amount") < 10000), True).otherwise(False)
).withColumn(
    "is_large_transaction",
    F.when(F.col("amount") >= 50000, True).otherwise(False)
).withColumn(
    "account_risk_tier",
    F.when(F.col("account_age_days") < 30, "HIGH")
     .when(F.col("account_age_days") < 90, "MEDIUM")
     .otherwise("LOW")
).withColumn(
    "merchant_risk_tier",
    F.when(F.col("merchant_category").isin("CASINO", "CRYPTO_EXCHANGE", "MONEY_SERVICE"), "HIGH")
     .when(F.col("merchant_category").isin("TRAVEL"), "MEDIUM")
     .otherwise("LOW")
)

# Write to Silver Delta table
df_silver.write.format("delta") \
    .mode("overwrite") \
    .save(f"{SILVER_PATH}/transactions_enriched")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS silver_transactions
    USING DELTA
    LOCATION '{SILVER_PATH}/transactions_enriched'
""")

print(f"✅ Silver layer created with enriched features")

# COMMAND ----------

# Review enriched data
display(spark.table("silver_transactions").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary and Next Steps
# MAGIC
# MAGIC ### What we've accomplished:
# MAGIC 1. ✅ Set up Anthropic API connection
# MAGIC 2. ✅ Loaded transaction data into Bronze layer
# MAGIC 3. ✅ Performed exploratory data analysis
# MAGIC 4. ✅ Created Silver layer with enriched risk features
# MAGIC
# MAGIC ### Key observations from the data:
# MAGIC - ~20% of transactions flagged as suspicious
# MAGIC - Structuring (multiple deposits near $10K) is the most common pattern
# MAGIC - High-risk countries (CY, PA, VG) have elevated suspicious transaction rates
# MAGIC - New accounts (<30 days) show higher risk profiles
# MAGIC
# MAGIC ### Next notebook: 02_transaction_analysis_with_claude
# MAGIC We'll use Claude to analyze individual transactions and generate human-readable risk explanations.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC *Demo notebook - Claude + Databricks Financial Services Risk Analysis*
