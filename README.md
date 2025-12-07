# ML-Deployment-Plan


# Lead Scoring Model: Production Deployment & MLOps Strategy

**Case Study Proposal**  
*Proposal for Deploying, Testing, and Maintaining ML Models in Production*

---

## Executive Summary

This proposal outlines a robust, enterprise-grade MLOps solution for deploying a lead scoring model to production. The strategy encompasses:

- **Deployment Architecture**: Databricks + MLflow + Docker-based serving
- **Online Testing**: Staged rollout with shadow deployment → A/B testing → canary release
- **Monitoring Framework**: Real-time data/prediction drift detection, automated alerting, business metrics tracking
- **Automation**: CI/CD pipelines, reproducible workflows, and autonomous retraining triggers

The solution prioritizes **safety, auditability, and business impact** while maintaining flexibility for rapid iteration.

---
## 1. DEPLOYMENT STRATEGY

### 1.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   DATABRICKS WORKSPACE                      │
│  (Unified platform for data + ML + serving)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Data Layer          Model Layer         Serving Layer      │
│  ├─ Raw Lead Data    ├─ MLflow           ├─ REST API        │
│  ├─ Feature Store    │  Experiments      ├─ Batch Jobs      │
│  └─ Delta Lake       └─ Model Registry   └─ Real-time       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         ↓                    ↓                   ↓
    Data Ops          MLOps (Monitoring)    API Gateway
    (DLT)             (Drift, Performance)    (CRM/Dashboard)
```
#### Architecture - HLD

![Architecture](https://github.com/shreyaschim/ML-Deployment-Plan/blob/main/mlops_architecture.png?raw=true)

### 1.2 Deployment Framework

**Technology Stack:**
- **Container**: Docker + Docker Registry (for reproducible serving environments)
- **Orchestration**: Kubernetes (optional, for high-scale serving) OR Databricks Model Serving (recommended—simpler)
- **Model Format**: MLflow Model (ensures portability)
- **API Server**: FastAPI (lightweight, production-ready)
- **Database**: PostgreSQL (for metadata, predictions, feedback)

**Why this stack:**
- MLflow provides native Databricks integration (no friction)
- Model Serving eliminates infrastructure management
- Docker ensures consistent dev → prod environments
- FastAPI offers built-in validation, async support, and auto-generated OpenAPI docs

### 1.3 Model Versioning & Auditability

**MLflow Model Registry Strategy:**

```
┌──────────────────────────────────────────────┐
│       MLflow Model Registry                  │
├──────────────────────────────────────────────┤
│ Model Name: lead-scoring-v2                  │
│                                              │
│ Version 1  → Archived (accuracy: 0.82)       │
│ Version 2  → Production (accuracy: 0.87)     │
│ Version 3  → Staging (accuracy: 0.88)        │
│ Version 4  → None (experimental)             │
└──────────────────────────────────────────────┘
```

**Version Control Metadata:**

Each model version captures:
- **Training Data Snapshot**: Delta table version hash (reproducibility)
- **Hyperparameters**: Complete parameter grid used
- **Feature Set**: Feature engineering code + transformations
- **Performance Metrics**: Cross-validation scores, baseline comparison
- **Git Commit Hash**: Link to exact code used for training
- **Author & Timestamp**: Audit trail

**Rollback Mechanism:**

1. **Instant Rollback**: Update API serving to point to previous version (seconds)
2. **Gradual Rollback**: Canary release with 90% traffic on old model, 10% on new
3. **Data-Driven Decision**: Monitor metrics during rollback to confirm success

**Implementation:**
```python
# Example: Register model with audit trail
mlflow.sklearn.log_model(
    model,
    artifact_path="lead-scorer",
    registered_model_name="lead-scoring",
    metadata={
        "data_snapshot_version": "v20250206",
        "feature_set_hash": "abc123def456",
        "git_commit": "a7f3b2c9e1d4",
        "approved_by": "ml-review-team",
        "approval_date": "2025-02-06"
    }
)
```

### 1.4 Deployment Options

#### Option A: Databricks Model Serving (Recommended)

**Pros:**
- Zero infrastructure management
- Auto-scaling, high availability built-in
- Native integration with MLflow
- VPC endpoints for security

**Cons:**
- Less control over serving environment
- Slightly higher latency vs. optimized containers

#### Option B: Docker + FastAPI + Kubernetes

**Pros:**
- Maximum control and optimization
- Multi-cloud portability
- Custom preprocessing/postprocessing

**Cons:**
- Operational overhead (K8s management)
- Requires DevOps expertise

**Recommendation**: Start with **Databricks Model Serving** for simplicity. Migrate to Docker + K8s if latency or cost optimization becomes critical.

---

## 2. ONLINE TESTING STRATEGY

### 2.1 Three-Stage Rollout

```
┌─────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Stage 1: SHADOW DEPLOYMENT (1–2 weeks)                     │
│  ├─ New model runs in parallel, no traffic directed to it   │
│  ├─ Predictions logged but NOT shown to sales/users         │
│  └─ Validates: correctness, latency, stability              │
│                                                             │
│  Stage 2: CANARY RELEASE (1 week)                           │
│  ├─ Route 10% of traffic to new model, 90% to old           │
│  ├─ Monitor: performance drift, user feedback               │
│  └─ Success criteria: no metric degradation                 │
│                                                             │
│  Stage 3: A/B TEST (2–4 weeks)                              │
│  ├─ Randomly split leads: 50% new model, 50% old            │
│  ├─ Track: conversion rates, deal size, sales cycle time    │
│  └─ Statistical significance: p < 0.05                      │
│                                                             │
│  Stage 4: FULL ROLLOUT                                      │
│  └─ 100% traffic to new model                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Metrics for Success

#### Technical Metrics (Shadow Deployment)

| Metric | Threshold | Owner |
|--------|-----------|-------|
| **Latency (p99)** | < 200ms | Eng |
| **Throughput** | > 1000 req/s | Eng |
| **Error Rate** | < 0.1% | Eng |
| **Memory Usage** | < 2GB per instance | Eng |

#### Performance Metrics (Canary & A/B)

| Metric | Success Criteria | Owner |
|--------|------------------|-------|
| **Accuracy** (offline validation) | ≥ 87% (baseline: 82%) | DS/ML |
| **Calibration** | Predicted prob matches actual conversion rate | DS/ML |
| **Feature Distribution** | No significant drift from training data | DS/ML |
| **Prediction Spread** | Maintain quartile distribution of scores | DS/ML |

#### Business Metrics (A/B Test)

| Metric | Target | Owner |
|--------|--------|-------|
| **Conversion Rate** | +5% improvement (statistical significance) | Sales/Product |
| **Deal Size** | No negative impact (or +3% avg) | Sales/Finance |
| **Sales Cycle** | No increase in days to close | Sales/Operations |
| **Lead Quality Score** | Subjective: sales team satisfaction ≥ 8/10 | Sales/Product |
| **Time to Decision** | Sales team spends ≤ same time prioritizing | Sales/Operations |

#### Dashboard Design

![Dashboard Design](https://github.com/shreyaschim/ML-Deployment-Plan/blob/main/dashboard.png?raw=true)

### 2.3 Implementation: Shadow Deployment Example

```python
# Databricks Job: Log predictions from new model (no traffic impact)
@task
def shadow_predict(batch_df):
    """
    Run new model in parallel, log predictions to separate table.
    Sales/CRM still uses old model scores.
    """
    new_model = mlflow.pyfunc.load_model(
        model_uri="models:/lead-scoring/staging"
    )
    
    # Generate predictions
    shadow_predictions = new_model.predict(batch_df[FEATURE_COLUMNS])
    
    # Log to shadow table with timestamp
    shadow_results = batch_df.copy()
    shadow_results["shadow_score"] = shadow_predictions
    shadow_results["shadow_timestamp"] = current_timestamp()
    shadow_results["model_version"] = "v3-staging"
    
    # Append to Delta table for later analysis
    shadow_results.write.format("delta").mode("append").save(
        "dbfs:/production/shadow_predictions"
    )
    
    return {"rows_processed": len(shadow_results), "timestamp": current_timestamp()}

# Databricks Job: Compare shadow predictions with actual outcomes
@task
def shadow_validation():
    """
    Compare model predictions (from 1 week ago) with actual outcomes.
    Identify model accuracy, calibration issues, feature drift.
    """
    shadow_df = spark.read.delta("dbfs:/production/shadow_predictions")
    actuals_df = spark.read.delta("dbfs:/production/lead_outcomes")
    
    # Join on lead_id, activity_date
    comparison = shadow_df.join(
        actuals_df, 
        on=["lead_id"], 
        how="inner"
    )
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, calibration_curve
    
    auc = roc_auc_score(comparison["converted"], comparison["shadow_score"])
    precision = (comparison["shadow_score"] > 0.5).sum() / len(comparison)
    
    # Log results
    mlflow.log_metric("shadow_auc", auc)
    mlflow.log_metric("shadow_precision", precision)
    
    # Alert if metrics degrade
    if auc < 0.82:
        send_alert(f"Shadow AUC degraded: {auc}")
    
    return {"auc": auc, "precision": precision}
```

### 2.4 A/B Test Design

**Setup:**
- **Duration**: 2–4 weeks (sufficient for statistical power)
- **Sample Size**: Minimum 5,000 leads per variant (depends on conversion rate)
- **Randomization**: Hash(lead_id) % 2 → ensures reproducibility, no bias
- **Confidence Level**: 95% (p < 0.05)

**Analysis Plan:**

```python
import scipy.stats as stats

# After 2 weeks, analyze results
control_conversions = 250  # Old model
treatment_conversions = 275  # New model

control_size = 5000
treatment_size = 5000

# Chi-square test
chi2, p_value = stats.chi2_contingency([
    [control_conversions, control_size - control_conversions],
    [treatment_conversions, treatment_size - treatment_conversions]
])[:2]

lift = (treatment_conversions - control_conversions) / control_conversions * 100

if p_value < 0.05:
    print(f"✓ Statistically significant lift: {lift}%")
else:
    print(f"✗ Not significant. Continue test or investigate.")
```

**Guardrails:**

- **Stop test early if**: Metrics degrade by > 10% (p < 0.05)
- **Confidence interval**: Report ±95% CI for lift
- **Secondary analysis**: Segment by industry, company size, region

---

## 3. MONITORING & ALERTING

### 3.1 Monitoring Dashboard

**Real-time Dashboard Components:**

```
┌──────────────────────────────────────────────────────────────┐
│                  LEAD SCORING MONITORING DASHBOARD           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [System Health]              [Model Performance]            │
│  ├─ API Latency: 45ms ✓       ├─ AUC: 0.87 ✓                 │
│  ├─ Error Rate: 0.02% ✓       ├─ Precision@0.5: 0.85 ✓       │
│  ├─ Throughput: 850 req/s     ├─ Recall: 0.82 ✓              │
│  └─ Uptime: 99.95% ✓          └─ Calibration Error: 0.03 ✓   │
│                                                              │
│  [Data Health]                [Business Impact]              │
│  ├─ Data Freshness: 2h ✓      ├─ Conversion Rate: 12.5% ↑    │
│  ├─ Missing Values: 0.1% ✓    ├─ Avg Deal Size: $85K ↑       │
│  ├─ Feature Drift: None ✓     ├─ Sales Satisfaction: 8.2/10  │
│  └─ KS Statistic: 0.08        └─ Model Usage: 5,234 scores   │
│                                                              │
│  [Drift Detection]            [Recent Alerts]                │
│  ├─ Prediction Drift: LOW ✓   ├─ [OK] 12:45 - Perf nominal   │
│  ├─ Feature Shift: MEDIUM ⚠   ├─ [WARN] 10:20 - High latency │
│  └─ Distribution Delta: 0.12  └─ [OK] 08:15 - Retrain done   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Dashboard Tools:** Databricks SQL + Grafana OR Tableau (recommended: Databricks SQL for simplicity)

### 3.2 Data Drift Detection

**Methods:**

1. **Kolmogorov–Smirnov (KS) Test** (best for continuous features)
   - Compares feature distribution: current week vs. training baseline
   - Threshold: KS statistic > 0.15 → alert

2. **Jensen–Shannon Divergence** (for discrete features)
   - Symmetric, bounded divergence measure
   - Threshold: JS divergence > 0.1 → alert

3. **Evidently or Great Expectations Integration**
   - Open-source drift detection libraries
   - Automated report generation

**Implementation:**

```python
from scipy.stats import ks_2samp
import numpy as np

def detect_data_drift(current_data, baseline_data, threshold=0.15):
    """
    Detect feature drift using KS test.
    Returns drift severity and affected features.
    """
    drift_report = {}
    
    for feature in FEATURE_COLUMNS:
        ks_stat, p_value = ks_2samp(
            current_data[feature].values,
            baseline_data[feature].values
        )
        
        drift_report[feature] = {
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "drifted": ks_stat > threshold
        }
        
        if ks_stat > threshold:
            # Alert immediately
            send_slack_alert(
                f"⚠️ Data drift detected in '{feature}': "
                f"KS={ks_stat:.3f} (threshold={threshold})"
            )
    
    return drift_report
```

### 3.3 Prediction Drift Detection

**Metrics:**

- **Score Distribution Changes**: Compare percentiles of predicted scores
  - Alert if: 90th percentile score changes by > 20%
- **Prediction Entropy**: Measure "confidence" of model
  - Decreasing entropy → model becoming overconfident
- **Coverage**: % of leads scored (should be near 100%)
  - Alert if: Coverage drops below 95%

**SQL Query Example:**

```sql
-- Daily prediction drift monitoring
SELECT
  DATE(prediction_timestamp) as date,
  PERCENTILE_CONT(predicted_score, 0.25) as q1,
  PERCENTILE_CONT(predicted_score, 0.50) as median,
  PERCENTILE_CONT(predicted_score, 0.75) as q3,
  PERCENTILE_CONT(predicted_score, 0.90) as p90,
  COUNT(*) as num_predictions,
  ROUND(100.0 * COUNT(*) / LAG(COUNT(*)) OVER (ORDER BY DATE(prediction_timestamp)) - 100, 2) as volume_change_pct
FROM lead_scores
WHERE prediction_timestamp >= CURRENT_DATE - INTERVAL 30 DAY
GROUP BY 1
ORDER BY 1 DESC;

-- Alert if p90 changes significantly
CREATE OR REPLACE ALERT pred_drift_alert AS
SELECT
  CASE WHEN ABS((PERCENTILE_CONT(predicted_score, 0.90) OVER (ORDER BY DATE(prediction_timestamp) ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) - LAG(PERCENTILE_CONT(predicted_score, 0.90)) OVER (ORDER BY DATE(prediction_timestamp))) / LAG(PERCENTILE_CONT(predicted_score, 0.90)) OVER (ORDER BY DATE(prediction_timestamp))) > 0.20
  THEN 'ALERT'
  ELSE 'OK'
  END as status
FROM lead_scores
WHERE prediction_timestamp >= CURRENT_DATE - INTERVAL 1 DAY;
```

### 3.4 Latency & Throughput

**Metrics:**

- **P99 Latency**: < 200ms (acceptable for async sales workflows)
- **P95 Latency**: < 100ms
- **Throughput**: 1,000+ requests/second
- **Error Rate**: < 0.1%

**Implementation (Databricks Model Serving automatically tracked):**

```python
# Custom metric logging
import time
from datetime import datetime

def predict_with_monitoring(lead_data):
    start = time.time()
    
    predictions = model.predict(lead_data)
    
    latency_ms = (time.time() - start) * 1000
    
    # Log to monitoring table
    monitoring_log = {
        "timestamp": datetime.utcnow(),
        "num_records": len(lead_data),
        "latency_ms": latency_ms,
        "error": False
    }
    
    mlflow.log_metric("prediction_latency_ms", latency_ms)
    
    # Alert if latency degraded
    if latency_ms > 200:
        send_alert(f"Latency high: {latency_ms}ms")
    
    return predictions
```

### 3.5 Business Performance Metrics

**Track conversion outcomes vs. model predictions:**

```sql
-- Daily business metrics report
SELECT
  DATE(prediction_date) as date,
  COUNT(DISTINCT lead_id) as num_leads,
  SUM(CASE WHEN converted THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as conversion_rate,
  AVG(deal_size) as avg_deal_size,
  AVG(CASE WHEN converted THEN days_to_close ELSE NULL END) as avg_days_to_close,
  CORR(predicted_score, CASE WHEN converted THEN 1 ELSE 0 END) as score_conversion_correlation
FROM lead_scores
LEFT JOIN lead_outcomes USING (lead_id)
GROUP BY 1
ORDER BY 1 DESC;
```

### 3.6 Troubleshooting: When Model Scores Stop Helping

**Investigation Flowchart:**

```
Sales: "Model scores no longer help prioritize leads"
  ↓
[Step 1: Check system health]
  • Is model serving? (Check API logs)
  • Are predictions stale? (Check freshness)
  • Is coverage < 95%? (% of leads scored)
  → Fix if issues found, re-assess
  
  ↓
[Step 2: Check model performance metrics]
  • Has AUC degraded? (Compare to baseline)
  • Is calibration off? (Predicted ≠ actual conversion rate)
  • Is prediction distribution normal? (Check percentiles)
  
  IF metric degradation detected:
    → Root cause: MODEL DRIFT
    → Action: Investigate features → Retrain
    
  ↓
[Step 3: Check data quality]
  • Are input features drifting? (KS test on each feature)
  • Are missing values increasing? (% nulls per feature)
  • Are new categories appearing? (Categorical features)
  
  IF data drift detected:
    → Root cause: DATA DISTRIBUTION CHANGE
    → Action: Investigate feature sources → Retrain
    
  ↓
[Step 4: Check business alignment]
  • Have sales processes changed? (New qualification criteria)
  • Is lead source mix different? (Different channels)
  • Did market conditions shift? (External factors)
  
  IF business context changed:
    → Root cause: BUSINESS/CONTEXT SHIFT
    → Action: Gather feedback → Retrain with new labels
    
  ↓
[Step 5: Check model calibration & integration]
  • Are threshold defaults still appropriate?
    (e.g., is "high priority" still score > 0.7?)
  • Is CRM using model scores correctly?
  • Are sales team using model at all?
  
  IF integration issue:
    → Root cause: HUMAN/PROCESS ISSUE
    → Action: Retrain sales team, adjust threshold
```

---

## 4. AUTOMATION, REPRODUCIBILITY & RETRAINING

### 4.1 Reproducibility Framework

**Goal**: Train identical model with same results 6 months later.

**Approach:**

1. **Data Snapshot Versioning**

```python
# Data preparation job (Databricks)
raw_leads = spark.read.table("raw_leads")

# Create immutable snapshot with version hash
leads_snapshot = raw_leads.filter(
    col("created_date") >= "2025-01-01"
)

# Write to versioned Delta table
leads_snapshot.write \
  .format("delta") \
  .mode("overwrite") \
  .save(f"dbfs:/datasets/lead_features/v_{SNAPSHOT_VERSION}")

# Log snapshot metadata
mlflow.log_param("data_snapshot_version", SNAPSHOT_VERSION)
mlflow.log_param("data_row_count", leads_snapshot.count())
mlflow.log_param("feature_column_count", len(FEATURE_COLUMNS))
```

2. **Feature Engineering Determinism**

```python
# Features must be reproducible
# ✓ Good: hash-based train/test split
# ✗ Bad: random.seed() without context

def deterministic_train_test_split(df, test_ratio=0.2, seed=42):
    """
    Hash-based split ensures same records in test set regardless
    of DataFrame ordering.
    """
    from pyspark.sql.functions import md5, concat_ws
    
    # Create deterministic hash
    df = df.withColumn(
        "_split_hash",
        md5(concat_ws("|", col("lead_id"), col("created_date")))
    )
    
    # Use hash to assign to train/test
    test_df = df.filter(
        (unix_timestamp(col("_split_hash")) % 100) < int(test_ratio * 100)
    )
    train_df = df.filter(
        (unix_timestamp(col("_split_hash")) % 100) >= int(test_ratio * 100)
    )
    
    return train_df, test_df
```

3. **Hyperparameter & Version Control**

```python
# Store hyperparameters in Git (GitOps for ML)
# config.yaml
model:
  type: logistic_regression
  hyperparameters:
    C: 1.0
    max_iter: 100
    random_state: 42
    
preprocessing:
  scaler: StandardScaler
  feature_columns: [...]
  
data:
  snapshot_version: "v_20250206"
  train_ratio: 0.8
  
# In training code:
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Log config as artifact
mlflow.log_dict(config, "model_config.yaml")
```

### 4.2 CI/CD Pipeline

**GitHub Actions + Databricks Workflow:**

```yaml
# .github/workflows/mlops-pipeline.yml
name: Lead Scoring MLOps Pipeline

on:
  push:
    branches:
      - main
  schedule:
    - cron: '0 2 * * 0'  # Weekly retraining

jobs:
  validate-and-train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      # Step 1: Code quality checks
      - name: Lint Python code
        run: |
          pip install flake8 black mypy
          flake8 src/ --count --select=E9,F63,F7,F82 --show-source
          black --check src/
          mypy src/ --strict
      
      # Step 2: Unit tests
      - name: Run unit tests
        run: |
          pip install pytest pytest-cov
          pytest tests/ --cov=src/ --cov-report=xml
      
      # Step 3: Data validation
      - name: Validate data quality
        run: |
          python scripts/validate_data.py
          
      # Step 4: Trigger Databricks training job
      - name: Submit training job to Databricks
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
        run: |
          pip install databricks-cli
          databricks jobs run-now --job-id 123
          
      # Step 5: Wait for job completion & get metrics
      - name: Monitor training job
        run: python scripts/wait_for_databricks_job.py
        
      # Step 6: Model registry transitions
      - name: Promote model to staging
        if: success()
        run: |
          python scripts/promote_model.py \
            --model-name lead-scoring \
            --target-stage Staging
            
      # Step 7: Generate report
      - name: Generate deployment report
        run: |
          python scripts/generate_report.py > report.md
          
      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: deployment-report
          path: report.md
```

### 4.3 Automated Retraining Strategy

**Triggers:**

| Trigger | Condition | Action |
|---------|-----------|--------|
| **Scheduled** | Weekly (every Sunday 2am UTC) | Full retraining |
| **Drift Detection** | KS statistic > 0.15 for any feature | Alert + manual review |
| **Prediction Drift** | AUC < 0.85 (baseline 0.87) | Alert + auto-retrain |
| **Data Volume** | > 10,000 new leads collected | Incremental retrain |
| **Manual Trigger** | ML engineer request | On-demand retrain |

**Retraining Job (Databricks):**

```python
# Retraining job running on schedule
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Step 1: Load latest data
latest_data = spark.read.delta("dbfs:/datasets/lead_features/latest")

# Step 2: Feature preparation (deterministic)
X = latest_data[FEATURE_COLUMNS].toPandas()
y = latest_data["converted"].toPandas()

# Step 3: Data validation (fail fast)
assert X.isnull().sum().sum() == 0, "Null values found in features"
assert y.nunique() == 2, "Binary target required"

# Step 4: Train with MLflow tracking
with mlflow.start_run(experiment_id=EXPERIMENT_ID):
    # Log metadata
    mlflow.log_param("training_data_rows", len(X))
    mlflow.log_param("model_type", "LogisticRegression")
    
    # Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train
    model = LogisticRegression(C=1.0, max_iter=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Evaluate
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    train_auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
    
    mlflow.log_metric("train_auc", train_auc)
    mlflow.log_metric("train_precision", precision_score(y, model.predict(X_scaled)))
    mlflow.log_metric("train_recall", recall_score(y, model.predict(X_scaled)))
    
    # Step 5: Log model
    mlflow.sklearn.log_model(
        model,
        "lead_scorer",
        registered_model_name="lead-scoring",
        input_example=X_scaled[:5]
    )
    
    # Step 6: Automatic quality gates
    if train_auc >= 0.85:
        # Promote to staging for shadow deployment
        mlflow.set_model_version_tag(
            name="lead-scoring",
            version=mlflow.active_run().info.run_id,
            key="stage",
            value="Staging"
        )
        print("✓ Model promoted to Staging")
    else:
        mlflow.set_model_version_tag(
            name="lead-scoring",
            version=mlflow.active_run().info.run_id,
            key="status",
            value="RequiresReview"
        )
        # Notify ML team
        send_slack_alert(
            f"⚠️ New model AUC={train_auc:.3f} < 0.85. "
            f"Manual review required."
        )
```

### 4.4 Feedback Loop

**Collecting Ground Truth:**

```python
# CRM integration: feed actual outcomes back to ML pipeline
def record_lead_outcome(lead_id, converted, deal_size, days_to_close):
    """
    Called by CRM when lead closes or is marked as lost.
    Feeds back into retraining pipeline.
    """
    outcome_record = {
        "lead_id": lead_id,
        "converted": converted,
        "deal_size": deal_size,
        "days_to_close": days_to_close,
        "outcome_timestamp": datetime.utcnow()
    }
    
    # Write to Delta table (immutable log)
    spark.createDataFrame([outcome_record]).write \
        .format("delta") \
        .mode("append") \
        .save("dbfs:/production/lead_outcomes")
    
    # Trigger drift check if threshold met
    if get_unprocessed_outcomes_count() > 5000:
        trigger_drift_check()

# Match predictions to outcomes
def match_predictions_to_outcomes():
    """
    Join predictions (from 30 days ago) with actual outcomes.
    Calculate calibration, precision @ threshold, etc.
    """
    predictions_30d_ago = spark.read.delta(
        "dbfs:/production/lead_scores"
    ).filter(
        col("prediction_timestamp") >= current_date() - 30
    )
    
    outcomes = spark.read.delta(
        "dbfs:/production/lead_outcomes"
    ).filter(
        col("outcome_timestamp") >= current_date() - 30
    )
    
    matched = predictions_30d_ago.join(
        outcomes,
        on="lead_id",
        how="inner"
    )
    
    # Calculate metrics
    calibration = matched.groupBy(
        F.round(col("predicted_score"), 1).alias("score_bin")
    ).agg(
        F.avg(col("converted")).alias("actual_conversion_rate"),
        F.count("*").alias("count")
    )
    
    return calibration
```

---

## 5. MONITORING DASHBOARD SAMPLE CODE

**Databricks SQL:**

```sql
-- Dashboard: Lead Scoring Model Health

-- Panel 1: Daily Predictions
SELECT
  DATE(prediction_timestamp) as date,
  COUNT(*) as predictions,
  ROUND(100.0 * SUM(CASE WHEN predicted_score > 0.7 THEN 1 ELSE 0 END) / COUNT(*), 2) as pct_high_score
FROM lead_scores
WHERE prediction_timestamp >= CURRENT_DATE - 30
GROUP BY 1
ORDER BY 1 DESC;

-- Panel 2: Model Performance
SELECT
  model_version,
  COUNT(*) as num_predictions,
  ROUND(AVG(CASE WHEN converted THEN 1 ELSE 0 END), 4) as actual_conversion_rate,
  ROUND(AVG(CASE WHEN converted THEN predicted_score ELSE NULL END), 4) as avg_score_converted,
  ROUND(AVG(CASE WHEN NOT converted THEN predicted_score ELSE NULL END), 4) as avg_score_not_converted
FROM lead_scores
LEFT JOIN lead_outcomes USING (lead_id)
WHERE prediction_timestamp >= CURRENT_DATE - 30
GROUP BY 1;

-- Panel 3: Latency Monitoring
SELECT
  DATE_TRUNC('hour', prediction_timestamp) as hour,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) as p50_latency,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency,
  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99_latency,
  MAX(latency_ms) as max_latency
FROM prediction_logs
WHERE prediction_timestamp >= CURRENT_TIMESTAMP - INTERVAL 1 DAY
GROUP BY 1
ORDER BY 1 DESC;

-- Panel 4: Data Drift Detection
WITH baseline AS (
  SELECT
    feature_name,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY feature_value) as baseline_q1,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY feature_value) as baseline_median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY feature_value) as baseline_q3
  FROM training_data
  GROUP BY 1
),
current AS (
  SELECT
    feature_name,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY feature_value) as current_q1,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY feature_value) as current_median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY feature_value) as current_q3
  FROM lead_scores
  WHERE prediction_timestamp >= CURRENT_DATE - 7
  GROUP BY 1
)
SELECT
  b.feature_name,
  ABS((c.current_median - b.baseline_median) / b.baseline_median) as median_drift_pct,
  CASE WHEN ABS((c.current_median - b.baseline_median) / b.baseline_median) > 0.15 THEN 'ALERT' ELSE 'OK' END as drift_status
FROM baseline b
JOIN current c ON b.feature_name = c.feature_name
ORDER BY median_drift_pct DESC;
```

---

## 6. INTEGRATION WITH CRM & BUSINESS DASHBOARDS

### 6.1 Real-time Score Exposure

**Option 1: REST API (Recommended for real-time)**

```python
# FastAPI endpoint in Databricks
from fastapi import FastAPI, HTTPException
import mlflow

app = FastAPI()
model = mlflow.pyfunc.load_model("models:/lead-scoring/Production")

@app.post("/predict")
async def predict(lead_data: LeadRequest):
    """
    Real-time lead scoring endpoint.
    """
    try:
        # Validate input
        features = lead_data.to_features()
        
        # Score
        prediction = model.predict(features)
        score = float(prediction[0])
        
        # Log to monitoring
        log_prediction(lead_data.lead_id, score)
        
        return {
            "lead_id": lead_data.lead_id,
            "predicted_score": score,
            "priority": "high" if score > 0.7 else "medium" if score > 0.4 else "low",
            "confidence": "high" if 0.3 < score < 0.7 else "medium"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Option 2: Batch Scoring (for daily lead list updates)**

```python
# Daily batch job
def batch_score_all_leads():
    """
    Score all open leads daily, update CRM.
    """
    model = mlflow.pyfunc.load_model("models:/lead-scoring/Production")
    open_leads = spark.read.table("crm_open_leads")
    
    # Score
    scores = model.predict(open_leads[FEATURE_COLUMNS])
    
    # Prepare output
    results = open_leads.select("lead_id", "company_name", "contact_email") \
        .withColumn("predicted_score", lit(scores)) \
        .withColumn("priority", F.when(col("predicted_score") > 0.7, "High")
                              .when(col("predicted_score") > 0.4, "Medium")
                              .otherwise("Low")) \
        .withColumn("updated_timestamp", F.current_timestamp())
    
    # Push to CRM (via API or direct DB write)
    push_to_salesforce(results)
```

### 6.2 CRM Integration (Salesforce Example)

```python
# Sync scores back to Salesforce
from salesforce_bulk import SalesforceQueryType
from simple_salesforce import Salesforce

def sync_scores_to_salesforce():
    sf = Salesforce(username=SFDC_USER, password=SFDC_PASSWORD, security_token=SFDC_TOKEN)
    
    # Get scored leads
    scored_leads = spark.read.delta("dbfs:/production/lead_scores").toPandas()
    
    # Batch update (avoid API rate limits)
    for i in range(0, len(scored_leads), 200):
        batch = scored_leads.iloc[i:i+200]
        
        for _, lead in batch.iterrows():
            sf.Lead.update(
                lead["salesforce_id"],
                {
                    "ML_Score__c": lead["predicted_score"],
                    "ML_Priority__c": lead["priority"],
                    "ML_Updated__c": datetime.utcnow().isoformat()
                }
            )
```

---

## 7. RISK MITIGATION & FALLBACK PLANS

| Risk | Mitigation |
|------|-----------|
| **Model degradation** | Weekly drift checks, A/B testing, automatic rollback |
| **Data quality issues** | Great Expectations validation, schema enforcement |
| **Infrastructure failure** | Multi-region setup, database failover |
| **Incorrect integration** | Sales team training, CRM validation tests |
| **Unforeseen business change** | Regular stakeholder feedback loops, model retraining triggers |

---

## 8. SUCCESS METRICS & KPIs

**By End of Month 1:**
- ✓ Model deployed in shadow mode
- ✓ Monitoring dashboard operational
- ✓ Zero critical incidents

**By End of Month 2:**
- ✓ A/B test running (2,000+ leads per variant)
- ✓ Conversion lift ≥ 3% (p < 0.05)
- ✓ Sales team satisfied (feedback score ≥ 8/10)

**By End of Month 3:**
- ✓ Full rollout to 100% of leads
- ✓ Automated retraining operational
- ✓ Model ROI: > $500K in incremental revenue

### Timeline 

![Timeline](https://github.com/shreyaschim/ML-Deployment-Plan/blob/main/deployment_timeline.png?raw=true)


---

## 9. APPENDIX: QUICK REFERENCE

**Key Tools:**
- MLflow: Model registry, experiment tracking
- Databricks: Data + model training unified platform
- Docker: Reproducible serving environments
- FastAPI: REST API for real-time scoring
- Grafana/Databricks SQL: Monitoring dashboards

**Key Metrics to Monitor:**
- AUC, precision, recall, calibration (model)
- KS statistic, feature distributions (data drift)
- P99 latency, error rate (system)
- Conversion rate, deal size (business)

**Decision Trees:**
- Deploy? → Check: technical metrics + manual review
- Retrain? → Check: drift alerts + business feedback
- Rollback? → Check: > 10% metric degradation + p < 0.05

---

## NEXT STEPS

1. **Week 1**: Set up Databricks workspace, MLflow, monitoring
2. **Week 2**: Implement shadow deployment, validation scripts
3. **Week 3**: Launch canary release (10% traffic)
4. **Week 4**: Begin A/B test (50/50 split)
5. **Week 6**: Evaluate results, make rollout/rollback decision
6. **Week 8**: Full production deployment
7. **Ongoing**: Weekly monitoring, monthly retraining

---

**Document Version**: 1.0  
**Last Updated**: December 7, 2025  
**Status**: Ready for Implementation  
**Confidence Level**: High
