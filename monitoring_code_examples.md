# Lead Scoring Model - Monitoring Dashboard Code Examples

## 1. Databricks SQL Monitoring Queries

### 1.1 Daily Model Performance Metrics

```sql
-- Query: Daily Model Performance Summary
-- Execution: Daily at 2 AM UTC (via Databricks Job)
-- Owner: MLOps Team
-- Alert Threshold: AUC < 0.85

SELECT
  DATE(prediction_date) as prediction_date,
  model_version,
  COUNT(DISTINCT lead_id) as num_leads_scored,
  COUNT(DISTINCT CASE WHEN predicted_score > 0.5 THEN lead_id END) as high_priority_leads,
  ROUND(100.0 * COUNT(DISTINCT CASE WHEN predicted_score > 0.5 THEN lead_id END) / COUNT(DISTINCT lead_id), 2) as pct_high_priority
FROM lead_scores
WHERE prediction_date >= CURRENT_DATE - 30
GROUP BY 1, 2
ORDER BY 1 DESC, 2;

-- Join with actual outcomes for performance metrics
WITH scores_and_outcomes AS (
  SELECT
    DATE(s.prediction_date) as prediction_date,
    s.model_version,
    s.predicted_score,
    COALESCE(o.converted, FALSE) as actual_converted,
    o.deal_size,
    o.days_to_close
  FROM lead_scores s
  LEFT JOIN lead_outcomes o ON s.lead_id = o.lead_id
    AND DATE_DIFF(o.outcome_timestamp, s.prediction_date) BETWEEN 0 AND 60
  WHERE s.prediction_date >= CURRENT_DATE - 30
)
SELECT
  prediction_date,
  model_version,
  COUNT(*) as num_predictions,
  ROUND(100.0 * SUM(CASE WHEN actual_converted THEN 1 ELSE 0 END) / COUNT(*), 2) as actual_conversion_rate_pct,
  ROUND(AVG(CASE WHEN actual_converted THEN predicted_score ELSE NULL END), 4) as avg_score_converted,
  ROUND(AVG(CASE WHEN NOT actual_converted THEN predicted_score ELSE NULL END), 4) as avg_score_not_converted,
  ROUND(CORR(predicted_score, CAST(actual_converted AS INT)), 4) as score_outcome_correlation
FROM scores_and_outcomes
GROUP BY 1, 2
ORDER BY 1 DESC, 2;
```

### 1.2 Data Drift Detection

```sql
-- Query: Feature Distribution Drift Analysis (Kolmogorov-Smirnov style)
-- Compares current week vs. training baseline
-- Execution: Weekly

WITH training_baseline AS (
  -- Baseline from training data
  SELECT
    'company_size_log10' as feature_name,
    PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY company_size_log10) as p10,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY company_size_log10) as q1,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY company_size_log10) as median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY company_size_log10) as q3,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY company_size_log10) as p90,
    COUNT(*) as baseline_count
  FROM training_dataset
  GROUP BY 1
  
  UNION ALL
  
  SELECT
    'engagement_score',
    PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY engagement_score) as p10,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY engagement_score) as q1,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY engagement_score) as median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY engagement_score) as q3,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY engagement_score) as p90,
    COUNT(*) as baseline_count
  FROM training_dataset
  GROUP BY 1
),
current_distribution AS (
  -- Current week's data
  SELECT
    'company_size_log10' as feature_name,
    PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY company_size_log10) as p10,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY company_size_log10) as q1,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY company_size_log10) as median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY company_size_log10) as q3,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY company_size_log10) as p90,
    COUNT(*) as current_count
  FROM lead_features
  WHERE created_date >= CURRENT_DATE - 7
  GROUP BY 1
  
  UNION ALL
  
  SELECT
    'engagement_score',
    PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY engagement_score) as p10,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY engagement_score) as q1,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY engagement_score) as median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY engagement_score) as q3,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY engagement_score) as p90,
    COUNT(*) as current_count
  FROM lead_features
  WHERE created_date >= CURRENT_DATE - 7
  GROUP BY 1
)
SELECT
  b.feature_name,
  ROUND(ABS((c.median - b.median) / NULLIF(b.median, 0)) * 100, 2) as median_change_pct,
  ROUND(ABS((c.q1 - b.q1) / NULLIF(b.q1, 0)) * 100, 2) as q1_change_pct,
  ROUND(ABS((c.q3 - b.q3) / NULLIF(b.q3, 0)) * 100, 2) as q3_change_pct,
  CASE
    WHEN ABS((c.median - b.median) / NULLIF(b.median, 0)) > 0.20 THEN 'ALERT'
    WHEN ABS((c.median - b.median) / NULLIF(b.median, 0)) > 0.15 THEN 'WARNING'
    ELSE 'OK'
  END as drift_status,
  b.baseline_count,
  c.current_count
FROM training_baseline b
JOIN current_distribution c ON b.feature_name = c.feature_name
ORDER BY median_change_pct DESC;
```

### 1.3 Prediction Drift Detection

```sql
-- Query: Prediction Score Distribution Changes
-- Execution: Daily

WITH daily_predictions AS (
  SELECT
    DATE(prediction_date) as date,
    PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY predicted_score) as p10,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY predicted_score) as q1,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY predicted_score) as median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY predicted_score) as q3,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY predicted_score) as p90,
    COUNT(*) as num_predictions,
    SUM(CASE WHEN predicted_score > 0.7 THEN 1 ELSE 0 END) as high_priority_count
  FROM lead_scores
  WHERE prediction_date >= CURRENT_DATE - 30
  GROUP BY 1
)
SELECT
  date,
  ROUND(median, 4) as median_score,
  ROUND(p10, 4) as p10_score,
  ROUND(p90, 4) as p90_score,
  ROUND(p90 - p10, 4) as score_range_iqr,
  num_predictions,
  ROUND(100.0 * high_priority_count / num_predictions, 2) as pct_high_priority,
  ROUND(100.0 * (CAST(high_priority_count AS FLOAT) / LAG(CAST(high_priority_count AS FLOAT)) OVER (ORDER BY date) - 1), 2) as high_priority_change_pct,
  CASE
    WHEN ABS(ROUND(median, 4) - LAG(ROUND(median, 4)) OVER (ORDER BY date)) > 0.10 THEN 'ALERT'
    WHEN ABS(ROUND(median, 4) - LAG(ROUND(median, 4)) OVER (ORDER BY date)) > 0.05 THEN 'WARNING'
    ELSE 'OK'
  END as drift_status
FROM daily_predictions
ORDER BY date DESC;
```

### 1.4 Model Latency Monitoring

```sql
-- Query: API Latency Percentiles
-- Execution: Continuous (every 15 minutes)

SELECT
  DATE_TRUNC('hour', prediction_timestamp) as hour,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) as p50,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY latency_ms) as p90,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95,
  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99,
  MAX(latency_ms) as max_latency,
  COUNT(*) as num_requests,
  SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END) as num_errors,
  ROUND(100.0 * SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END) / COUNT(*), 4) as error_rate_pct
FROM prediction_logs
WHERE prediction_timestamp >= CURRENT_TIMESTAMP - INTERVAL 24 HOUR
GROUP BY 1
ORDER BY 1 DESC;

-- Alert if latency spikes
CREATE OR REPLACE ALERT latency_spike_alert AS
SELECT
  CASE
    WHEN PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) > 200 THEN 'ALERT'
    ELSE 'OK'
  END as alert_status
FROM prediction_logs
WHERE prediction_timestamp >= CURRENT_TIMESTAMP - INTERVAL 1 HOUR
HAVING alert_status = 'ALERT';
```

### 1.5 Business Metrics Report

```sql
-- Query: Conversion and Revenue Impact
-- Execution: Weekly

WITH scoring_and_outcomes AS (
  SELECT
    DATE(s.prediction_date) as week_start,
    s.model_version,
    s.lead_id,
    s.predicted_score,
    COALESCE(o.converted, FALSE) as converted,
    COALESCE(o.deal_size, 0) as deal_size,
    COALESCE(o.days_to_close, NULL) as days_to_close,
    CASE WHEN s.predicted_score > 0.7 THEN 'high' WHEN s.predicted_score > 0.4 THEN 'medium' ELSE 'low' END as priority_bin
  FROM lead_scores s
  LEFT JOIN lead_outcomes o ON s.lead_id = o.lead_id
  WHERE DATE(s.prediction_date) >= CURRENT_DATE - 30
)
SELECT
  week_start,
  model_version,
  priority_bin,
  COUNT(DISTINCT lead_id) as num_leads,
  SUM(CASE WHEN converted THEN 1 ELSE 0 END) as num_converted,
  ROUND(100.0 * SUM(CASE WHEN converted THEN 1 ELSE 0 END) / COUNT(DISTINCT lead_id), 2) as conversion_rate_pct,
  ROUND(AVG(deal_size), 2) as avg_deal_size,
  ROUND(AVG(CASE WHEN converted THEN days_to_close ELSE NULL END), 1) as avg_days_to_close,
  ROUND(SUM(CASE WHEN converted THEN deal_size ELSE 0 END), 2) as total_revenue
FROM scoring_and_outcomes
GROUP BY 1, 2, 3
ORDER BY 1 DESC, 2, 3;
```

---

## 2. Python Monitoring Functions

### 2.1 Data Drift Detection Function

```python
import numpy as np
from scipy.stats import ks_2samp, entropy
import pandas as pd
from typing import Dict, Tuple

def detect_data_drift(
    current_data: pd.DataFrame,
    baseline_data: pd.DataFrame,
    features: list,
    ks_threshold: float = 0.15,
    divergence_threshold: float = 0.10
) -> Dict:
    """
    Detect feature drift using Kolmogorov-Smirnov test.
    
    Args:
        current_data: Recent data (e.g., last week)
        baseline_data: Training data baseline
        features: List of feature names to monitor
        ks_threshold: Alert if KS statistic > threshold
        divergence_threshold: Alert if JS divergence > threshold
    
    Returns:
        Dictionary with drift report per feature
    """
    drift_report = {}
    
    for feature in features:
        # Skip if feature missing
        if feature not in current_data.columns or feature not in baseline_data.columns:
            continue
        
        current_values = current_data[feature].dropna().values
        baseline_values = baseline_data[feature].dropna().values
        
        # Kolmogorov-Smirnov test (continuous features)
        if np.issubdtype(current_data[feature].dtype, np.number):
            ks_stat, p_value = ks_2samp(current_values, baseline_values)
            
            drift_report[feature] = {
                "feature_type": "continuous",
                "ks_statistic": round(ks_stat, 4),
                "p_value": round(p_value, 6),
                "drifted": ks_stat > ks_threshold,
                "severity": "high" if ks_stat > 0.25 else "medium" if ks_stat > ks_threshold else "low"
            }
        else:
            # Jensen-Shannon for categorical features
            current_dist = current_data[feature].value_counts(normalize=True)
            baseline_dist = baseline_data[feature].value_counts(normalize=True)
            
            # Align indices
            all_categories = set(current_dist.index) | set(baseline_dist.index)
            current_dist = current_dist.reindex(all_categories, fill_value=0)
            baseline_dist = baseline_dist.reindex(all_categories, fill_value=0)
            
            # Jensen-Shannon divergence
            m = (current_dist + baseline_dist) / 2
            js_divergence = (entropy(current_dist, m) + entropy(baseline_dist, m)) / 2
            
            drift_report[feature] = {
                "feature_type": "categorical",
                "js_divergence": round(js_divergence, 4),
                "drifted": js_divergence > divergence_threshold,
                "severity": "high" if js_divergence > 0.20 else "medium" if js_divergence > divergence_threshold else "low"
            }
    
    # Summary
    drifted_features = [f for f, report in drift_report.items() if report.get("drifted", False)]
    
    return {
        "total_features_monitored": len(drift_report),
        "features_drifted": len(drifted_features),
        "drifted_feature_names": drifted_features,
        "feature_reports": drift_report,
        "overall_drift": "HIGH" if len(drifted_features) > 2 else "MEDIUM" if len(drifted_features) == 1 else "LOW"
    }

# Example usage:
# drift_report = detect_data_drift(recent_data, training_data, FEATURE_COLUMNS)
# if drift_report["overall_drift"] == "HIGH":
#     send_alert(f"Data drift detected: {drift_report['drifted_feature_names']}")
```

### 2.2 Prediction Drift Detection

```python
def detect_prediction_drift(
    current_scores: np.ndarray,
    baseline_scores: np.ndarray,
    threshold_percentile_change: float = 0.20
) -> Dict:
    """
    Detect prediction distribution drift.
    """
    current_scores = np.array(current_scores).flatten()
    baseline_scores = np.array(baseline_scores).flatten()
    
    metrics = {}
    
    # Percentile changes
    for p in [10, 25, 50, 75, 90]:
        current_p = np.percentile(current_scores, p)
        baseline_p = np.percentile(baseline_scores, p)
        change = abs((current_p - baseline_p) / (baseline_p + 1e-10))
        
        metrics[f"p{p}"] = {
            "current": round(current_p, 4),
            "baseline": round(baseline_p, 4),
            "change_pct": round(change * 100, 2),
            "drifted": change > threshold_percentile_change
        }
    
    # Model confidence entropy
    current_entropy = -np.mean(
        current_scores * np.log(current_scores + 1e-10) +
        (1 - current_scores) * np.log(1 - current_scores + 1e-10)
    )
    baseline_entropy = -np.mean(
        baseline_scores * np.log(baseline_scores + 1e-10) +
        (1 - baseline_scores) * np.log(1 - baseline_scores + 1e-10)
    )
    
    metrics["entropy"] = {
        "current": round(current_entropy, 4),
        "baseline": round(baseline_entropy, 4),
        "change_pct": round(abs(current_entropy - baseline_entropy) / baseline_entropy * 100, 2)
    }
    
    drifted_percentiles = [p for p in [10, 25, 50, 75, 90] if metrics[f"p{p}"].get("drifted")]
    
    return {
        "drifted": len(drifted_percentiles) > 0,
        "severity": "high" if len(drifted_percentiles) > 2 else "medium" if len(drifted_percentiles) > 0 else "low",
        "drifted_percentiles": drifted_percentiles,
        "metrics": metrics
    }
```

### 2.3 Latency Monitoring

```python
def log_prediction_latency(latency_ms: float, model_version: str, batch_size: int):
    """
    Log prediction latency with context.
    """
    import mlflow
    from datetime import datetime
    
    # Determine if latency is anomalous
    anomalous = latency_ms > 200  # p99 threshold
    
    # Log to MLflow
    mlflow.log_metric("prediction_latency_ms", latency_ms)
    mlflow.log_metric("batch_size", batch_size)
    mlflow.log_metric("latency_per_record_ms", latency_ms / batch_size)
    
    # Write to monitoring table
    monitoring_record = {
        "timestamp": datetime.utcnow(),
        "latency_ms": latency_ms,
        "batch_size": batch_size,
        "model_version": model_version,
        "latency_per_record_ms": latency_ms / batch_size,
        "anomalous": anomalous
    }
    
    # Spark write to Delta table
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("monitoring").getOrCreate()
    spark.createDataFrame([monitoring_record]).write.format("delta").mode("append").save(
        "dbfs:/monitoring/prediction_latency"
    )
    
    # Alert if anomalous
    if anomalous:
        send_slack_alert(f"⚠️ Latency spike: {latency_ms}ms for batch size {batch_size}")

def calculate_latency_percentiles(hours: int = 24) -> Dict:
    """
    Calculate latency percentiles for reporting.
    """
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    
    spark = SparkSession.builder.appName("monitoring").getOrCreate()
    
    df = spark.read.format("delta").load("dbfs:/monitoring/prediction_latency") \
        .filter(F.col("timestamp") >= F.current_timestamp() - F.lit(f"{hours} hours"))
    
    percentiles = df.approxQuantile(
        "latency_ms",
        [0.50, 0.90, 0.95, 0.99],
        0.01
    )
    
    return {
        "p50_ms": round(percentiles[0], 2),
        "p90_ms": round(percentiles[1], 2),
        "p95_ms": round(percentiles[2], 2),
        "p99_ms": round(percentiles[3], 2),
        "max_ms": round(df.agg(F.max("latency_ms")).collect()[0][0], 2),
        "records_monitored": df.count()
    }
```

### 2.4 Automated Alerting

```python
import json
from slack_sdk import WebClient
from typing import Optional

def send_slack_alert(message: str, severity: str = "warning", channel: str = "#ml-alerts"):
    """
    Send alert to Slack.
    """
    client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
    
    color_map = {
        "critical": "danger",
        "warning": "warning",
        "info": "good"
    }
    
    response = client.chat_postMessage(
        channel=channel,
        attachments=[
            {
                "color": color_map.get(severity, "warning"),
                "title": f"🚨 {severity.upper()} - Lead Scoring Model",
                "text": message,
                "ts": int(datetime.now().timestamp())
            }
        ]
    )
    return response

def send_email_alert(
    subject: str,
    body: str,
    recipients: list,
    severity: str = "warning"
):
    """
    Send alert via email.
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    msg = MIMEMultipart()
    msg['Subject'] = f"[{severity.upper()}] {subject}"
    msg['From'] = os.getenv("ALERT_EMAIL_FROM")
    msg['To'] = ", ".join(recipients)
    
    html = f"""
    <html>
      <body>
        <h2>{subject}</h2>
        <p>{body}</p>
        <hr/>
        <p><small>Automated alert from Lead Scoring MLOps system</small></p>
      </body>
    </html>
    """
    
    msg.attach(MIMEText(html, 'html'))
    
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(os.getenv("ALERT_EMAIL_USER"), os.getenv("ALERT_EMAIL_PASSWORD"))
        server.send_message(msg)

# Alert orchestration
def run_monitoring_checks():
    """
    Run all monitoring checks and alert if needed.
    """
    # Check 1: Data drift
    drift_report = detect_data_drift(current_data, baseline_data, FEATURE_COLUMNS)
    if drift_report["overall_drift"] == "HIGH":
        send_slack_alert(
            f"Data drift detected in features: {', '.join(drift_report['drifted_feature_names'])}",
            severity="critical"
        )
    
    # Check 2: Model performance
    recent_auc = calculate_auc(recent_predictions, recent_actuals)
    if recent_auc < 0.85:
        send_slack_alert(
            f"Model AUC degraded: {recent_auc:.3f} < 0.85 threshold",
            severity="critical"
        )
    
    # Check 3: Latency
    latency_percentiles = calculate_latency_percentiles()
    if latency_percentiles["p99_ms"] > 200:
        send_slack_alert(
            f"P99 latency spike: {latency_percentiles['p99_ms']}ms",
            severity="warning"
        )
    
    print("Monitoring checks completed")
```

---

## 3. Grafana Dashboard JSON (Optional)

For teams using Grafana instead of Databricks SQL, here's a sample dashboard configuration.

```json
{
  "dashboard": {
    "title": "Lead Scoring Model Monitoring",
    "panels": [
      {
        "title": "Daily AUC Score",
        "targets": [
          {
            "expr": "SELECT DATE(prediction_date), AVG(auc) FROM model_metrics GROUP BY 1"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Feature Drift (KS Statistic)",
        "targets": [
          {
            "expr": "SELECT feature_name, MAX(ks_statistic) FROM feature_drift GROUP BY 1"
          }
        ],
        "type": "table"
      },
      {
        "title": "API Latency (P99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, latency_ms)"
          }
        ],
        "type": "gauge",
        "thresholds": ["200"]
      },
      {
        "title": "Conversion Rate Trend",
        "targets": [
          {
            "expr": "SELECT DATE(outcome_date), conversion_rate FROM business_metrics ORDER BY 1"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

---

## Summary

This code provides:
- ✓ Production-ready SQL queries for Databricks
- ✓ Python functions for drift detection
- ✓ Latency monitoring and alerting
- ✓ Slack/email integration
- ✓ Grafana dashboard config

**Deploy these:**
1. SQL queries → Databricks jobs (scheduled)
2. Python functions → MLflow projects or Databricks notebooks
3. Alerting → GitHub Actions CI/CD or Databricks workflows