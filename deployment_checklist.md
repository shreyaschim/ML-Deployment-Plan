# QUICK REFERENCE: Lead Scoring MLOps Deployment Checklist

## Pre-Deployment (Week 1)

- [ ] **Infrastructure Setup**
  - [ ] Databricks workspace provisioned
  - [ ] MLflow tracking server configured
  - [ ] PostgreSQL database for monitoring data
  - [ ] Docker registry access (for container serving)
  - [ ] Slack workspace integration for alerts

- [ ] **Data Preparation**
  - [ ] Training data snapshot versioned (Delta Lake)
  - [ ] Feature engineering code in Git
  - [ ] Data validation rules implemented (Great Expectations)
  - [ ] Baseline metrics calculated from training data

- [ ] **Model Registry**
  - [ ] MLflow tracking experiment created
  - [ ] Model registered with version metadata
  - [ ] Hyperparameters and config stored in Git
  - [ ] Training data snapshot hash recorded

- [ ] **Monitoring Setup**
  - [ ] Databricks SQL workspace created
  - [ ] Monitoring tables schema defined
  - [ ] Drift detection functions implemented
  - [ ] Alert notification channels configured

---

## Stage 1: Shadow Deployment (Week 1-2)

**Objective**: Validate new model runs without impacting business

### Deployment Steps
- [ ] Deploy model as REST API (Databricks Model Serving)
- [ ] Create shadow prediction job (scheduled daily)
- [ ] Route predictions to separate "shadow" table (not shown to users)
- [ ] Log all predictions with timestamp and latency

### Validation Steps
- [ ] **Check system health**
  - [ ] API latency: p99 < 200ms? ✓
  - [ ] Throughput: > 1000 req/s? ✓
  - [ ] Error rate: < 0.1%? ✓
  - [ ] Memory usage: < 2GB per instance? ✓
  
- [ ] **Check prediction quality** (vs. actual outcomes from week prior)
  - [ ] AUC: ≥ 0.85? ✓
  - [ ] Precision @ 0.5 threshold: ≥ 0.80? ✓
  - [ ] Recall: ≥ 0.80? ✓
  - [ ] Calibration error: < 0.05? ✓
  
- [ ] **Check data quality**
  - [ ] Feature distributions normal? ✓
  - [ ] Missing values < 1%? ✓
  - [ ] No unexpected categories? ✓

### Success Criteria
- ✓ 2 weeks of shadow predictions logged
- ✓ No critical issues detected
- ✓ ML team sign-off on metrics

### If Issues Found
- [ ] Debug and fix issues
- [ ] Retrain model if needed
- [ ] Extend shadow phase another week
- [ ] Escalate to leadership if model quality poor

---

## Stage 2: Canary Release (Week 3)

**Objective**: Route small % of traffic to new model, monitor closely

### Deployment Steps
- [ ] Configure traffic split: 90% old model, 10% new model
- [ ] Ensure reproducible randomization (hash-based per lead_id)
- [ ] Log all predictions (both old and new) with variant label

### Monitoring (Daily Check-in)
- [ ] **Performance metrics**: AUC, precision, recall stable?
- [ ] **Latency**: Any increase? p99 still < 200ms?
- [ ] **Data drift**: Any unexpected feature changes?
- [ ] **Business feedback**: Sales team notice anything odd?

### Success Criteria
- ✓ No metrics degrade by > 10%
- ✓ Error rate remains < 0.1%
- ✓ Sales team reports no issues
- ✓ 1 week of canary data collected

### If Issues Found
- [ ] Immediate rollback to old model (100% traffic)
- [ ] Investigate root cause
- [ ] Fix and retrain
- [ ] Go back to shadow phase

---

## Stage 3: A/B Test (Week 4-6)

**Objective**: Measure business impact at 50/50 traffic split

### Deployment Steps
- [ ] Configure 50/50 traffic split (random, hash-based)
- [ ] Label all predictions with variant (A = old model, B = new model)
- [ ] Log predictions + outcomes for 2-4 weeks

### Success Metrics to Track (Weekly Updates)
| Metric | Target | Status |
|--------|--------|--------|
| Conversion Rate | +5% (p < 0.05) | _____ |
| Avg Deal Size | No decline | _____ |
| Sales Cycle Time | No increase | _____ |
| Sales Satisfaction | ≥ 8/10 | _____ |
| Model Accuracy | ≥ 0.87 AUC | _____ |

### Analysis Checklist
- [ ] Minimum 5,000 leads per variant
- [ ] Run chi-square test for statistical significance
- [ ] Calculate 95% confidence intervals
- [ ] Segment results by lead source, industry, region
- [ ] Check for any negative subgroup impacts

### Success Criteria
- ✓ Conversion lift ≥ 5% (p < 0.05)
- ✓ No decline in deal size or cycle time
- ✓ Sales team satisfied (8/10 or higher)
- ✓ Statistical significance achieved

### If Results Show Success
- ✓ Proceed to Full Rollout (Stage 4)

### If Results Are Inconclusive
- [ ] Extend test by 1 week
- [ ] Increase sample size
- [ ] Review segment-level results

### If Results Show No Lift or Decline
- [ ] Stop test immediately
- [ ] Rollback to old model
- [ ] Post-mortem: what went wrong?
- [ ] Investigate feature sources, data quality, business changes

---

## Stage 4: Full Rollout (Week 7+)

**Objective**: Deploy to 100% of leads, establish monitoring baseline

### Deployment Steps
- [ ] Update API routing: 100% traffic to new model
- [ ] Monitor old model logs (archive, don't delete)
- [ ] Update CRM/dashboard to show new scores

### Ongoing Monitoring (Daily)

**Every Morning Standup:**
- [ ] Check monitoring dashboard for red flags
- [ ] Review overnight logs for errors or anomalies
- [ ] Confirm predictions being written to Delta tables

**Weekly Review:**
- [ ] Data drift report (KS test on each feature)
- [ ] Prediction drift report (percentile changes)
- [ ] Performance metrics (AUC, precision, recall)
- [ ] Business metrics (conversion rate, deal size)
- [ ] API health (latency, throughput, errors)

**Monthly Deep-Dive:**
- [ ] Calibration analysis (predicted vs. actual conversion)
- [ ] Feature importance review
- [ ] Lead source segmentation analysis
- [ ] Competitive analysis (vs. human scoring)
- [ ] Retraining decision (scheduled for next month?)

### Automated Alerts (Active)
- [ ] ⚠️ If AUC drops below 0.85
- [ ] ⚠️ If KS statistic > 0.15 for any feature
- [ ] ⚠️ If p99 latency > 200ms
- [ ] ⚠️ If error rate > 0.5%
- [ ] ⚠️ If model coverage < 95% (% leads scored)

---

## Retraining Schedule

### Automatic Triggers
- [ ] **Weekly retraining** (Sunday 2 AM UTC)
  - Collect all outcomes from past 30 days
  - Run training pipeline (data prep → model fit → evaluation)
  - Compare new model to current production model
  - If AUC ≥ 0.87, promote to Staging for shadow deployment
  - If AUC < 0.85, alert ML team for investigation

- [ ] **Emergency retraining** if any trigger fires:
  - AUC degrades to < 0.85
  - Data drift detected in > 2 features
  - Sales team reports model is unhelpful

### Manual Retraining
- [ ] **When needed**: Request from ML engineer
  - New feature engineered
  - Business process changed
  - Quarterly model refresh

### Retraining Workflow
1. Load recent lead data (past 30 days)
2. Join with ground truth outcomes
3. Engineer features (same code as training)
4. Train model (same hyperparameters)
5. Evaluate vs. baseline
6. Log metrics to MLflow
7. Automatic promotion to Staging if AUC ≥ 0.87
8. Manual shadow deployment + canary if approved

---

## Troubleshooting Guide

### Issue: Sales says model scores no longer help prioritize

**Step 1: Check System Health** (5 min)
- [ ] Is API returning predictions? Check logs for errors
- [ ] Are predictions fresh? (< 4 hours old?)
- [ ] Is coverage high? (> 95% of leads scored?)

**Step 2: Check Model Performance** (10 min)
- [ ] Run: `SELECT AVG(auc) FROM model_metrics WHERE date >= TODAY() - 7`
- [ ] AUC degraded from baseline (0.87)?
  - YES → Go to Step 3
  - NO → Go to Step 4

**Step 3: Model Drift Investigation** (30 min)
- [ ] Run data drift detection (KS test)
- [ ] Which features drifted?
- [ ] Are drifted features high-importance?
- [ ] Action: Retrain model with recent data

**Step 4: Data Quality Investigation** (30 min)
- [ ] Check for missing values increase
- [ ] Check for new categories in categorical features
- [ ] Check for outliers in numerical features
- [ ] Action: Investigate data source, fix quality issues

**Step 5: Business Alignment Check** (20 min)
- [ ] Did sales processes change? (new qualification rules?)
- [ ] Did lead sources change? (new channels?)
- [ ] Did market conditions shift? (external factors?)
- [ ] Action: Gather feedback, retrain with updated labels

**Step 6: Integration Check** (15 min)
- [ ] Are sales using model correctly?
- [ ] Is CRM showing correct scores?
- [ ] Are threshold defaults still appropriate?
- [ ] Action: Retrain sales, adjust threshold, or fix CRM integration

### Issue: API latency spike

**Diagnosis:**
```sql
-- Check latency trend
SELECT
  DATE_TRUNC('hour', timestamp) as hour,
  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99
FROM prediction_logs
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL 6 HOUR
GROUP BY 1
ORDER BY 1 DESC;
```

**Possible Causes:**
1. **High batch size** → Check if batch_size increased
2. **Model complexity** → Check if model retraining added features
3. **Infrastructure load** → Check Databricks cluster utilization
4. **Network latency** → Check CRM → Databricks connection

**Resolution:**
- [ ] Check batch processing time: `SELECT AVG(latency_ms / batch_size) FROM prediction_logs`
- [ ] Scale up Databricks cluster if CPU high
- [ ] Optimize feature engineering code
- [ ] Use caching for frequent feature lookups

### Issue: Data drift detected in multiple features

**Response:**
1. [ ] Alert ML team immediately (Slack)
2. [ ] Check if this is expected (new lead source, market change?)
3. [ ] If unexpected, trigger emergency retraining
4. [ ] Monitor model performance after retraining
5. [ ] Document root cause for future reference

---

## Metrics Dashboard Checklist

**System Health Tab:**
- [ ] API Latency (p50, p99) ← < 200ms
- [ ] Throughput (req/s) ← > 1000
- [ ] Error Rate (%) ← < 0.1%
- [ ] Uptime (%) ← > 99.9%

**Model Performance Tab:**
- [ ] AUC Score ← > 0.85
- [ ] Precision @ threshold ← > 0.80
- [ ] Recall ← > 0.80
- [ ] Calibration Error ← < 0.05

**Data Quality Tab:**
- [ ] Data Freshness (hours) ← < 4
- [ ] Missing Values (%) ← < 1%
- [ ] Feature Drift (KS stat) ← < 0.15
- [ ] Coverage (% leads scored) ← > 95%

**Business Impact Tab:**
- [ ] Conversion Rate (%) ← tracking
- [ ] Avg Deal Size ($) ← tracking
- [ ] Sales Satisfaction (1-10) ← ≥ 8
- [ ] Model Usage (scores/day) ← > 1000

---

## Escalation Path

**Green Light (All Good)**
- Daily monitoring continues
- Weekly metrics review
- No action needed

**Yellow Alert (Warning)**
- One metric in yellow zone
- Example: AUC = 0.84 (below 0.85 threshold)
- Action: Notify ML team, plan retraining
- Timeline: Investigate within 48 hours

**Red Alert (Critical)**
- Multiple metrics degraded or system failure
- Example: API error rate > 5%, latency > 500ms
- Action: Immediate rollback to previous model version
- Timeline: Restore service within 1 hour
- Follow-up: Post-mortem with team

---

## CI/CD Pipeline Checklist

**Pre-Commit:**
- [ ] Code linting (flake8)
- [ ] Type checking (mypy)
- [ ] Unit tests pass (pytest)

**Pre-Training:**
- [ ] Data validation (Great Expectations)
- [ ] Feature engineering determinism check
- [ ] Config matches Git version

**Post-Training:**
- [ ] Model metrics logged to MLflow
- [ ] Comparison with baseline model
- [ ] Model pushed to registry

**Pre-Deployment:**
- [ ] Manual review of metrics
- [ ] Approval from ML lead
- [ ] Staging slot assignment

**Post-Deployment:**
- [ ] Shadow deployment validation (1 week)
- [ ] Canary release (10% traffic, 1 week)
- [ ] A/B test (50/50 split, 2-4 weeks)
- [ ] Full rollout decision

---

## Success Metrics (90-Day Goals)

**Month 1:**
- ✓ Shadow deployment operational
- ✓ Zero critical incidents
- ✓ Monitoring dashboard live

**Month 2:**
- ✓ A/B test shows +5% conversion lift (p < 0.05)
- ✓ Sales team satisfaction ≥ 8/10
- ✓ Full rollout approved

**Month 3:**
- ✓ Model deployed to 100% of leads
- ✓ Automated retraining operational
- ✓ Incremental revenue: > $500K

---

## Key Contacts

| Role | Name | Email | Slack |
|------|------|-------|-------|
| ML Lead | _____ | _____ | _____ |
| MLOps Engineer | _____ | _____ | _____ |
| Data Engineer | _____ | _____ | _____ |
| Sales Lead | _____ | _____ | _____ |
| Product Manager | _____ | _____ | _____ |

---

## Useful Links

- MLflow Documentation: https://mlflow.org/docs/latest/
- Databricks Model Serving: https://docs.databricks.com/machine-learning/model-serving/
- Great Expectations: https://greatexpectations.io/
- Evidently AI (Drift Monitoring): https://www.evidentlyai.com/

---

**Last Updated:** December 7, 2025
**Version:** 1.0
**Status:** Ready for Deployment
