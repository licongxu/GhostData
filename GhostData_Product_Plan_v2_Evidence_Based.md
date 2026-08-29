# GhostData — Evidence-Based Hackathon Product Plan v2

> **Status:** redesigned from the original GhostData concept after reviewing public production-ML incidents and existing validation / model-testing products.  
> **Hackathon scope:** tabular ML, pre-deployment / pipeline-change testing, Daytona-powered counterexample search.

---

# 0. Executive decision

## Final product definition

**GhostData is adversarial CI for ML data pipelines.**

A team already has:

- a production or candidate ML model,
- a reference / replay dataset with labels,
- an existing data or feature pipeline,
- and a set of safeguards they already trust:
  - schema checks,
  - null checks,
  - range checks,
  - distribution checks,
  - data contracts,
  - drift checks,
  - custom validation code.

GhostData asks a different question:

> **Can we construct a plausible pipeline failure that still passes every safeguard you already have, but materially degrades the downstream model?**

If the answer is yes, GhostData has found a **Ghost**.

A Ghost is not an LLM warning. It is an executable counterexample:

- a reproducible failure mechanism,
- the transformed / replay dataset,
- the exact existing checks it survived,
- measured downstream model damage,
- a minimal reproducer,
- and a suggested new regression test or data contract.

The long-term loop is:

```text
existing safeguards
        |
        v
search for a counterexample
        |
        v
all safeguards PASS?
      /     \
    no       yes
    |         |
 discard   model breaks?
             /    \
           no      yes
           |        |
        discard    GHOST
                    |
                    v
          promote into new test
                    |
                    v
            safeguards improve
```

## Primary pitch line

> **Your tests can only catch failures you encoded. GhostData searches for a failure that satisfies every test you have.**

## Secondary pitch line

> **Move the postmortem before the incident.**

## Daytona pitch line

> **Each possible failure is executable code. Daytona turns counterexample generation into a real parallel search instead of a handful of hand-written tests.**

---

# 1. The industry problem is real

GhostData should not be pitched as if production ML teams do not validate data. They do.

The actual problem is narrower:

> **A data / feature pipeline can remain technically valid while the semantics seen by a downstream model become wrong enough to degrade performance.**

This class of failure is well documented.

## 1.1 Google: serving bug silently pins a feature to `-1`

Google publicly described a production ML incident in which:

- a model was retrained daily,
- a serving-stack refactor accidentally pinned one feature to `-1`,
- the model continued returning predictions rather than crashing,
- accuracy degraded,
- serving data was reused for training,
- so the problem persisted and worsened until discovered.

The important lesson is not the exact `-1` value.

It is:

> **ML systems can fail silently because corrupted semantics can still be valid machine-readable inputs.**

Source:

- Google Cloud, *Monitor models for training-serving skew with Vertex AI*  
  https://cloud.google.com/blog/topics/developers-practitioners/monitor-models-training-serving-skew-vertex-ai

Google's *Rules of Machine Learning* also states that production systems at Google have experienced training-serving skew that negatively affected performance, and teams were sometimes surprised when they explicitly measured training/serving consistency.

Source:

- Google for Developers, *Rules of Machine Learning*  
  https://developers.google.com/machine-learning/guides/rules-of-ml/

---

## 1.2 Uber: the most impactful data regressions can be silent

Uber has publicly written that bad data quality affects ML outputs such as fares, ETAs, and products, and that some data issues are only discovered weeks or months after they begin.

Their D3 system specifically targets silent data regressions.

Uber lists causes including:

- logging changes,
- schema changes,
- changed field meaning,
- ETL logic changes,
- erroneous joins,
- partial / incomplete data.

Uber also reports that partial-data incidents had substantially longer time-to-detection than obvious infrastructure / ETL outages.

Sources:

- Uber, *D3: An Automated System to Detect Data Drifts*  
  https://www.uber.com/gb/en/blog/d3-an-automated-system-to-detect-data-drifts/

- Uber, *How Uber Achieves Operational Excellence in the Data Quality Experience*  
  https://www.uber.com/en-EE/blog/operational-excellence-data-quality/

The important lesson:

> **The dangerous incidents are often not the ones that crash the pipeline. They are the ones that continue producing plausible-looking data.**

---

## 1.3 LinkedIn: duplicates, skew, and feature-pipeline complexity are production risks

LinkedIn's Data Sentinel has caught:

- duplicated work-anniversary data,
- duplicated organization primary keys,
- corrupted member / jobs data,
- duplicate recruiter records that could otherwise bias ML models.

Source:

- LinkedIn Engineering, *Data Sentinel: Automating data validation*  
  https://www.linkedin.com/blog/engineering/data-management/data-sentinel-automating-data-validation

LinkedIn's Feathr feature-store work also emphasizes that feature pipelines must:

- combine time-sensitive data from many sources,
- perform point-in-time-correct joins,
- produce consistent feature definitions for training and inference,
- avoid training-serving skew.

Source:

- LinkedIn Engineering, *Open sourcing Feathr – LinkedIn’s feature store for productive machine learning*  
  https://www.linkedin.com/blog/engineering/open-source/open-sourcing-feathr--linkedin-s-feature-store-for-productive-m

The important lesson:

> **Entity alignment, temporal semantics, feature definitions, and joins are model-critical—not merely schema-level concerns.**

---

## 1.4 Avanade: an important ML feature silently became zero

A public Great Expectations case study describes an Avanade data-science team dealing with frequent upstream taxonomy and data-model changes.

One incident was noticed only coincidentally: a top model feature had fallen to zero because of an issue deep in the data warehouse.

They also describe dummy / outlier values entering data without downstream teams necessarily knowing.

Source:

- Great Expectations, *How Avanade Detects Data Drift with Great Expectations*  
  https://greatexpectations.io/case-studies/how-avanade-uses-gx-to-detect-data-drift-from-upstream-model-changes-in/

The important lesson:

> **Many safeguards are added after a failure mode becomes known.**

GhostData's opportunity is to search for the next missing safeguard before the incident.

---

# 2. What industry already solves

This section is essential for credibility.

GhostData is **not** based on the claim that nobody validates production data.

## 2.1 Data validation tools

Systems such as Great Expectations can validate:

- schema,
- ranges,
- means,
- standard deviations,
- quantiles,
- KL-divergence against reference distributions,
- other user-defined expectations.

Source:

- Great Expectations, *Validate data distribution with GX*  
  https://docs.greatexpectations.io/docs/reference/learn/data_quality_use_cases/distribution/

These tools answer:

> **Does the observed dataset violate a check we defined?**

They are complementary to GhostData.

---

## 2.2 Feature stores and training-serving consistency

Feature stores such as Feathr exist partly to make feature computation consistent between offline training and online inference, including point-in-time correctness.

They answer questions such as:

> **Are we computing and serving this feature consistently?**

Again, this is complementary.

---

## 2.3 Model/data testing systems such as Deepchecks

Deepchecks already supports sophisticated checks including:

- feature drift,
- prediction drift,
- feature-label correlation change,
- segment performance,
- weak-segment discovery,
- performance bias,
- model validation in CI/CD.

Sources:

- Deepchecks, *Using Deepchecks in CI/CD*  
  https://docs.deepchecks.com/stable/general/usage/ci_cd.html

- Deepchecks, *Weak Segments Performance*  
  https://docs.deepchecks.com/stable/tabular/auto_checks/model_evaluation/plot_weak_segments_performance.html

Therefore GhostData must **not** claim:

> "Existing tools only check feature means or marginal distributions."

That is false and easy for a technical judge to challenge.

---

## 2.4 Robust Intelligence is an important adjacent competitor

Robust Intelligence / RIME already provides AI Stress Testing.

Its public documentation describes:

- hundreds of pre-configured tests,
- abnormal-input tests,
- distribution-shift tests,
- model-behavior tests,
- transformations,
- adversarial attacks,
- data-cleanliness tests,
- tabular / NLP / vision support,
- pre-deployment stress testing,
- scheduled / continuous testing.

Its model interface explicitly perturbs model inputs and examines black-box model behavior.

Sources:

- Robust Intelligence, *What is Robust Intelligence?*  
  https://docs.rime.dev/en/latest/documentation_home/robust_intelligence_intro.html

- Robust Intelligence, *Fraud Classification Walkthrough*  
  https://docs.rime.dev/en/latest/notebooks/demo_notebooks/RI_Fraud_Classification_Walkthrough.html

This is close enough that GhostData cannot simply be:

> "Generate perturbations and see whether the model is robust."

That would be weak differentiation.

---

# 3. The remaining wedge

The opportunity is a different optimization problem.

## Traditional validation

```text
known check
   |
   v
does current data violate it?
```

## Traditional stress testing

```text
library of stress tests / perturbations
   |
   v
which weaknesses does the model have?
```

## GhostData

```text
all safeguards the company already has
           |
           v
find a plausible failure that survives them
           |
           v
maximize downstream model damage
```

Formally:

```text
maximize    downstream_model_damage(T)

subject to  every_existing_check(T(data)) == PASS
            pipeline_invariants(T(data)) == VALID
            plausibility(T) >= threshold
```

This is **counterexample discovery under customer-specific safeguards**.

The key inversion is:

> Existing testing asks, **"Does this candidate fail my safeguards?"**

GhostData asks:

> **"What candidate can defeat my safeguards?"**

That is the product.

---

# 4. The unit of search is a pipeline failure, not random noise

The original GhostData idea used broad "data worlds."

The revised version should be more concrete:

> **A candidate world is a parameterized, executable production failure mechanism.**

Do not lead with:

- Gaussian noise,
- arbitrary label flipping,
- random row mutation,
- generic distribution drift.

Lead with failure classes that map to public production-ML problems.

---

# 5. Production Failure Library

The MVP should have a small deterministic library of parameterized failure operators.

An LLM may select and parameterize operators from context, but it should not invent arbitrary Python as the main path.

## 5.1 Temporal feature misalignment

Real-world motivation:

- point-in-time correctness,
- stale features,
- delayed data,
- training-serving mismatch.

Example:

```text
correct:
customer feature at prediction time t

failure:
customer feature from t - Δt
```

Parameters:

```yaml
type: temporal_misalignment

feature: income
affected_segment:
  employment_type: self_employed

fraction: 0.12
lag_hours: 48
```

Possible preserved checks:

- schema,
- row count,
- null rate,
- feature range,
- approximate univariate distribution.

Potential model effect:

- destroys the relationship between current debt state and current income,
- while the values remain individually plausible.

---

## 5.2 Entity alignment / join corruption

Real-world motivation:

- erroneous joins,
- entity matching,
- provider-integration changes.

Example:

```text
customer 103 -> valid income belonging to customer 204
customer 204 -> valid income belonging to customer 391
...
```

The important property:

> **The values remain valid; ownership / semantics become wrong.**

A permutation inside a subgroup can exactly preserve:

- income histogram,
- mean,
- variance,
- missingness,
- range,
- category frequencies.

Parameters:

```yaml
type: entity_alignment

feature: income
segment:
  employment_type: self_employed

mismatch_fraction: 0.25

preserve:
  - marginal_distribution
  - null_rate
  - schema
```

This is an excellent hackathon Ghost because it creates a visually obvious gap between:

```text
what data tests see
```

and

```text
what the model depends on
```

---

## 5.3 Conditional sentinel/default value injection

Real-world motivation:

- Google feature pinned to `-1`,
- dummy values,
- fallback behavior.

Do not simply replace an entire feature.

Search for subtle conditions:

```yaml
type: conditional_sentinel

feature: transaction_velocity
segment:
  region: north

fraction: 0.08
sentinel: -1
```

The search should prefer candidates that remain below configured alert thresholds.

---

## 5.4 Controlled duplication / dedup failure

Real-world motivation:

- LinkedIn Data Sentinel duplicate examples,
- bias from duplicated records.

Parameters:

```yaml
type: duplication

segment:
  acquisition_channel: affiliate

fraction: 0.07

mode: duplicate_rows
```

Checks may include:

- total-row-count tolerance,
- duplicate thresholds,
- class / category share tolerances.

---

## 5.5 Conditional semantic remapping

Real-world motivation:

- changed field meaning,
- taxonomy changes,
- categorical mapping changes.

Example:

```text
employment_code 4 used to mean "self-employed"
upstream version changes what code 4 means
```

The values are syntactically valid but semantically different.

---

## 5.6 Aggregation-window / boundary shift

Real-world motivation:

- feature logic changes,
- time-window differences.

Example:

```text
30-day spend
```

accidentally becomes:

```text
28-day spend
```

or uses a changed boundary.

Search parameters:

- window size,
- affected segment,
- affected time region.

---

# 6. AI versus deterministic system

## AI may do

- read schema and feature descriptions,
- inspect model importance,
- inspect existing safeguards,
- choose promising failure classes,
- propose parameter ranges,
- explain a discovered Ghost,
- suggest a new safeguard.

## Deterministic code must do

- apply the transformation,
- enforce hard invariants,
- execute existing checks,
- evaluate the frozen model,
- compute model damage,
- rank candidates,
- save reproducible artifacts.

The LLM is **not allowed to declare success**.

A Ghost exists only if the measurements prove:

```text
existing checks PASS
+
model performance materially worse
```

---

# 7. Inputs

For the hackathon MVP require four inputs.

## A. Reference / replay dataset

A labelled tabular dataset:

```text
entity_id
feature_1
feature_2
...
label
```

Labels are needed because GhostData is a pre-deployment / replay testing tool, not a label-free live drift monitor.

---

## B. Frozen downstream model

MVP:

```python
pred = model.predict_proba(X)
```

Support:

- scikit-learn,
- XGBoost,
- LightGBM if easy.

Primary metric:

- ROC AUC for binary classification.

---

## C. Existing safeguards

Use a simple adapter interface:

```python
result = checks.run(df)
```

MVP built-ins:

- schema unchanged,
- missing rate below threshold,
- range checks,
- feature-mean tolerance,
- PSI / distribution tolerance,
- category-share tolerance.

Important:

The architecture should allow customers to bring arbitrary custom validation code later.

GhostData becomes stronger when the customer already has more checks.

---

## D. Pipeline context

A small structured manifest:

```yaml
features:
  income:
    source: payroll_provider
    entity_key: customer_id
    timestamped: true

  debt_ratio:
    source: credit_bureau
    entity_key: customer_id
    timestamped: true

segments:
  - employment_type
  - region
```

This makes candidate failures more semantically grounded.

For the hackathon, this can be a static YAML file.

---

# 8. Outputs

A winning Ghost should be stored as:

```text
ghost_0183/
├── ghost.yaml
├── transform.py
├── ghost_dataset.parquet
├── existing_checks.json
├── model_report.json
├── explanation.md
└── regression_test.py
```

Example `ghost.yaml`:

```yaml
id: ghost_0183

failure_class: entity_alignment

affected_feature:
  income

affected_segment:
  employment_type: self_employed

mismatch_fraction:
  0.23

existing_checks:
  passed: 27
  failed: 0

model:
  metric: roc_auc
  baseline: 0.864
  ghost: 0.731

damage:
  -0.133
```

This is important:

> **GhostData output is an engineering asset, not a report.**

---

# 9. Search strategy

Do not build a fully general optimizer during the hackathon.

Use a robust hybrid.

## Round 0 — baseline

Measure:

```text
baseline metric
baseline data profile
existing-check status
```

Require all baseline checks to pass.

---

## Round 1 — broad failure-family search

Launch candidate parameterizations across:

- temporal misalignment,
- entity alignment,
- conditional sentinel values,
- controlled duplication,
- semantic remapping,
- aggregation-window shift.

Example:

```text
24 total candidates
6 active concurrently
```

Each candidate runs independently.

---

## Candidate evaluation

```python
def score(candidate):
    ghost_df = candidate.apply(reference_df)

    invariant_result = validate_invariants(ghost_df)
    if not invariant_result.pass_:
        return REJECT

    existing_result = customer_checks(ghost_df)
    if not existing_result.pass_:
        return REJECT

    metric = evaluate_model(model, ghost_df)

    damage = baseline_metric - metric

    if damage <= minimum_damage:
        return REJECT

    return damage
```

The core constraint remains:

```text
ALL CUSTOMER CHECKS PASS
```

---

## Round 2 — local refinement

Take top surviving candidates.

Mutate:

- affected fraction,
- segment,
- lag,
- mismatch rate,
- window boundary,
- conditional region.

Example:

```text
top 4 mechanisms
x
4 refinements each
=
16 candidate worlds
```

---

## Round 3 — minimize the failure

Once a damaging Ghost is found, optionally search downward for the **smallest** failure that still causes meaningful damage.

This makes the result more credible.

Instead of:

```text
50% of rows corrupted
```

prefer:

```text
7.8% of one segment
```

if it still causes substantial model damage.

This can become a strong product metric:

> **Minimum failure required to break the model while all safeguards remain green.**

---

# 10. Objective

For AUC:

```text
damage = baseline_auc - ghost_auc
```

Hard constraint:

```text
existing_checks == PASS
```

Optional secondary penalties:

```text
score =
    model_damage
    - λ1 * affected_fraction
    - λ2 * plausibility_penalty
```

This encourages:

- damaging failures,
- smaller interventions,
- more plausible counterexamples.

For the hackathon, hard ranking can simply be:

```text
1. reject every monitor-failing candidate
2. among survivors, maximize AUC drop
```

Do not overengineer.

---

# 11. Where GhostData sits in the ML workflow

The product should **not** be presented as another dashboard users watch every day.

The cleanest insertion point is CI / pre-deployment validation.

```text
engineer changes:

new SQL
new join
new feature
new vendor
new data source
new mapping
new aggregation
       |
       v
pipeline PR / release candidate
       |
       v
existing CI + data contracts
       |
       | PASS
       v
GhostData adversarial search
       |
       +---- no Ghost ----> merge / release
       |
       +---- Ghost --------> block / investigate
```

Primary users:

- ML platform engineers,
- ML infrastructure engineers,
- data platform engineers,
- senior data scientists responsible for production models,
- model-risk / reliability teams.

---

# 12. Hero demo

## Industry scenario

Use a fictional fintech:

```text
ACME CREDIT
```

This company and incident are fictional.

**Do not claim the exact incident happened at a named real company.**

The failure class is grounded in real publicly documented classes of production issues:

- erroneous joins,
- feature skew,
- stale data,
- changed semantics,
- point-in-time errors.

---

## System

The company predicts loan default.

Pipeline:

```text
Application DB ---------\
                         \
Credit Bureau ------------> Feature Pipeline ---> Risk Model v17
                         /
Payroll Provider --------/
```

Reference model:

```text
ROC AUC = 0.86
```

Existing data tests:

```text
schema
missingness
feature ranges
mean / variance
PSI
category shares
row count
```

Everything passes.

---

# 13. The winning demo Ghost

Use **entity alignment corruption inside a subgroup** as the default hero failure.

Example:

A provider / join implementation change causes a fraction of self-employed customers to receive valid income values belonging to other self-employed customers.

Conceptually:

```text
before:

customer A -> income A
customer B -> income B
customer C -> income C

after:

customer A -> income C
customer B -> income A
customer C -> income B
```

Because the operation permutes real values within the segment:

```text
income histogram       unchanged
income mean            unchanged
income variance        unchanged
income missingness     unchanged
income range           unchanged
segment size           unchanged
```

But this changes:

```text
P(default | income, debt_ratio, employment_type)
```

from the model's point of view because the entity-to-feature relationship is corrupted.

This is the demo's conceptual reveal:

> **The values were valid. Their meaning wasn't.**

---

# 14. Demo screen flow

## Screen 1 — Pipeline PR

Do not show a generic "ML dashboard."

Show a concrete engineering workflow.

```text
ACME CREDIT
Feature Pipeline PR #1842

Change:
payroll_provider_v2

Application DB ---------\
                         \
Credit Bureau ------------> Feature Pipeline ---> Risk Model v17
                         /
Payroll Provider --------/


EXISTING DATA CI

Schema                         PASS
Missingness                    PASS
Feature ranges                 PASS
Distribution checks            PASS
Category frequencies           PASS
Custom data contracts          PASS

27 / 27 CHECKS                 PASS


[ RED-TEAM THIS PIPELINE ]
```

Opening line:

> "This pipeline change passes every data test the team already wrote."

---

## Screen 2 — Daytona counterexample search

This is the sponsor / technical wow screen.

Do **not** make it look like an agent chat log.

Make it look like an experiment search.

```text
GHOSTDATA SEARCH
Round 2 / 3

Candidate worlds searched       31
Currently running                6
Rejected by existing tests      18
Model survived                   8
Counterexamples                  5
```

Candidate stream:

```text
W021  temporal join skew       RUNNING
W022  entity misalignment      RUNNING
W023  conditional sentinel     RUNNING
W024  aggregation boundary     RUNNING
W025  semantic remap           RUNNING
W026  duplication              RUNNING
```

Completed candidates:

```text
W017  DATA TEST FAIL              reject
W018  27/27 PASS | AUC -0.01      harmless
W019  27/27 PASS | AUC -0.08      survivor
W020  27/27 PASS | AUC -0.13      survivor
```

Explain:

> "Every candidate is an executable production-failure hypothesis. We run it with the real checks and frozen downstream model in an isolated Daytona sandbox."

---

## Screen 3 — Ghost found

```text
GHOST FOUND

Failure class:
ENTITY / FEATURE MISALIGNMENT

Affected:
income
self-employed applicants
23% of segment


YOUR EXISTING DATA TESTS

Schema                       PASS
Missingness                  PASS
PSI                          PASS
Mean / variance              PASS
Category share               PASS
Custom contracts             PASS

27 / 27                      PASS


DOWNSTREAM MODEL

Reference AUC               0.86
Ghost AUC                   0.73

Damage                      -0.13
```

The exact numbers must come from the real demo run.

Never hard-code a more dramatic AUC drop than the system actually finds.

---

# 15. The reveal visualization

This is the most important explanatory visual.

## Left: what existing checks see

Show reference and Ghost histograms of `income`.

They should overlap almost perfectly.

Display:

```text
PSI                 PASS
mean drift          PASS
missingness         PASS
range               PASS
```

Caption:

> **Same values.**

## Right: what the model depended on

Show a conditional relationship or segment-level model performance.

Examples:

```text
income vs debt_ratio
colored by default
```

or:

```text
AUC on self-employed segment
reference vs Ghost
```

Caption:

> **Different relationships.**

Center line:

# **Same values. Different relationships.**

Alternative line:

# **The values were valid. Their meaning wasn't.**

---

# 16. The demo must not end at the scare

Click:

```text
[ PROMOTE TO REGRESSION TEST ]
```

Generate:

```text
Ghost #0183

Failure:
entity-feature alignment corruption

Reproducer:
ghost_0183.py

Regression fixture:
ghost_0183.parquet
```

Suggested safeguard:

```text
entity / feature alignment invariant

or

conditional relationship check
for income × debt_ratio
within self-employed segment
```

Then show:

```text
Future PR #1927
payroll_provider_v3

27 existing checks         PASS
Ghost #0183                FAIL

MERGE BLOCKED
```

This is the business-value moment.

---

# 17. Product loop

Today, many reliability workflows are:

```text
incident
  |
  v
postmortem
  |
  v
write new check
```

GhostData moves the loop earlier:

```text
search
  |
  v
counterexample
  |
  v
write new check
  |
  v
incident prevented
```

This supports the line:

# **Move the postmortem before the incident.**

---

# 18. Why Daytona matters

Daytona is not merely where Python executes.

GhostData is structurally a search over executable candidate failures.

Each candidate contains:

```text
failure specification
+
transformation code
+
reference / replay data
+
customer safeguards
+
frozen model
+
evaluation
```

Candidate code may be:

- generated dynamically,
- buggy,
- expensive,
- unsafe to run in-process,
- independently disposable.

Therefore:

```text
1 candidate failure = 1 isolated Daytona sandbox
```

and:

```text
1 search round = N independently measurable sandbox runs
```

Daytona enables:

- isolation,
- parallel execution,
- clean failure boundaries,
- disposable environments,
- artifact capture,
- reproducibility.

Without disposable execution, GhostData degrades into:

```text
a few hand-written tests
```

With Daytona, it becomes:

```text
an actual search for counterexamples
```

---

# 19. Daytona implementation strategy

Do not require 24 or 50 simultaneous sandboxes.

The UI can show:

```text
31 worlds searched
6 active now
```

Use controlled concurrency.

Recommended controller:

```text
Search Controller
      |
      +---- queue
      |
      +---- max_concurrency = 4-6
                |
       +--------+--------+--------+
       |        |        |        |
       v        v        v        v
    Daytona  Daytona  Daytona  Daytona
      W1       W2       W3       W4
       |        |        |        |
       +--------+--------+--------+
                |
                v
            Result store
                |
                v
          rank / refine
```

If snapshots / prewarmed environments are available, use them.

The demo should optimize for reliability, not maximal concurrency.

---

# 20. MVP architecture

```text
                              FRONTEND
                                  |
                                  v
                        GHOSTDATA CONTROLLER
                                  |
             +--------------------+--------------------+
             |                    |                    |
             v                    v                    v
      Dataset Profiler      Check Adapter        Model Adapter
             |                    |                    |
             +--------------------+--------------------+
                                  |
                                  v
                       Pipeline Context Loader
                                  |
                                  v
                       Failure Library / Search
                                  |
                                  v
                         Candidate WorldSpec
                                  |
                                  v
                         Deterministic Compiler
                                  |
                                  v
                        DAYTONA WORLD RUNNER
                                  |
                   +--------------+--------------+
                   |              |              |
                   v              v              v
                  W1             W2             W3 ...
                   |              |              |
                   +--------------+--------------+
                                  |
                                  v
                          Result Evaluator
                                  |
                                  v
                           Search Policy
                                  |
                      refine / minimize / stop
                                  |
                                  v
                                GHOST
                                  |
                                  v
                     Regression Test Generator
```

---

# 21. WorldSpec

Use a small structured schema.

Example:

```json
{
  "failure_class": "entity_alignment",
  "target_feature": "income",
  "segment": {
    "employment_type": "self_employed"
  },
  "parameters": {
    "mismatch_fraction": 0.23
  },
  "preserve": [
    "schema",
    "marginal_distribution",
    "missing_rate",
    "category_share"
  ]
}
```

Benefits:

- reproducible,
- safe,
- easy to validate,
- easy to mutate,
- easy to display,
- less fragile than arbitrary LLM-generated Python.

---

# 22. Candidate result contract

Every Daytona world returns a compact JSON object.

```json
{
  "world_id": "W020",
  "failure_class": "entity_alignment",
  "status": "completed",
  "existing_checks": {
    "passed": 27,
    "failed": 0
  },
  "model": {
    "metric": "roc_auc",
    "baseline": 0.864,
    "candidate": 0.731
  },
  "damage": 0.133,
  "affected_fraction": 0.054,
  "artifact_paths": {
    "world_spec": "ghost.yaml",
    "transformation": "transform.py",
    "report": "report.json"
  }
}
```

The controller should never need to inspect a candidate's full dataframe unless the candidate becomes a finalist.

---

# 23. What to build during the hackathon

## P0 — must work

### 1. Reference fixture

- one public credit-risk / fraud-like tabular dataset,
- one trained binary classifier,
- reproducible baseline AUC.

### 2. Data checks

Implement:

- schema,
- null rate,
- range,
- mean tolerance,
- PSI or KS-style distribution check,
- category share.

### 3. Failure operators

At minimum:

- entity alignment,
- temporal / stale-feature proxy,
- conditional sentinel/default value,
- controlled duplication.

If time:

- semantic remapping,
- aggregation-window proxy.

### 4. Daytona runner

Must prove:

```text
candidate -> isolated sandbox -> real checks -> real model metric -> result
```

### 5. Search controller

Must support:

```text
broad candidates
-> reject
-> rank
-> refine
-> best Ghost
```

### 6. Frontend

Three core states:

```text
Pipeline PR
-> Counterexample Search
-> Ghost Found
```

### 7. Regression promotion

Button can generate a concrete test fixture / Python check.

This is part of the core value, not decorative polish.

---

# 24. P1 only if P0 is stable

- LLM chooses failure classes from context.
- natural-language explanation.
- suggested data contract.
- one-click replay against Model v18.
- downloadable Ghost bundle.
- minimization search.
- richer relationship visualization.

---

# 25. Do not build

Do not spend hackathon time on:

- generic time-series support,
- image models,
- LLM evaluation,
- full observability dashboard,
- Datadog integration,
- Snowflake integration,
- Kubernetes,
- generic authentication,
- online live monitoring,
- full DAG ingestion,
- arbitrary user-generated Python as the primary search mode,
- foundation-model corpus training.

---

# 26. Demo acceptance criteria

The demo is successful only if the search genuinely discovers a counterexample.

Target:

```text
baseline AUC                    measured
candidate worlds searched       >= 20
existing-check-safe worlds      >= 2
damaging counterexamples        >= 1
best AUC drop                   >= 0.05 ideally >= 0.10
existing checks on winner       ALL PASS
```

The exact numbers should be real.

A modest real result is stronger than a dramatic hard-coded result.

---

# 27. 90-second demo narrative

## 0–12 s — concrete workflow

Show the feature-pipeline PR.

Say:

> "This credit model depends on data from an application database, a credit bureau, and a payroll provider. A new payroll integration passes all 27 data checks the team already wrote."

Click:

```text
RED-TEAM THIS PIPELINE
```

---

## 12–35 s — search

Show candidate worlds running.

Say:

> "GhostData does not run another fixed checklist. It searches for a plausible production failure that can satisfy the checklist."

Show:

```text
entity misalignment
temporal skew
sentinel fallback
duplication
...
```

Say:

> "Each candidate is executable. It runs with the real safeguards and frozen model in an isolated Daytona environment."

---

## 35–55 s — reveal

Show:

```text
GHOST FOUND

27 / 27 checks       PASS
AUC                   0.86 -> 0.73
```

Say:

> "It found an entity-alignment failure. The income values themselves are valid, so the existing distribution checks remain green. But some values now belong to the wrong customers."

---

## 55–70 s — explain

Show overlapping income histograms.

Then show changed conditional relationship / subgroup performance.

Say:

> "Same values. Different relationships. The values were valid; their meaning wasn't."

---

## 70–84 s — convert to engineering value

Click:

```text
PROMOTE TO REGRESSION TEST
```

Show next PR blocked by Ghost #0183.

Say:

> "Now this exact blind spot becomes a permanent test every future pipeline and model must survive."

---

## 84–90 s — close

> **"Your tests can only catch failures you encoded. GhostData searches for the next one."**

Alternative final line:

> **"GhostData moves the postmortem before the incident."**

---

# 28. Judge Q&A

## Q: Is this just data drift monitoring?

No.

Monitoring observes real incoming data and asks whether predefined statistics changed.

GhostData runs before release / deployment and asks:

> **Can a plausible failure pass the exact checks I already use and still harm the model?**

It is counterexample search, not passive drift detection.

---

## Q: Don't Deepchecks / Great Expectations already solve this?

They solve important pieces.

They provide checks such as:

- schema validation,
- distributions,
- feature drift,
- label relationships,
- weak segments,
- model performance.

GhostData can treat all of those as **constraints**.

If a company already has 27 excellent checks, GhostData's job is:

> **Find check #28.**

---

## Q: Isn't this Robust Intelligence?

Robust Intelligence is the most important adjacent comparison.

Its public product runs large pre-configured stress-test suites and adversarial / transformation tests to discover model weaknesses.

GhostData's proposed wedge is narrower:

> **Customer-safeguard-constrained counterexample search.**

The optimization explicitly searches for candidates that:

1. survive the customer's own existing checks,
2. still degrade the downstream model,
3. become a new customer-specific regression test.

Do not claim no overlap.

The distinction is the search objective and the safeguard-improvement loop.

---

## Q: How do you know a generated failure is realistic?

Do not claim GhostData predicts the future.

Say:

> "We search within an explicit library of production failure mechanisms and hard pipeline constraints."

MVP plausibility comes from:

- real failure classes,
- valid schema,
- parameter bounds,
- segment constraints,
- allowed transformations,
- small affected fractions.

Long term, plausibility can incorporate:

- company incident history,
- lineage,
- historical pipeline changes,
- vendor behavior,
- production logs,
- domain simulators.

---

## Q: What if the company adds a correlation check?

Good.

That check becomes another GhostData constraint.

GhostData reruns the search and looks for another counterexample.

The product should get **harder to beat as the customer's safeguards improve**.

That is intentional.

---

## Q: Why do you need labels?

The MVP is a pre-deployment / replay testing system.

It needs a labelled reference set to measure downstream model degradation.

This is not meant to replace live label-free drift monitoring.

---

## Q: Why Daytona?

Because each candidate is executable, independent, and disposable.

The product needs to run many possible pipeline-failure variants safely and measure each one.

Daytona makes:

```text
one candidate = one isolated experiment
```

practical.

---

# 29. Claims we can make

Supported / defensible:

- Production ML systems can suffer silent data / feature failures.
- Google, Uber, LinkedIn and others have publicly described classes of such failures.
- Data validation, feature stores, drift monitoring and model stress testing already exist and are important.
- Existing safeguards are necessarily finite and encode particular assumptions / checks.
- GhostData proposes to search specifically for counterexamples that survive those safeguards.
- A discovered counterexample can be promoted into a new regression test.

---

# 30. Claims we should NOT make

Do not say:

> "Nobody validates production ML data."

False.

Do not say:

> "Current monitoring only checks marginals."

False; modern products can check correlations, drift, weak segments, model behavior, etc.

Do not say:

> "Nobody has ever done automated stress testing."

False; Robust Intelligence and others do this.

Do not say:

> "Google / Uber / LinkedIn experienced our exact payroll permutation example."

We do not have evidence for that exact incident.

Do not say:

> "GhostData guarantees future failures."

It finds counterexamples under a chosen failure model / search space.

Do not say:

> "Every Ghost is a realistic future event."

Use:

> plausible stress world under explicit constraints.

---

# 31. Long-term product

The real moat is not the initial failure operators.

It is the accumulated closed loop.

## A. Company-specific Ghost library

Every discovered blind spot becomes a permanent regression asset.

```text
Ghost #002  stale location feature
Ghost #014  temporal join skew
Ghost #031  category semantic remap
Ghost #044  entity alignment
...
```

---

## B. Incident-informed failure library

Curate parameterized failure mechanisms from:

- public engineering postmortems,
- internal company incidents,
- data lineage,
- feature-store metadata,
- vendor changes,
- domain-specific failure modes.

This improves priors for the search.

---

## C. Safeguard-aware search policy

As a company's tests become stronger:

```text
more safeguards
      ->
harder search problem
      ->
more informative Ghosts
      ->
better safeguards
```

This is a natural feedback loop.

---

## D. Minimal counterexamples

Search for the smallest affected segment / fraction that still causes material model damage.

This gives teams prioritization:

```text
"Only 4.2% entity misalignment in this segment is enough to reduce AUC by 8 points."
```

---

## E. Pipeline-aware integrations

Future integrations:

- Great Expectations suites,
- dbt tests,
- feature stores,
- data contracts,
- model registries,
- GitHub CI,
- lineage systems.

GhostData should sit **on top of** existing infrastructure, not replace it.

---

# 32. Relationship to IBM Data Scout

IBM Research's Data Scout is a different but strategically related direction.

Its public NeurIPS 2025 description focuses on:

```text
high-level domain intent
       |
       v
taxonomy expansion
       |
       v
diversified web search
       |
       v
candidate seed URLs
       |
       v
filter for relevance / licensing / crawlability
       |
       v
domain corpus
```

IBM reports that selected sources, once crawled, can yield 40%+ relevant pages for the intended topic, compared with less than 1% in general web-scale corpora.

Source:

- IBM Research, *Data Scout*, NeurIPS 2025  
  https://research.ibm.com/events/neurips-2025

Data Scout's question is approximately:

> **Where should I acquire relevant domain data?**

A future GhostData extension could ask:

> **Which apparently valid data-acquisition / mixture recipe can still damage downstream model behavior?**

Conceptually:

```text
Data Scout
find candidate data sources
       |
       v
candidate corpus recipes
       |
       v
GhostData
search source-mix / filtering / dedup / contamination worlds
       |
       v
downstream training / eval
       |
       v
tested corpus recipe
```

This supports a broader thesis:

> **Data engineering should optimize downstream model behavior, not only data quality or relevance.**

However:

**Do not build the Data Scout extension during the Daytona hackathon.**

It introduces:

- crawling,
- corpus building,
- training / fine-tuning,
- expensive evaluation,
- much slower search loops.

Use it only as a future direction in the pitch if useful.

---

# 33. Strategic positioning

Bad positioning:

> "AI monitoring platform."

Bad positioning:

> "Synthetic data generator."

Bad positioning:

> "Automated drift detector."

Bad positioning:

> "LLM agent for data science."

Better:

# **Adversarial CI for ML data pipelines**

Technical:

# **Customer-safeguard-constrained counterexample search for production ML systems**

Outcome-oriented:

# **Find the pipeline failure your current tests would approve and your model would not survive.**

---

# 34. Final product definition

**GhostData is an adversarial CI layer for ML data and feature pipelines.**

Given:

1. a labelled reference / replay dataset,
2. a frozen downstream model,
3. a pipeline context,
4. the safeguards a team already trusts,

GhostData searches a library of plausible, executable production failure mechanisms.

Every candidate is run independently against:

```text
pipeline invariants
+
existing customer checks
+
downstream model
```

Candidates that trip existing checks are uninteresting and discarded.

Candidates that pass the checks but do not harm the model are harmless and discarded.

A candidate becomes a **Ghost** only when:

```text
ALL EXISTING SAFEGUARDS PASS
+
DOWNSTREAM MODEL MATERIALLY DEGRADES
```

The Ghost is then saved as:

- a reproducer,
- a regression dataset,
- an impact report,
- and a new test / contract candidate.

The product loop is:

```text
KNOWN SAFEGUARDS
      |
      v
SEARCH FOR WHAT THEY MISS
      |
      v
COUNTEREXAMPLE
      |
      v
PROMOTE INTO NEW SAFEGUARD
      |
      v
STRONGER CI
```

## Final pitch

> **Your tests can only catch failures you encoded. GhostData searches for a failure that satisfies every test you have. When it finds one, that counterexample becomes your next test.**

---

# 35. Source notes

The industry evidence and competitive positioning in this document were checked against public sources including:

1. Google Cloud — training-serving skew and the feature-pinned-to-`-1` production example.  
   https://cloud.google.com/blog/topics/developers-practitioners/monitor-models-training-serving-skew-vertex-ai

2. Google — *Rules of Machine Learning*.  
   https://developers.google.com/machine-learning/guides/rules-of-ml/

3. Uber — D3 automated data-drift detection and silent regressions.  
   https://www.uber.com/gb/en/blog/d3-an-automated-system-to-detect-data-drifts/

4. Uber — Unified Data Quality / operational data-quality discussion.  
   https://www.uber.com/en-EE/blog/operational-excellence-data-quality/

5. LinkedIn — Data Sentinel.  
   https://www.linkedin.com/blog/engineering/data-management/data-sentinel-automating-data-validation

6. LinkedIn — Feathr feature store.  
   https://www.linkedin.com/blog/engineering/open-source/open-sourcing-feathr--linkedin-s-feature-store-for-productive-m

7. Great Expectations — Avanade case study.  
   https://greatexpectations.io/case-studies/how-avanade-uses-gx-to-detect-data-drift-from-upstream-model-changes-in/

8. Great Expectations — distribution validation.  
   https://docs.greatexpectations.io/docs/reference/learn/data_quality_use_cases/distribution/

9. Deepchecks — CI/CD and model/data checks.  
   https://docs.deepchecks.com/stable/general/usage/ci_cd.html

10. Robust Intelligence / RIME — Stress Testing.  
    https://docs.rime.dev/en/latest/documentation_home/robust_intelligence_intro.html

11. IBM Research — Data Scout, NeurIPS 2025.  
    https://research.ibm.com/events/neurips-2025

These sources support the existence of the underlying failure classes and existing industry tooling.

They do **not** establish that the exact fictional ACME CREDIT hero incident occurred at any named real company.

That distinction should be preserved in the hackathon pitch.
