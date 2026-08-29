# GhostData — Hackathon Short Spec

## 1. One-line product

**GhostData is adversarial CI for ML data pipelines.**

A company already has:

- a labelled reference / replay dataset,
- a frozen downstream model,
- an existing data / feature pipeline,
- data checks and contracts it already trusts.

GhostData asks:

> **Can we find a plausible pipeline failure that passes every existing check, but still breaks the downstream model?**

If yes, that failure is a **Ghost**.

A Ghost becomes:

- a reproducible transformation,
- a failing dataset,
- a model-impact report,
- and a new regression test / data contract.

### Best pitch line

> **Your tests can only catch failures you encoded. GhostData searches for the next one.**

Alternative:

> **Move the postmortem before the incident.**

---

# 2. Why this is a real industry problem

Production ML can fail without the pipeline crashing.

Public examples include:

- **Google:** a serving bug pinned a feature to `-1`; predictions continued, but model accuracy degraded.
- **Uber:** publicly discusses silent data regressions caused by schema changes, ETL changes, erroneous joins, partial data and changed field meaning.
- **LinkedIn:** has caught duplicate / corrupted datasets and built feature infrastructure around training-serving consistency and point-in-time correctness.
- **Avanade:** a top ML feature silently became zero because of an upstream warehouse problem and was noticed only coincidentally.

So the real problem is:

> **Technically valid data does not necessarily preserve the semantics a model depends on.**

---

# 3. What GhostData is NOT

Do not pitch GhostData as:

- another drift dashboard,
- another Great Expectations,
- another Deepchecks,
- generic synthetic-data generation,
- random noise injection,
- generic adversarial examples,
- an LLM inventing scary failure stories.

Existing tools already do:

- schema / null / range validation,
- distribution and drift checks,
- feature-label and segment checks,
- model monitoring,
- stress testing.

GhostData's wedge is narrower:

> **Search specifically for a counterexample that survives the customer's own safeguards.**

---

# 4. Core optimization

Traditional validation asks:

```text
Does this dataset violate my tests?
```

GhostData asks:

```text
What plausible dataset / pipeline failure
can pass all my tests
and still damage my model?
```

Conceptually:

```text
maximize    model damage

subject to  all existing checks PASS
            pipeline invariants remain valid
            failure is plausible
```

A candidate is interesting only if:

```text
DATA CHECKS     PASS
MODEL           FAIL
```

---

# 5. Search real pipeline failures, not random noise

MVP failure library:

### 1. Entity / join misalignment
Valid feature values become attached to the wrong entity.

Example:

```text
customer A -> income C
customer B -> income A
customer C -> income B
```

This can preserve:

- histogram,
- mean,
- variance,
- null rate,
- range,

while breaking relationships the model relies on.

### 2. Temporal / stale-feature skew

Use:

```text
feature(t - Δt)
```

instead of:

```text
feature(t)
```

for a subset of rows.

### 3. Conditional sentinel / fallback values

A small segment gets a default such as `-1`, but not enough to trigger existing thresholds.

### 4. Controlled duplication

Duplicate a targeted subgroup while staying inside current row-count / duplicate tolerances.

Optional later:

- semantic remapping,
- aggregation-window shift,
- taxonomy changes.

---

# 6. Hero demo

## Scenario

Fictional fintech:

```text
ACME CREDIT
```

Pipeline:

```text
Application DB ---------\
                         \
Credit Bureau ------------> Feature Pipeline ---> Risk Model v17
                         /
Payroll Provider --------/
```

A PR updates:

```text
payroll_provider_v2
```

Existing CI:

```text
Schema                  PASS
Missingness             PASS
Ranges                  PASS
PSI / distributions     PASS
Category shares         PASS
Custom contracts        PASS

27 / 27                 PASS
```

Button:

```text
[ RED-TEAM THIS PIPELINE ]
```

---

# 7. Daytona search screen

GhostData launches executable failure candidates.

Example UI:

```text
GHOSTDATA SEARCH

Worlds searched              31
Currently running             6
Rejected by data checks      18
Model survived                8
Counterexamples               5

W021  temporal skew          RUNNING
W022  entity misalignment    RUNNING
W023  sentinel fallback      RUNNING
W024  duplication            RUNNING

W017  DATA TEST FAIL            reject
W018  27/27 PASS | AUC -0.01    harmless
W019  27/27 PASS | AUC -0.08    survivor
W020  27/27 PASS | AUC -0.13    survivor
```

Explain:

> **Each candidate is an executable production-failure hypothesis running with the real safeguards and frozen model in an isolated Daytona sandbox.**

---

# 8. Winning Ghost

Default hero failure:

## Entity / feature misalignment

A fraction of self-employed applicants receive valid income values belonging to other applicants.

Because values are permuted inside the same segment:

```text
income histogram       unchanged
mean                   unchanged
variance               unchanged
missingness            unchanged
range                   unchanged
```

But:

```text
income ↔ debt ↔ default
```

relationships break.

Reveal:

```text
GHOST FOUND

Existing checks             27 / 27 PASS

Reference AUC               0.86
Ghost AUC                   0.73
Damage                      -0.13
```

Use **real measured numbers** from the demo.

Never fake a dramatic AUC drop.

---

# 9. Key visual

Left:

## What existing tests see

Reference and Ghost income histograms overlap.

```text
PSI             PASS
mean            PASS
missingness     PASS
range           PASS
```

Right:

## What the model depended on

Show changed:

```text
income × debt_ratio relationship
```

or subgroup model performance.

Center message:

# **Same values. Different relationships.**

Alternative:

# **The values were valid. Their meaning wasn't.**

---

# 10. Do not stop at finding the problem

Button:

```text
[ PROMOTE TO REGRESSION TEST ]
```

Output:

```text
Ghost #0183

Failure:
entity-feature alignment corruption

Reproducer:
ghost_0183.py

Regression fixture:
ghost_0183.parquet
```

Then show:

```text
Future PR #1927

27 existing checks      PASS
Ghost #0183             FAIL

MERGE BLOCKED
```

This is what turns GhostData from a demo into an engineering product.

---

# 11. Where GhostData lives

Not as a dashboard people watch all day.

It sits in CI / pre-deployment:

```text
new SQL / join / vendor / feature / ETL
                  |
                  v
            pipeline PR
                  |
                  v
         existing data CI
                  |
                PASS
                  |
                  v
             GhostData
              /     \
        no Ghost    Ghost
           |          |
         merge       block
```

Primary users:

- ML platform teams,
- data platform teams,
- ML infrastructure engineers,
- senior production data scientists,
- model reliability / risk teams.

---

# 12. Why Daytona matters

GhostData is a **search problem**, not one inference call.

Each candidate contains:

```text
failure spec
+
transformation code
+
reference data
+
customer checks
+
frozen model
```

Candidate code is:

- independent,
- disposable,
- potentially buggy,
- naturally parallel.

Therefore:

```text
1 candidate failure = 1 Daytona sandbox
```

Use controlled concurrency:

```text
31 worlds searched
4–6 active at once
```

Do not depend on dozens of simultaneous environments.

---

# 13. MVP architecture

```text
Frontend
   |
GhostData Controller
   |
   +-- Dataset Profiler
   +-- Check Adapter
   +-- Model Adapter
   +-- Pipeline Context
   |
Failure Library
   |
WorldSpec
   |
Daytona Runner
   |
Checks + Model Eval
   |
Rank / Refine
   |
Ghost
   |
Regression Test Generator
```

---

# 14. WorldSpec

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
  }
}
```

Prefer structured failure specs over arbitrary LLM-written Python.

AI can choose / parameterize failure classes.

Deterministic code decides whether a Ghost actually exists.

---

# 15. What to build

## P0

1. Public tabular credit / fraud dataset.
2. Binary classifier with reproducible baseline AUC.
3. Checks:
   - schema,
   - null rate,
   - range,
   - mean tolerance,
   - PSI / distribution,
   - category share.
4. Four failure operators:
   - entity alignment,
   - stale feature,
   - conditional sentinel,
   - duplication.
5. Daytona candidate runner.
6. Search:
   - broad candidates,
   - reject,
   - rank,
   - refine,
   - winner.
7. Three frontend states:
   - Pipeline PR,
   - Search,
   - Ghost Found.
8. Promote winner to regression test.

## Only if P0 works

- LLM failure selection,
- automatic explanation,
- suggested contract,
- prettier relationship plots.

## Do not build

- full observability platform,
- live monitoring,
- Snowflake / Datadog integrations,
- auth,
- time series,
- vision,
- LLM eval,
- IBM Data Scout integration.

---

# 16. 90-second demo

### 0–12 s

> "This pipeline change passes every data test the team already wrote."

Show:

```text
27 / 27 PASS
```

Click:

```text
RED-TEAM THIS PIPELINE
```

### 12–35 s

> "GhostData does not run another fixed checklist. It searches for a plausible production failure that can satisfy the checklist."

Show Daytona worlds running.

### 35–55 s

Reveal:

```text
GHOST FOUND

27 / 27 checks      PASS
AUC                 0.86 -> 0.73
```

### 55–70 s

Show overlapping histograms and broken relationship.

> **"Same values. Different relationships."**

### 70–84 s

Click:

```text
PROMOTE TO REGRESSION TEST
```

Show future PR blocked.

### 84–90 s

> **"Your tests can only catch failures you encoded. GhostData searches for the next one."**

---

# 17. Judge Q&A

### Isn't this just drift monitoring?

No.

Monitoring observes real production data and checks predefined statistics.

GhostData actively searches for a plausible counterexample that passes those exact checks.

### Isn't this Great Expectations / Deepchecks?

They provide excellent checks.

GhostData treats those checks as constraints.

> **If the company has 27 checks, GhostData tries to find check #28.**

### Isn't this Robust Intelligence?

It is adjacent.

Robust Intelligence already does model stress testing and adversarial transformations.

GhostData's proposed wedge is:

> **customer-safeguard-constrained counterexample search**

and then converting the counterexample into a new customer-specific regression test.

Do not claim there is zero overlap.

### How do you know the failure is realistic?

We do not claim to predict the future.

We search within a parameterized library of known production failure classes and explicit pipeline constraints.

### Why labels?

This MVP is pre-deployment / replay testing, not label-free live drift detection.

### Why Daytona?

Because each candidate is executable, isolated, disposable and independently measurable.

---

# 18. Claims to avoid

Do **not** say:

- nobody validates ML data,
- monitoring only checks marginals,
- nobody does stress testing,
- Google/Uber/LinkedIn had the exact fictional ACME incident,
- GhostData predicts future failures,
- every generated Ghost is realistic.

Say:

> **GhostData searches plausible counterexamples under explicit constraints.**

---

# 19. Long-term direction

Moat:

```text
company-specific Ghost library
+
incident-informed failure library
+
customer safeguards
+
learned search priors
```

Over time:

```text
more tests
   ->
harder counterexamples
   ->
better Ghosts
   ->
stronger tests
```

Future integrations could include:

- Great Expectations,
- dbt tests,
- feature stores,
- GitHub CI,
- model registries,
- lineage systems.

---

# 20. IBM Data Scout

IBM Data Scout is strategically related but **not part of the hackathon MVP**.

Data Scout roughly asks:

> **Where should I acquire relevant domain data?**

A future GhostData extension could ask:

> **Which apparently valid corpus / data recipe can still harm downstream model behaviour?**

Long-term thesis:

> **Data engineering should optimize downstream model behaviour, not only data quality or relevance.**

But for Daytona:

**stay focused on ML data-pipeline counterexamples.**

---

# Final definition

## GhostData

**Adversarial CI for ML data pipelines.**

Given:

```text
reference data
+
frozen model
+
pipeline context
+
existing safeguards
```

GhostData searches for:

```text
plausible executable failure
+
all safeguards PASS
+
downstream model FAIL
```

and turns the result into:

```text
reproducer
+
regression fixture
+
new safeguard
```

### Final line

> **Your tests can only catch failures you encoded. GhostData searches for the next one.**
