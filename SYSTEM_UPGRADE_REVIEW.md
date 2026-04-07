# BCFT Placement Stability System - Upgrade Review and Implementation Steps

Date: 2026-04-07
Scope: `src/` only

## 1) What was reviewed

The upgrade review covered:

- `src/app/routes.py` (endpoints, role permissions, prediction/comparison flows)
- `src/app/forms.py` (schema and validation)
- `src/app/utils.py` (input preparation, prediction ranking, bulk upload)
- `src/app/models.py` (DB schema, persistence, auditability, analysis)
- `src/app/train_models.py` (preprocessing and model retraining lifecycle)
- `src/templates/*.html` for prediction/comparison/analysis/admin pages

---

## 2) Requirements traceability - before vs after

Legend:
- Implemented = delivered in code
- Partial = some support exists, further enhancement remains
- Deferred = intentionally left for future iteration/time constraints

### Functional requirements (FR)

| ID | Requirement | Before | After | Notes |
|---|---|---|---|---|
| FR01 | CSV upload with BCFT schema | Partial | Implemented | Bulk upload now validates BCFT-aligned schema and maps all key fields. |
| FR02 | Predict breakdown likelihood | Partial | Implemented | Added `rf_breakdown_classifier.pkl` support and breakdown likelihood output. |
| FR03 | Rank options by stability + expected duration | Partial | Implemented | Ranking now uses breakdown risk + stability + duration ordering. |
| FR04 | Compare up to 4 placement options | Missing | Implemented | Compare route/form/template now supports 2-4 options. |
| FR05 | Explain key influencing variables | Partial | Implemented | Prediction payload now includes top explanation factors per option. |
| FR06 | Role-based access control | Implemented | Implemented | Existing RBAC retained and used for new routes. |
| FR07 | Refresh with new uploads / retraining | Missing | Implemented | Added admin retrain endpoint to trigger model retraining and hot reload. |
| FR08 | Breakdown patterns by duration bands | Partial | Implemented | Added duration band analytics (`<1 year`, `1-3 years`, `3+ years`). |
| FR09 | Export summaries/reports (CSV/PDF) | Missing | Partial | CSV exports added for predictions, comparisons, breakdown analysis. PDF deferred. |
| FR10 | Admin config/users/categories/settings | Partial | Partial | Users/settings retained; retraining added. Placement-category management UI still basic. |
| FR11 | Handle missing key attributes | Missing | Implemented | Prediction/comparison forms now allow optional input; defaults/imputation applied. |
| FR12 | Refined BCFT placement categories | Implemented | Implemented | Current categories preserved as Kinship/External/In-House Fostering. |
| FR13 | Separate fostering vs residential models | Missing | Deferred | Documented as future work (time-constrained). |
| FR14 | Contextualised matching factors | Partial | Partial | Some contextual factors are supported; advanced skill/disability matching deferred. |
| FR15 | Suggest suitable carer characteristics | Missing | Deferred | Not yet implemented in this iteration. |
| FR16 | High-risk placement identification | Partial | Partial | Breakdown likelihood + risk tables implemented; deeper risk module deferred. |
| FR17 | Use breakdown reasons in predictive modelling | Missing | Partial | Move reason is captured in schema; model augmentation still pending. |
| FR18 | Multi-factor pattern detection | Missing | Partial | Duration-band and risk analysis improved; advanced interaction mining deferred. |

### Non-functional requirements (NFR)

| ID | Requirement | Status after upgrade | Notes |
|---|---|---|---|
| NFR1 | 99% business-hours availability | Partial | Depends on deployment/hosting redundancy and monitoring outside app code. |
| NFR2 | Intuitive/accessibile interface | Partial | Existing UI retained with minimal workflow changes and added export/retrain actions. |
| NFR3 | <=10s processing per case | Partial | Inference remains lightweight; production performance testing still required. |
| NFR4 | >=10 concurrent users | Partial | Requires WSGI deployment config and load testing. |
| NFR5 | Secure username/password auth | Implemented | Password hashing + Flask-Login retained. |
| NFR6 | GDPR/safeguarding compliance | Partial | Audit logs and controlled access exist; full compliance process is operational/governance work. |

---

## 3) Main code changes completed

### A) Data/model pipeline

- Updated `src/app/train_models.py` to train and save:
  - `lr_regressor.pkl`
  - `rf_regressor.pkl`
  - `rf_classifier.pkl` (placement suitability)
  - `rf_breakdown_classifier.pkl` (breakdown likelihood)
- Regenerated explainability artifacts:
  - `rf_feature_importance.csv`
  - `rf_breakdown_feature_importance.csv`
  - `lr_coefficients.csv`
  - `model_metadata.json` with feature sets

### B) Preprocessing and inference

- Updated `src/app/utils.py`:
  - BCFT schema validation for CSV upload
  - imputable input handling for missing form values
  - breakdown likelihood scoring
  - explanation factors per predicted option
  - ranking logic aligned to stakeholder feedback

### C) Persistence and analytics

- Updated `src/app/models.py`:
  - expanded `placements` schema with BCFT fields
  - expanded `predictions` schema with breakdown + payload
  - migration helper for older DBs
  - added `get_breakdown_patterns_by_duration`
  - added getters for export routes

### D) Endpoints and workflow

- Updated `src/app/routes.py`:
  - model asset loader with metadata support
  - compare limit increased to 4 options
  - retrain endpoint: `POST /admin/retrain-models`
  - CSV export endpoints:
    - `/export/prediction/<id>.csv`
    - `/export/comparison/<id>.csv`
    - `/export/breakdown-analysis.csv`

### E) UI updates (visual style preserved)

- Updated templates:
  - `src/templates/results.html`
  - `src/templates/comparison_results.html`
  - `src/templates/compare.html`
  - `src/templates/breakdown_analysis.html`
  - `src/templates/manager_dashboard.html`
  - `src/templates/admin_dashboard.html`
- BCFT colors/assets were not changed.

---

## 4) How to run the upgraded workflow

1. Ensure updated dataset is in:
   - `src/app/static/dataset.csv`
2. Retrain models:
   - run `src/app/train_models.py`
   - or use admin button "Retrain Models From Latest Dataset"
3. Start Flask app via your existing run method.
4. Validate key flows:
   - upload CSV with BCFT schema
   - run prediction with some missing inputs
   - compare 2-4 placement options
   - export CSV outputs
   - check duration-band analysis views

---

## 5) Remaining backlog (recommended next)

1. Add PDF export pipeline (FR09 full completion).
2. Add separate residential model path (FR13).
3. Add carer-characteristic recommendation module (FR15).
4. Add richer multi-factor interaction analysis (FR18).
5. Add load/performance tests for NFR3/NFR4 and uptime monitoring for NFR1.

---

## 6) Notes on validation

- Static code updates have been applied to the files listed above.
- Where runtime validation was not completed in-editor, run the training and app smoke tests locally in your environment to confirm artifact generation and route behavior.

