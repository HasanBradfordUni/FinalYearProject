# AI-Powered Placement Prediction System for Children's Social Care

**A Comprehensive ML-Enabled Flask Web Application for Placement Stability Prediction**

[![University of Bradford](https://img.shields.io/badge/University-Bradford-blue)](https://www.bradford.ac.uk/)
[![BSc Computer Science](https://img.shields.io/badge/Degree-BSc%20Computer%20Science-green)](https://www.bradford.ac.uk/courses/ug/computer-science-bsc/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-yellow)](#)
[![License](https://img.shields.io/badge/License-Private-red)](#)

**GitHub Repository**: [HasanBradfordUni/FinalYearProject](https://github.com/HasanBradfordUni/FinalYearProject)

---

## Overview

This final year project implements an **AI-powered system** for predicting placement stability in children's social care, specifically designed for **BCFT (Bradford Children & Families Trust)**. The system leverages machine learning models with Flask web framework to provide data-driven insights for social workers, placement officers, and administrators.

### Key Goals

✅ Predict optimal placement types (Kinship, External Fostering, In-House Fostering)  
✅ Estimate placement duration using regression models  
✅ Identify breakdown risk factors and trends  
✅ Enable data-driven decision-making in social care placement  
✅ Provide role-based access with comprehensive audit trails  
✅ Support model retraining with CSV field mapping  

---

## Headline Features

### 🤖 Machine Learning Models
- **Random Forest Classifier** for placement type prediction
- **Random Forest Regressor** for placement duration estimation
- Comprehensive feature importance analysis
- Confusion matrix and predictive performance visualization

### 📊 Predictive Analytics
- Single placement predictions with explainability summaries
- Multi-option placement comparison tool
- Placement outcome tracking (breakdown flags, duration bands, end reasons)
- Breakdown analysis with risk factors and recommendations
- Stability trends over time

### 👥 Role-Based Access Control (RBAC)
6 distinct user roles with granular permissions:
- **IT & App Support** - System administration, model retraining
- **Data & Performance Team** - Analytics, model evaluation, system configuration
- **Placement Officers** - Placement uploads, predictions, outcome tracking
- **Social Work Teams** - Predictions, breakdown analysis, read-only access
- **Service Managers & Team Managers** - Overview dashboards, management insights
- **Residential Placement Leads** - Placement management and tracking

### 🔐 Security & User Management
- User authentication with force password reset capability
- Password change functionality for all user types
- Temporary password generation with email distribution
- "Remember Me" session management
- Comprehensive audit logging with timestamps and user attribution

### 📁 Model Retraining Workflow
- CSV upload with automatic field detection
- Intelligent field mapping with auto-suggestion
- Per-row random default value generation (ethnicities, age ranges, custom numeric)
- Saved mapping profiles for future reuse
- Critical field validation and missing data handling

### 📈 Comprehensive Analysis
- Breakdown analysis by placement type and duration
- Risk factor identification
- Placement stability metrics
- Duration band categorization (7 bands: <1 month to 4+ years)
- Trend visualization and recommendations

### 🔍 Explainability Engine
- Unified explainability summary paragraph synthesizing:
  - Top feature predictors
  - Profile-to-average deltas
  - Prediction spread across options
  - Placement type preference drivers

---

## Technology Stack

### Backend
- **Python 3.13** - Core language
- **Flask 3.1.1** - Web framework
- **SQLite3** - Relational database (local deployment)
- **Flask-Login** - User authentication
- **Flask-WTF** - Form validation and CSRF protection

### Machine Learning
- **scikit-learn 1.7** - ML algorithms (Random Forest, Linear Regression)
- **pandas 2.3** - Data manipulation and preprocessing
- **numpy 2.3** - Numerical computing
- **joblib 1.4** - Model artifact serialization

### Data Visualization
- **matplotlib** - Static plotting
- **seaborn** - Statistical visualizations
- **BCFT Color Palette** - Branded styling (#07375f, #ce0f69, #ffdd00, #a2c7e2)

### Testing & Quality
- **pytest 8.3** - Unit testing framework (around 60 test cases)
- **pytest-mock** - Test mocking and monkeypatching

### Frontend (Template Layer)
- **Jinja2** - Flask templating
- **HTML5** - Semantic markup
- **CSS3** - Styling with BCFT branding
- **Bootstrap** - Responsive layout
- **JavaScript** - Client-side interactivity

---

## Project Structure

```
new/
├── src/
│   ├── app/
│   │   ├── __init__.py                      # Flask app factory
│   │   ├── routes.py                        # All route handlers (1184 lines)
│   │   ├── models.py                        # Database & ML operations
│   │   ├── forms.py                         # WTForms validation
│   │   ├── utils.py                         # Helper functions
│   │   ├── permissions.py                   # RBAC and role mapping
│   │   ├── train_models.py                  # Model training pipeline (412 lines)
│   │   ├── test_*.py                        # Unit tests (70+ tests)
│   │   ├── static/
│   │   │   ├── models/                      # ML artifacts (*.pkl)
│   │   │   ├── dataset.csv                  # Training dataset
│   │   │   ├── uploads/retraining/          # CSV uploads for retraining
│   │   │   ├── visuals/                     # Generated visualizations
│   │   │   └── placements.db                # SQLite database
│   │   └── templates/                       # Jinja2 templates
│   ├── generate_model_visuals.py            # Visualization generation
│   ├── conftest.py                          # Pytest configuration
│   └── requirements.txt                     # Python dependencies
├── Prototype/                               # Early prototype (legacy)
├── Copilot Documentation/                   # Development progress notes
├── README.md                                # This file
└── .git/                                    # Version control

```

---

## Installation & Setup

### Prerequisites
- Python 3.13+
- pip package manager
- Git

### Quick Start

```bash
# Clone repository
git clone https://github.com/HasanBradfordUni/FinalYearProject.git
cd FinalYearProject

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
cd src
pip install -r requirements.txt

# Initialize database
python app/create_admin.py

# Run application
python run.py
```

Access at: `http://localhost:5000`

### Run Tests

```bash
pytest -v --tb=short
```

---

## Running the Application

### Development
```bash
cd src
python run.py
```

### Production
```bash
cd src
gunicorn -w 4 -b 0.0.0.0:8000 run:app
```

---

## Key Features in Action

### 1. Make a Placement Prediction
1. Navigate to "Predict Placement"
2. Enter child profile (optional numeric fields auto-fill)
3. View ranked placement options with explainability summary

### 2. Compare Placement Options
1. Navigate to "Compare Placements"
2. Select 2+ placement types
3. View comparative metrics and unified explainability

### 3. Track Placement Outcomes
1. Navigate to "Update Placement Outcome"
2. Select placement record
3. Set end reason and days lasted
4. System auto-flags breakdowns

### 4. Analyze Breakdown Patterns
1. Navigate to "Breakdown Analysis"
2. View breakdown rates, risk factors, duration correlations
3. Export analysis as CSV (admin only)

### 5. Retrain Models
1. Navigate to "Model Retraining"
2. Upload CSV with new placement data
3. Map fields and configure default strategies
4. Retrain models with new data

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/login` | GET, POST | User authentication |
| `/logout` | GET | User logout |
| `/dashboard` | GET | Role-appropriate dashboard |
| `/predict` | GET, POST | Single placement prediction |
| `/compare` | GET, POST | Multi-placement comparison |
| `/upload-placement` | GET, POST | Upload placement record |
| `/breakdown-analysis` | GET | Breakdown pattern analysis |
| `/stability-trends` | GET | Stability trend visualization |
| `/placement-outcome` | GET | Track placement outcomes |
| `/model-retraining` | GET, POST | Model retraining workflow |
| `/users` | GET | User management |
| `/audit-logs` | GET | Audit log viewing |
| `/export/prediction/<id>.csv` | GET | Export prediction |

---

## Testing

- **Total Tests**: 70+
- **Coverage**: Routes, models, forms, ML pipeline, RBAC
- **Status**: All tests passing ✅

```bash
pytest -v
pytest --cov=app --cov-report=html
```

---

## Development Progress

### ✅ Completed
- [x] Core Flask application with SQLite
- [x] ML models (Random Forest, Linear Regression)
- [x] User authentication & password management
- [x] 6-role RBAC with permission matrix
- [x] Placement upload & prediction pipeline
- [x] Breakdown analysis & stability trends
- [x] CSV field mapping & model retraining workflow
- [x] Unified explainability engine
- [x] 70+ comprehensive unit tests
- [x] Audit logging system

### 📋 Dataset
- **Child Features**: Age, Gender, Ethnicity, Prior Placements, Returning Child, Missing Episodes, Sibling Group Size, Placed With Siblings
- **Carer Features**: Age, Gender, Ethnicity
- **Placement Features**: Type (21.22% Kinship, 33.64% External Fostering, 45.14% In-House Fostering), Sequence Number, Involvement Flags, Duration, Outcome, Breakdown Status

---

## Author

**Hasan Akhtar**  
BSc Computer Science (Final Year)  
University of Bradford, 2025-2026

**GitHub**: [HasanBradfordUni/FinalYearProject](https://github.com/HasanBradfordUni/FinalYearProject)

---

## License

Private. Redistribution without permission is prohibited.

**Last Updated**: April 29, 2026  
**Status**: Active Development  
**Python**: 3.13+  
**Version**: 2.1.0  

## Conclusion

This project demonstrates how machine learning and role-aware web engineering can be combined to support more informed placement decisions in children's social care. By integrating prediction, explainability, outcome tracking, and retraining workflows into one platform, the system provides a practical foundation for data-driven practice at BCFT.  
As development continues, the architecture is designed to support further enhancements in analytics, scalability, and real-world deployment readiness.