# Setup and Testing Guide

## Initial Setup

### 1. Install Dependencies
```powershell
cd "C:\Users\hakhta26\OneDrive - University of Bradford\Documents\new\src"
pip install -r requirements.txt
```

### 2. Create Initial Admin User
You'll need to create an admin user to access the system. Add this script to create one:

**create_admin.py** (create in src folder):
```python
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from app.models import create_connection, create_tables, create_user

db_path = os.path.join(os.path.dirname(__file__), 'app', 'static', 'placements.db')
connection = create_connection(db_path)
create_tables(connection)

# Create admin user
admin_id = create_user(
    connection=connection,
    username='admin',
    email='admin@bcft.local',
    password='Admin123!',  # Change this!
    role='admin'
)

print(f"Admin user created with ID: {admin_id}")
print("Username: admin")
print("Password: Admin123!")
print("\n⚠️ IMPORTANT: Change the password after first login!")
```

Run it:
```powershell
python create_admin.py
```

### 3. Set Secret Key (Production)
```powershell
$env:SECRET_KEY = "your-very-secret-key-here-change-this"
```

### 4. Run the Application
```powershell
python run.py
```

## Testing Checklist

### Authentication Tests
- [ ] Login with admin credentials
- [ ] Logout functionality
- [ ] Access control (try accessing admin routes as staff)
- [ ] Invalid credentials handling

### Placement Management Tests
- [ ] Upload individual placement record
- [ ] View placement details
- [ ] Bulk upload CSV file

### Prediction Tests
- [ ] Generate prediction for various demographics
- [ ] Compare multiple placement types
- [ ] Verify predictions are saved to database

### User Management Tests (Admin Only)
- [ ] Create new user (staff, manager, admin roles)
- [ ] Edit existing user
- [ ] Delete user
- [ ] Verify audit logs

### Analysis Tests (Manager/Admin)
- [ ] View breakdown analysis
- [ ] View stability trends
- [ ] Check risk factors identification

## Sample Test Data

### Sample CSV for Bulk Upload
Create a file named `sample_placements.csv`:

```csv
Child Age At Placement,Child Gender,Child Ethnicity,Carer Age,Carer Gender Composition,Carer Ethnicity Or Religion,Placement Type
5,Female,White - British,35,Female,White - British,Fostering - Long Term
12,Male,Asian/British Asian - Pakistani,42,Male,Asian/British Asian - Pakistani,Kinship
8,Non binary,Mixed - White/Asian,38,Female,White - British,Fostering - Short Term
15,Male,Black/Black British - African,45,Male,Black/Black British - African,Residential
3,Female,White - British,28,Female,White - British,Adoption
```

### Sample Prediction Input
- Child Age: 7
- Child Gender: Female
- Child Ethnicity: White - British
- Carer Age: 35
- Carer Gender: Female
- Carer Ethnicity: White - British

Expected output: Predictions for all 8 placement types ranked by stability score

## Common Issues & Solutions

### Issue: Models not loading
**Error:** `Warning: Could not load AI models`
**Solution:** Ensure the Prototype/models folder contains:
- lr_regressor.pkl
- rf_classifier.pkl
- feature_encoders.pkl
- placement_encoder.pkl

If missing, run the training script:
```powershell
cd "C:\Users\fifau\OneDrive - University of Bradford\Documents\new\Prototype"
python train_models.py
```

### Issue: Database not found
**Error:** `Connection to SQLite DB successful` not appearing
**Solution:** The database is created automatically. Check that `src/app/static/` directory exists.

### Issue: Template not found
**Error:** `TemplateNotFound`
**Solution:** Templates need to be created in `src/templates/` folder. See ROUTES_UPDATE_SUMMARY.md for list of required templates.

### Issue: Import errors
**Error:** `ImportError: cannot import name 'X'`
**Solution:** Make sure all dependencies are installed:
```powershell
pip install -r requirements.txt
```

### Issue: CSRF token missing
**Error:** `CSRF token missing`
**Solution:** Ensure all forms include `{{ form.csrf_token }}` or use `{{ form.hidden_tag() }}`

## API Endpoints Summary

### Public Routes
- `GET /login` - Login page
- `POST /login` - Login submission

### Staff Routes (Login Required)
- `GET /staff-dashboard` - Staff dashboard
- `GET /upload-placement` - Upload placement form
- `POST /upload-placement` - Submit placement
- `GET /upload-bulk` - Bulk upload form
- `POST /upload-bulk` - Submit CSV file
- `GET /predict` - Prediction form
- `POST /predict` - Generate prediction
- `GET /compare` - Comparison form
- `POST /compare` - Compare placements
- `GET /placement/<id>` - View placement details

### Manager Routes (Manager/Admin Only)
- `GET /manager-dashboard` - Manager dashboard
- `GET /breakdown-analysis` - Breakdown patterns
- `GET /stability-trends` - Stability trends

### Admin Routes (Admin Only)
- `GET /admin-dashboard` - Admin dashboard
- `GET /users` - User management
- `GET /users/add` - Add user form
- `POST /users/add` - Create user
- `GET /users/<id>/edit` - Edit user form
- `POST /users/<id>/edit` - Update user
- `POST /users/<id>/delete` - Delete user
- `GET /settings` - System settings
- `POST /settings/update` - Update settings
- `GET /audit-logs` - View audit logs

## Performance Tips

1. **Database Indexing**: Add indexes for frequently queried columns:
```sql
CREATE INDEX idx_placements_type ON placements(placement_type);
CREATE INDEX idx_placements_uploaded_by ON placements(uploaded_by);
CREATE INDEX idx_audit_logs_user ON audit_logs(user_id);
```

2. **Caching**: Consider implementing Flask-Caching for statistics queries

3. **Pagination**: Large datasets should use pagination (already implemented for audit logs)

## Security Checklist

- [ ] Change default admin password
- [ ] Set strong SECRET_KEY in production
- [ ] Enable HTTPS in production
- [ ] Implement rate limiting for login attempts
- [ ] Regular security audits via audit logs
- [ ] Backup database regularly
- [ ] Validate and sanitize all user inputs
- [ ] Use parameterized SQL queries (already implemented)
- [ ] Implement session timeout
- [ ] Add CSRF protection (already enabled via Flask-WTF)

## Next Development Steps

1. **Create HTML Templates** - Priority: High
2. **Add Data Validation** - Implement custom validators
3. **Improve Error Handling** - User-friendly error messages
4. **Add Email Notifications** - For important events
5. **Implement Data Export** - CSV/Excel export functionality
6. **Add Reporting** - PDF report generation
7. **Dashboard Visualizations** - Charts and graphs
8. **Mobile Responsiveness** - Optimize for mobile devices
9. **Automated Testing** - Unit and integration tests
10. **API Documentation** - OpenAPI/Swagger documentation

## Support & Maintenance

### Log Locations
- Application logs: Check console output
- Audit logs: Database table `audit_logs`
- Error logs: Add file logging in production

### Backup Strategy
```powershell
# Backup database
Copy-Item "src\app\static\placements.db" "backups\placements_$(Get-Date -Format 'yyyyMMdd_HHmmss').db"
```

### Monitoring Queries
```sql
-- Check total users by role
SELECT role, COUNT(*) FROM users GROUP BY role;

-- Check placement statistics
SELECT placement_type, COUNT(*) FROM placements GROUP BY placement_type;

-- Recent predictions
SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10;

-- Active users
SELECT username, last_login FROM users WHERE is_active = 1 ORDER BY last_login DESC;
```

## Troubleshooting Commands

```powershell
# Check Python version (should be 3.7+)
python --version

# Check installed packages
pip list

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check if app can start
python -c "from app import create_app; app = create_app(); print('✓ App created successfully')"

# Test database connection
python -c "from app.models import create_connection, create_tables; import os; conn = create_connection(os.path.join('app', 'static', 'placements.db')); create_tables(conn); print('✓ Database connected')"
```

## Quick Start (Complete Workflow)

```powershell
# 1. Navigate to project
cd "C:\Users\fifau\OneDrive - University of Bradford\Documents\new\src"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create admin user
python create_admin.py

# 4. Run application
python run.py

# 5. Open browser
# Navigate to: http://127.0.0.1:5000

# 6. Login
# Username: admin
# Password: Admin123!

# 7. Start testing!
```

Good luck with your BCFT Placement Stability Prediction System! 🎯

