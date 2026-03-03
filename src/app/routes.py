from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user, login_user, logout_user
from functools import wraps
from .forms import (LoginForm, UserForm, UserEditForm, PlacementUploadForm,
                    BulkUploadForm, PredictionForm, ComparisonForm)
from .models import (create_connection, create_tables, authenticate_user,
                     get_user_by_id, get_all_users, create_user, update_user,
                     delete_user_by_id, add_placement_record, get_placement_by_id,
                     get_recent_placements, get_placement_statistics, save_prediction,
                     save_comparison, analyze_breakdown_patterns, get_stability_trends,
                     identify_risk_factors, calculate_stability_metrics, log_audit,
                     get_recent_audit_logs, get_audit_logs_paginated, get_system_statistics,
                     get_system_settings, update_system_settings, generate_breakdown_recommendations)
from .utils import (prepare_prediction_input, generate_predictions_list,
                    extract_profile_from_form, compare_placement_options,
                    process_bulk_upload)
import os
import joblib

app = Blueprint('app', __name__)
db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'placements.db')
connection = create_connection(db_path)
if connection:
    create_tables(connection)

# Load AI models for placement predictions
try:
    models_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Prototype', 'models')
    lr_model = joblib.load(os.path.join(models_path, "lr_regressor.pkl"))
    rf_model = joblib.load(os.path.join(models_path, "rf_classifier.pkl"))
    feature_encoders = joblib.load(os.path.join(models_path, "feature_encoders.pkl"))
    placement_encoder = joblib.load(os.path.join(models_path, "placement_encoder.pkl"))
except Exception as e:
    print(f"Warning: Could not load AI models: {e}")
    lr_model = rf_model = feature_encoders = placement_encoder = None

# Role-based access control decorator
def role_required(*roles):
    def decorator(f):
        @wraps(f)
        @login_required
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                flash('Please log in to access this page.', 'warning')
                return redirect(url_for('app.login'))
            if current_user.role not in roles:
                flash('You do not have permission to access this page.', 'danger')
                return redirect(url_for('app.dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ============== Authentication Routes ==============

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if current_user.is_authenticated:
        return redirect(url_for('app.dashboard'))

    form = LoginForm()
    if form.validate_on_submit():
        user = authenticate_user(connection, form.username.data, form.password.data)
        if user:
            login_user(user)
            flash(f'Welcome back, {user.username}!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page if next_page else url_for('app.dashboard'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('app.login'))

# ============== Dashboard Routes ==============

@app.route('/')
@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard - role-based view"""
    if current_user.role == 'admin':
        return redirect(url_for('app.admin_dashboard'))
    elif current_user.role == 'manager':
        return redirect(url_for('app.manager_dashboard'))
    else:  # staff role
        return redirect(url_for('app.staff_dashboard'))

@app.route('/staff-dashboard')
@role_required('staff', 'manager', 'admin')
def staff_dashboard():
    """Staff dashboard - view placements and make predictions"""
    recent_placements = get_recent_placements(connection, limit=10)
    stats = get_placement_statistics(connection)
    return render_template('staff_dashboard.html',
                         recent_placements=recent_placements,
                         stats=stats)

@app.route('/manager-dashboard')
@role_required('manager', 'admin')
def manager_dashboard():
    """Manager dashboard - view analytics and breakdown patterns"""
    stats = get_placement_statistics(connection)
    breakdown_patterns = analyze_breakdown_patterns(connection)
    stability_trends = get_stability_trends(connection)

    return render_template('manager_dashboard.html',
                         stats=stats,
                         breakdown_patterns=breakdown_patterns,
                         stability_trends=stability_trends)

@app.route('/admin-dashboard')
@role_required('admin')
def admin_dashboard():
    """Admin dashboard - manage users and system configuration"""
    users = get_all_users(connection)
    system_stats = get_system_statistics(connection)
    audit_logs = get_recent_audit_logs(connection, limit=20)

    return render_template('admin_dashboard.html',
                         users=users,
                         system_stats=system_stats,
                         audit_logs=audit_logs)

# ============== Placement Data Routes ==============

@app.route('/upload-placement', methods=['GET', 'POST'])
@role_required('staff', 'manager', 'admin')
def upload_placement():
    """Upload individual placement record"""
    form = PlacementUploadForm()
    if form.validate_on_submit():
        placement_data = {
            'child_age': form.child_age.data,
            'child_gender': form.child_gender.data,
            'child_ethnicity': form.child_ethnicity.data,
            'carer_age': form.carer_age.data,
            'carer_gender': form.carer_gender.data,
            'carer_ethnicity': form.carer_ethnicity.data,
            'placement_type': form.placement_type.data,
            'uploaded_by': current_user.id
        }

        placement_id = add_placement_record(connection, placement_data)
        log_audit(connection, current_user.id, 'placement_upload', placement_id)

        flash('Placement record uploaded successfully!', 'success')
        return redirect(url_for('app.view_placement', placement_id=placement_id))

    return render_template('upload_placement.html', form=form)

@app.route('/upload-bulk', methods=['GET', 'POST'])
@role_required('staff', 'manager', 'admin')
def upload_bulk():
    """Bulk upload placement data from CSV"""
    form = BulkUploadForm()
    if form.validate_on_submit():
        file = form.csv_file.data
        results = process_bulk_upload(connection, file, current_user.id)

        log_audit(connection, current_user.id, 'bulk_upload',
                 f"{results['success']} records uploaded")

        flash(f"Successfully uploaded {results['success']} records. "
              f"{results['failed']} failed.", 'info')
        return redirect(url_for('app.staff_dashboard'))

    return render_template('bulk_upload.html', form=form)

@app.route('/placement/<int:placement_id>')
@login_required
def view_placement(placement_id):
    """View individual placement details"""
    placement = get_placement_by_id(connection, placement_id)
    if not placement:
        flash('Placement not found.', 'warning')
        return redirect(url_for('app.dashboard'))

    return render_template('placement_detail.html', placement=placement)

# ============== Prediction Routes ==============

@app.route('/predict', methods=['GET', 'POST'])
@role_required('staff', 'manager', 'admin')
def predict():
    """Generate AI-powered placement stability prediction"""
    form = PredictionForm()
    if form.validate_on_submit():
        input_data = prepare_prediction_input(form, feature_encoders)
        predictions = generate_predictions_list(input_data, rf_model, lr_model, placement_encoder)

        # Save prediction to database
        prediction_id = save_prediction(connection, form.data, predictions, current_user.id)
        log_audit(connection, current_user.id, 'prediction_generated', prediction_id)

        return render_template('results.html',
                             child_age=form.child_age.data,
                             child_gender=form.child_gender.data,
                             child_ethnicity=form.child_ethnicity.data,
                             carer_age=form.carer_age.data,
                             carer_gender=form.carer_gender.data,
                             carer_ethnicity=form.carer_ethnicity.data,
                             predictions=predictions)

    return render_template('index.html', form=form)

@app.route('/compare', methods=['GET', 'POST'])
@role_required('staff', 'manager', 'admin')
def compare():
    """Compare multiple placement options"""
    form = ComparisonForm()
    if form.validate_on_submit():
        selected_types = form.placement_types.data
        profile_data = extract_profile_from_form(form)

        # Generate predictions for each selected placement type
        comparisons = compare_placement_options(
            profile_data, selected_types,
            rf_model, lr_model,
            feature_encoders, placement_encoder
        )

        # Save comparison to database
        comparison_id = save_comparison(connection, profile_data, comparisons, current_user.id)
        log_audit(connection, current_user.id, 'comparison_generated', comparison_id)

        return render_template('comparison_results.html',
                             profile=profile_data,
                             predictions=comparisons)

    return render_template('compare.html', form=form)

# ============== Analysis Routes ==============

@app.route('/breakdown-analysis')
@role_required('manager', 'admin')
def breakdown_analysis():
    """Analyze placement breakdown patterns"""
    breakdown_data = analyze_breakdown_patterns(connection)
    risk_factors = identify_risk_factors(connection)
    recommendations = generate_breakdown_recommendations(breakdown_data)

    return render_template('breakdown_analysis.html',
                         breakdown_data=breakdown_data,
                         risk_factors=risk_factors,
                         recommendations=recommendations)

@app.route('/stability-trends')
@role_required('manager', 'admin')
def stability_trends():
    """View placement stability trends over time"""
    trends = get_stability_trends(connection)
    metrics = calculate_stability_metrics(connection)

    return render_template('stability_trends.html',
                         trends=trends,
                         metrics=metrics)

# ============== User Management Routes ==============

@app.route('/users')
@role_required('admin')
def manage_users():
    """User management page"""
    users = get_all_users(connection)
    return render_template('manage_users.html', users=users)

@app.route('/users/add', methods=['GET', 'POST'])
@role_required('admin')
def add_user():
    """Add new user"""
    form = UserForm()
    if form.validate_on_submit():
        user_id = create_user(connection, form.username.data,
                             form.email.data, form.password.data, form.role.data)
        log_audit(connection, current_user.id, 'user_created', user_id)

        flash(f'User {form.username.data} created successfully!', 'success')
        return redirect(url_for('app.manage_users'))

    return render_template('add_user.html', form=form)

@app.route('/users/<int:user_id>/edit', methods=['GET', 'POST'])
@role_required('admin')
def edit_user(user_id):
    """Edit existing user"""
    user = get_user_by_id(connection, user_id)
    if not user:
        flash('User not found.', 'warning')
        return redirect(url_for('app.manage_users'))

    form = UserEditForm(obj=user)
    if form.validate_on_submit():
        update_user(connection, user_id, form.data)
        log_audit(connection, current_user.id, 'user_updated', user_id)

        flash('User updated successfully!', 'success')
        return redirect(url_for('app.manage_users'))

    return render_template('edit_user.html', form=form, user=user)

@app.route('/users/<int:user_id>/delete', methods=['POST'])
@role_required('admin')
def delete_user(user_id):
    """Delete user"""
    if user_id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('app.manage_users'))

    delete_user_by_id(connection, user_id)
    log_audit(connection, current_user.id, 'user_deleted', user_id)

    flash('User deleted successfully!', 'success')
    return redirect(url_for('app.manage_users'))

# ============== System Configuration Routes ==============

@app.route('/settings')
@role_required('admin')
def system_settings():
    """System configuration page"""
    settings = get_system_settings(connection)
    return render_template('settings.html', settings=settings)

@app.route('/settings/update', methods=['POST'])
@role_required('admin')
def update_settings():
    """Update system settings"""
    update_system_settings(connection, request.form)
    log_audit(connection, current_user.id, 'settings_updated', None)

    flash('Settings updated successfully!', 'success')
    return redirect(url_for('app.system_settings'))

@app.route('/audit-logs')
@role_required('admin')
def audit_logs():
    """View audit logs"""
    page = request.args.get('page', 1, type=int)
    logs = get_audit_logs_paginated(connection, page, per_page=50)

    return render_template('audit_logs.html', logs=logs)
