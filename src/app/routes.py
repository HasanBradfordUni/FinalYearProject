from flask import Blueprint, render_template, request, redirect, url_for, flash, Response, current_app
from flask_login import login_required, current_user, login_user, logout_user
from functools import wraps
from .forms import (LoginForm, UserForm, UserEditForm, PlacementUploadForm,
                    BulkUploadForm, PredictionForm, ComparisonForm,
                    ForgotPasswordForm, ChangePasswordForm, ResetPasswordForm)
from .models import (create_connection, create_tables, authenticate_user,
                     get_user_by_id, get_all_users, create_user, update_user,
                     delete_user_by_id, add_placement_record, get_placement_by_id,
                     get_recent_placements, get_placement_statistics, save_prediction,
                     save_comparison, analyze_breakdown_patterns, get_stability_trends,
                     identify_risk_factors, calculate_stability_metrics, log_audit,
                     get_recent_audit_logs, get_audit_logs_paginated, get_system_statistics,
                     get_system_settings, update_system_settings, generate_breakdown_recommendations,
                     get_prediction_by_id, get_comparison_by_id, get_breakdown_patterns_by_duration,
                     get_user_by_identifier, verify_user_password, update_user_password,
                     issue_temporary_password)
from .utils import (prepare_prediction_input, generate_predictions_list,
                    extract_profile_from_form, compare_placement_options,
                    process_bulk_upload)
import os
import joblib
import json
import csv
import smtplib
import secrets
from io import StringIO
from email.message import EmailMessage
from . import train_models

# Calculate template folder relative to routes.py
template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')

app = Blueprint('app', __name__, template_folder=template_dir)
db_path = os.path.join(os.path.dirname(__file__), 'static', 'placements.db')
connection = create_connection(db_path)
if connection:
    create_tables(connection)


def _generate_temporary_password(length=12):
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _send_temporary_password_email(to_email, username, temporary_password):
    """Attempt to send temp password email; returns True when SMTP send succeeds."""
    smtp_host = current_app.config.get('MAIL_SERVER')
    if not smtp_host:
        return False

    smtp_port = int(current_app.config.get('MAIL_PORT', 587))
    smtp_username = current_app.config.get('MAIL_USERNAME')
    smtp_password = current_app.config.get('MAIL_PASSWORD')
    sender = current_app.config.get('MAIL_DEFAULT_SENDER', smtp_username or 'no-reply@bcft.local')
    use_tls = bool(current_app.config.get('MAIL_USE_TLS', True))

    message = EmailMessage()
    message['Subject'] = 'BCFT Placement System - Temporary Password'
    message['From'] = sender
    message['To'] = to_email
    message.set_content(
        f"Hello {username},\n\n"
        "A temporary password has been generated for your BCFT account.\n"
        f"Temporary Password: {temporary_password}\n\n"
        "Sign in using this temporary password and set a new password immediately.\n"
    )

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as smtp:
            if use_tls:
                smtp.starttls()
            if smtp_username and smtp_password:
                smtp.login(smtp_username, smtp_password)
            smtp.send_message(message)
        return True
    except Exception as exc:
        print(f"Warning: failed to send temporary password email: {exc}")
        return False

def _load_models():
    model_search_paths = [
        os.path.join(os.path.dirname(__file__), 'static', 'models'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'Prototype', 'models'),
    ]
    models_path = next((p for p in model_search_paths if os.path.exists(os.path.join(p, "lr_regressor.pkl"))), None)
    if not models_path:
        raise FileNotFoundError("No trained model artifacts found.")

    metadata = {}
    metadata_path = os.path.join(models_path, "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    return {
        "models_path": models_path,
        "lr_model": joblib.load(os.path.join(models_path, "lr_regressor.pkl")),
        "rf_model": joblib.load(os.path.join(models_path, "rf_classifier.pkl")),
        "rf_reg_model": joblib.load(os.path.join(models_path, "rf_regressor.pkl")) if os.path.exists(os.path.join(models_path, "rf_regressor.pkl")) else None,
        "breakdown_model": joblib.load(os.path.join(models_path, "rf_breakdown_classifier.pkl")) if os.path.exists(os.path.join(models_path, "rf_breakdown_classifier.pkl")) else None,
        "feature_encoders": joblib.load(os.path.join(models_path, "feature_encoders.pkl")),
        "placement_encoder": joblib.load(os.path.join(models_path, "placement_encoder.pkl")),
        "metadata": metadata,
    }


try:
    model_assets = _load_models()
except Exception as e:
    print(f"Warning: Could not load AI models: {e}")
    model_assets = {
        "models_path": None,
        "lr_model": None,
        "rf_model": None,
        "rf_reg_model": None,
        "breakdown_model": None,
        "feature_encoders": None,
        "placement_encoder": None,
        "metadata": {},
    }

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


@app.before_app_request
def enforce_temporary_password_reset():
    """Users with temporary passwords can only access password reset until updated."""
    if not current_user.is_authenticated or not getattr(current_user, 'must_reset_password', False):
        return None

    allowed_endpoints = {
        'app.force_password_reset',
        'app.logout',
        'app.login',
        'app.forgot_password',
        'static',
    }
    if request.endpoint in allowed_endpoints:
        return None

    flash('Please set a new password before continuing.', 'warning')
    return redirect(url_for('app.force_password_reset'))

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
            login_user(user, remember=bool(form.remember_me.data))
            flash(f'Welcome back, {user.username}!', 'success')
            if user.must_reset_password:
                flash('You are signed in with a temporary password. Please set a new password.', 'warning')
                return redirect(url_for('app.force_password_reset'))
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


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Generate one-time temporary password and email it to the account holder."""
    if current_user.is_authenticated:
        return redirect(url_for('app.dashboard'))

    form = ForgotPasswordForm()
    if form.validate_on_submit():
        user = get_user_by_identifier(connection, form.identifier.data.strip())
        if not user:
            flash('If an account exists for those details, a temporary password has been generated.', 'info')
            return redirect(url_for('app.login'))

        temporary_password = _generate_temporary_password()
        issue_temporary_password(connection, user['id'], temporary_password)
        email_sent = _send_temporary_password_email(user['email'], user['username'], temporary_password)
        log_audit(connection, user['id'], 'temporary_password_issued', 'forgot_password_flow')

        return render_template(
            'temp_password_result.html',
            username=user['username'],
            temporary_password=temporary_password,
            email_sent=email_sent,
        )

    return render_template('forgot_password.html', form=form)


@app.route('/account/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Allow any authenticated user to update their own password."""
    form = ChangePasswordForm()
    if form.validate_on_submit():
        if not verify_user_password(connection, current_user.id, form.current_password.data):
            flash('Current password is incorrect.', 'danger')
            return render_template('change_password.html', form=form)

        update_user_password(connection, current_user.id, form.new_password.data)
        log_audit(connection, current_user.id, 'password_changed', 'self_service')
        flash('Password updated successfully.', 'success')
        return redirect(url_for('app.dashboard'))

    return render_template('change_password.html', form=form)


@app.route('/reset-password-temp', methods=['GET', 'POST'])
@login_required
def force_password_reset():
    """Enforce new password setup after temporary password login."""
    if not getattr(current_user, 'must_reset_password', False):
        return redirect(url_for('app.dashboard'))

    form = ResetPasswordForm()
    if form.validate_on_submit():
        update_user_password(connection, current_user.id, form.new_password.data)
        log_audit(connection, current_user.id, 'password_reset_completed', 'temporary_password')
        flash('Your password has been updated. You can now use the full system.', 'success')
        return redirect(url_for('app.dashboard'))

    return render_template('reset_password.html', form=form)

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
    duration_band_patterns = get_breakdown_patterns_by_duration(connection)
    stability_trends = get_stability_trends(connection)

    return render_template('manager_dashboard.html',
                         stats=stats,
                         breakdown_patterns=breakdown_patterns,
                         duration_band_patterns=duration_band_patterns,
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
        child_prior_placements = form.child_prior_placements.data or 0
        placement_data = {
            'child_age': form.child_age.data,
            'child_gender': form.child_gender.data,
            'child_ethnicity': form.child_ethnicity.data,
            'child_prior_placements': child_prior_placements,
            'returning_child': int(str(form.returning_child.data).strip().lower() == 'true'),
            'missing_episodes': form.missing_episodes.data or 0,
            'sibling_group_size': form.sibling_group_size.data or 0,
            'placed_with_siblings': int(str(form.placed_with_siblings.data).strip().lower() == 'true'),
            'carer_age': form.carer_age.data,
            'carer_gender': form.carer_gender.data,
            'carer_ethnicity': form.carer_ethnicity.data,
            'eh_involvement': int(str(form.eh_involvement.data).strip().lower() == 'true'),
            'yot_involvement': int(str(form.yot_involvement.data).strip().lower() == 'true'),
            'placement_sequence_number': child_prior_placements + 1,
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
        input_data = prepare_prediction_input(form, model_assets["feature_encoders"])
        predictions = generate_predictions_list(
            input_data,
            model_assets["rf_model"],
            model_assets["rf_reg_model"] or model_assets["lr_model"],
            model_assets["placement_encoder"],
            breakdown_model=model_assets["breakdown_model"],
            placement_feature_names=model_assets["metadata"].get("classification_features"),
            breakdown_feature_names=model_assets["metadata"].get("breakdown_features"),
        )

        # Save prediction to database
        prediction_id = save_prediction(connection, form.data, predictions, current_user.id)
        log_audit(connection, current_user.id, 'prediction_generated', prediction_id)

        return render_template('results.html',
                             prediction_id=prediction_id,
                             child_age=form.child_age.data,
                             child_gender=form.child_gender.data,
                             child_ethnicity=form.child_ethnicity.data,
                             child_prior_placements=form.child_prior_placements.data,
                             returning_child=form.returning_child.data,
                             missing_episodes=form.missing_episodes.data,
                             sibling_group_size=form.sibling_group_size.data,
                             placed_with_siblings=form.placed_with_siblings.data,
                             carer_age=form.carer_age.data,
                             carer_gender=form.carer_gender.data,
                             carer_ethnicity=form.carer_ethnicity.data,
                             eh_involvement=form.eh_involvement.data,
                             yot_involvement=form.yot_involvement.data,
                             predictions=predictions)

    return render_template('index.html', form=form)

@app.route('/compare', methods=['GET', 'POST'])
@role_required('staff', 'manager', 'admin')
def compare():
    """Compare multiple placement options"""
    form = ComparisonForm()
    if form.validate_on_submit():
        selected_types = form.placement_types.data
        if len(selected_types) < 2 or len(selected_types) > 4:
            flash('Please select between 2 and 4 placement types.', 'warning')
            return render_template('compare.html', form=form)

        profile_data = extract_profile_from_form(form)

        # Generate predictions for each selected placement type
        comparisons = compare_placement_options(
            profile_data, selected_types,
            model_assets["rf_model"], model_assets["rf_reg_model"] or model_assets["lr_model"],
            model_assets["feature_encoders"], model_assets["placement_encoder"],
            breakdown_model=model_assets["breakdown_model"],
            placement_feature_names=model_assets["metadata"].get("classification_features"),
            breakdown_feature_names=model_assets["metadata"].get("breakdown_features"),
        )

        # Save comparison to database
        comparison_id = save_comparison(connection, profile_data, comparisons, current_user.id)
        log_audit(connection, current_user.id, 'comparison_generated', comparison_id)

        return render_template('comparison_results.html',
                             comparison_id=comparison_id,
                             profile=profile_data,
                             predictions=comparisons)

    return render_template('compare.html', form=form)

# ============== Analysis Routes ==============

@app.route('/breakdown-analysis')
@role_required('manager', 'admin')
def breakdown_analysis():
    """Analyze placement breakdown patterns"""
    breakdown_data = analyze_breakdown_patterns(connection)
    duration_band_patterns = get_breakdown_patterns_by_duration(connection)
    risk_factors = identify_risk_factors(connection)
    recommendations = generate_breakdown_recommendations(breakdown_data)

    return render_template('breakdown_analysis.html',
                         breakdown_data=breakdown_data,
                         duration_band_patterns=duration_band_patterns,
                         risk_factors=risk_factors,
                         recommendations=recommendations)


@app.route('/admin/retrain-models', methods=['POST'])
@role_required('admin')
def retrain_models():
    """Retrain all models from the latest uploaded dataset and reload in-memory artifacts."""
    global model_assets
    try:
        train_models.main()
        model_assets = _load_models()
        log_audit(connection, current_user.id, 'models_retrained', model_assets.get('models_path'))
        flash('Models retrained and reloaded successfully.', 'success')
    except Exception as exc:
        flash(f'Model retraining failed: {exc}', 'danger')
    return redirect(url_for('app.admin_dashboard'))


@app.route('/export/prediction/<int:prediction_id>.csv')
@role_required('staff', 'manager', 'admin')
def export_prediction_csv(prediction_id):
    prediction = get_prediction_by_id(connection, prediction_id)
    if not prediction:
        flash('Prediction not found.', 'warning')
        return redirect(url_for('app.dashboard'))

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Prediction ID', prediction['id']])
    writer.writerow(['Created At', prediction['created_at']])
    writer.writerow([])
    writer.writerow(['Top Placement Type', 'Estimated Duration (days)', 'Stability (%)', 'Breakdown Likelihood (%)'])
    writer.writerow([
        prediction['predicted_type'],
        prediction['predicted_duration'],
        prediction['stability_score'],
        prediction['breakdown_likelihood'],
    ])

    payload = prediction['prediction_payload']
    if payload:
        writer.writerow([])
        writer.writerow(['Placement Type', 'Estimated Duration (days)', 'Stability (%)', 'Breakdown Likelihood (%)', 'Net Stability'])
        for row in json.loads(payload):
            writer.writerow([
                row.get('type'),
                row.get('duration'),
                row.get('stability'),
                row.get('breakdown_likelihood'),
                row.get('net_stability'),
            ])

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=prediction_{prediction_id}.csv'}
    )


@app.route('/export/comparison/<int:comparison_id>.csv')
@role_required('staff', 'manager', 'admin')
def export_comparison_csv(comparison_id):
    comparison = get_comparison_by_id(connection, comparison_id)
    if not comparison:
        flash('Comparison not found.', 'warning')
        return redirect(url_for('app.dashboard'))

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Comparison ID', comparison['id']])
    writer.writerow(['Created At', comparison['created_at']])
    writer.writerow([])
    writer.writerow(['Placement Type', 'Estimated Duration (days)', 'Stability (%)', 'Breakdown Likelihood (%)', 'Net Stability'])

    for row in json.loads(comparison['comparison_results'] or '[]'):
        writer.writerow([
            row.get('type'),
            row.get('duration'),
            row.get('stability'),
            row.get('breakdown_likelihood'),
            row.get('net_stability'),
        ])

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=comparison_{comparison_id}.csv'}
    )


@app.route('/export/breakdown-analysis.csv')
@role_required('manager', 'admin')
def export_breakdown_analysis_csv():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Placement Type', 'Total', 'Breakdowns', 'Breakdown Rate (%)'])

    for row in analyze_breakdown_patterns(connection):
        writer.writerow([row['placement_type'], row['total'], row['breakdowns'], row['breakdown_rate']])

    writer.writerow([])
    writer.writerow(['Duration Band', 'Total', 'Breakdowns', 'Breakdown Rate (%)'])
    for row in get_breakdown_patterns_by_duration(connection):
        writer.writerow([row['duration_band'], row['total'], row['breakdowns'], row['breakdown_rate']])

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=breakdown_analysis.csv'}
    )

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
    settings_payload = {}

    for key, value in request.form.items():
        if key in {'csrf_token', 'new_setting_key', 'new_setting_value'}:
            continue
        settings_payload[key] = value

    new_key = request.form.get('new_setting_key', '').strip()
    new_value = request.form.get('new_setting_value', '').strip()
    if new_key:
        settings_payload[new_key] = new_value

    update_system_settings(connection, settings_payload)
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
