from flask import Blueprint, render_template, request, redirect, url_for, flash, Response, current_app
from flask_login import login_required, current_user, login_user, logout_user
from functools import wraps
from .forms import (LoginForm, UserForm, UserEditForm, PlacementUploadForm,
                    BulkUploadForm, PredictionForm, ComparisonForm,
                    ForgotPasswordForm, ChangePasswordForm, ResetPasswordForm,
                    PlacementOutcomeForm,
                    GENDER_CHOICES, ETHNICITY_CHOICES)
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
                     issue_temporary_password, get_prediction_numeric_averages,
                     update_placement_outcome as update_placement_outcome_record)
from .utils import (prepare_prediction_input, generate_predictions_list,
                    extract_profile_from_form, compare_placement_options,
                    process_bulk_upload, generate_explainability_summary)
import os
import joblib
import json
import csv
import re
import hashlib
import smtplib
import secrets
from pathlib import Path
from datetime import datetime, timezone
from io import StringIO
from email.message import EmailMessage
import pandas as pd
from . import train_models
from .permissions import has_permission, normalize_role

# Calculate template folder relative to routes.py
template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')

app = Blueprint('app', __name__, template_folder=template_dir)
db_path = os.path.join(os.path.dirname(__file__), 'static', 'placements.db')
connection = create_connection(db_path)
if connection:
    create_tables(connection)

MODEL_RETRAIN_UPLOAD_DIR = Path(__file__).resolve().parent / "static" / "uploads" / "retraining"
RETRAIN_REQUIRED_FIELDS = train_models.FEATURE_COLUMNS + [train_models.PLACEMENT_COLUMN, train_models.REGRESSION_TARGET]
RETRAIN_CRITICAL_FIELDS = list(getattr(train_models, "DEFAULT_CRITICAL_FIELDS", [train_models.PLACEMENT_COLUMN, train_models.REGRESSION_TARGET]))
RETRAIN_MAPPING_CONFIG_PATH = MODEL_RETRAIN_UPLOAD_DIR / "mapping_profiles.json"
RETRAIN_DEFAULT_MODE_CHOICES = [
    ("", "No default"),
    ("ethnicities", "Ethnicities"),
    ("genders", "Genders"),
    ("child_ages", "Child Ages (0-18)"),
    ("carer_ages", "Carer Ages (25-75)"),
    ("boolean", "Boolean (True/False)"),
    ("custom", "Custom numeric range"),
]
RETRAIN_DEFAULT_MODE_SET = {mode for mode, _ in RETRAIN_DEFAULT_MODE_CHOICES if mode}
RETRAIN_GENDER_VALUES = [value for value, _ in GENDER_CHOICES]
RETRAIN_ETHNICITY_VALUES = [value for value, _ in ETHNICITY_CHOICES]


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


def _normalize_column_name(column_name):
    return re.sub(r"[^a-z0-9]+", "", str(column_name).strip().lower())


def _suggest_column_mapping(headers):
    header_lookup = {_normalize_column_name(header): header for header in headers}
    aliases = {
        "Carer Gender": ["Carer Gender Composition"],
        "Carer Ethnicity": ["Carer Ethnicity Or Religion"],
    }

    suggested = {}
    for field in RETRAIN_REQUIRED_FIELDS:
        default_match = header_lookup.get(_normalize_column_name(field))
        if default_match:
            suggested[field] = default_match
            continue

        alias_match = None
        for alias in aliases.get(field, []):
            alias_match = header_lookup.get(_normalize_column_name(alias))
            if alias_match:
                break
        suggested[field] = alias_match

    return suggested


def _headers_signature(headers):
    normalized = [_normalize_column_name(header) for header in headers if str(header).strip()]
    payload = "|".join(sorted(normalized))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_mapping_profiles():
    if not RETRAIN_MAPPING_CONFIG_PATH.exists():
        return {}
    try:
        with open(RETRAIN_MAPPING_CONFIG_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_mapping_profiles(payload):
    MODEL_RETRAIN_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with open(RETRAIN_MAPPING_CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _get_saved_mapping_profile(headers):
    profiles = _load_mapping_profiles()
    return profiles.get(_headers_signature(headers), {})


def _persist_mapping_profile(headers, profile):
    profiles = _load_mapping_profiles()
    signature = _headers_signature(headers)
    profiles[signature] = {
        **profile,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_mapping_profiles(profiles)


def _read_csv_headers(dataset_path):
    return [str(col) for col in pd.read_csv(dataset_path, nrows=1).columns]


def _summarize_dataset_fields(headers):
    required_set = set(RETRAIN_REQUIRED_FIELDS)
    present_required = [field for field in RETRAIN_REQUIRED_FIELDS if field in headers]
    missing_required = [field for field in RETRAIN_REQUIRED_FIELDS if field not in headers]
    extra_fields = [header for header in headers if header not in required_set]
    return present_required, missing_required, extra_fields


def _normalize_default_config(raw_value):
    """Normalize saved default configuration so templates can render consistently."""
    if not isinstance(raw_value, dict):
        return {"mode": "", "start": "", "end": ""}

    mode = str(raw_value.get("mode", "")).strip().lower()
    if mode not in RETRAIN_DEFAULT_MODE_SET:
        mode = ""

    start = raw_value.get("start", "")
    end = raw_value.get("end", "")
    return {
        "mode": mode,
        "start": "" if start is None else str(start),
        "end": "" if end is None else str(end),
    }

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


def _run_model_retraining(
    dataset_path=None,
    column_mapping=None,
    missing_defaults=None,
    critical_fields=None,
    exclude_missing_critical=False,
):
    """Run training, reload in-memory assets, and return success state plus message."""
    global model_assets
    try:
        train_models.main(
            dataset_path=dataset_path,
            column_mapping=column_mapping,
            missing_defaults=missing_defaults,
            critical_fields=critical_fields,
            exclude_missing_critical=exclude_missing_critical,
        )
        model_assets = _load_models()
        log_audit(connection, current_user.id, 'models_retrained', model_assets.get('models_path'))
        return True, 'Models retrained and reloaded successfully.'
    except Exception as exc:
        log_audit(connection, current_user.id, 'models_retrain_failed', str(exc))
        return False, f'Model retraining failed: {exc}'


# Permission-based access control decorator
def permission_required(permission):
    def decorator(f):
        @wraps(f)
        @login_required
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                flash('Please log in to access this page.', 'warning')
                return redirect(url_for('app.login'))
            if not has_permission(current_user.role, permission):
                flash('You do not have permission to access this page.', 'danger')
                return redirect(url_for('app.dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def role_required(*roles):
    """Backward-compatible wrapper for legacy role checks."""
    allowed = {normalize_role(role) for role in roles}

    def decorator(f):
        @wraps(f)
        @login_required
        def decorated_function(*args, **kwargs):
            if normalize_role(current_user.role) not in allowed:
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
    if has_permission(current_user.role, 'access_admin_dashboard'):
        return redirect(url_for('app.admin_dashboard'))
    if has_permission(current_user.role, 'access_manager_dashboard'):
        return redirect(url_for('app.manager_dashboard'))
    return redirect(url_for('app.staff_dashboard'))

@app.route('/staff-dashboard')
@permission_required('access_staff_dashboard')
def staff_dashboard():
    """Staff dashboard - view placements and make predictions"""
    uploaded_by = None if has_permission(current_user.role, 'view_all_placements') else current_user.id
    recent_placements = get_recent_placements(connection, limit=10, uploaded_by=uploaded_by)
    # Backward compatibility: older records may not have uploader metadata.
    if not recent_placements and uploaded_by is not None:
        recent_placements = get_recent_placements(connection, limit=10)
    stats = get_placement_statistics(connection)
    return render_template('staff_dashboard.html',
                         recent_placements=recent_placements,
                         stats=stats)

@app.route('/manager-dashboard')
@permission_required('access_manager_dashboard')
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
@permission_required('access_admin_dashboard')
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
@permission_required('upload_placements')
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

        try:
            placement_id = add_placement_record(connection, placement_data)
            if not placement_id:
                raise ValueError('No placement identifier returned from storage.')
            log_audit(connection, current_user.id, 'placement_upload', placement_id)
            flash('Placement record uploaded successfully!', 'success')
            return redirect(url_for('app.view_placement', placement_id=placement_id))
        except Exception as exc:
            flash(f'Could not upload placement record: {exc}', 'danger')
    elif request.method == 'POST':
        invalid_fields = [name for name, errors in form.errors.items() if errors]
        if invalid_fields:
            flash(f"Please fix the highlighted fields: {', '.join(invalid_fields)}", 'warning')

    return render_template('upload_placement.html', form=form)

@app.route('/upload-bulk', methods=['GET', 'POST'])
@permission_required('upload_placements')
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
@permission_required('view_placement_record')
def view_placement(placement_id):
    """View individual placement details"""
    uploaded_by = None if has_permission(current_user.role, 'view_all_placements') else current_user.id
    placement = get_placement_by_id(connection, placement_id, uploaded_by=uploaded_by)
    if not placement and uploaded_by is not None:
        placement = get_placement_by_id(connection, placement_id)
    if not placement:
        flash('Placement not found.', 'warning')
        return redirect(url_for('app.dashboard'))

    return render_template('placement_detail.html', placement=placement)


@app.route('/placement-outcome', methods=['GET', 'POST'])
@permission_required('update_placement_outcome')
def update_placement_outcome():
    """Update closure outcomes for placements created in the system."""
    uploaded_by = None if has_permission(current_user.role, 'view_all_placements') else current_user.id
    placements = get_recent_placements(connection, limit=500, uploaded_by=uploaded_by)

    form = PlacementOutcomeForm()
    form.placement_id.choices = [
        (int(row['id']), f"#{row['id']} - {row['placement_type']} (Created {row['created_at']})")
        for row in placements
    ]

    if not form.placement_id.choices:
        flash('No placement records are available to update.', 'warning')
        return redirect(url_for('app.staff_dashboard'))

    selectable_ids = {choice[0] for choice in form.placement_id.choices}
    requested_id = request.args.get('placement_id', type=int)
    if request.method == 'GET':
        if requested_id in selectable_ids:
            form.placement_id.data = requested_id
        elif not form.placement_id.data:
            form.placement_id.data = form.placement_id.choices[0][0]

    selected_placement = get_placement_by_id(connection, form.placement_id.data, uploaded_by=uploaded_by)

    if form.validate_on_submit():
        placement_id = form.placement_id.data
        selected_placement = get_placement_by_id(connection, placement_id, uploaded_by=uploaded_by)
        if not selected_placement:
            flash('Placement record not found or not accessible.', 'warning')
            return redirect(url_for('app.update_placement_outcome'))

        update_success = update_placement_outcome_record(
            connection,
            placement_id=placement_id,
            end_reason=form.placement_end_reason.data,
            days_lasted=form.days_placement_lasted.data,
            notes=(form.outcome_notes.data or '').strip(),
            uploaded_by=uploaded_by,
        )
        if not update_success:
            flash('Could not update placement outcome. Please try again.', 'danger')
            return redirect(url_for('app.update_placement_outcome', placement_id=placement_id))

        log_audit(connection, current_user.id, 'placement_outcome_updated', placement_id)
        flash('Placement outcome updated successfully.', 'success')
        return redirect(url_for('app.view_placement', placement_id=placement_id))

    return render_template(
        'placement_outcome_update.html',
        form=form,
        selected_placement=selected_placement,
    )

# ============== Prediction Routes ==============

@app.route('/predict', methods=['GET', 'POST'])
@permission_required('predict')
def predict():
    """Generate AI-powered placement stability prediction"""
    form = PredictionForm()
    if form.validate_on_submit():
        numeric_defaults = get_prediction_numeric_averages(connection)
        profile_data = extract_profile_from_form(form)
        input_data = prepare_prediction_input(
            form,
            model_assets["feature_encoders"],
            numeric_defaults=numeric_defaults,
        )
        predictions = generate_predictions_list(
            input_data,
            model_assets["rf_model"],
            model_assets["rf_reg_model"] or model_assets["lr_model"],
            model_assets["placement_encoder"],
            breakdown_model=model_assets["breakdown_model"],
            placement_feature_names=model_assets["metadata"].get("classification_features"),
            breakdown_feature_names=model_assets["metadata"].get("breakdown_features"),
        )
        explainability_summary = generate_explainability_summary(
            user_profile=profile_data,
            predictions=predictions,
            feature_names=model_assets["metadata"].get("classification_features"),
            feature_importances=getattr(model_assets["rf_model"], "feature_importances_", None),
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
                             explainability_summary=explainability_summary,
                             predictions=predictions)

    return render_template('index.html', form=form)

@app.route('/compare', methods=['GET', 'POST'])
@permission_required('compare')
def compare():
    """Compare multiple placement options"""
    form = ComparisonForm()
    if form.validate_on_submit():
        selected_types = form.placement_types.data
        if len(selected_types) < 2 or len(selected_types) > 4:
            flash('Please select between 2 and 4 placement types.', 'warning')
            return render_template('compare.html', form=form)

        profile_data = extract_profile_from_form(form)
        numeric_defaults = get_prediction_numeric_averages(connection)

        # Generate predictions for each selected placement type
        comparisons = compare_placement_options(
            profile_data, selected_types,
            model_assets["rf_model"], model_assets["rf_reg_model"] or model_assets["lr_model"],
            model_assets["feature_encoders"], model_assets["placement_encoder"],
            numeric_defaults=numeric_defaults,
            breakdown_model=model_assets["breakdown_model"],
            placement_feature_names=model_assets["metadata"].get("classification_features"),
            breakdown_feature_names=model_assets["metadata"].get("breakdown_features"),
        )
        explainability_summary = generate_explainability_summary(
            user_profile=profile_data,
            predictions=comparisons,
            feature_names=model_assets["metadata"].get("classification_features"),
            feature_importances=getattr(model_assets["rf_model"], "feature_importances_", None),
        )

        # Save comparison to database
        comparison_id = save_comparison(connection, profile_data, comparisons, current_user.id)
        log_audit(connection, current_user.id, 'comparison_generated', comparison_id)

        return render_template('comparison_results.html',
                             comparison_id=comparison_id,
                             profile=profile_data,
                             explainability_summary=explainability_summary,
                             predictions=comparisons)

    return render_template('compare.html', form=form)

# ============== Analysis Routes ==============

@app.route('/breakdown-analysis')
@permission_required('breakdown_full')
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
                         recommendations=recommendations,
                         read_only=False)


@app.route('/breakdown-analysis/staff')
@permission_required('breakdown_read')
def breakdown_analysis_staff():
    """Read-only breakdown analysis view for operational users."""
    breakdown_data = analyze_breakdown_patterns(connection)
    duration_band_patterns = get_breakdown_patterns_by_duration(connection)
    risk_factors = identify_risk_factors(connection)

    return render_template(
        'breakdown_analysis.html',
        breakdown_data=breakdown_data,
        duration_band_patterns=duration_band_patterns,
        risk_factors=risk_factors,
        recommendations=[],
        read_only=True,
    )


@app.route('/model-retraining', methods=['GET', 'POST'])
@permission_required('manage_model_retraining')
def model_retraining():
    """Upload retraining CSV and forward users to the field-mapping workflow."""
    metadata = model_assets.get('metadata') or {}

    if request.method == 'POST':
        csv_file = request.files.get('training_csv')
        if not csv_file or not csv_file.filename:
            flash('Please choose a CSV file to continue.', 'warning')
            return redirect(url_for('app.model_retraining'))
        if not csv_file.filename.lower().endswith('.csv'):
            flash('Only CSV files are supported for retraining.', 'warning')
            return redirect(url_for('app.model_retraining'))

        try:
            uploaded_df = pd.read_csv(csv_file.stream)
            headers = [str(col) for col in uploaded_df.columns]
            if not headers:
                raise ValueError('Uploaded CSV has no header row.')

            MODEL_RETRAIN_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            dataset_token = f"retrain_{current_user.id}_{secrets.token_hex(8)}.csv"
            dataset_path = MODEL_RETRAIN_UPLOAD_DIR / dataset_token
            uploaded_df.to_csv(dataset_path, index=False)

            return redirect(url_for('app.model_retraining_mapping', dataset_token=dataset_token))
        except Exception as exc:
            flash(f'Could not process uploaded CSV: {exc}', 'danger')
            return redirect(url_for('app.model_retraining'))

    return render_template(
        'model_retraining.html',
        models_path=model_assets.get('models_path'),
        metadata=metadata,
    )


@app.route('/model-retraining/mapping/<dataset_token>', methods=['GET', 'POST'])
@permission_required('manage_model_retraining')
def model_retraining_mapping(dataset_token):
    """Map CSV fields to model fields, handle missing data, and execute retraining."""
    safe_token = os.path.basename(dataset_token)
    dataset_path = MODEL_RETRAIN_UPLOAD_DIR / safe_token
    if not safe_token or safe_token != dataset_token or not dataset_path.exists():
        flash('Uploaded dataset could not be found. Please upload the CSV again.', 'warning')
        return redirect(url_for('app.model_retraining'))

    metadata = model_assets.get('metadata') or {}

    try:
        csv_headers = _read_csv_headers(dataset_path)
    except Exception as exc:
        flash(f'Could not read uploaded dataset headers: {exc}', 'danger')
        return redirect(url_for('app.model_retraining'))

    present_required, missing_required, extra_fields = _summarize_dataset_fields(csv_headers)

    saved_profile = _get_saved_mapping_profile(csv_headers)
    suggested_mapping = _suggest_column_mapping(csv_headers)
    saved_mapping = saved_profile.get('column_mapping', {}) if isinstance(saved_profile, dict) else {}
    suggested_mapping.update({field: value for field, value in saved_mapping.items() if value in csv_headers})
    raw_saved_defaults = saved_profile.get('missing_defaults', {}) if isinstance(saved_profile, dict) else {}
    suggested_default_configs = {
        field: _normalize_default_config(raw_saved_defaults.get(field))
        for field in RETRAIN_REQUIRED_FIELDS
    }
    saved_exclude_missing = bool(saved_profile.get('exclude_missing_critical', False)) if isinstance(saved_profile, dict) else False
    saved_ignore_extra = bool(saved_profile.get('ignore_extra_fields', True)) if isinstance(saved_profile, dict) else True

    if request.method == 'POST':
        column_mapping = {}
        missing_defaults = {}
        default_config_errors = []
        for field in RETRAIN_REQUIRED_FIELDS:
            selected_column = request.form.get(f"map__{field}", '').strip()
            default_mode = request.form.get(f"default_mode__{field}", '').strip().lower()
            custom_start = request.form.get(f"default_custom_start__{field}", '').strip()
            custom_end = request.form.get(f"default_custom_end__{field}", '').strip()

            if selected_column:
                column_mapping[field] = selected_column

            if not default_mode:
                continue

            if default_mode not in RETRAIN_DEFAULT_MODE_SET:
                default_config_errors.append(f"{field}: unsupported default mode '{default_mode}'")
                continue

            default_config = {"mode": default_mode}
            if default_mode == 'custom':
                if not custom_start or not custom_end:
                    default_config_errors.append(f"{field}: custom range requires both start and end")
                    continue
                try:
                    start_value = float(custom_start)
                    end_value = float(custom_end)
                except ValueError:
                    default_config_errors.append(f"{field}: custom range values must be numeric")
                    continue
                if start_value > end_value:
                    default_config_errors.append(f"{field}: custom range start must be <= end")
                    continue
                default_config["start"] = start_value
                default_config["end"] = end_value

            missing_defaults[field] = default_config

        exclude_missing_critical = request.form.get('exclude_missing_critical') == 'on'
        ignore_extra_fields = request.form.get('ignore_extra_fields') == 'on'

        invalid_mappings = [
            f"{field} -> {source}"
            for field, source in column_mapping.items()
            if source not in csv_headers
        ]
        unresolved_critical = [
            field for field in RETRAIN_CRITICAL_FIELDS
            if not column_mapping.get(field) and not missing_defaults.get(field)
        ]

        if invalid_mappings:
            flash(f"Some mappings are invalid for the uploaded CSV: {', '.join(invalid_mappings)}", 'warning')
        if default_config_errors:
            flash(f"Please fix default range settings: {', '.join(default_config_errors)}", 'warning')
        if unresolved_critical:
            flash(
                f"Critical fields must be mapped or have defaults before retraining: {', '.join(unresolved_critical)}",
                'danger',
            )
        if extra_fields and not ignore_extra_fields:
            flash(
                'Extra CSV fields were detected. Tick "Ignore extra CSV fields" to continue with retraining.',
                'warning',
            )

        if default_config_errors or invalid_mappings or unresolved_critical or (extra_fields and not ignore_extra_fields):
            current_default_configs = {
                field: _normalize_default_config(missing_defaults.get(field))
                for field in RETRAIN_REQUIRED_FIELDS
            }
            return render_template(
                'model_retraining_mapping.html',
                models_path=model_assets.get('models_path'),
                metadata=metadata,
                dataset_token=safe_token,
                csv_headers=csv_headers,
                required_fields=RETRAIN_REQUIRED_FIELDS,
                critical_fields=RETRAIN_CRITICAL_FIELDS,
                present_required=present_required,
                missing_required=missing_required,
                extra_fields=extra_fields,
                suggested_mapping=column_mapping,
                suggested_default_configs=current_default_configs,
                exclude_missing_critical=exclude_missing_critical,
                ignore_extra_fields=ignore_extra_fields,
                default_mode_choices=RETRAIN_DEFAULT_MODE_CHOICES,
            )

        success, message = _run_model_retraining(
            dataset_path=str(dataset_path),
            column_mapping=column_mapping,
            missing_defaults=missing_defaults,
            critical_fields=RETRAIN_CRITICAL_FIELDS,
            exclude_missing_critical=exclude_missing_critical,
        )
        flash(message, 'success' if success else 'danger')

        if success:
            _persist_mapping_profile(
                csv_headers,
                {
                    'column_mapping': column_mapping,
                    'missing_defaults': missing_defaults,
                    'exclude_missing_critical': exclude_missing_critical,
                    'ignore_extra_fields': ignore_extra_fields,
                },
            )
            return redirect(url_for('app.model_retraining'))

        return render_template(
            'model_retraining_mapping.html',
            models_path=model_assets.get('models_path'),
            metadata=metadata,
            dataset_token=safe_token,
            csv_headers=csv_headers,
            required_fields=RETRAIN_REQUIRED_FIELDS,
            critical_fields=RETRAIN_CRITICAL_FIELDS,
            present_required=present_required,
            missing_required=missing_required,
            extra_fields=extra_fields,
            suggested_mapping=column_mapping,
            suggested_default_configs={
                field: _normalize_default_config(missing_defaults.get(field))
                for field in RETRAIN_REQUIRED_FIELDS
            },
            exclude_missing_critical=exclude_missing_critical,
            ignore_extra_fields=ignore_extra_fields,
            default_mode_choices=RETRAIN_DEFAULT_MODE_CHOICES,
        )

    if missing_required:
        flash(
            f"Uploaded CSV is missing required headers: {', '.join(missing_required)}. Map these fields or provide defaults.",
            'warning',
        )

    return render_template(
        'model_retraining_mapping.html',
        models_path=model_assets.get('models_path'),
        metadata=metadata,
        dataset_token=safe_token,
        csv_headers=csv_headers,
        required_fields=RETRAIN_REQUIRED_FIELDS,
        critical_fields=RETRAIN_CRITICAL_FIELDS,
        present_required=present_required,
        missing_required=missing_required,
        extra_fields=extra_fields,
        suggested_mapping=suggested_mapping,
        suggested_default_configs=suggested_default_configs,
        exclude_missing_critical=saved_exclude_missing,
        ignore_extra_fields=saved_ignore_extra,
        default_mode_choices=RETRAIN_DEFAULT_MODE_CHOICES,
    )


@app.route('/admin/retrain-models', methods=['POST'])
@permission_required('manage_model_retraining')
def retrain_models():
    """Backward-compatible endpoint that redirects users into the mapping workflow."""
    flash('Upload a CSV and map fields before retraining models.', 'info')
    return redirect(url_for('app.model_retraining'))


@app.route('/admin/retrain-models', methods=['GET'])
@permission_required('manage_model_retraining')
def retrain_models_legacy_get():
    """Legacy GET alias for environments linking directly to the old endpoint."""
    return redirect(url_for('app.model_retraining'))


@app.route('/export/prediction/<int:prediction_id>.csv')
@permission_required('export_ai_outputs')
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
        writer.writerow(['Placement Type', 'Estimated Duration (days)', 'Stability (%)', 'Breakdown Likelihood (%)'])
        for row in json.loads(payload):
            writer.writerow([
                row.get('type'),
                row.get('duration'),
                row.get('stability'),
                row.get('breakdown_likelihood'),
            ])

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=prediction_{prediction_id}.csv'}
    )


@app.route('/export/comparison/<int:comparison_id>.csv')
@permission_required('export_ai_outputs')
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
    writer.writerow(['Placement Type', 'Estimated Duration (days)', 'Stability (%)', 'Breakdown Likelihood (%)'])

    for row in json.loads(comparison['comparison_results'] or '[]'):
        writer.writerow([
            row.get('type'),
            row.get('duration'),
            row.get('stability'),
            row.get('breakdown_likelihood'),
        ])

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=comparison_{comparison_id}.csv'}
    )


@app.route('/export/breakdown-analysis.csv')
@permission_required('breakdown_export')
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
@permission_required('stability_trends')
def stability_trends():
    """View placement stability trends over time"""
    trends = get_stability_trends(connection)
    metrics = calculate_stability_metrics(connection)

    return render_template('stability_trends.html',
                         trends=trends,
                         metrics=metrics)

# ============== User Management Routes ==============

@app.route('/users')
@permission_required('admin_manage_system')
def manage_users():
    """User management page"""
    users = get_all_users(connection)
    return render_template('manage_users.html', users=users)

@app.route('/users/add', methods=['GET', 'POST'])
@permission_required('admin_manage_system')
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
@permission_required('admin_manage_system')
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
@permission_required('admin_manage_system')
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
@permission_required('admin_manage_system')
def system_settings():
    """System configuration page"""
    settings = get_system_settings(connection)
    return render_template('settings.html', settings=settings)

@app.route('/settings/update', methods=['POST'])
@permission_required('admin_manage_system')
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
@permission_required('view_audit_logs')
def audit_logs():
    """View audit logs"""
    page = request.args.get('page', 1, type=int)
    logs = get_audit_logs_paginated(connection, page, per_page=50)

    return render_template('audit_logs.html', logs=logs)
