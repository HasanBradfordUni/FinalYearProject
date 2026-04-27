import sqlite3
from sqlite3 import Error
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

def create_connection(path):
    connection = None
    try:
        connection = sqlite3.connect(path, check_same_thread=False)
        connection.row_factory = sqlite3.Row  # Enable dict-like access
        print("Connection to SQLite DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

def create_tables(connection):
    """Create all required database tables"""

    # Users table with role-based access
    query = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'placement_officer',
        is_active INTEGER DEFAULT 1,
        must_reset_password INTEGER DEFAULT 0,
        temp_password_issued_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    );
    """
    execute_query(connection, query)

    # Placements table
    query = """
    CREATE TABLE IF NOT EXISTS placements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        child_age REAL,
        child_gender TEXT,
        child_ethnicity TEXT,
        child_prior_placements INTEGER DEFAULT 0,
        returning_child INTEGER DEFAULT 0,
        missing_episodes INTEGER DEFAULT 0,
        sibling_group_size INTEGER DEFAULT 1,
        placed_with_siblings INTEGER DEFAULT 0,
        carer_age REAL,
        carer_gender TEXT,
        carer_ethnicity TEXT,
        placement_type TEXT,
        placement_start_date TEXT,
        move_date TEXT,
        move_reason TEXT,
        distance_from_home REAL,
        eh_involvement INTEGER DEFAULT 0,
        yot_involvement INTEGER DEFAULT 0,
        placement_sequence_number INTEGER DEFAULT 1,
        placement_duration INTEGER,
        breakdown_occurred INTEGER DEFAULT 0,
        uploaded_by INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (uploaded_by) REFERENCES users (id)
    );
    """
    execute_query(connection, query)

    # Predictions table
    query = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        child_age REAL,
        child_gender TEXT,
        child_ethnicity TEXT,
        carer_age REAL,
        carer_gender TEXT,
        carer_ethnicity TEXT,
        predicted_type TEXT,
        predicted_duration REAL,
        stability_score REAL,
        breakdown_likelihood REAL DEFAULT 0,
        prediction_payload TEXT,
        user_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """
    execute_query(connection, query)

    # Comparisons table
    query = """
    CREATE TABLE IF NOT EXISTS comparisons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_data TEXT,
        comparison_results TEXT,
        user_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """
    execute_query(connection, query)

    # Audit logs table
    query = """
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        action TEXT NOT NULL,
        details TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """
    execute_query(connection, query)

    # System settings table
    query = """
    CREATE TABLE IF NOT EXISTS system_settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        setting_key TEXT UNIQUE NOT NULL,
        setting_value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    execute_query(connection, query)

    # Lightweight in-place migrations for older databases.
    _ensure_column(connection, "placements", "child_prior_placements", "INTEGER DEFAULT 0")
    _ensure_column(connection, "placements", "returning_child", "INTEGER DEFAULT 0")
    _ensure_column(connection, "placements", "missing_episodes", "INTEGER DEFAULT 0")
    _ensure_column(connection, "placements", "sibling_group_size", "INTEGER DEFAULT 0")
    _ensure_column(connection, "placements", "placed_with_siblings", "INTEGER DEFAULT 0")
    _ensure_column(connection, "placements", "placement_start_date", "TEXT")
    _ensure_column(connection, "placements", "move_date", "TEXT")
    _ensure_column(connection, "placements", "move_reason", "TEXT")
    _ensure_column(connection, "placements", "distance_from_home", "REAL")
    _ensure_column(connection, "placements", "eh_involvement", "INTEGER DEFAULT 0")
    _ensure_column(connection, "placements", "yot_involvement", "INTEGER DEFAULT 0")
    _ensure_column(connection, "placements", "placement_sequence_number", "INTEGER DEFAULT 1")
    _ensure_column(connection, "placements", "placement_end_reason", "TEXT")
    _ensure_column(connection, "placements", "breakdown_flag", "INTEGER DEFAULT 0")
    _ensure_column(connection, "placements", "days_placement_lasted", "INTEGER")
    _ensure_column(connection, "placements", "outcome_notes", "TEXT")
    _ensure_column(connection, "predictions", "breakdown_likelihood", "REAL DEFAULT 0")
    _ensure_column(connection, "predictions", "prediction_payload", "TEXT")
    _ensure_column(connection, "users", "must_reset_password", "INTEGER DEFAULT 0")
    _ensure_column(connection, "users", "temp_password_issued_at", "TIMESTAMP")


def _ensure_column(connection, table_name, column_name, column_definition):
    cursor = connection.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if column_name not in existing_columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")
        connection.commit()


def execute_query(connection, query, params=None):
    cursor = connection.cursor()
    try:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        connection.commit()
        return cursor.fetchall()
    except Error as e:
        print(f"The error '{e}' occurred")
        return None

# ============== User Management Functions ==============

class User(UserMixin):
    """User model for Flask-Login"""
    def __init__(self, id, username, email, role, is_active=True, must_reset_password=False):
        self.id = id
        self.username = username
        self.email = email
        self.role = role
        # Keep DB active flag in a private field to avoid assigning to UserMixin property.
        self._is_active = bool(is_active)
        self.must_reset_password = bool(must_reset_password)

    @property
    def is_active(self):
        return self._is_active

def create_user(connection, username, email, password, role='placement_officer'):
    """Create a new user"""
    password_hash = generate_password_hash(password)
    query = """
    INSERT INTO users (username, email, password_hash, role)
    VALUES (?, ?, ?, ?)
    """
    cursor = connection.cursor()
    cursor.execute(query, (username, email, password_hash, role))
    connection.commit()
    return cursor.lastrowid

def authenticate_user(connection, username, password):
    """Authenticate user and return User object"""
    query = "SELECT * FROM users WHERE username = ? AND is_active = 1"
    cursor = connection.cursor()
    cursor.execute(query, (username,))
    row = cursor.fetchone()

    if row and check_password_hash(row['password_hash'], password):
        # Update last login
        update_query = "UPDATE users SET last_login = ? WHERE id = ?"
        cursor.execute(update_query, (datetime.now(), row['id']))
        connection.commit()

        return User(
            row['id'],
            row['username'],
            row['email'],
            row['role'],
            row['is_active'],
            row['must_reset_password']
        )
    return None

def get_user_by_id(connection, user_id):
    """Get user by ID for Flask-Login"""
    query = "SELECT * FROM users WHERE id = ?"
    cursor = connection.cursor()
    cursor.execute(query, (user_id,))
    row = cursor.fetchone()

    if row:
        return User(
            row['id'],
            row['username'],
            row['email'],
            row['role'],
            row['is_active'],
            row['must_reset_password']
        )
    return None


def get_user_by_identifier(connection, identifier):
    """Get active user by username or email for account recovery."""
    query = "SELECT * FROM users WHERE is_active = 1 AND (username = ? OR email = ?)"
    cursor = connection.cursor()
    cursor.execute(query, (identifier, identifier))
    return cursor.fetchone()


def verify_user_password(connection, user_id, password):
    """Verify whether provided password matches stored hash for a user."""
    query = "SELECT password_hash FROM users WHERE id = ?"
    cursor = connection.cursor()
    cursor.execute(query, (user_id,))
    row = cursor.fetchone()
    if not row:
        return False
    return check_password_hash(row['password_hash'], password)


def update_user_password(connection, user_id, new_password, must_reset_password=False):
    """Update user password hash and reset any temporary password state."""
    query = """
    UPDATE users
    SET password_hash = ?, must_reset_password = ?, temp_password_issued_at = NULL
    WHERE id = ?
    """
    cursor = connection.cursor()
    cursor.execute(query, (generate_password_hash(new_password), int(bool(must_reset_password)), user_id))
    connection.commit()


def issue_temporary_password(connection, user_id, temporary_password):
    """Set a one-time temporary password and force password change on next login."""
    query = """
    UPDATE users
    SET password_hash = ?, must_reset_password = 1, temp_password_issued_at = ?
    WHERE id = ?
    """
    cursor = connection.cursor()
    cursor.execute(query, (generate_password_hash(temporary_password), datetime.now(), user_id))
    connection.commit()

def get_all_users(connection):
    """Get all users"""
    query = "SELECT id, username, email, role, is_active, created_at FROM users ORDER BY created_at DESC"
    cursor = connection.cursor()
    cursor.execute(query)
    return cursor.fetchall()

def update_user(connection, user_id, data):
    """Update user details"""
    query = """
    UPDATE users 
    SET username = ?, email = ?, role = ?, is_active = ?
    WHERE id = ?
    """
    cursor = connection.cursor()
    cursor.execute(query, (data['username'], data['email'], data['role'], data['is_active'], user_id))
    connection.commit()

def delete_user_by_id(connection, user_id):
    """Delete user"""
    query = "DELETE FROM users WHERE id = ?"
    cursor = connection.cursor()
    cursor.execute(query, (user_id,))
    connection.commit()

# ============== Placement Management Functions ==============

def add_placement_record(connection, placement_data):
    """Add a new placement record"""
    query = """
    INSERT INTO placements 
    (
        child_age, child_gender, child_ethnicity,
        child_prior_placements, returning_child, missing_episodes, sibling_group_size, placed_with_siblings,
        carer_age, carer_gender, carer_ethnicity,
        placement_type, placement_start_date, move_date, move_reason, distance_from_home,
        eh_involvement, yot_involvement, placement_sequence_number,
        placement_duration, breakdown_occurred, uploaded_by
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor = connection.cursor()
    cursor.execute(query, (
        placement_data['child_age'],
        placement_data['child_gender'],
        placement_data['child_ethnicity'],
        placement_data.get('child_prior_placements', 0),
        placement_data.get('returning_child', 0),
        placement_data.get('missing_episodes', 0),
        max(1, int(placement_data.get('sibling_group_size', 1) or 1)),
        placement_data.get('placed_with_siblings', 0),
        placement_data['carer_age'],
        placement_data['carer_gender'],
        placement_data['carer_ethnicity'],
        placement_data['placement_type'],
        placement_data.get('placement_start_date'),
        placement_data.get('move_date'),
        placement_data.get('move_reason'),
        placement_data.get('distance_from_home', 0),
        placement_data.get('eh_involvement', 0),
        placement_data.get('yot_involvement', 0),
        placement_data.get('placement_sequence_number', 1),
        placement_data.get('placement_duration', 0),
        placement_data.get('breakdown_occurred', 0),
        placement_data['uploaded_by']
    ))
    connection.commit()
    return cursor.lastrowid


def update_placement_outcome(connection, placement_id, end_reason, days_lasted, notes, uploaded_by=None):
    """Update closure outcome details for an existing placement."""
    if uploaded_by is not None:
        existing = get_placement_by_id(connection, placement_id, uploaded_by=uploaded_by)
    else:
        existing = get_placement_by_id(connection, placement_id)
    if not existing:
        return False

    breakdown_flag = int(str(end_reason).strip().lower() == "breakdown")
    query = """
    UPDATE placements
    SET placement_end_reason = ?,
        breakdown_flag = ?,
        breakdown_occurred = ?,
        days_placement_lasted = ?,
        placement_duration = ?,
        outcome_notes = ?
    WHERE id = ?
    """
    cursor = connection.cursor()
    cursor.execute(
        query,
        (
            end_reason,
            breakdown_flag,
            breakdown_flag,
            days_lasted,
            days_lasted,
            notes,
            placement_id,
        ),
    )
    connection.commit()
    return cursor.rowcount > 0

def get_placement_by_id(connection, placement_id, uploaded_by=None):
    """Get placement by ID"""
    query = "SELECT * FROM placements WHERE id = ?"
    params = [placement_id]
    if uploaded_by is not None:
        query += " AND uploaded_by = ?"
        params.append(uploaded_by)
    cursor = connection.cursor()
    cursor.execute(query, tuple(params))
    return cursor.fetchone()

def get_recent_placements(connection, limit=10, uploaded_by=None):
    """Get recent placement records"""
    query = "SELECT * FROM placements"
    params = []
    if uploaded_by is not None:
        query += " WHERE uploaded_by = ?"
        params.append(uploaded_by)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    cursor = connection.cursor()
    cursor.execute(query, tuple(params))
    return cursor.fetchall()

def get_placement_statistics(connection):
    """Get placement statistics"""
    query = """
    SELECT 
        COUNT(*) as total_placements,
        SUM(COALESCE(breakdown_flag, breakdown_occurred, 0)) as total_breakdowns,
        AVG(COALESCE(days_placement_lasted, placement_duration)) as avg_duration,
        COUNT(DISTINCT placement_type) as placement_types
    FROM placements
    WHERE uploaded_by IS NOT NULL
    """
    cursor = connection.cursor()
    cursor.execute(query)
    return cursor.fetchone()


def get_prediction_numeric_averages(connection):
    """Get numeric fallback values for prediction fields from current placement data."""
    query = """
    SELECT
        AVG(child_age) AS child_age,
        AVG(child_prior_placements) AS child_prior_placements,
        AVG(missing_episodes) AS missing_episodes,
        AVG(sibling_group_size) AS sibling_group_size,
        AVG(carer_age) AS carer_age
    FROM placements
    """
    cursor = connection.cursor()
    cursor.execute(query)
    row = cursor.fetchone()

    defaults = {
        'child_age': 12,
        'child_prior_placements': 1,
        'missing_episodes': 1,
        'sibling_group_size': 1,
        'carer_age': 45,
    }
    if not row:
        return defaults

    for field_name, fallback in defaults.items():
        value = row[field_name]
        if value is None:
            continue
        rounded = int(round(float(value)))
        if field_name == 'child_age':
            defaults[field_name] = min(17, max(0, rounded))
        elif field_name == 'child_prior_placements':
            defaults[field_name] = min(4, max(0, rounded))
        elif field_name == 'missing_episodes':
            defaults[field_name] = min(7, max(0, rounded))
        elif field_name == 'sibling_group_size':
            defaults[field_name] = min(5, max(1, rounded))
        elif field_name == 'carer_age':
            defaults[field_name] = min(75, max(25, rounded))

    return defaults

# ============== Prediction Functions ==============

def save_prediction(connection, form_data, predictions, user_id):
    """Save prediction to database"""
    import json
    query = """
    INSERT INTO predictions 
    (child_age, child_gender, child_ethnicity, carer_age, carer_gender, carer_ethnicity, 
     predicted_type, predicted_duration, stability_score, breakdown_likelihood, prediction_payload, user_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor = connection.cursor()
    # Use first prediction for main record
    pred = predictions[0] if predictions else {}
    cursor.execute(query, (
        form_data.get('child_age'),
        form_data.get('child_gender'),
        form_data.get('child_ethnicity'),
        form_data.get('carer_age'),
        form_data.get('carer_gender'),
        form_data.get('carer_ethnicity'),
        pred.get('type', ''),
        pred.get('duration', 0),
        pred.get('stability', 0),
        pred.get('breakdown_likelihood', 0),
        json.dumps(predictions),
        user_id
    ))
    connection.commit()
    return cursor.lastrowid

def save_comparison(connection, profile_data, comparisons, user_id):
    """Save comparison to database"""
    import json
    query = """
    INSERT INTO comparisons (profile_data, comparison_results, user_id)
    VALUES (?, ?, ?)
    """
    cursor = connection.cursor()
    cursor.execute(query, (json.dumps(profile_data), json.dumps(comparisons), user_id))
    connection.commit()
    return cursor.lastrowid


def get_prediction_by_id(connection, prediction_id):
    query = "SELECT * FROM predictions WHERE id = ?"
    cursor = connection.cursor()
    cursor.execute(query, (prediction_id,))
    return cursor.fetchone()


def get_comparison_by_id(connection, comparison_id):
    query = "SELECT * FROM comparisons WHERE id = ?"
    cursor = connection.cursor()
    cursor.execute(query, (comparison_id,))
    return cursor.fetchone()

# ============== Analysis Functions ==============

def analyze_breakdown_patterns(connection):
    """Analyze placement breakdown patterns"""
    query = """
    SELECT 
        placement_type,
        COUNT(*) as total,
        SUM(COALESCE(breakdown_flag, breakdown_occurred, 0)) as breakdowns,
        CAST(SUM(COALESCE(breakdown_flag, breakdown_occurred, 0)) AS FLOAT) / COUNT(*) * 100 as breakdown_rate
    FROM placements
    WHERE uploaded_by IS NOT NULL
    GROUP BY placement_type
    ORDER BY breakdown_rate DESC
    """
    cursor = connection.cursor()
    cursor.execute(query)
    return cursor.fetchall()


def get_breakdown_patterns_by_duration(connection):
    """Breakdown rates across granular duration bands."""
    query = """
    SELECT
        CASE
            WHEN COALESCE(days_placement_lasted, placement_duration, 0) < 30 THEN '<1 month'
            WHEN COALESCE(days_placement_lasted, placement_duration, 0) BETWEEN 30 AND 89 THEN '1-3 months'
            WHEN COALESCE(days_placement_lasted, placement_duration, 0) BETWEEN 90 AND 179 THEN '3-6 months'
            WHEN COALESCE(days_placement_lasted, placement_duration, 0) BETWEEN 180 AND 364 THEN '6 months - 1 year'
            WHEN COALESCE(days_placement_lasted, placement_duration, 0) BETWEEN 365 AND 729 THEN '1-2 years'
            WHEN COALESCE(days_placement_lasted, placement_duration, 0) BETWEEN 730 AND 1459 THEN '2-4 years'
            ELSE '4+ years'
        END AS duration_band,
        COUNT(*) AS total,
        SUM(COALESCE(breakdown_flag, breakdown_occurred, 0)) AS breakdowns,
        CASE WHEN COUNT(*) = 0 THEN 0
             ELSE CAST(SUM(COALESCE(breakdown_flag, breakdown_occurred, 0)) AS FLOAT) / COUNT(*) * 100
        END AS breakdown_rate
    FROM placements
    WHERE uploaded_by IS NOT NULL
    GROUP BY duration_band
    ORDER BY CASE duration_band
        WHEN '<1 month' THEN 1
        WHEN '1-3 months' THEN 2
        WHEN '3-6 months' THEN 3
        WHEN '6 months - 1 year' THEN 4
        WHEN '1-2 years' THEN 5
        WHEN '2-4 years' THEN 6
        ELSE 7
    END
    """
    cursor = connection.cursor()
    cursor.execute(query)
    return cursor.fetchall()

def get_stability_trends(connection):
    """Get stability trends over time"""
    query = """
    SELECT 
        DATE(created_at) as date,
        COUNT(*) as placements,
        SUM(COALESCE(breakdown_flag, breakdown_occurred, 0)) as breakdowns
    FROM placements
    WHERE uploaded_by IS NOT NULL
    GROUP BY DATE(created_at)
    ORDER BY date DESC
    LIMIT 30
    """
    cursor = connection.cursor()
    cursor.execute(query)
    return cursor.fetchall()

def identify_risk_factors(connection):
    """Identify common risk factors in breakdown placements"""
    query = """
    SELECT 
        child_ethnicity,
        carer_ethnicity,
        placement_type,
        AVG(child_age) as avg_child_age,
        COUNT(*) as count
    FROM placements
    WHERE COALESCE(breakdown_flag, breakdown_occurred, 0) = 1
      AND uploaded_by IS NOT NULL
    GROUP BY child_ethnicity, carer_ethnicity, placement_type
    ORDER BY count DESC
    LIMIT 10
    """
    cursor = connection.cursor()
    cursor.execute(query)
    return cursor.fetchall()

def calculate_stability_metrics(connection):
    """Calculate various stability metrics"""
    query = """
    SELECT 
        AVG(CASE WHEN COALESCE(breakdown_flag, breakdown_occurred, 0) = 0 THEN COALESCE(days_placement_lasted, placement_duration) END) as avg_stable_duration,
        AVG(CASE WHEN COALESCE(breakdown_flag, breakdown_occurred, 0) = 1 THEN COALESCE(days_placement_lasted, placement_duration) END) as avg_breakdown_duration,
        COUNT(CASE WHEN COALESCE(days_placement_lasted, placement_duration, 0) > 365 THEN 1 END) as long_term_placements
    FROM placements
    WHERE uploaded_by IS NOT NULL
    """
    cursor = connection.cursor()
    cursor.execute(query)
    return cursor.fetchone()

# ============== Audit & System Functions ==============

def log_audit(connection, user_id, action, details):
    """Log user action for audit trail"""
    query = """
    INSERT INTO audit_logs (user_id, action, details)
    VALUES (?, ?, ?)
    """
    cursor = connection.cursor()
    cursor.execute(query, (user_id, action, str(details)))
    connection.commit()

def get_recent_audit_logs(connection, limit=20):
    """Get recent audit logs"""
    query = """
    SELECT a.*, u.username 
    FROM audit_logs a
    LEFT JOIN users u ON a.user_id = u.id
    ORDER BY a.timestamp DESC
    LIMIT ?
    """
    cursor = connection.cursor()
    cursor.execute(query, (limit,))
    return cursor.fetchall()

def get_audit_logs_paginated(connection, page, per_page=50):
    """Get paginated audit logs"""
    offset = (page - 1) * per_page
    query = """
    SELECT a.*, u.username 
    FROM audit_logs a
    LEFT JOIN users u ON a.user_id = u.id
    ORDER BY a.timestamp DESC
    LIMIT ? OFFSET ?
    """
    cursor = connection.cursor()
    cursor.execute(query, (per_page, offset))
    return cursor.fetchall()

def get_system_statistics(connection):
    """Get system-wide statistics"""
    query = """
    SELECT 
        (SELECT COUNT(*) FROM users) as total_users,
        (SELECT COUNT(*) FROM placements) as total_placements,
        (SELECT COUNT(*) FROM predictions) as total_predictions,
        (SELECT COUNT(*) FROM comparisons) as total_comparisons
    """
    cursor = connection.cursor()
    cursor.execute(query)
    return cursor.fetchone()

def get_system_settings(connection):
    """Get all system settings"""
    query = "SELECT * FROM system_settings"
    cursor = connection.cursor()
    cursor.execute(query)
    return cursor.fetchall()

def update_system_settings(connection, settings_dict):
    """Update system settings"""
    for key, value in settings_dict.items():
        query = """
        INSERT INTO system_settings (setting_key, setting_value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(setting_key) DO UPDATE SET 
        setting_value = excluded.setting_value,
        updated_at = excluded.updated_at
        """
        cursor = connection.cursor()
        cursor.execute(query, (key, value, datetime.now()))
    connection.commit()

def generate_breakdown_recommendations(breakdown_data):
    """Generate recommendations based on breakdown analysis"""
    recommendations = []
    for row in breakdown_data:
        if row['breakdown_rate'] > 30:
            recommendations.append({
                'placement_type': row['placement_type'],
                'severity': 'high',
                'message': f"High breakdown rate ({row['breakdown_rate']:.1f}%) - requires immediate attention"
            })
        elif row['breakdown_rate'] > 15:
            recommendations.append({
                'placement_type': row['placement_type'],
                'severity': 'medium',
                'message': f"Moderate breakdown rate ({row['breakdown_rate']:.1f}%) - monitor closely"
            })
    return recommendations

