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
        role TEXT NOT NULL DEFAULT 'staff',
        is_active INTEGER DEFAULT 1,
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
        carer_age REAL,
        carer_gender TEXT,
        carer_ethnicity TEXT,
        placement_type TEXT,
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
    def __init__(self, id, username, email, role, is_active=True):
        self.id = id
        self.username = username
        self.email = email
        self.role = role
        self.is_active = is_active

def create_user(connection, username, email, password, role='staff'):
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

        return User(row['id'], row['username'], row['email'], row['role'], row['is_active'])
    return None

def get_user_by_id(connection, user_id):
    """Get user by ID for Flask-Login"""
    query = "SELECT * FROM users WHERE id = ?"
    cursor = connection.cursor()
    cursor.execute(query, (user_id,))
    row = cursor.fetchone()

    if row:
        return User(row['id'], row['username'], row['email'], row['role'], row['is_active'])
    return None

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
    (child_age, child_gender, child_ethnicity, carer_age, carer_gender, carer_ethnicity, placement_type, uploaded_by)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor = connection.cursor()
    cursor.execute(query, (
        placement_data['child_age'],
        placement_data['child_gender'],
        placement_data['child_ethnicity'],
        placement_data['carer_age'],
        placement_data['carer_gender'],
        placement_data['carer_ethnicity'],
        placement_data['placement_type'],
        placement_data['uploaded_by']
    ))
    connection.commit()
    return cursor.lastrowid

def get_placement_by_id(connection, placement_id):
    """Get placement by ID"""
    query = "SELECT * FROM placements WHERE id = ?"
    cursor = connection.cursor()
    cursor.execute(query, (placement_id,))
    return cursor.fetchone()

def get_recent_placements(connection, limit=10):
    """Get recent placement records"""
    query = "SELECT * FROM placements ORDER BY created_at DESC LIMIT ?"
    cursor = connection.cursor()
    cursor.execute(query, (limit,))
    return cursor.fetchall()

def get_placement_statistics(connection):
    """Get placement statistics"""
    query = """
    SELECT 
        COUNT(*) as total_placements,
        SUM(breakdown_occurred) as total_breakdowns,
        AVG(placement_duration) as avg_duration,
        COUNT(DISTINCT placement_type) as placement_types
    FROM placements
    """
    cursor = connection.cursor()
    cursor.execute(query)
    return cursor.fetchone()

# ============== Prediction Functions ==============

def save_prediction(connection, form_data, predictions, user_id):
    """Save prediction to database"""
    import json
    query = """
    INSERT INTO predictions 
    (child_age, child_gender, child_ethnicity, carer_age, carer_gender, carer_ethnicity, 
     predicted_type, predicted_duration, stability_score, user_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

# ============== Analysis Functions ==============

def analyze_breakdown_patterns(connection):
    """Analyze placement breakdown patterns"""
    query = """
    SELECT 
        placement_type,
        COUNT(*) as total,
        SUM(breakdown_occurred) as breakdowns,
        CAST(SUM(breakdown_occurred) AS FLOAT) / COUNT(*) * 100 as breakdown_rate
    FROM placements
    GROUP BY placement_type
    ORDER BY breakdown_rate DESC
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
        SUM(breakdown_occurred) as breakdowns
    FROM placements
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
    WHERE breakdown_occurred = 1
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
        AVG(CASE WHEN breakdown_occurred = 0 THEN placement_duration END) as avg_stable_duration,
        AVG(CASE WHEN breakdown_occurred = 1 THEN placement_duration END) as avg_breakdown_duration,
        COUNT(CASE WHEN placement_duration > 365 THEN 1 END) as long_term_placements
    FROM placements
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

