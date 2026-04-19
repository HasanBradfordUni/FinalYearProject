import sqlite3

from src.app import models


def _seed_user(conn, username="user1", email="user1@example.com", password="Password123", role="staff"):
    return models.create_user(conn, username, email, password, role)


def _seed_placement(conn, uploaded_by=1, placement_type="Kinship", duration=100, breakdown=0):
    return models.add_placement_record(
        conn,
        {
            "child_age": 12,
            "child_gender": "Male",
            "child_ethnicity": "White - British",
            "child_prior_placements": 1,
            "returning_child": 0,
            "missing_episodes": 0,
            "sibling_group_size": 1,
            "placed_with_siblings": 1,
            "carer_age": 40,
            "carer_gender": "Female",
            "carer_ethnicity": "White - British",
            "placement_type": placement_type,
            "placement_start_date": "2024-01-01",
            "move_date": "2024-03-01",
            "move_reason": "planned move",
            "distance_from_home": 10,
            "eh_involvement": 0,
            "yot_involvement": 0,
            "placement_sequence_number": 2,
            "placement_duration": duration,
            "breakdown_occurred": breakdown,
            "uploaded_by": uploaded_by,
        },
    )


def _fresh_connection():
    conn = models.create_connection(":memory:")
    models.create_tables(conn)
    return conn


def test_create_connection_returns_sqlite_connection_expected_behavior():
    conn = models.create_connection(":memory:")
    assert isinstance(conn, sqlite3.Connection)


def test_execute_query_invalid_sql_returns_none_expected_behavior():
    conn = _fresh_connection()
    assert models.execute_query(conn, "BAD SQL") is None


def test_user_crud_and_auth_functions_expected_behavior():
    conn = _fresh_connection()
    user_id = _seed_user(conn, "staff1", "staff1@example.com")

    authed = models.authenticate_user(conn, "staff1", "Password123")
    assert authed is not None
    assert models.get_user_by_id(conn, user_id).username == "staff1"

    row = models.get_user_by_identifier(conn, "staff1@example.com")
    assert row["id"] == user_id
    assert models.verify_user_password(conn, user_id, "Password123") is True

    models.update_user_password(conn, user_id, "NewPassword123")
    assert models.verify_user_password(conn, user_id, "NewPassword123") is True

    models.issue_temporary_password(conn, user_id, "TempPass123")
    user_after_temp = models.authenticate_user(conn, "staff1", "TempPass123")
    assert user_after_temp.must_reset_password is True

    all_users = models.get_all_users(conn)
    assert len(all_users) == 1

    models.update_user(conn, user_id, {"username": "staff2", "email": "s2@example.com", "role": "manager", "is_active": 1})
    assert models.get_user_by_identifier(conn, "staff2") is not None

    models.delete_user_by_id(conn, user_id)
    assert models.get_user_by_id(conn, user_id) is None


def test_placement_and_prediction_persistence_functions_expected_behavior():
    conn = _fresh_connection()
    uploader = _seed_user(conn, "uploader", "up@example.com")
    placement_id = _seed_placement(conn, uploaded_by=uploader, duration=300)
    assert models.get_placement_by_id(conn, placement_id)["id"] == placement_id
    assert len(models.get_recent_placements(conn, limit=5)) >= 1
    stats = models.get_placement_statistics(conn)
    assert stats["total_placements"] >= 1

    prediction_id = models.save_prediction(
        conn,
        {"child_age": 12, "child_gender": "Male", "child_ethnicity": "White - British", "carer_age": 40, "carer_gender": "Female", "carer_ethnicity": "White - British"},
        [{"type": "Kinship", "duration": 100, "stability": 70.0, "breakdown_likelihood": 10.0}],
        uploader,
    )
    comparison_id = models.save_comparison(conn, {"child_age": 12}, [{"type": "Kinship"}], uploader)
    assert models.get_prediction_by_id(conn, prediction_id)["id"] == prediction_id
    assert models.get_comparison_by_id(conn, comparison_id)["id"] == comparison_id


def test_analysis_and_system_functions_expected_behavior():
    conn = _fresh_connection()
    admin_id = _seed_user(conn, "admin1", "admin1@example.com", role="admin")
    _seed_placement(conn, uploaded_by=admin_id, placement_type="Kinship", duration=100, breakdown=1)
    _seed_placement(conn, uploaded_by=admin_id, placement_type="External Fostering", duration=500, breakdown=0)

    assert len(models.analyze_breakdown_patterns(conn)) >= 1
    assert len(models.get_breakdown_patterns_by_duration(conn)) >= 1
    assert len(models.get_stability_trends(conn)) >= 1
    assert isinstance(models.identify_risk_factors(conn), list)
    assert models.calculate_stability_metrics(conn) is not None

    models.log_audit(conn, admin_id, "action", {"x": 1})
    assert len(models.get_recent_audit_logs(conn, limit=5)) >= 1
    assert isinstance(models.get_audit_logs_paginated(conn, page=1, per_page=10), list)

    sys_stats = models.get_system_statistics(conn)
    assert sys_stats["total_users"] >= 1

    models.update_system_settings(conn, {"theme": "dark", "max_upload": "100"})
    settings = models.get_system_settings(conn)
    assert any(s["setting_key"] == "theme" for s in settings)


def test_generate_breakdown_recommendations_returns_severity_expected_behavior():
    data = [
        {"placement_type": "A", "breakdown_rate": 40.0},
        {"placement_type": "B", "breakdown_rate": 20.0},
        {"placement_type": "C", "breakdown_rate": 10.0},
    ]
    recommendations = models.generate_breakdown_recommendations(data)
    severities = {r["severity"] for r in recommendations}
    assert severities == {"high", "medium"}


def test_ensure_column_adds_missing_column_expected_behavior():
    conn = _fresh_connection()
    models._ensure_column(conn, "placements", "new_debug_col", "TEXT")
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(placements)")
    cols = {row[1] for row in cursor.fetchall()}
    assert "new_debug_col" in cols
