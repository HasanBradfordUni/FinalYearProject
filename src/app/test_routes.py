import io
import json
import types

import pytest
from flask import Response
from flask_login import UserMixin

import src.app.routes as routes
from src.app import create_app, login_manager


class _User(UserMixin):
    def __init__(self, user_id, role="staff", username="tester", must_reset_password=False):
        self.id = user_id
        self.role = role
        self.username = username
        self.email = f"{username}@example.com"
        self.must_reset_password = must_reset_password
        self._active = True

    @property
    def is_active(self):
        return self._active


@pytest.fixture
def app_client(monkeypatch):
    app = create_app()
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False

    users = {}

    def _loader(user_id):
        return users.get(int(user_id))

    monkeypatch.setattr(login_manager, "_user_callback", _loader)
    monkeypatch.setattr(routes, "connection", object())
    monkeypatch.setattr(routes, "flash", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(routes, "render_template", lambda template, **_ctx: f"TEMPLATE:{template}")

    client = app.test_client()

    def login_as(role="staff", user_id=1, must_reset_password=False):
        users[user_id] = _User(user_id=user_id, role=role, username=f"u{user_id}", must_reset_password=must_reset_password)
        with client.session_transaction() as session:
            session["_user_id"] = str(user_id)
            session["_fresh"] = True
        return users[user_id]

    return app, client, login_as


def test_generate_temporary_password_expected_behavior():
    value = routes._generate_temporary_password(length=16)
    assert len(value) == 16
    assert all(ch not in "O0Il" for ch in value)


def test_send_temporary_password_email_without_server_returns_false_expected_behavior(app_client):
    app, _, _ = app_client
    with app.app_context():
        assert routes._send_temporary_password_email("a@b.com", "user", "Temp12345") is False


def test_load_models_without_artifacts_raises_expected_behavior(monkeypatch):
    monkeypatch.setattr(routes.os.path, "exists", lambda *_args, **_kwargs: False)
    with pytest.raises(FileNotFoundError):
        routes._load_models()


def test_login_get_and_failed_post_expected_behavior(app_client, monkeypatch):
    _, client, _ = app_client
    monkeypatch.setattr(routes, "authenticate_user", lambda *_args, **_kwargs: None)

    response_get = client.get("/login")
    response_post = client.post("/login", data={"username": "bad", "password": "bad"})

    assert response_get.status_code == 200
    assert response_post.status_code == 200


def test_login_post_success_redirects_dashboard_expected_behavior(app_client, monkeypatch):
    _, client, _ = app_client
    user = _User(99, role="staff", username="ok")
    monkeypatch.setattr(routes, "authenticate_user", lambda *_args, **_kwargs: user)

    response = client.post("/login", data={"username": "ok", "password": "pass123456", "remember_me": "y"})
    assert response.status_code == 302
    assert "/dashboard" in response.headers["Location"]


def test_logout_redirects_login_expected_behavior(app_client):
    _, client, login_as = app_client
    login_as("staff")
    response = client.get("/logout")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_forgot_password_get_and_post_behaviors_expected_behavior(app_client, monkeypatch):
    _, client, _ = app_client
    monkeypatch.setattr(routes, "get_user_by_identifier", lambda *_args, **_kwargs: None)
    response_get = client.get("/forgot-password")
    response_post_unknown = client.post("/forgot-password", data={"identifier": "unknown"})
    assert response_get.status_code == 200
    assert response_post_unknown.status_code == 302

    monkeypatch.setattr(routes, "get_user_by_identifier", lambda *_args, **_kwargs: {"id": 1, "username": "u1", "email": "u1@example.com"})
    monkeypatch.setattr(routes, "issue_temporary_password", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(routes, "_send_temporary_password_email", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(routes, "log_audit", lambda *_args, **_kwargs: None)
    response_post_known = client.post("/forgot-password", data={"identifier": "u1"})
    assert response_post_known.status_code == 200


def test_change_password_post_branches_expected_behavior(app_client, monkeypatch):
    _, client, login_as = app_client
    user = login_as("manager", user_id=2)
    monkeypatch.setattr(routes, "verify_user_password", lambda *_args, **_kwargs: False)
    response_bad = client.post(
        "/account/change-password",
        data={"current_password": "x", "new_password": "NewPassword1", "confirm_password": "NewPassword1"},
    )
    assert response_bad.status_code == 200

    monkeypatch.setattr(routes, "verify_user_password", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(routes, "update_user_password", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(routes, "log_audit", lambda *_args, **_kwargs: None)
    response_ok = client.post(
        "/account/change-password",
        data={"current_password": "x", "new_password": "NewPassword1", "confirm_password": "NewPassword1"},
    )
    assert user.id == 2
    assert response_ok.status_code == 302


def test_force_password_reset_behaviors_expected_behavior(app_client, monkeypatch):
    _, client, login_as = app_client
    login_as("staff", user_id=3, must_reset_password=False)
    response_no_reset = client.get("/reset-password-temp")
    assert response_no_reset.status_code == 302

    login_as("staff", user_id=4, must_reset_password=True)
    monkeypatch.setattr(routes, "update_user_password", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(routes, "log_audit", lambda *_args, **_kwargs: None)
    response_get = client.get("/reset-password-temp")
    response_post = client.post("/reset-password-temp", data={"new_password": "NewPassword1", "confirm_password": "NewPassword1"})
    assert response_get.status_code == 200
    assert response_post.status_code == 302


@pytest.mark.parametrize(
    "role,expected",
    [
        ("admin", "/admin-dashboard"),
        ("manager", "/manager-dashboard"),
        ("staff", "/staff-dashboard"),
    ],
)
def test_dashboard_redirects_by_role_expected_behavior(app_client, role, expected):
    _, client, login_as = app_client
    login_as(role, user_id=10)
    response = client.get("/dashboard")
    assert response.status_code == 302
    assert expected in response.headers["Location"]


def test_staff_manager_admin_dashboard_render_expected_behavior(app_client, monkeypatch):
    _, client, login_as = app_client
    monkeypatch.setattr(routes, "get_recent_placements", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(routes, "get_placement_statistics", lambda *_args, **_kwargs: {"total": 1})
    monkeypatch.setattr(routes, "analyze_breakdown_patterns", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(routes, "get_breakdown_patterns_by_duration", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(routes, "get_stability_trends", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(routes, "get_all_users", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(routes, "get_system_statistics", lambda *_args, **_kwargs: {"total_users": 1})
    monkeypatch.setattr(routes, "get_recent_audit_logs", lambda *_args, **_kwargs: [])

    login_as("staff", user_id=11)
    assert client.get("/staff-dashboard").status_code == 200
    login_as("manager", user_id=12)
    assert client.get("/manager-dashboard").status_code == 200
    login_as("admin", user_id=13)
    assert client.get("/admin-dashboard").status_code == 200


def test_upload_routes_expected_behavior(app_client, monkeypatch):
    _, client, login_as = app_client
    login_as("staff", user_id=20)
    monkeypatch.setattr(routes, "add_placement_record", lambda *_args, **_kwargs: 123)
    monkeypatch.setattr(routes, "log_audit", lambda *_args, **_kwargs: None)
    get_resp = client.get("/upload-placement")
    post_resp = client.post(
        "/upload-placement",
        data={
            "child_age": 10,
            "child_gender": "Male",
            "child_ethnicity": "White - British",
            "child_prior_placements": 1,
            "returning_child": "False",
            "missing_episodes": 0,
            "sibling_group_size": 0,
            "placed_with_siblings": "False",
            "carer_age": 40,
            "carer_gender": "Female",
            "carer_ethnicity": "White - British",
            "eh_involvement": "False",
            "yot_involvement": "False",
            "placement_type": "Kinship",
        },
    )
    assert get_resp.status_code == 200
    assert post_resp.status_code == 302

    monkeypatch.setattr(routes, "process_bulk_upload", lambda *_args, **_kwargs: {"success": 1, "failed": 0})
    bulk_post = client.post(
        "/upload-bulk",
        data={"csv_file": (io.BytesIO(b"a,b\n1,2"), "data.csv")},
        content_type="multipart/form-data",
    )
    assert bulk_post.status_code in {200, 302}


def test_view_placement_get_expected_behavior(app_client, monkeypatch):
    _, client, login_as = app_client
    login_as("staff", user_id=22)
    monkeypatch.setattr(routes, "get_placement_by_id", lambda *_args, **_kwargs: None)
    not_found = client.get("/placement/999")
    assert not_found.status_code == 302
    monkeypatch.setattr(routes, "get_placement_by_id", lambda *_args, **_kwargs: {"id": 1})
    found = client.get("/placement/1")
    assert found.status_code == 200


def test_predict_and_compare_routes_expected_behavior(app_client, monkeypatch):
    _, client, login_as = app_client
    login_as("staff", user_id=30)
    monkeypatch.setattr(routes, "prepare_prediction_input", lambda *_args, **_kwargs: [[1] * 15])
    monkeypatch.setattr(routes, "generate_predictions_list", lambda *_args, **_kwargs: [{"type": "Kinship", "duration": 10, "stability": 50, "breakdown_likelihood": 20, "net_stability": 30}])
    monkeypatch.setattr(routes, "save_prediction", lambda *_args, **_kwargs: 5)
    monkeypatch.setattr(routes, "log_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(routes, "save_comparison", lambda *_args, **_kwargs: 9)
    monkeypatch.setattr(routes, "extract_profile_from_form", lambda *_args, **_kwargs: {"child_age": 10})
    monkeypatch.setattr(routes, "compare_placement_options", lambda *_args, **_kwargs: [{"type": "Kinship"}, {"type": "External Fostering"}])

    predict_get = client.get("/predict")
    predict_post = client.post("/predict", data={"child_age": 10})
    assert predict_get.status_code == 200
    assert predict_post.status_code == 200

    compare_bad = client.post("/compare", data={"placement_types": ["Kinship"]})
    compare_ok = client.post("/compare", data={"placement_types": ["Kinship", "External Fostering"]})
    assert compare_bad.status_code == 200
    assert compare_ok.status_code == 200


def test_analysis_and_retrain_routes_expected_behavior(app_client, monkeypatch):
    _, client, login_as = app_client
    login_as("admin", user_id=40)
    monkeypatch.setattr(routes, "analyze_breakdown_patterns", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(routes, "get_breakdown_patterns_by_duration", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(routes, "identify_risk_factors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(routes, "generate_breakdown_recommendations", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(routes, "calculate_stability_metrics", lambda *_args, **_kwargs: {"avg": 1})
    monkeypatch.setattr(routes, "get_stability_trends", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(routes.train_models, "main", lambda: None)
    monkeypatch.setattr(routes, "_load_models", lambda: routes.model_assets)
    monkeypatch.setattr(routes, "log_audit", lambda *_args, **_kwargs: None)

    assert client.get("/breakdown-analysis").status_code == 200
    assert client.get("/stability-trends").status_code == 200
    retrain = client.post("/admin/retrain-models")
    assert retrain.status_code == 302


def test_export_routes_return_csv_expected_behavior(app_client, monkeypatch):
    _, client, login_as = app_client
    login_as("manager", user_id=50)
    monkeypatch.setattr(
        routes,
        "get_prediction_by_id",
        lambda *_args, **_kwargs: {
            "id": 1,
            "created_at": "2026-01-01",
            "predicted_type": "Kinship",
            "predicted_duration": 100,
            "stability_score": 70,
            "breakdown_likelihood": 20,
            "prediction_payload": json.dumps([{"type": "Kinship", "duration": 100, "stability": 70, "breakdown_likelihood": 20, "net_stability": 50}]),
        },
    )
    monkeypatch.setattr(
        routes,
        "get_comparison_by_id",
        lambda *_args, **_kwargs: {
            "id": 2,
            "created_at": "2026-01-01",
            "comparison_results": json.dumps([{"type": "Kinship", "duration": 10, "stability": 40, "breakdown_likelihood": 20, "net_stability": 20}]),
        },
    )
    monkeypatch.setattr(routes, "analyze_breakdown_patterns", lambda *_args, **_kwargs: [{"placement_type": "Kinship", "total": 1, "breakdowns": 0, "breakdown_rate": 0}])
    monkeypatch.setattr(routes, "get_breakdown_patterns_by_duration", lambda *_args, **_kwargs: [{"duration_band": "<1 year", "total": 1, "breakdowns": 0, "breakdown_rate": 0}])

    prediction_csv = client.get("/export/prediction/1.csv")
    comparison_csv = client.get("/export/comparison/2.csv")
    breakdown_csv = client.get("/export/breakdown-analysis.csv")

    assert prediction_csv.status_code == 200
    assert comparison_csv.status_code == 200
    assert breakdown_csv.status_code == 200
    assert "text/csv" in prediction_csv.content_type


def test_user_management_and_settings_routes_expected_behavior(app_client, monkeypatch):
    _, client, login_as = app_client
    login_as("admin", user_id=60)
    monkeypatch.setattr(routes, "get_all_users", lambda *_args, **_kwargs: [{"id": 1}])
    monkeypatch.setattr(routes, "create_user", lambda *_args, **_kwargs: 2)
    monkeypatch.setattr(routes, "update_user", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(routes, "delete_user_by_id", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(routes, "get_user_by_id", lambda *_args, **_kwargs: _User(70, role="staff"))
    monkeypatch.setattr(routes, "log_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(routes, "get_system_settings", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(routes, "update_system_settings", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(routes, "get_audit_logs_paginated", lambda *_args, **_kwargs: [])

    assert client.get("/users").status_code == 200
    assert client.get("/users/add").status_code == 200
    assert client.post(
        "/users/add",
        data={"username": "newuser", "email": "new@example.com", "password": "Password123", "confirm_password": "Password123", "role": "placement_officer"},
    ).status_code == 302
    assert client.get("/users/70/edit").status_code == 200
    assert client.post("/users/70/edit", data={"username": "u", "email": "u@example.com", "role": "placement_officer"}).status_code == 302
    assert client.post("/users/70/delete").status_code == 302

    assert client.get("/settings").status_code == 200
    assert client.post("/settings/update", data={"site_name": "BCFT", "new_setting_key": "k", "new_setting_value": "v"}).status_code == 302
    assert client.get("/audit-logs?page=1").status_code == 200

