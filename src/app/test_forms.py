from flask import Flask
from werkzeug.datastructures import MultiDict

from src.app.forms import (
    BulkUploadForm,
    ChangePasswordForm,
    ComparisonForm,
    ForgotPasswordForm,
    LoginForm,
    PlacementUploadForm,
    PredictionForm,
    ResetPasswordForm,
    UserEditForm,
    UserForm,
)


def _app_ctx():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "test"
    app.config["WTF_CSRF_ENABLED"] = False
    return app


def test_loginform_valid_data_passes_expected_behavior():
    app = _app_ctx()
    with app.test_request_context(method="POST"):
        form = LoginForm(formdata=MultiDict({"username": "abc", "password": "pass12345", "remember_me": "y"}))
        assert form.validate() is True


def test_forgotpasswordform_missing_identifier_fails_expected_behavior():
    app = _app_ctx()
    with app.test_request_context(method="POST"):
        form = ForgotPasswordForm(formdata=MultiDict({}))
        assert form.validate() is False


def test_changepasswordform_mismatch_confirmation_fails_expected_behavior():
    app = _app_ctx()
    with app.test_request_context(method="POST"):
        form = ChangePasswordForm(
            formdata=MultiDict(
                {
                    "current_password": "oldpass123",
                    "new_password": "newpass123",
                    "confirm_password": "different123",
                }
            )
        )
        assert form.validate() is False


def test_resetpasswordform_valid_data_passes_expected_behavior():
    app = _app_ctx()
    with app.test_request_context(method="POST"):
        form = ResetPasswordForm(formdata=MultiDict({"new_password": "newpass123", "confirm_password": "newpass123"}))
        assert form.validate() is True


def test_userform_invalid_email_fails_expected_behavior():
    app = _app_ctx()
    with app.test_request_context(method="POST"):
        form = UserForm(
            formdata=MultiDict(
                {
                    "username": "tester",
                    "email": "bad-email",
                    "password": "Password123",
                    "confirm_password": "Password123",
                    "role": "staff",
                }
            )
        )
        assert form.validate() is False


def test_usereditform_valid_data_passes_expected_behavior():
    app = _app_ctx()
    with app.test_request_context(method="POST"):
        form = UserEditForm(formdata=MultiDict({"username": "tester", "email": "t@example.com", "role": "manager", "is_active": "y"}))
        assert form.validate() is True


def test_placementuploadform_out_of_range_age_fails_expected_behavior():
    app = _app_ctx()
    with app.test_request_context(method="POST"):
        payload = {
            "child_age": "18",
            "child_gender": "Male",
            "child_ethnicity": "White - British",
            "child_prior_placements": "1",
            "returning_child": "False",
            "missing_episodes": "0",
            "sibling_group_size": "0",
            "placed_with_siblings": "False",
            "carer_age": "35",
            "carer_gender": "Female",
            "carer_ethnicity": "White - British",
            "eh_involvement": "False",
            "yot_involvement": "False",
            "placement_type": "Kinship",
        }
        form = PlacementUploadForm(formdata=MultiDict(payload))
        assert form.validate() is False


def test_bulkuploadform_missing_file_fails_expected_behavior():
    app = _app_ctx()
    with app.test_request_context(method="POST"):
        form = BulkUploadForm(formdata=MultiDict({}))
        assert form.validate() is False


def test_predictionform_optional_fields_allow_empty_expected_behavior():
    app = _app_ctx()
    with app.test_request_context(method="POST"):
        form = PredictionForm(formdata=MultiDict({}))
        assert form.validate() is True


def test_comparisonform_requires_placement_types_expected_behavior():
    app = _app_ctx()
    with app.test_request_context(method="POST"):
        form = ComparisonForm(formdata=MultiDict({}))
        assert form.validate() is False
