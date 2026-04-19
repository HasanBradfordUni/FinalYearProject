import runpy
import sys
import types

import pytest


def _build_fake_modules(connection_obj, created_user_id=7):
    fake_models = types.ModuleType("app.models")
    fake_models.create_connection = lambda _path: connection_obj
    fake_models.create_tables = lambda _conn: None
    fake_models.create_user = lambda **_kwargs: created_user_id

    fake_app = types.ModuleType("app")
    fake_app.models = fake_models
    return fake_app, fake_models


def test_create_admin_creates_tables_and_user_expected_behavior(project_root, capsys):
    fake_connection = types.SimpleNamespace(close=lambda: None)
    fake_app, fake_models = _build_fake_modules(fake_connection, created_user_id=11)
    sys.modules["app"] = fake_app
    sys.modules["app.models"] = fake_models

    try:
        module_globals = runpy.run_path(str(project_root / "src" / "create_admin.py"))
        module_globals["create_admin"]()
    finally:
        sys.modules.pop("app", None)
        sys.modules.pop("app.models", None)

    out = capsys.readouterr().out
    assert "Creating database tables" in out
    assert "Admin user created successfully" in out


def test_create_admin_exits_when_connection_missing_expected_behavior(project_root):
    fake_app, fake_models = _build_fake_modules(None)
    sys.modules["app"] = fake_app
    sys.modules["app.models"] = fake_models

    try:
        module_globals = runpy.run_path(str(project_root / "src" / "create_admin.py"))
        with pytest.raises(SystemExit):
            module_globals["create_admin"]()
    finally:
        sys.modules.pop("app", None)
        sys.modules.pop("app.models", None)


def test_create_admin_main_closes_connection_expected_behavior(project_root):
    state = {"closed": False}

    class Conn:
        def close(self):
            state["closed"] = True

    fake_app, fake_models = _build_fake_modules(Conn())
    sys.modules["app"] = fake_app
    sys.modules["app.models"] = fake_models
    try:
        runpy.run_path(str(project_root / "src" / "create_admin.py"), run_name="__main__")
    finally:
        sys.modules.pop("app", None)
        sys.modules.pop("app.models", None)

    assert state["closed"] is True
