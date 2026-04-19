import types


def test_create_app_sets_expected_config_and_blueprint_behavior(monkeypatch):
	monkeypatch.setenv("REMEMBER_COOKIE_DAYS", "10")
	monkeypatch.setenv("REMEMBER_COOKIE_SECURE", "1")
	monkeypatch.setenv("MAIL_PORT", "2525")
	monkeypatch.setenv("MAIL_USE_TLS", "0")

	from src.app import create_app

	app = create_app()
	assert app.config["REMEMBER_COOKIE_DURATION"].days == 10
	assert app.config["REMEMBER_COOKIE_SECURE"] is True
	assert app.config["MAIL_PORT"] == 2525
	assert app.config["MAIL_USE_TLS"] is False
	assert "app" in app.blueprints


def test_create_app_user_loader_uses_models_expected_behavior(monkeypatch):
	from src.app import create_app, login_manager

	fake_user = types.SimpleNamespace(id=1, username="u")

	def fake_create_connection(_db_path):
		return object()

	def fake_get_user_by_id(_conn, user_id):
		assert user_id == 1
		return fake_user

	monkeypatch.setattr("src.app.models.create_connection", fake_create_connection)
	monkeypatch.setattr("src.app.models.get_user_by_id", fake_get_user_by_id)

	app = create_app()
	with app.app_context():
		loaded = login_manager._user_callback("1")

	assert loaded is fake_user

