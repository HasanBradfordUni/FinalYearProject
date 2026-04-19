import runpy
import sys
import types


def test_run_module_creates_app_expected_behavior(project_root):
	created = {"called": False}

	def _create_app():
		created["called"] = True
		return object()

	fake_app_module = types.SimpleNamespace(create_app=_create_app)
	sys.modules["app"] = fake_app_module
	try:
		module_globals = runpy.run_path(str(project_root / "src" / "run.py"))
	finally:
		sys.modules.pop("app", None)

	assert created["called"] is True
	assert "app" in module_globals


def test_run_main_block_runs_server_expected_behavior(project_root):
	called = {"run": False}

	class FakeFlaskApp:
		def run(self, host=None, debug=None):
			called["run"] = True
			called["host"] = host
			called["debug"] = debug

	fake_app = FakeFlaskApp()
	fake_app_module = types.SimpleNamespace(create_app=lambda: fake_app)
	sys.modules["app"] = fake_app_module
	try:
		runpy.run_path(str(project_root / "src" / "run.py"), run_name="__main__")
	finally:
		sys.modules.pop("app", None)

	assert called["run"] is True
	assert called["host"] == "localhost"
	assert called["debug"] is True

