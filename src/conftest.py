import types
from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class DummyUser:
    def __init__(self, user_id=1, role="staff", username="tester", must_reset_password=False):
        self.id = user_id
        self.role = role
        self.username = username
        self.email = f"{username}@example.com"
        self.must_reset_password = must_reset_password
        self.is_authenticated = True

    @property
    def is_active(self):
        return True


@pytest.fixture
def dummy_user_cls():
    return DummyUser


@pytest.fixture
def field():
    def _make(value):
        return types.SimpleNamespace(data=value)

    return _make
