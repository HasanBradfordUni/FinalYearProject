from src.conftest import DummyUser


def test_dummy_user_is_active_expected_behavior():
    user = DummyUser(user_id=1, role="staff", username="alice")
    assert user.is_active is True
    assert user.username == "alice"
