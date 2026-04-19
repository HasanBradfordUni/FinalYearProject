def test_tests_package_imports_expected_behavior():
    import src.tests

    assert src.tests is not None
