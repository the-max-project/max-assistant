import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize("enabled", [False, True])
def test_passing_stdout_is_opt_in(pytester, enabled):
    pytester.makeconftest('''
pytest_plugins = ["tests.function.conftest"]

import pytest

@pytest.fixture(scope="session", autouse=True)
def ensure_required_services_are_alive():
    pass

@pytest.fixture(scope="session", autouse=True)
def cleanup_ollama_models_at_session_end():
    pass
''')
    pytester.makepyfile('''
import pytest

@pytest.fixture
def output_fixture():
    print("SETUP_MARKER")
    yield
    print("TEARDOWN_MARKER")

def test_success(output_fixture):
    print("SUCCESS_MARKER")

def test_failure():
    print("FAILURE_MARKER")
    assert False
''')
    args = ["-q", "--tb=no", "--show-capture=no"]
    if enabled:
        args.append("--show-passed-stdout")
    result = pytester.runpytest(*args)
    result.assert_outcomes(passed=1, failed=1)
    output = result.stdout.str()
    for marker in ("SETUP_MARKER", "SUCCESS_MARKER", "TEARDOWN_MARKER"):
        assert (marker in output) is enabled
    assert "FAILURE_MARKER" not in output