# tests/conftest.py
import os
import json
import socket
from urllib import request
import pytest
from pathlib import Path
from datetime import datetime

import yaml
from dotenv import load_dotenv

# Find the workspace root based on this file's position
TESTS_DIR = Path(__file__).parent.resolve()
WORKSPACE_ROOT = TESTS_DIR.parent
SCENARIOS_DIR = TESTS_DIR / 'ro_scenarios'

# set up filename to record results
RESULTS_DIR = TESTS_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"test_run_{timestamp}.jsonl"
RESULTS_FILE = RESULTS_DIR / filename

# Initialize local env variables before importing local files
local_env = WORKSPACE_ROOT / '..' / '.env.local'

if local_env.exists():
    load_dotenv(local_env)
else:
    print(f"[TEST SETUP] Warning: Root env file not found at {local_env}")

from tests.function.validators import build_steps_from_yaml
from max_assistant.config import OLLAMA_MODEL_NAME, NEO4J_URI, OLLAMA_BASE_URL

# Define models to test (override via env var or list here)
MODELS_TO_TEST = [
    model.strip()
    for model in os.getenv("TEST_MODELS", OLLAMA_MODEL_NAME).split(",")
]

SEMANTIC_VALIDATOR_MODEL = os.getenv("SEMANTIC_VALIDATOR_MODEL")


def verify_neo4j_connectivity(uri: str) -> bool:
    # noinspection PyBroadException
    try:
        clean_address = uri.replace("bolt://", "").replace("neo4j://", "")
        host, port = clean_address.split(":")
        with socket.create_connection((host, int(port)), timeout=2.0):
            return True
    except Exception:
        return False


def verify_ollama_connectivity(url: str) -> bool:
    # noinspection PyBroadException
    try:
        base_url = url.rstrip("/")
        with request.urlopen(f"{base_url}/", timeout=2.0) as response:
            return response.getcode() == 200
    except Exception:
        return False


def verify_ollama_model_pulled(url: str, model_name: str) -> bool:
    """Hits the local Ollama registry catalog to ensure the requested model is pulled."""
    # noinspection PyBroadException
    try:
        base_url = url.rstrip("/")
        # Hit Ollama's local tags listing endpoint
        with request.urlopen(f"{base_url}/api/tags", timeout=3.0) as response:
            if response.getcode() != 200:
                return False

            data = json.loads(response.read().decode())
            # Extract names from the list of downloaded models
            local_models = [model["name"] for model in data.get("models", [])]

            # Check for direct match or tagless match (Ollama assumes :latest if omitted)
            if model_name in local_models:
                return True
            if ":" not in model_name and f"{model_name}:latest" in local_models:
                return True

            return False
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def ensure_required_services_are_alive():
    """Pre-flight environment sanity validator."""
    print("\n[PRE-FLIGHT] Auditing required infrastructure services...")

    # 1. Audit Neo4j Connection
    if not verify_neo4j_connectivity(NEO4J_URI):
        pytest.exit(
            f"\n❌ ENVIRONMENT FAILURE: Cannot link to Neo4j instance at {NEO4J_URI}.\n"
            f"Please verify your database service or container is active locally.\n",
            returncode=1
        )

    # 2. Audit Ollama Server Connection
    if not verify_ollama_connectivity(OLLAMA_BASE_URL):
        pytest.exit(
            f"\n❌ ENVIRONMENT FAILURE: Cannot communicate with Ollama service at {OLLAMA_BASE_URL}.\n"
            f"Please run `ollama serve` to boot up the endpoint.\n",
            returncode=1
        )

    # 3. Audit Target LLM Availability
    missing_models = [m for m in MODELS_TO_TEST if not verify_ollama_model_pulled(OLLAMA_BASE_URL, m)]
    if missing_models:
        pytest.exit(
            f"\n❌ The following models are not pulled in Ollama: {missing_models}\n"
            f"Please run `ollama pull <model>` for missing models.\n",
            returncode=1
        )

    print("[PRE-FLIGHT] Environment checks healthy. Launching framework test grid.\n")


def pytest_generate_tests(metafunc):
    """
        This pytest hook performs the following steps:
        - Discovery: It scans the tests/function/ro_scenarios/ directory for all .yaml and .yml files.
        - Ordering: It explicitly sorts these files alphabetically to ensure a consistent execution order.
        - Data Extraction: It opens each YAML file and iterates through the scenarios defined inside.
        - Processing: For each model: runs all scenarios for that model.
                For each scenario, it:
                - Extracts the username.
                - Processes the steps using a build_steps_from_yaml utility.
                - Creates a scenario_payload.
        - Injection: It uses metafunc.parametrize("dynamic_scenario_data", argvalues, ids=ids) to inject this data into
                 any test function that requests the dynamic_scenario_data fixture.
    """
    # This will print out exactly what pytest considers the file/test targets
    print(f"\nDEBUG: Pytest targets: {metafunc.config.args}")
    print(f"\nDEBUG: Full option dict: {metafunc.config.option}")

    # 1. Parametrize Models
    if "model_name" in metafunc.fixturenames:
        metafunc.parametrize("model_name", MODELS_TO_TEST)

    # 2. Parametrize YAML Scenarios
    if "dynamic_scenario_data" in metafunc.fixturenames:
        # Fetch files and sort them alphabetically
        raw_files = list(SCENARIOS_DIR.glob("*.yaml")) + list(SCENARIOS_DIR.glob("*.yml"))
        yaml_files = sorted(raw_files, key=lambda p: p.name)  # <--- ENFORCES ALPHABETICAL ORDER

        argvalues = []
        ids = []

        for file_path in yaml_files:
            with open(file_path, "r") as f:
                raw_data = yaml.safe_load(f) or {}

            # Loops through keys exactly in the order they appear inside the YAML file
            for scenario_key, data in raw_data.items():
                processed_steps = build_steps_from_yaml(data["steps"])
                scenario_payload = {
                    "username": data["username"],
                    "steps": processed_steps
                }
                argvalues.append(scenario_payload)
                ids.append(f"{file_path.stem}-{scenario_key}")

        metafunc.parametrize("dynamic_scenario_data", argvalues, ids=ids)


# noinspection PyUnusedLocal
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    # We only want to process the data when the main test execution phase finishes
    if report.when == "call":

        # Extract the dictionary you attached in the test function
        results = getattr(item, "results", None)

        if results:
            # noinspection PyUnresolvedReferences
            results["pytest_status"] = report.outcome

            # Append the dictionary to the JSONL file safely
            with open(RESULTS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(results) + "\n")