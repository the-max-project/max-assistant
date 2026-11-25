import pytest


from tests.function.test_runner import execute_scenario_workflow
from tests.function.types import ScenarioResult

@pytest.mark.asyncio
async def test_execute_yaml_workflow(dynamic_scenario_data, model_name, request):
    """
    Executes a single test scenario sourced from YAML.
    The pytest_generate_tests hook populates 'dynamic_scenario_data'.
    """
    results: ScenarioResult = {}

    request.node.results = results
    await execute_scenario_workflow(
        username=dynamic_scenario_data["username"],
        steps=dynamic_scenario_data["steps"],
        model_name=model_name,
        scenario_request=request,
        results=results,
    )