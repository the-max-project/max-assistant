import pytest
from contextlib import nullcontext
from freezegun import freeze_time

from max_assistant.clients.neo4j_client import Neo4jClient
from max_assistant.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from tests.function.test_runner import execute_scenario_workflow
from tests.function.types import ScenarioResult


async def _cleanup_stale_playbook_tasks() -> None:
    """Removes leftover PENDING playbook tasks from previous test runs."""
    async with await Neo4jClient.create(
        uri=NEO4J_URI,
        user=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    ) as db_client:
        await db_client.execute_query(
            "MATCH (t:Task {source: 'PLAYBOOK', status: 'PENDING'}) DETACH DELETE t",
            {},
        )


@pytest.mark.asyncio
async def test_execute_playbook_yaml_workflow(playbook_scenario_data, model_name, request):
    """
    Executes playbook test scenarios sourced from the playbook_scenarios directory.
    """
    await _cleanup_stale_playbook_tasks()

    results: ScenarioResult = {}
    request.node.results = results

    frozen_time = playbook_scenario_data.get("freeze_time")
    time_ctx = freeze_time(frozen_time, tick=True, ignore=["logging"]) if frozen_time else nullcontext()

    with time_ctx as frozen_clock:
        await execute_scenario_workflow(
            username=playbook_scenario_data["username"],
            steps=playbook_scenario_data["steps"],
            model_name=model_name,
            scenario_request=request,
            results=results,
            frozen_time=frozen_clock,
        )