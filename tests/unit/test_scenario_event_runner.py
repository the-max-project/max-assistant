import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from max_assistant.events.bus import LocalEventBus
from max_assistant.services.engagement_manager import EngagementManager
from tests.function import test_runner
from tests.function.validators import build_steps_from_yaml


@pytest.mark.asyncio
async def test_event_injection_waits_for_agent_response(monkeypatch):
    agent = MagicMock()
    agent.get_thread_id.return_value = "thread"
    agent.handle_push_event = AsyncMock(return_value="Let's talk about gardening")
    bus = LocalEventBus()
    db = MagicMock()
    db.close = AsyncMock()
    db.execute_query = AsyncMock(side_effect=[
        {"data": [{"presence": "present", "sleep_state": "awake"}]},
        {"data": [{"item": {"category": "OPTIONAL_ACTIVITY", "title": "gardening", "id": "item-1"}}]},
    ])
    services = SimpleNamespace(event_bus=bus, db_client=db, reasoning_engine=object(),
                               llm_ready_event=asyncio.Event())
    services.llm_ready_event.set()
    monkeypatch.setattr(test_runner.AppServices, "create", AsyncMock(return_value=services))
    monkeypatch.setattr(test_runner, "Agent", MagicMock(return_value=agent))
    monkeypatch.setattr(test_runner, "PersonTools", lambda db: SimpleNamespace(
        get_user_info_internal=AsyncMock(return_value={"id": "user-1"})))
    monkeypatch.setattr(test_runner.session_manager, "register", MagicMock())
    monkeypatch.setattr(test_runner.session_manager, "unregister", MagicMock())
    sessions = SimpleNamespace(get_agents=MagicMock(return_value=[agent]))
    router = SimpleNamespace(is_user_in_active_turn=MagicMock(return_value=False))
    EngagementManager(bus, sessions, db, router)
    steps = build_steps_from_yaml([
        {"inject_event": {"topic": "user.engagement.idle", "settle_time": 0}},
        {"wait_for_response": {"timeout": 0.1},
         "validators": [{"substring_present": "gardening"}]},
    ])
    results = {}
    await test_runner.execute_scenario_workflow(
        "user", steps, "model", SimpleNamespace(node=SimpleNamespace(name="scenario")), results)

    payload = agent.handle_push_event.await_args.args[0]
    assert "gardening" in payload["text"]
    assert "item-1" in payload["text"]
    assert db.execute_query.await_args_list[0].args[1] == {"user_id": "user-1"}
    assert results["step_results"][-1]["actual_result"] == "Let's talk about gardening"


@pytest.mark.asyncio
async def test_wait_for_response_times_out(monkeypatch):
    agent = MagicMock()
    agent.get_thread_id.return_value = "thread"
    db = MagicMock()
    db.close = AsyncMock()
    ready = asyncio.Event()
    ready.set()
    services = SimpleNamespace(event_bus=MagicMock(), db_client=db, reasoning_engine=object(),
                               llm_ready_event=ready)
    monkeypatch.setattr(test_runner.AppServices, "create", AsyncMock(return_value=services))
    monkeypatch.setattr(test_runner, "Agent", MagicMock(return_value=agent))
    monkeypatch.setattr(test_runner, "PersonTools", lambda db: SimpleNamespace(
        get_user_info_internal=AsyncMock(return_value={"id": "user-1"})))
    monkeypatch.setattr(test_runner.session_manager, "register", MagicMock())
    monkeypatch.setattr(test_runner.session_manager, "unregister", MagicMock())
    steps = build_steps_from_yaml([{"wait_for_response": {"timeout": 0.01}}])
    with pytest.raises(TimeoutError, match="response"):
        await test_runner.execute_scenario_workflow(
            "user", steps, "model", SimpleNamespace(node=SimpleNamespace(name="scenario")))