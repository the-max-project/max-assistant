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
    def _mock_scheduler():
        return SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), wake=MagicMock())

    agent = MagicMock()
    agent.get_thread_id.return_value = "thread"
    agent.handle_push_event = AsyncMock(return_value="Let's talk about gardening")
    bus = LocalEventBus()
    db = MagicMock()
    db.close = AsyncMock()
    db.execute_query = AsyncMock(side_effect=[
        {"data": [{"presence": "present", "sleep_state": "awake"}]},
        {"data": [{"item": {"category": "OPTIONAL_ACTIVITY", "title": "gardening", "id": "item-1"}}]},
        {"data": []},
    ])
    services = SimpleNamespace(event_bus=bus, db_client=db, reasoning_engine=object(),
                               llm_ready_event=asyncio.Event(), time_scheduler=_mock_scheduler())
    services.llm_ready_event.set()
    monkeypatch.setattr(test_runner.AppServices, "create", AsyncMock(return_value=services))
    monkeypatch.setattr(test_runner, "Agent", MagicMock(return_value=agent))
    monkeypatch.setattr(test_runner, "PersonTools", lambda db: SimpleNamespace(
        get_user_info_internal=AsyncMock(return_value={"user": {"id": "user-1"}, "location": None})))
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
        "user", steps, "model", SimpleNamespace(node=SimpleNamespace(name="scenario"),
                                                config=SimpleNamespace(getoption=MagicMock(return_value=False))), results)

    payload = agent.handle_push_event.await_args.args[0]
    assert "gardening" in payload["text"]
    assert "item-1" in payload["text"]
    assert db.execute_query.await_args_list[0].args[1] == {"user_id": "user-1"}
    assert "PROMPTED" in db.execute_query.await_args_list[2].args[0]
    assert db.execute_query.await_args_list[2].args[1]["id"] == "item-1"
    test_runner.session_manager.register.assert_called_once_with("user-1", agent)
    test_runner.session_manager.unregister.assert_called_once_with("user-1", agent)
    assert results["step_results"][-1]["actual_result"] == "Let's talk about gardening"


@pytest.mark.asyncio
async def test_wait_for_response_times_out(monkeypatch):
    def _mock_scheduler():
        return SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), wake=MagicMock())

    agent = MagicMock()
    agent.get_thread_id.return_value = "thread"
    db = MagicMock()
    db.close = AsyncMock()
    ready = asyncio.Event()
    ready.set()
    services = SimpleNamespace(event_bus=MagicMock(), db_client=db, reasoning_engine=object(),
                               llm_ready_event=ready, time_scheduler=_mock_scheduler())
    monkeypatch.setattr(test_runner.AppServices, "create", AsyncMock(return_value=services))
    monkeypatch.setattr(test_runner, "Agent", MagicMock(return_value=agent))
    monkeypatch.setattr(test_runner, "PersonTools", lambda db: SimpleNamespace(
        get_user_info_internal=AsyncMock(return_value={"user": {"id": "user-1"}, "location": None})))
    monkeypatch.setattr(test_runner.session_manager, "register", MagicMock())
    monkeypatch.setattr(test_runner.session_manager, "unregister", MagicMock())
    steps = build_steps_from_yaml([{"wait_for_response": {"timeout": 0.01}}])
    with pytest.raises(TimeoutError, match="response"):
        await test_runner.execute_scenario_workflow(
            "user", steps, "model", SimpleNamespace(node=SimpleNamespace(name="scenario"),
                                                    config=SimpleNamespace(getoption=MagicMock(return_value=False))))
    services.time_scheduler.stop.assert_awaited_once()
    db.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("user_data", [
    {"error": "User not found", "details": "No matching user"},
    {},
    {"user": None},
    {"user": {}},
    {"user": {"id": ""}},
])
async def test_missing_user_id_aborts_scenario_and_cleans_up(monkeypatch, user_data):
    def _mock_scheduler():
        return SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), wake=MagicMock())

    db = MagicMock()
    db.close = AsyncMock()
    ready = asyncio.Event()
    ready.set()
    services = SimpleNamespace(event_bus=MagicMock(), db_client=db, reasoning_engine=object(),
                               llm_ready_event=ready, time_scheduler=_mock_scheduler())
    monkeypatch.setattr(test_runner.AppServices, "create", AsyncMock(return_value=services))
    agent_factory = MagicMock()
    monkeypatch.setattr(test_runner, "Agent", agent_factory)
    monkeypatch.setattr(test_runner, "PersonTools", lambda db: SimpleNamespace(
        get_user_info_internal=AsyncMock(return_value=user_data)))
    monkeypatch.setattr(test_runner.session_manager, "register", MagicMock())
    monkeypatch.setattr(test_runner.session_manager, "unregister", MagicMock())
    results = {}

    with pytest.raises(ValueError, match="user.*alice|alice.*user"):
        await test_runner.execute_scenario_workflow(
            "alice", [{"user_input": "hello"}], "model",
            SimpleNamespace(node=SimpleNamespace(name="scenario")), results)

    agent_factory.assert_not_called()
    test_runner.session_manager.register.assert_not_called()
    test_runner.session_manager.unregister.assert_not_called()
    db.close.assert_awaited_once()
    assert results["success"] is False
    assert results["step_results"] == []