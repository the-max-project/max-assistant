import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import ToolException

from max_assistant.tools.playbook_tools import PlaybookTools


NOW = datetime(2026, 11, 17, 10, 0)
USER = {"user": {"id": "user-1"}}


@pytest.fixture
def mock_db() -> MagicMock:
    db = MagicMock()
    db.execute_query = AsyncMock()
    return db


@pytest.fixture
def mock_service() -> MagicMock:
    service = MagicMock()
    service.get_remaining_items = AsyncMock()
    service.create_and_schedule_task = AsyncMock()
    return service


@pytest.fixture
def tools(mock_db: MagicMock, mock_service: MagicMock) -> PlaybookTools:
    return PlaybookTools(mock_db, mock_service)


@pytest.fixture(autouse=True)
def fixed_clock():
    with patch("max_assistant.tools.playbook_tools.datetime", wraps=datetime) as clock:
        clock.now.return_value = NOW
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("result_tuple", [
    (None, []),
    (SimpleNamespace(morning_summary="Quiet day", items=[]), []),
])
async def test_empty_daily_plan(tools: PlaybookTools, mock_service: MagicMock, result_tuple):
    mock_service.get_remaining_items.return_value = result_tuple
    result = json.loads(await tools.get_remaining_daily_plans(USER))
    assert result == {"summary": "You have a completely free schedule for the rest of the day."}
    mock_service.get_remaining_items.assert_awaited_once_with("user-1", NOW)


@pytest.mark.asyncio
async def test_daily_plan_formats_remaining_items(tools: PlaybookTools, mock_service: MagicMock):
    playbook = SimpleNamespace(morning_summary="A busy day")
    items = [
        SimpleNamespace(event_time="11:00", title="Visit", category="REQUIRED", status="SCHEDULED"),
        SimpleNamespace(event_time="14:00", title="Lunch", category="MEAL", status="SCHEDULED"),
        SimpleNamespace(event_time=None, title="News", category="FAMILY_NEWS", status="QUEUED"),
        SimpleNamespace(event_time="", title="Message", category="FAMILY_MESSAGE", status="QUEUED"),
    ]
    mock_service.get_remaining_items.return_value = (playbook, items)

    result = json.loads(await tools.get_remaining_daily_plans(USER))
    assert result == {
        "daily_summary": "A busy day",
        "upcoming_events": [
            {"time": "11:00", "title": "Visit", "category": "REQUIRED", "status": "SCHEDULED"},
            {"time": "14:00", "title": "Lunch", "category": "MEAL", "status": "SCHEDULED"},
            {"time": "Anytime", "title": "News", "category": "FAMILY_NEWS", "status": "QUEUED"},
            {"time": "Anytime", "title": "Message", "category": "FAMILY_MESSAGE", "status": "QUEUED"},
        ],
        "instruction": (
            "Respond warmly using the daily_summary and upcoming_events. "
            "Do not mention internal statuses or IDs."
        ),
    }
    mock_service.get_remaining_items.assert_awaited_once_with("user-1", NOW)


@pytest.mark.asyncio
@pytest.mark.parametrize("user_info", [None, {}, {"user": {}}, {"user": {"id": ""}}])
@pytest.mark.parametrize("method", ["get_remaining_daily_plans", "respond_to_playbook_item"])
async def test_missing_user_rejected_before_side_effects(
    tools: PlaybookTools, mock_db: MagicMock, mock_service: MagicMock, user_info, method
):
    args = (user_info,) if method == "get_remaining_daily_plans" else ("item-1", True, user_info)
    with pytest.raises(ToolException, match="User ID is missing"):
        await getattr(tools, method)(*args)
    mock_db.execute_query.assert_not_awaited()
    mock_service.get_remaining_items.assert_not_awaited()
    mock_service.create_and_schedule_task.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [{}, {"data": []}])
async def test_item_not_found(tools: PlaybookTools, mock_db: MagicMock, mock_service: MagicMock, response):
    mock_db.execute_query.return_value = response
    assert json.loads(await tools.respond_to_playbook_item("missing", True, USER)) == {
        "error": "Playbook item not found."
    }
    mock_service.create_and_schedule_task.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("event_time", ["11:30:00", "11:30", " 11:30 am ", "11:30am"])
@pytest.mark.parametrize("lead_minutes", [None, 30])
async def test_accept_schedules_reminder_for_supported_times(
    tools: PlaybookTools, mock_db: MagicMock, mock_service: MagicMock, event_time, lead_minutes
):
    item: dict[str, Any] = {"title": "Music", "event_time": event_time}
    if lead_minutes is not None:
        item["lead_minutes"] = lead_minutes
    mock_db.execute_query.return_value = {"data": [item]}

    result = json.loads(await tools.respond_to_playbook_item("item-1", True, USER))

    due = datetime(2026, 11, 17, 11, 0 if lead_minutes == 30 else 15)
    assert result == {"success": True, "message": f"Activity accepted. Reminder set for {due:%I:%M %p}."}
    query, params = mock_db.execute_query.await_args.args
    assert "MATCH (u:User {id: $user_id})-[:HAS_PLAYBOOK]->(:Playbook)-[:HAS_ITEM]->(pi:PlaybookItem {id: $item_id})" in query
    assert "SET pi.status = $new_status" in query
    assert params == {
        "user_id": "user-1",
        "item_id": "item-1",
        "new_status": "ACCEPTED",
        "updated_at": NOW.isoformat(),
    }
    mock_service.create_and_schedule_task.assert_awaited_once_with(
        user_id="user-1", playbook_item_id="item-1",
        text="It is almost time for Music. It starts at 11:30 AM.",
        due_dt=due, validity_minutes=45, current_dt=NOW,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted,event_time,lead", [
    (False, "11:30", 15),
    (True, None, 15),
    (True, "", 15),
    (True, "invalid", 15),
    (True, "10:15", 15),
    (True, "10:14", 15),
])
async def test_no_reminder_for_declined_missing_invalid_or_nonfuture_due_time(
    tools: PlaybookTools, mock_db: MagicMock, mock_service: MagicMock, accepted, event_time, lead
):
    mock_db.execute_query.return_value = {"data": [
        {"title": "Music", "event_time": event_time, "lead_minutes": lead}
    ]}
    status = "ACCEPTED" if accepted else "DECLINED"
    assert json.loads(await tools.respond_to_playbook_item("item-1", accepted, USER)) == {
        "success": True, "message": f"Activity marked as {status}."
    }
    assert mock_db.execute_query.await_args.args[1] == {
        "user_id": "user-1",
        "item_id": "item-1",
        "new_status": status,
        "updated_at": NOW.isoformat(),
    }
    mock_service.create_and_schedule_task.assert_not_awaited()


def test_get_tools_registers_coroutines_and_error_handler(tools: PlaybookTools):
    registered = {tool.name: tool for tool in tools.get_tools()}
    assert set(registered) == {"get_remaining_daily_plans", "respond_to_playbook_item"}
    for name, tool in registered.items():
        assert tool.coroutine == getattr(tools, name)
        assert tool.handle_tool_error == tools.format_system_tool_error