import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import ToolException

from max_assistant.tools.playbook_tools import PlaybookTools


NOW = datetime(2026, 11, 17, 10, 0)
USER = {"user": {"id": "user-1"}}


@pytest.fixture
def tools():
    db = MagicMock()
    db.execute_query = AsyncMock()
    service = MagicMock()
    service.generate_daily_playbook = AsyncMock()
    service.create_and_schedule_task = AsyncMock()
    return PlaybookTools(db, service)


@pytest.fixture(autouse=True)
def fixed_clock():
    with patch("max_assistant.tools.playbook_tools.datetime", wraps=datetime) as clock:
        clock.now.return_value = NOW
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("playbook", [None, SimpleNamespace(items=[])])
async def test_empty_daily_plan(tools, playbook):
    tools.playbook_service.generate_daily_playbook.return_value = playbook
    result = json.loads(await tools.get_remaining_daily_plans(USER))
    assert result == {"summary": "You have a completely free schedule for the rest of the day."}
    tools.playbook_service.generate_daily_playbook.assert_awaited_once_with("user-1", NOW)


@pytest.mark.asyncio
async def test_daily_plan_sorts_and_formats_items(tools):
    tools.playbook_service.generate_daily_playbook.return_value = SimpleNamespace(
        morning_summary="A busy day", items=[
            SimpleNamespace(event_time=None, title="News", category="FAMILY_NEWS"),
            SimpleNamespace(event_time="14:00", title="Lunch", category="MEAL"),
            SimpleNamespace(event_time="09:00", title="Visit", category="REQUIRED"),
            SimpleNamespace(event_time="", title="Message", category="FAMILY_MESSAGE"),
        ])
    result = json.loads(await tools.get_remaining_daily_plans(USER))
    assert result == {
        "daily_summary": "A busy day",
        "upcoming_events": [
            {"time": "09:00", "title": "Visit", "category": "REQUIRED"},
            {"time": "14:00", "title": "Lunch", "category": "MEAL"},
            {"time": "Anytime", "title": "News", "category": "FAMILY_NEWS"},
            {"time": "Anytime", "title": "Message", "category": "FAMILY_MESSAGE"},
        ],
        "instruction": "Respond warmly using the daily_summary. You may mention the upcoming_events.",
    }
    tools.playbook_service.generate_daily_playbook.assert_awaited_once_with("user-1", NOW)


@pytest.mark.asyncio
@pytest.mark.parametrize("user_info", [None, {}, {"user": {}}, {"user": {"id": ""}}])
@pytest.mark.parametrize("method", ["get_remaining_daily_plans", "respond_to_playbook_item"])
async def test_missing_user_rejected_before_side_effects(tools, user_info, method):
    args = (user_info,) if method == "get_remaining_daily_plans" else ("item-1", True, user_info)
    with pytest.raises(ToolException, match="User ID is missing"):
        await getattr(tools, method)(*args)
    tools.db_client.execute_query.assert_not_awaited()
    tools.playbook_service.generate_daily_playbook.assert_not_awaited()
    tools.playbook_service.create_and_schedule_task.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [{}, {"data": []}])
async def test_item_not_found(tools, response):
    tools.db_client.execute_query.return_value = response
    assert json.loads(await tools.respond_to_playbook_item("missing", True, USER)) == {
        "error": "Playbook item not found."
    }
    tools.playbook_service.create_and_schedule_task.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("event_time", ["11:30:00", "11:30", " 11:30 am ", "11:30am"])
@pytest.mark.parametrize("lead_minutes", [None, 30])
async def test_accept_schedules_reminder_for_supported_times(tools, event_time, lead_minutes):
    item = {"title": "Music", "event_time": event_time}
    if lead_minutes is not None:
        item["lead_minutes"] = lead_minutes
    tools.db_client.execute_query.return_value = {"data": [item]}

    result = json.loads(await tools.respond_to_playbook_item("item-1", True, USER))

    due = datetime(2026, 11, 17, 11, 0 if lead_minutes == 30 else 15)
    assert result == {"success": True, "message": f"Activity accepted. Reminder set for {due:%I:%M %p}."}
    query, params = tools.db_client.execute_query.await_args.args
    assert "MATCH (pi:PlaybookItem {id: $item_id})" in query
    assert "SET pi.status = $new_status" in query
    assert params == {"item_id": "item-1", "new_status": "ACCEPTED", "updated_at": NOW.isoformat()}
    tools.playbook_service.create_and_schedule_task.assert_awaited_once_with(
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
async def test_no_reminder_for_declined_missing_invalid_or_nonfuture_due_time(tools, accepted, event_time, lead):
    tools.db_client.execute_query.return_value = {"data": [
        {"title": "Music", "event_time": event_time, "lead_minutes": lead}
    ]}
    status = "ACCEPTED" if accepted else "DECLINED"
    assert json.loads(await tools.respond_to_playbook_item("item-1", accepted, USER)) == {
        "success": True, "message": f"Activity marked as {status}."
    }
    assert tools.db_client.execute_query.await_args.args[1] == {
        "item_id": "item-1", "new_status": status, "updated_at": NOW.isoformat()
    }
    tools.playbook_service.create_and_schedule_task.assert_not_awaited()


def test_get_tools_registers_coroutines_and_error_handler(tools):
    registered = {tool.name: tool for tool in tools.get_tools()}
    assert set(registered) == {"get_remaining_daily_plans", "respond_to_playbook_item"}
    for name, tool in registered.items():
        assert tool.coroutine == getattr(tools, name)
        assert tool.handle_tool_error == tools.format_system_tool_error