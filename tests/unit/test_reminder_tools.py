import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import ToolException

from max_assistant.clients.neo4j_client import Neo4jCircuitBreakerError, Neo4jClientError
from max_assistant.tools import reminder_tools
from max_assistant.tools.reminder_tools import ReminderTools


NOW = datetime(2026, 12, 31, 13, 30)


@pytest.fixture
def tools(monkeypatch):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    monkeypatch.setattr(reminder_tools, "datetime", FixedDatetime)
    db = MagicMock()
    db.execute_query = AsyncMock(return_value={"data": [{"task_id": "task-1"}]})
    scheduler = MagicMock()
    scheduler.schedule = AsyncMock()
    return ReminderTools(db, scheduler)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" 2:15 pm ", datetime(2026, 12, 31, 14, 15)),
        ("2:15PM", datetime(2026, 12, 31, 14, 15)),
        ("2 PM", datetime(2026, 12, 31, 14)),
        ("14:15:30", datetime(2026, 12, 31, 14, 15, 30)),
        ("14:15", datetime(2026, 12, 31, 14, 15)),
        ("13:30", datetime(2027, 1, 1, 13, 30)),
        ("08:00", datetime(2027, 1, 1, 8)),
        ("2025-01-02T03:04:05", datetime(2025, 1, 2, 3, 4, 5)),
        ("2025-01-02 03:04:05", datetime(2025, 1, 2, 3, 4, 5)),
        ("2025-01-02 03:04", datetime(2025, 1, 2, 3, 4)),
        ("25:00", None),
        ("tomorrow", None),
        ("", None),
    ],
)
def test_parse_target_time(value, expected):
    assert ReminderTools._parse_target_time(value, NOW) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("time_args", "due", "wake"),
    [
        ({"delay_minutes": 1.5}, datetime(2026, 12, 31, 13, 31, 30), True),
        ({"target_time_str": "08:00"}, datetime(2027, 1, 1, 8), False),
        ({"delay_minutes": 0, "target_time_str": "invalid"}, NOW, False),
    ],
)
async def test_schedule_reminder_persists_and_registers(tools, sample_user_info, time_args, due, wake):
    result = json.loads(await tools.schedule_reminder(
        "Take medicine", 2.5, wake, sample_user_info, **time_args
    ))
    expires = due.timestamp() + 150
    query, params = tools.db_client.execute_query.await_args.args
    assert "CREATE (t:Task" in query
    assert "'REMINDER'" in query and "'PENDING'" in query
    assert params == {
        "text": "Take medicine",
        "due_time": due.isoformat(),
        "expires_at": datetime.fromtimestamp(expires).isoformat(),
        "can_wake_user": wake,
        "user_id": "1",
    }
    tools.time_scheduler.schedule.assert_awaited_once_with(
        task_id="task-1", user_id="1", text="Take medicine",
        due_epoch=due.timestamp(), expires_epoch=expires, can_wake_user=wake,
    )
    assert result == {
        "success": True,
        "task_id": "task-1",
        "message": f"Successfully scheduled reminder for {due.strftime('%I:%M %p').lstrip('0')}: 'Take medicine'",
        "due_time": params["due_time"],
        "expires_at": params["expires_at"],
        "can_wake_user": wake,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("time_args", "error"),
    [({}, "Missing_Time_Parameter"), ({"target_time_str": ""}, "Missing_Time_Parameter"),
     ({"target_time_str": "not a time"}, "Invalid_Time_Format")],
)
async def test_invalid_time_does_not_write(tools, sample_user_info, time_args, error):
    result = json.loads(await tools.schedule_reminder("test", 10, False, sample_user_info, **time_args))
    assert result["error"] == error
    assert result["instruction"]
    tools.db_client.execute_query.assert_not_awaited()
    tools.time_scheduler.schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_user_raises_before_writing(tools):
    with pytest.raises(ToolException, match="User ID is missing"):
        await tools.schedule_reminder("test", 10, False, {}, delay_minutes=1)
    tools.db_client.execute_query.assert_not_awaited()
    tools.time_scheduler.schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_database_returns_configuration_error(sample_user_info):
    scheduler = MagicMock(schedule=AsyncMock())
    tools = ReminderTools(None, scheduler)
    result = json.loads(await tools.schedule_reminder("test", 10, False, sample_user_info, delay_minutes=1))
    assert result["error"] == "Configuration_Error"
    scheduler.schedule.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("db_result", [{}, {"data": []}])
async def test_user_not_found_does_not_register(tools, sample_user_info, db_result):
    tools.db_client.execute_query.return_value = db_result
    result = json.loads(await tools.schedule_reminder("test", 10, False, sample_user_info, delay_minutes=1))
    assert result["error"] == "User_Not_Found"
    tools.db_client.execute_query.assert_awaited_once()
    tools.time_scheduler.schedule.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception", "error"),
    [(Neo4jCircuitBreakerError("offline"), "Database_Offline_Circuit_Open"),
     (Neo4jClientError("offline"), "Database_Unavailable"),
     (RuntimeError("offline"), "Failed to schedule reminder")],
)
async def test_database_errors_do_not_register(tools, sample_user_info, exception, error):
    tools.db_client.execute_query.side_effect = exception
    result = json.loads(await tools.schedule_reminder("test", 10, False, sample_user_info, delay_minutes=1))
    assert result["error"] == error
    assert result["details"] == "offline"
    tools.time_scheduler.schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_failure_reports_error_after_database_write(tools, sample_user_info):
    tools.time_scheduler.schedule.side_effect = RuntimeError("scheduler failed")
    result = json.loads(await tools.schedule_reminder("test", 10, False, sample_user_info, delay_minutes=1))
    assert result == {"error": "Failed to schedule reminder", "details": "scheduler failed"}
    tools.db_client.execute_query.assert_awaited_once()
    tools.time_scheduler.schedule.assert_awaited_once()


def test_get_tools_exposes_bound_reminder_tool(tools):
    registered, = tools.get_tools()
    assert registered.name == "schedule_reminder"
    assert registered.coroutine == tools.schedule_reminder
    assert registered.handle_tool_error == tools.format_system_tool_error