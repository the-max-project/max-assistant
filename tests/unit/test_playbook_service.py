import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import json

from max_assistant.services.playbook_service import PlaybookService
from max_assistant.models.playbook_models import CareAdminConfig, DailyPlaybook, PlaybookItem


@pytest.fixture
def mock_db_client():
    client = MagicMock()
    client.execute_query = AsyncMock()
    return client


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def mock_time_scheduler():
    scheduler = MagicMock()
    scheduler.schedule = AsyncMock()
    return scheduler


@pytest.fixture
def mock_event_bus():
    bus = MagicMock()
    bus.publish = AsyncMock()
    bus.subscribe = MagicMock()
    return bus


@pytest.fixture
def playbook_service(mock_db_client, mock_llm, mock_time_scheduler, mock_event_bus):
    return PlaybookService(
        db_client=mock_db_client,
        llm=mock_llm,
        time_scheduler=mock_time_scheduler,
        event_bus=mock_event_bus
    )


@pytest.mark.asyncio
async def test_gather_daily_context_filters_past_events(playbook_service, mock_db_client):
    """Verifies that events occurring before current_dt are excluded from LLM context."""
    current_dt = datetime(2026, 11, 17, 10, 0, 0)
    target_date = "2026-11-17"

    async def db_side_effect(query, params):
        if "Appointment" in query:
            return {"data": [
                {"id": "1", "title": "Past Appt", "time": "09:00:00"},
                {"id": "2", "title": "Future Appt", "time": "11:00:00"}
            ]}
        elif "DailyRoutine" in query:
            return {"data": [
                {"id": "3", "title": "Past Routine", "time": "08:00:00"},
                {"id": "4", "title": "Future Routine", "time": "12:00:00"}
            ]}
        return {"data": []}

    mock_db_client.execute_query.side_effect = db_side_effect

    context = await playbook_service._gather_daily_context("user_1", target_date, current_dt)

    assert len(context["appointments"]) == 1
    assert context["appointments"][0]["title"] == "Future Appt"

    assert len(context["routines"]) == 1
    assert context["routines"][0]["title"] == "Future Routine"


@pytest.mark.asyncio
async def test_generate_daily_playbook_llm_fallback(playbook_service, mock_db_client, mock_llm):
    """Verifies the deterministic fallback generates PlaybookItems when LLM fails."""
    mock_llm.ainvoke.side_effect = Exception("LLM error")

    with patch.object(playbook_service, '_gather_daily_context', new_callable=AsyncMock) as mock_gather:
        mock_gather.return_value = {
            "appointments": [{"id": "1", "title": "Dentist", "time": "14:00:00"}],
            "routines": [{"id": "2", "title": "Lunch", "time": "12:00:00", "type": "meal"}],
            "family_messages": [{"id": "3", "sender": "John", "text": "Hello"}],
            "family_news": [{"id": "4", "headline": "Baby born!", "details": "A new baby"}]
        }

        # Default empty queries (e.g. status updates or initial loads)
        mock_db_client.execute_query.return_value = {"data": []}

        playbook = await playbook_service.generate_daily_playbook("user_1", datetime(2026, 11, 17, 8, 0, 0))

        assert playbook is not None
        assert len(playbook.items) == 4

        categories = {item.category for item in playbook.items}
        assert "REQUIRED" in categories
        assert "MEAL" in categories
        assert "FAMILY_MESSAGE" in categories
        assert "FAMILY_NEWS" in categories


@pytest.mark.asyncio
async def test_persist_playbook_is_idempotent(playbook_service, mock_db_client):
    """Verifies that the save query executes a DETACH DELETE on old tasks."""
    pb = DailyPlaybook(
        id="pb-user_1-2026-11-17",
        user_id="user_1",
        date="2026-11-17",
        morning_summary="Good morning",
        items=[]
    )

    await playbook_service._persist_playbook(pb, datetime.now())

    call_args = mock_db_client.execute_query.call_args[0]
    query = call_args[0]

    assert "DETACH" in query
    assert "DELETE old_item, old_task" in query


@pytest.mark.asyncio
async def test_get_playbook_filters_and_sorts_items(playbook_service, mock_db_client):
    mock_db_client.execute_query.return_value = {"data": [{
        "id": "pb-1", "date": "2026-11-17", "morning_summary": None,
        "items": [None, {},
                  {"id": "late", "category": "MEAL", "title": "Lunch", "summary": "", "priority": 8, "event_time": "12:00"},
                  {"id": "untimed", "category": "MEAL", "title": "Meal", "summary": "", "priority": 8},
                  {"id": "early", "category": "REQUIRED", "title": "Visit", "summary": "", "priority": 8, "event_time": "09:00"},
                  {"id": "low", "category": "FAMILY_NEWS", "title": "News", "summary": "", "priority": 3}],
    }]}
    result = await playbook_service.get_playbook("u1", "2026-11-17")
    assert [item.id for item in result.items] == ["early", "late", "untimed", "low"]
    assert result.morning_summary == ""
    assert mock_db_client.execute_query.call_args.args[1] == {"user_id": "u1", "date": "2026-11-17"}


@pytest.mark.asyncio
async def test_get_or_create_uses_existing_or_generates(playbook_service):
    now = datetime(2026, 11, 17, 10)
    existing = DailyPlaybook(id="existing", user_id="u1", date="2026-11-17", morning_summary="Hi")
    playbook_service.get_playbook = AsyncMock(side_effect=[existing, None])
    playbook_service.generate_daily_playbook = AsyncMock(return_value=existing)
    assert await playbook_service.get_or_create_playbook("u1", now) is existing
    playbook_service.generate_daily_playbook.assert_not_awaited()
    assert await playbook_service.get_or_create_playbook("u1", now) is existing
    playbook_service.get_playbook.assert_awaited_with(user_id="u1", target_date="2026-11-17")
    playbook_service.generate_daily_playbook.assert_awaited_once_with(user_id="u1", current_dt=now)


@pytest.mark.asyncio
async def test_synthesis_parses_wrapped_json_and_defaults(playbook_service, mock_llm):
    mock_llm.ainvoke.return_value.content = '```json\n{"items": [{"title": "Music", "priority": "7"}]}\n```'
    result = await playbook_service._synthesize_with_llm("u1", "2026-11-17", CareAdminConfig(), {})
    assert result.id == "pb-u1-2026-11-17"
    assert "relaxed schedule" in result.morning_summary
    item = result.items[0]
    assert (item.category, item.summary, item.priority, item.status) == ("OPTIONAL_ACTIVITY", "Music", 7, "QUEUED")
    messages = mock_llm.ainvoke.await_args.args[0]
    assert json.loads(messages[1].content) == {"date": "2026-11-17", "context": {}}


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ["not json", '{"items": [{"category": "INVALID"}]}'])
async def test_invalid_llm_output_uses_fallback(playbook_service, mock_llm, response):
    mock_llm.ainvoke.return_value.content = response
    context = {"routines": [
        {"title": "Lunch", "time": "12:00"},
        {"title": "Medication", "type": "medication", "time": "09:00"},
        {"title": "Bingo", "rating": "Dislike"},
        {"title": "Music", "room": "Lounge"},
    ]}
    result = await playbook_service._synthesize_with_llm("u1", "2026-11-17", CareAdminConfig(), context)
    assert [(i.title, i.category, i.lead_minutes) for i in result.items] == [
        ("Lunch", "MEAL", 15), ("Medication", "MEDICATION", 5), ("Music", "OPTIONAL_ACTIVITY", 60),
    ]
    assert result.items[-1].room == "Lounge"
    assert "Medication at 09:00" in result.morning_summary


@pytest.mark.asyncio
async def test_reminders_only_schedule_future_valid_required_items(playbook_service, mock_db_client):
    now = datetime(2026, 11, 17, 10)
    items = [
        PlaybookItem(id="visit", category="REQUIRED", title="Visit", summary="", event_time="10:30", lead_minutes=30),
        PlaybookItem(id="meal", category="MEAL", title="Lunch", summary="", event_time="11:00", lead_minutes=10),
        PlaybookItem(id="bad", category="MEDICATION", title="Medicine", summary="", event_time="invalid"),
        PlaybookItem(id="optional", category="OPTIONAL_ACTIVITY", title="Music", summary="", event_time="11:00"),
        PlaybookItem(id="untimed", category="REQUIRED", title="Untimed", summary=""),
    ]
    pb = DailyPlaybook(id="pb", user_id="u1", date="2026-11-17", morning_summary="", items=items)
    playbook_service.create_and_schedule_task = AsyncMock()
    await playbook_service._schedule_required_reminders(pb, CareAdminConfig(), now)
    calls = playbook_service.create_and_schedule_task.await_args_list
    assert [(c.kwargs["playbook_item_id"], c.kwargs["due_dt"], c.kwargs["validity_minutes"]) for c in calls] == [
        ("visit", datetime(2026, 11, 17, 10, 15), 30),
        ("meal", datetime(2026, 11, 17, 10, 50), 15),
    ]
    assert [c.args[1]["id"] for c in mock_db_client.execute_query.await_args_list] == ["visit", "meal"]


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [[{"task_id": "task-1"}], []])
async def test_create_task_schedules_only_when_database_returns_id(playbook_service, mock_db_client, mock_time_scheduler, rows):
    mock_db_client.execute_query.return_value = {"data": rows}
    now = datetime(2026, 11, 17, 9)
    due = datetime(2026, 11, 17, 10)
    await playbook_service.create_and_schedule_task("u1", "item", "Get ready", due, 20, now)
    params = mock_db_client.execute_query.await_args.args[1]
    assert params == {"user_id": "u1", "item_id": "item", "text": "Get ready", "due_time": due.isoformat(),
                      "expires_at": "2026-11-17T10:20:00", "created_at": now.isoformat()}
    if rows:
        mock_time_scheduler.schedule.assert_awaited_once_with(
            task_id="task-1", user_id="u1", text="Get ready", due_epoch=due.timestamp(),
            expires_epoch=datetime(2026, 11, 17, 10, 20).timestamp(), can_wake_user=False)
    else:
        mock_time_scheduler.schedule.assert_not_awaited()


@pytest.mark.parametrize("value,expected", [
    (" 9:05 pm ", datetime(2026, 11, 17, 21, 5)),
    ("09:05:30", datetime(2026, 11, 17, 9, 5, 30)),
    ("2026-11-18T09:05:00", datetime(2026, 11, 18, 9, 5)),
    ("25:00", None),
])
def test_resolve_event_datetime_formats(playbook_service, value, expected):
    assert playbook_service._resolve_event_datetime(datetime(2026, 11, 17).date(), value) == expected


@pytest.mark.asyncio
async def test_event_routes_single_user_and_all_users(playbook_service, mock_event_bus):
    now = datetime(2026, 11, 17, 10)
    mock_event_bus.subscribe.assert_called_once()
    playbook_service.generate_daily_playbook = AsyncMock()
    playbook_service.generate_for_all_users = AsyncMock()
    await playbook_service.on_playbook_generate("playbook.generate", {"date": now.isoformat(), "user_id": 42})
    playbook_service.generate_daily_playbook.assert_awaited_once_with(user_id="42", current_dt=now)
    await playbook_service.on_playbook_generate("playbook.generate", {"date": now.isoformat()})
    playbook_service.generate_for_all_users.assert_awaited_once_with(current_dt=now)


@pytest.mark.asyncio
async def test_context_retains_boundary_missing_and_unparseable_times(playbook_service, mock_db_client):
    mock_db_client.execute_query.side_effect = [
        {"data": [{"id": "past", "time": "09:59"}, {"id": "now", "time": "10:00"},
                  {"id": "missing"}, {"id": "invalid", "time": "unknown"}]},
        {"data": []}, {"data": [{"id": "message"}]}, {"data": [{"id": "news"}]},
    ]
    context = await playbook_service._gather_daily_context("u1", "2026-11-17", datetime(2026, 11, 17, 10))
    assert [item["id"] for item in context["appointments"]] == ["now", "missing", "invalid"]
    assert context["family_messages"] == [{"id": "message"}]
    assert context["family_news"] == [{"id": "news"}]
    assert all(call.args[1] == {"user_id": "u1", "target_date": "2026-11-17"}
               for call in mock_db_client.execute_query.await_args_list)