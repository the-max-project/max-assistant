import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch
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
    """Verifies that events occurring before current_dt are excluded from the LLM context."""
    current_dt = datetime(2026, 11, 17, 10, 0, 0)
    target_date = "2026-11-17"

    async def db_side_effect(query, _params):
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

        # Default empty queries (e.g., status updates or initial loads)
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

    assert "DETACH DELETE oi" in query
    assert "DETACH DELETE ot" in query


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
    assert result is not None
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
async def test_get_remaining_items_returns_none_when_playbook_missing(playbook_service):
    now = datetime(2026, 11, 17, 10)
    playbook_service.get_or_create_playbook = AsyncMock(return_value=None)

    pb, items = await playbook_service.get_remaining_items("u1", now)

    assert pb is None
    assert items == []
    playbook_service.get_or_create_playbook.assert_awaited_once_with(user_id="u1", current_dt=now)


@pytest.mark.asyncio
async def test_get_remaining_items_filters_past_and_terminal_statuses_while_preserving_opt_ins(playbook_service):
    now = datetime(2026, 11, 17, 10, 0)
    pb = DailyPlaybook(
        id="pb-u1-2026-11-17",
        user_id="u1",
        date="2026-11-17",
        morning_summary="Calm morning",
        items=[
            PlaybookItem(id="past-meal", category="MEAL", title="Breakfast", summary="", event_time="08:30", status="SCHEDULED"),
            PlaybookItem(id="declined-act", category="OPTIONAL_ACTIVITY", title="Bingo", summary="", event_time="15:00", status="DECLINED"),
            PlaybookItem(id="expired-act", category="OPTIONAL_ACTIVITY", title="Walk", summary="", event_time="11:00", status="EXPIRED"),
            PlaybookItem(id="completed-act", category="REQUIRED", title="Vitals", summary="", event_time="10:30", status="COMPLETED"),
            PlaybookItem(id="accepted-act", category="OPTIONAL_ACTIVITY", title="Chair Pilates", summary="", event_time="14:00", status="ACCEPTED", priority=5),
            PlaybookItem(id="upcoming-meal", category="MEAL", title="Lunch", summary="", event_time="12:00", status="SCHEDULED", priority=8),
            PlaybookItem(id="now-appt", category="REQUIRED", title="Checkup", summary="", event_time="10:00", status="SCHEDULED", priority=10),
            PlaybookItem(id="prompted-news", category="FAMILY_NEWS", title="Sarah's Update", summary="", status="PROMPTED", priority=3),
            PlaybookItem(id="queued-msg", category="FAMILY_MESSAGE", title="Note from John", summary="", status="QUEUED", priority=4),
        ],
    )
    playbook_service.get_or_create_playbook = AsyncMock(return_value=pb)

    returned_pb, remaining = await playbook_service.get_remaining_items("u1", now)

    assert returned_pb is pb
    assert [(i.id, i.status) for i in remaining] == [
        ("now-appt", "SCHEDULED"),
        ("upcoming-meal", "SCHEDULED"),
        ("accepted-act", "ACCEPTED"),
        ("queued-msg", "QUEUED"),
        ("prompted-news", "PROMPTED"),
    ]


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
@pytest.mark.parametrize("care_cfg, activity_lead", [(CareAdminConfig(), 15), (CareAdminConfig(activity_ask_ahead_minutes=60), 60)])
async def test_invalid_llm_output_uses_fallback(playbook_service, mock_llm, response, care_cfg, activity_lead):
    mock_llm.ainvoke.return_value.content = response
    context = {"routines": [
        {"title": "Lunch", "time": "12:00"},
        {"title": "Medication", "type": "medication", "time": "09:00"},
        {"title": "Bingo", "rating": "Dislike"},
        {"title": "Music", "room": "Lounge"},
    ]}
    result = await playbook_service._synthesize_with_llm("u1", "2026-11-17", care_cfg, context)
    assert [(i.title, i.category, i.lead_minutes) for i in result.items] == [
        ("Lunch", "MEAL", 15), ("Medication", "MEDICATION", 5), ("Music", "OPTIONAL_ACTIVITY", activity_lead),
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
    assert playbook_service.resolve_event_datetime(datetime(2026, 11, 17).date(), value) == expected


@pytest.mark.asyncio
async def test_event_routes_single_user_and_all_users(playbook_service, mock_event_bus):
    now = datetime(2026, 11, 17, 10)
    mock_event_bus.subscribe.assert_has_calls([
        call("playbook.generate", playbook_service.on_playbook_generate),
        call("schedule.updated", playbook_service.on_schedule_updated),
    ])
    assert mock_event_bus.subscribe.call_count == 2

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

@pytest.mark.asyncio
async def test_get_remaining_items_preserves_accepted_activities(playbook_service):
    """Verifies mid-day schedule reads return ACCEPTED items intact without regenerating."""
    now = datetime(2026, 11, 17, 13, 0)
    accepted_item = PlaybookItem(
        id="item-pilates",
        category="OPTIONAL_ACTIVITY",
        title="Chair Pilates",
        summary="Chair Pilates at 14:00",
        event_time="14:00",
        status="ACCEPTED",
        priority=5,
    )
    pb = DailyPlaybook(
        id="pb-u1-2026-11-17",
        user_id="u1",
        date="2026-11-17",
        morning_summary="Calm afternoon",
        items=[accepted_item],
    )
    playbook_service.get_playbook = AsyncMock(return_value=pb)
    playbook_service.generate_daily_playbook = AsyncMock()

    returned_pb, remaining = await playbook_service.get_remaining_items("u1", now)

    assert returned_pb is pb
    assert len(remaining) == 1
    assert remaining[0].id == "item-pilates"
    assert remaining[0].status == "ACCEPTED"
    playbook_service.generate_daily_playbook.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_daily_playbook_adds_new_appointment_and_preserves_declined_activity(
    playbook_service, mock_db_client, mock_time_scheduler
):
    """
    Verifies sync_daily_playbook:
    - Preserves DECLINED/ACCEPTED items without overwriting status
    - Adds newly created afternoon appointments and schedules their reminders
    - Expires removed SCHEDULED items and cancels their spawned tasks in time_scheduler
    """
    now = datetime(2026, 11, 17, 11, 0)
    existing_pb = DailyPlaybook(
        id="pb-u1-2026-11-17",
        user_id="u1",
        date="2026-11-17",
        morning_summary="Morning summary",
        items=[
            PlaybookItem(
                id="item-declined",
                category="OPTIONAL_ACTIVITY",
                title="Bingo",
                summary="Bingo at 15:00",
                event_time="15:00",
                status="DECLINED",
                priority=5,
            ),
            PlaybookItem(
                id="item-cancelled",
                category="REQUIRED",
                title="Old Lab Visit",
                summary="Old Lab Visit at 13:00",
                event_time="13:00",
                status="SCHEDULED",
                priority=10,
            ),
        ],
    )

    playbook_service.get_playbook = AsyncMock(return_value=existing_pb)
    playbook_service._load_care_config = AsyncMock(return_value=CareAdminConfig(appointment_lead_minutes=30, departure_lead_minutes=0))
    playbook_service._gather_daily_context = AsyncMock(return_value={
        "appointments": [{"id": "appt-new", "title": "Eye Doctor", "time": "16:00"}],
        "routines": [{"id": "r-bingo", "title": "Bingo", "time": "15:00", "type": "activity"}],
        "family_messages": [],
        "family_news": [],
    })
    playbook_service.create_and_schedule_task = AsyncMock()

    mock_db_client.execute_query.side_effect = [
        {"data": [{"deleted_task_ids": ["task-old-lab"]}]},  # cancel removed item + task
        {"data": []},  # insert new item
        {"data": []},  # mark new item SCHEDULED
    ]

    await playbook_service.sync_daily_playbook("u1", now)

    # 1. Cancelled item's task evicted from heap
    mock_time_scheduler.cancel_tasks.assert_called_once_with(["task-old-lab"])

    # 2. Only the newly added appointment was inserted (declined Bingo was untouched)
    insert_call_params = mock_db_client.execute_query.await_args_list[1].args[1]
    inserted_titles = [item["title"] for item in insert_call_params["items"]]
    assert inserted_titles == ["Eye Doctor"]

    # 3. Prep reminder scheduled for the newly added appointment (16:00 - 30m = 15:30)
    playbook_service.create_and_schedule_task.assert_awaited_once()
    assert playbook_service.create_and_schedule_task.await_args.kwargs["due_dt"] == datetime(2026, 11, 17, 15, 30)