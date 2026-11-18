import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from max_assistant.services.engagement_manager import EngagementManager
from max_assistant.events.bus import TOPIC_CHAT_TURN_COMPLETED, TOPIC_USER_ENGAGEMENT_IDLE


@pytest.fixture
def mock_event_bus():
    bus = MagicMock()
    bus.subscribe = MagicMock()
    return bus


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.connection_manager = MagicMock()
    agent.connection_manager.submit_external_event = AsyncMock()
    return agent


@pytest.fixture
def mock_session_mgr(mock_agent):
    mgr = MagicMock()
    mgr.get_agents = MagicMock(return_value=[mock_agent])
    return mgr


@pytest.fixture
def mock_db_client():
    client = MagicMock()
    client.execute_query = AsyncMock()
    return client


@pytest.fixture
def mock_notification_router():
    router = MagicMock()
    router.is_user_in_active_turn = MagicMock(return_value=False)
    return router


@pytest.fixture
def engagement_manager(mock_event_bus, mock_session_mgr, mock_db_client, mock_notification_router):
    return EngagementManager(
        event_bus=mock_event_bus,
        session_mgr=mock_session_mgr,
        db_client=mock_db_client,
        notification_router=mock_notification_router
    )


@pytest.mark.asyncio
async def test_sleep_state_awake_delivers_briefing_once(engagement_manager, mock_db_client, mock_agent):
    """Verifies wake-up briefing is delivered once per day when sleep state changes to awake."""
    mock_db_client.execute_query.return_value = {
        "data": [{"summary": "Good morning! You have Bingo today at 4:00 PM."}]
    }

    await engagement_manager.on_sleep_state_change("state.sleep", {"state": "awake", "user_id": "user123"})

    # Assertion verifies it was wrapped into an LLM directive
    mock_agent.connection_manager.submit_external_event.assert_awaited_once()
    call_args = mock_agent.connection_manager.submit_external_event.call_args[0][0]
    assert call_args["type"] == "WAKE_UP_BRIEFING"
    assert "PROACTIVE PROMPT DIRECTIVE" in call_args["text"]
    assert "Good morning! You have Bingo today at 4:00 PM." in call_args["text"]


@pytest.mark.asyncio
async def test_user_idle_delivers_prompt_when_present_and_awake(engagement_manager, mock_db_client, mock_agent):
    """Verifies idle event delivers the highest priority queued playbook item."""

    async def db_side_effect(query, params):
        if "presence" in query:
            return {"data": [{"presence": "present", "sleep_state": "awake"}]}
        if "QUEUED" in query:
            return {"data": [{
                "item": {
                    "id": "item_abc",
                    "category": "OPTIONAL_ACTIVITY",
                    "title": "Bingo",
                    "summary": "Afternoon Bingo"
                }
            }]}
        return {"data": []}

    mock_db_client.execute_query.side_effect = db_side_effect

    await engagement_manager.on_user_idle(TOPIC_USER_ENGAGEMENT_IDLE, {"user_id": "user123"})

    mock_agent.connection_manager.submit_external_event.assert_awaited_once()
    call_args = mock_agent.connection_manager.submit_external_event.call_args[0][0]
    assert "Bingo" in call_args["text"]
    assert "item_abc" in call_args["text"]


@pytest.mark.asyncio
async def test_user_idle_suppresses_prompt_when_away(engagement_manager, mock_db_client, mock_agent):
    """Verifies idle event is ignored if user is away or asleep."""
    mock_db_client.execute_query.return_value = {"data": [{"presence": "away", "sleep_state": "awake"}]}

    await engagement_manager.on_user_idle(TOPIC_USER_ENGAGEMENT_IDLE, {"user_id": "user123"})

    mock_agent.connection_manager.submit_external_event.assert_not_called()
    assert mock_db_client.execute_query.call_count == 1


@pytest.mark.asyncio
async def test_user_idle_suppresses_prompt_in_active_turn(engagement_manager, mock_db_client, mock_notification_router,
                                                          mock_agent):
    """Verifies idle event is ignored if user is currently in an active chat turn."""
    mock_notification_router.is_user_in_active_turn.return_value = True

    await engagement_manager.on_user_idle(TOPIC_USER_ENGAGEMENT_IDLE, {"user_id": "user123"})

    mock_db_client.execute_query.assert_not_called()
    mock_agent.connection_manager.submit_external_event.assert_not_called()


@pytest.mark.asyncio
async def test_chat_turn_completed_delivers_prompt_after_delay(engagement_manager, mock_db_client, mock_agent):
    """Verifies chat turn completed waits and then delivers prompt if still idle."""

    async def db_side_effect(query, params):
        if "presence" in query:
            return {"data": [{"presence": "present", "sleep_state": "awake"}]}
        if "QUEUED" in query:
            return {"data": [{
                "item": {
                    "id": "item_xyz",
                    "category": "FAMILY_MESSAGE",
                    "title": "Message from John",
                    "summary": "Hi Mom!",
                    "sender": "John"
                }
            }]}
        return {"data": []}

    mock_db_client.execute_query.side_effect = db_side_effect

    # Patch asyncio.sleep to not actually wait 3 seconds during the test
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await engagement_manager.on_chat_turn_completed(TOPIC_CHAT_TURN_COMPLETED, {"user_id": "user123"})

        mock_sleep.assert_awaited_once_with(3.0)

        mock_agent.connection_manager.submit_external_event.assert_awaited_once()
        call_args = mock_agent.connection_manager.submit_external_event.call_args[0][0]
        assert call_args["type"] == "PLAYBOOK_PROMPT"
        assert call_args["item_id"] == "item_xyz"
        assert "John" in call_args["text"]
        assert "Hi Mom!" in call_args["text"]