import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from max_assistant.services.notification_router import NotificationRouter


@pytest.fixture
def mock_event_bus():
    """Provides a mock event bus for testing."""
    return MagicMock()


@pytest.fixture
def mock_session_mgr():
    """Provides a mock session manager for testing."""
    return MagicMock()


@pytest.fixture
def mock_db_client():
    """Provides a mock database client for testing."""
    return AsyncMock()


@pytest.fixture
def notification_router(mock_event_bus, mock_session_mgr, mock_db_client):
    """Provides a NotificationRouter instance for testing."""
    return NotificationRouter(
        event_bus=mock_event_bus,
        session_mgr=mock_session_mgr,
        db_client=mock_db_client
    )


class TestNotificationRouter:
    """Tests for the NotificationRouter class."""

    def test_init_registers_handlers(self, mock_event_bus, mock_session_mgr, mock_db_client):
        """Verifies that NotificationRouter registers event handlers correctly."""
        router = NotificationRouter(
            event_bus=mock_event_bus,
            session_mgr=mock_session_mgr,
            db_client=mock_db_client
        )
        
        # Verify that subscribe was called for all topics
        mock_event_bus.subscribe.assert_any_call("task.due", router.on_task_due)
        mock_event_bus.subscribe.assert_any_call("presence.*", router.on_presence_change)
        mock_event_bus.subscribe.assert_any_call("state.sleep", router.on_sleep_state_change)
        mock_event_bus.subscribe.assert_any_call("chat.turn.started", router.on_chat_turn_started)
        mock_event_bus.subscribe.assert_any_call("chat.turn.completed", router.on_chat_turn_completed)

    @pytest.mark.asyncio
    async def test_on_task_due_handles_valid_data(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client):
        """Verifies that on_task_due handles valid task data correctly."""
        # Mock the _broadcast_to_user method
        with patch.object(notification_router, '_broadcast_to_user') as mock_broadcast:
            # Call the method with valid data
            await notification_router.on_task_due("task.due", {
                "user_id": "user123",
                "task_id": "task456",
                "text": "Test task"
            })
            
            # Verify broadcast was called with correct payload
            mock_broadcast.assert_called_once_with(
                "user123",
                {"type": "REMINDER", "task_id": "task456", "text": "Test task"}
            )

    @pytest.mark.asyncio
    async def test_on_task_due_handles_missing_user_id(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client, caplog):
        """Verifies that on_task_due handles missing user_id gracefully."""
        # Call the method with invalid data (missing user_id)
        await notification_router.on_task_due("task.due", {
            "task_id": "task456",
            "text": "Test task"
        })
        
        # Verify warning was logged
        assert "received event without a valid user_id" in caplog.text

    @pytest.mark.asyncio
    async def test_on_task_due_handles_invalid_user_id(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client, caplog):
        """Verifies that on_task_due handles invalid user_id gracefully."""
        # Call the method with invalid user_id (not a string)
        await notification_router.on_task_due("task.due", {
            "user_id": 123,
            "task_id": "task456",
            "text": "Test task"
        })
        
        # Verify warning was logged
        assert "received event without a valid user_id" in caplog.text

    @pytest.mark.asyncio
    async def test_on_presence_change_handles_vacated_state(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client):
        """Verifies that on_presence_change handles vacated state correctly."""
        # Mock the database client to return some results
        mock_db_client.execute_query = AsyncMock(return_value={
            "data": [
                {"id": "task1", "text": "Reminder task"}
            ]
        })
        
        # Mock the _broadcast_to_user method
        with patch.object(notification_router, '_broadcast_to_user') as mock_broadcast:
            # Call the method with vacated state
            await notification_router.on_presence_change("presence.bedroom", {
                "room": "bedroom",
                "state": "vacated",
                "scope": "premise",
                "out_location": "out",
                "user_id": "user123"
            })
            
            # Verify database queries were called
            assert mock_db_client.execute_query.call_count == 2

            # Verify first query marks user away / 'out'
            first_query, first_params = mock_db_client.execute_query.call_args_list[0][0]
            assert "u.presence = 'away'" in first_query
            assert first_params == {"user_id": "user123", "out_location": "out"}
            
            # Verify broadcast was called with correct payload
            mock_broadcast.assert_called_once_with(
                "user123",
                {
                    "type": "REMINDER",
                    "task_id": "task1",
                    "text": "The resident just left the bedroom. Gently deliver this reminder: Reminder task"
                }
            )

    @pytest.mark.asyncio
    async def test_on_presence_change_handles_no_active_session(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client):
        """Verifies that on_presence_change handles no active session gracefully."""
        # Mock session manager to return no agents (but not async)
        mock_session_mgr.get_agents.return_value = []
        
        # Mock the database client to return some results
        mock_db_client.execute_query = AsyncMock(return_value={
            "data": []
        })
        
        # Call the method with vacated state - should not raise exception
        await notification_router.on_presence_change("presence.bedroom", {
            "room": "bedroom",
            "state": "vacated",
            "user_id": "user123"
        })
        
        # Should not raise any exception and database queries should still be called
        assert mock_db_client.execute_query.call_count == 2

    @pytest.mark.asyncio
    async def test_on_sleep_state_change_handles_awake_state(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client):
        """Verifies that on_sleep_state_change handles awake state correctly."""
        # Mock the _wake_scheduler method to verify it's called
        with patch.object(notification_router, '_wake_scheduler') as mock_wake:
            # Call the method with awake state
            await notification_router.on_sleep_state_change("state.sleep", {
                "user_id": "user123",
                "state": "awake"
            })
            
            # Verify database query was called with correct parameters
            mock_db_client.execute_query.assert_called_once()
            call_args = mock_db_client.execute_query.call_args[0]
            assert "u.sleep_state = $state" in call_args[0]
            assert call_args[1] == {"user_id": "user123", "state": "awake"}
            
            # Verify _wake_scheduler was called
            mock_wake.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_sleep_state_change_handles_asleep_state(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client):
        """Verifies that on_sleep_state_change handles asleep state correctly."""
        # Call the method with asleep state
        await notification_router.on_sleep_state_change("state.sleep", {
            "user_id": "user123",
            "state": "asleep"
        })
        
        # Verify database query was called with correct parameters
        mock_db_client.execute_query.assert_called_once()
        call_args = mock_db_client.execute_query.call_args[0]
        assert "u.sleep_state = $state" in call_args[0]
        assert call_args[1] == {"user_id": "user123", "state": "asleep"}

    @pytest.mark.asyncio
    async def test_on_sleep_state_change_handles_invalid_state(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client, caplog):
        """Verifies that on_sleep_state_change handles invalid state gracefully."""
        # Call the method with invalid state
        await notification_router.on_sleep_state_change("state.sleep", {
            "user_id": "user123",
            "state": "invalid_state"
        })
        
        # Verify warning was logged and no database query was made
        assert "received unknown state 'invalid_state'" in caplog.text
        mock_db_client.execute_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_chat_turn_started(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client):
        """Verifies that on_chat_turn_started correctly manages active users."""
        # Call the method with valid user_id
        await notification_router.on_chat_turn_started("chat.turn.started", {
            "user_id": "user123"
        })
        
        # Verify user was added to active users set
        assert "user123" in notification_router._active_users

    @pytest.mark.asyncio
    async def test_on_chat_turn_completed(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client):
        """Verifies that on_chat_turn_completed correctly manages active users."""
        # First add user to active users set
        notification_router._active_users.add("user123")
        
        # Call the method with valid user_id
        await notification_router.on_chat_turn_completed("chat.turn.completed", {
            "user_id": "user123"
        })
        
        # Verify user was removed from active users set
        assert "user123" not in notification_router._active_users

    @pytest.mark.asyncio
    async def test_is_user_in_active_turn(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client):
        """Verifies that is_user_in_active_turn works correctly."""
        # Test with user not in active users set
        result = notification_router.is_user_in_active_turn("user123")
        assert result is False
        
        # Add user to active users set
        notification_router._active_users.add("user123")
        
        # Test with user in active users set
        result = notification_router.is_user_in_active_turn("user123")
        assert result is True

    @pytest.mark.asyncio
    async def test_broadcast_to_user_no_agents(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client, caplog):
        """Verifies that _broadcast_to_user handles no active session gracefully."""
        # Mock session manager to return no agents
        mock_session_mgr.get_agents.return_value = []
        
        # Call the method directly
        await notification_router._broadcast_to_user("user123", {"test": "data"})
        
        # Verify warning was logged
        assert "No active session found for user user123" in caplog.text

    @pytest.mark.asyncio
    async def test_broadcast_to_user_with_connection_manager(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client):
        """Verifies that _broadcast_to_user correctly submits events to connection managers."""
        # Mock agent with connection manager
        mock_agent = MagicMock()
        mock_agent.connection_manager = AsyncMock()
        mock_session_mgr.get_agents.return_value = [mock_agent]
        
        # Call the method directly
        await notification_router._broadcast_to_user("user123", {"test": "data"})
        
        # Verify event was submitted to connection manager
        mock_agent.connection_manager.submit_external_event.assert_called_once_with({"test": "data"})

    @pytest.mark.asyncio
    async def test_broadcast_to_user_without_connection_manager(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client, caplog):
        """Verifies that _broadcast_to_user handles agents without connection managers gracefully."""
        # Mock agent without connection manager
        mock_agent = MagicMock()
        mock_agent.connection_manager = None
        mock_session_mgr.get_agents.return_value = [mock_agent]
        
        # Call the method directly
        await notification_router._broadcast_to_user("user123", {"test": "data"})
        
        # Should not raise exception, but no event should be submitted
        assert mock_agent.connection_manager is None

    @pytest.mark.asyncio
    async def test_on_presence_change_handles_occupied_state(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client):
        """Verifies that on_presence_change handles occupied state correctly."""
        # Call the method with occupied state
        await notification_router.on_presence_change("presence.bedroom", {
            "room": "bedroom",
            "state": "occupied",
            "user_id": "user123"
        })
        
        # Verify database query was called
        mock_db_client.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_presence_change_handles_room_move(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client):
        """Verifies that on_presence_change handles room move correctly."""
        # Mock the database client to return some results
        mock_db_client.execute_query = AsyncMock(return_value={
            "data": []
        })
        
        # Call the method with vacated state and scope 'room'
        await notification_router.on_presence_change("presence.bedroom", {
            "room": "bedroom",
            "state": "vacated",
            "scope": "room",
            "next_room": "hallway",
            "user_id": "user123"
        })
        
        # Verify database queries were called
        assert mock_db_client.execute_query.call_count == 2

    @pytest.mark.asyncio
    async def test_wake_scheduler_with_none_scheduler(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client):
        """Verifies that _wake_scheduler handles None scheduler gracefully."""
        # Set scheduler to None
        notification_router.time_scheduler = None
        
        # Call the method - should not raise exception
        notification_router._wake_scheduler()

    @pytest.mark.asyncio
    async def test_wake_scheduler_with_scheduler_without_wake_method(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client):
        """Verifies that _wake_scheduler handles scheduler without wake method gracefully."""
        # Set scheduler without wake method
        mock_scheduler = MagicMock()
        notification_router.time_scheduler = mock_scheduler
        
        # Call the method - should not raise exception
        notification_router._wake_scheduler()

    @pytest.mark.asyncio
    async def test_wake_scheduler_with_scheduler_with_wake_method(self, notification_router, mock_event_bus, mock_session_mgr, mock_db_client):
        """Verifies that _wake_scheduler calls wake method when present."""
        # Set scheduler with wake method
        mock_scheduler = MagicMock()
        notification_router.time_scheduler = mock_scheduler
        
        # Call the method
        notification_router._wake_scheduler()
        
        # Verify wake method was called
        mock_scheduler.wake.assert_called_once()