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
        
        # Verify that subscribe was called for both topics
        mock_event_bus.subscribe.assert_any_call("task.due", router.on_task_due)
        mock_event_bus.subscribe.assert_any_call("presence.*", router.on_presence_change)

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
                {"task_id": "task456", "text": "Test task"}
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
                "user_id": "user123"
            })
            
            # Verify database queries were called
            assert mock_db_client.execute_query.call_count == 2
            
            # Verify broadcast was called with correct payload
            mock_broadcast.assert_called_once_with(
                "user123",
                {
                    "task_id": "task1",
                    "text": "You just left the bedroom. Reminder: Reminder task"
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