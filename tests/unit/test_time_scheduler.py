import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from max_assistant.services.time_scheduler import PersistentTimeScheduler


@pytest.fixture
def mock_db_client():
    """Provides a mock database client for testing."""
    return AsyncMock()


@pytest.fixture
def mock_event_bus():
    """Provides a mock event bus for testing."""
    return AsyncMock()


@pytest.fixture
def time_scheduler(mock_db_client, mock_event_bus):
    """Provides a PersistentTimeScheduler instance for testing."""
    return PersistentTimeScheduler(
        db_client=mock_db_client,
        event_bus=mock_event_bus
    )


class TestPersistentTimeScheduler:
    """Tests for the PersistentTimeScheduler class."""

    def test_init_sets_default_values(self, mock_db_client, mock_event_bus):
        """Verifies that PersistentTimeScheduler initializes with correct default values."""
        scheduler = PersistentTimeScheduler(
            db_client=mock_db_client,
            event_bus=mock_event_bus
        )
        
        assert scheduler.db_client == mock_db_client
        assert scheduler.event_bus == mock_event_bus
        assert scheduler._heap == []
        assert isinstance(scheduler._wake_event, asyncio.Event)
        assert scheduler._running is False
        assert scheduler._loop_task is None

    @pytest.mark.asyncio
    async def test_schedule_adds_to_heap(self, time_scheduler):
        """Verifies that schedule adds items to the heap correctly."""
        # Add a task
        await time_scheduler.schedule(
            task_id="task123",
            user_id="user456",
            text="Test reminder",
            due_epoch=1000.0,
            expires_epoch=2000.0,
            can_wake_user=True
        )
        
        # Verify heap contains the item
        assert len(time_scheduler._heap) == 1
        assert time_scheduler._heap[0] == (1000.0, "task123", "user456", "Test reminder", 2000.0, True)
        # Verify wake event was set
        assert time_scheduler._wake_event.is_set() is True

    @pytest.mark.asyncio
    async def test_start_initializes_and_starts_loop(self, time_scheduler, mock_db_client):
        """Verifies that start initializes and starts the scheduler loop."""
        with patch.object(time_scheduler, '_hydrate_from_neo4j') as mock_hydrate:
            with patch.object(time_scheduler, '_run_loop') as mock_run_loop:
                # Mock the run loop to complete quickly
                mock_run_loop.return_value = asyncio.sleep(0.01)
                
                await time_scheduler.start()
                
                # Verify initialization
                assert time_scheduler._running is True
                assert mock_hydrate.called
                assert mock_run_loop.called

    @pytest.mark.asyncio
    async def test_stop_sets_running_false_and_cancels_task(self, time_scheduler):
        """Verifies that stop correctly stops the scheduler."""
        # Mock a task
        mock_task = AsyncMock()
        time_scheduler._loop_task = mock_task
        
        await time_scheduler.stop()
        
        # Verify state change
        assert time_scheduler._running is False
        assert time_scheduler._wake_event.is_set() is True
        assert mock_task.cancel.called

    @pytest.mark.asyncio
    async def test_hydrate_from_neo4j_handles_empty_results(self, time_scheduler, mock_db_client):
        """Verifies that hydrate_from_neo4j handles empty results gracefully."""
        # Mock empty result
        mock_db_client.execute_query.return_value = {"data": []}
        
        await time_scheduler._hydrate_from_neo4j()
        
        # Verify no items were added to heap
        assert len(time_scheduler._heap) == 0

    @pytest.mark.asyncio
    async def test_hydrate_from_neo4j_handles_db_errors(self, time_scheduler, mock_db_client, caplog):
        """Verifies that hydrate_from_neo4j handles database errors gracefully."""
        # Mock a database error
        from max_assistant.clients.neo4j_client import Neo4jClientError
        mock_db_client.execute_query.side_effect = Neo4jClientError("Database error")
        
        await time_scheduler._hydrate_from_neo4j()
        
        # Verify error was logged
        assert "Failed to hydrate from Neo4j" in caplog.text

    @pytest.mark.asyncio
    async def test_get_user_state_handles_empty_results(self, time_scheduler, mock_db_client):
        """Verifies that get_user_state handles empty results gracefully."""
        # Mock empty result
        mock_db_client.execute_query.return_value = {"data": []}
        
        result = await time_scheduler._get_user_state("user123")
        
        # Should return default values
        assert result == {"presence": "present", "sleep_state": "awake"}

    @pytest.mark.asyncio
    async def test_set_task_status_executes_query(self, time_scheduler, mock_db_client):
        """Verifies that set_task_status executes the correct query."""
        await time_scheduler._set_task_status("task123", "COMPLETED")
        
        # Verify query was executed
        mock_db_client.execute_query.assert_called_once()