import asyncio
from unittest.mock import AsyncMock

import pytest

from max_assistant.events.bus import LocalEventBus


@pytest.fixture
def event_bus():
    """Provides a fresh EventBus instance for each test."""
    return LocalEventBus()


class TestLocalEventBus:
    """Tests for the LocalEventBus class."""

    def test_subscribe_and_unsubscribe(self, event_bus):
        """Verifies that handlers can be subscribed and unsubscribed correctly."""
        handler = AsyncMock()
        
        # Subscribe
        event_bus.subscribe("task.*", handler)
        assert len(event_bus._subscribers["task.*"]) == 1
        assert handler in event_bus._subscribers["task.*"]
        
        # Unsubscribe
        event_bus.unsubscribe("task.*", handler)
        assert len(event_bus._subscribers["task.*"]) == 0
        assert handler not in event_bus._subscribers["task.*"]

    @pytest.mark.asyncio
    async def test_publish_single_pattern_match(self, event_bus):
        """Verifies that events are published to matching subscribers."""
        handler = AsyncMock()
        event_bus.subscribe("task.created", handler)
        
        await event_bus.publish("task.created", {"id": 1, "name": "Test task"})
        
        handler.assert_awaited_once_with("task.created", {"id": 1, "name": "Test task"})

    @pytest.mark.asyncio
    async def test_publish_wildcard_pattern_match(self, event_bus):
        """Verifies that wildcard patterns match correctly."""
        handler = AsyncMock()
        event_bus.subscribe("task.*", handler)
        
        await event_bus.publish("task.created", {"id": 1})
        handler.assert_awaited_once_with("task.created", {"id": 1})
        
        # Test another wildcard match
        handler.reset_mock()
        await event_bus.publish("task.completed", {"id": 2})
        handler.assert_awaited_once_with("task.completed", {"id": 2})

    @pytest.mark.asyncio
    async def test_publish_no_match(self, event_bus):
        """Verifies that events with no matching subscribers don't cause errors."""
        # No subscribers
        await event_bus.publish("task.created", {"id": 1})
        
        # Subscriber for different pattern
        handler = AsyncMock()
        event_bus.subscribe("presence.*", handler)
        await event_bus.publish("task.created", {"id": 1})
        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_publish_multiple_subscribers(self, event_bus):
        """Verifies that events are delivered to multiple subscribers."""
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        
        event_bus.subscribe("task.*", handler1)
        event_bus.subscribe("task.*", handler2)
        
        await event_bus.publish("task.created", {"id": 1})
        
        handler1.assert_awaited_once_with("task.created", {"id": 1})
        handler2.assert_awaited_once_with("task.created", {"id": 1})

    @pytest.mark.asyncio
    async def test_publish_multiple_patterns(self, event_bus):
        """Verifies that events are delivered to subscribers matching different patterns."""
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        
        event_bus.subscribe("task.*", handler1)
        event_bus.subscribe("presence.*", handler2)
        
        await event_bus.publish("task.created", {"id": 1})
        handler1.assert_awaited_once_with("task.created", {"id": 1})
        handler2.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_error_handling_in_handler(self, event_bus, caplog):
        """Verifies that exceptions in handlers don't crash the bus."""
        async def failing_handler(topic, data):
            raise ValueError("Test error")
        
        event_bus.subscribe("task.*", failing_handler)
        
        # Should not raise an exception
        await event_bus.publish("task.created", {"id": 1})
        
        # Check that error was logged
        assert "EventBus error" in caplog.text
        assert "Test error" in caplog.text

    @pytest.mark.asyncio
    async def test_async_execution(self, event_bus):
        """Verifies that handlers are executed asynchronously."""
        # This test is more about ensuring the method can be called without errors
        # The actual async behavior is tested through other methods
        handler = AsyncMock()
        event_bus.subscribe("task.*", handler)
        
        # Just verify it doesn't raise an exception
        await event_bus.publish("task.created", {"id": 1})
        handler.assert_awaited_once_with("task.created", {"id": 1})

    @pytest.mark.asyncio
    async def test_inject_test_event(self, event_bus):
        """Verifies that the test injection method works correctly."""
        handler = AsyncMock()
        event_bus.subscribe("test.event", handler)
        
        # This should publish and then wait briefly
        await event_bus.inject_test_event("test.event", {"data": "test"}, settle_time=0.01)
        
        handler.assert_awaited_once_with("test.event", {"data": "test"})
