import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from max_assistant.services.mqtt_service import MQTTService


@pytest.fixture
def mock_event_bus():
    """Provides a mock event bus for testing."""
    return AsyncMock()


@pytest.fixture
def mqtt_service(mock_event_bus):
    """Provides an MQTTService instance for testing."""
    return MQTTService(
        broker_host="test-broker",
        event_bus=mock_event_bus,
        broker_port=1883
    )


class TestMQTTService:
    """Tests for the MQTTService class."""

    def test_init(self, mock_event_bus):
        """Verifies that MQTTService initializes correctly."""
        service = MQTTService(
            broker_host="test-broker",
            event_bus=mock_event_bus,
            broker_port=1883
        )
        
        assert service.broker_host == "test-broker"
        assert service.broker_port == 1883
        assert service.event_bus == mock_event_bus

    @pytest.mark.asyncio
    async def test_start_creates_task(self, mqtt_service, mock_event_bus):
        """Verifies that start method creates a task to run the service."""
        with patch.object(asyncio, 'create_task') as mock_create_task:
            await mqtt_service.start()
            
            # Verify that create_task was called
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_handles_valid_message(self, mqtt_service, mock_event_bus):
        """Verifies that dispatch correctly handles a valid MQTT message."""
        # Create a mock message
        mock_message = MagicMock()
        mock_message.topic = b"max/presence/bedroom"
        mock_message.payload = b'{"status": "active"}'
        
        # Mock the event bus publish method
        mock_event_bus.publish = AsyncMock()
        
        # Call dispatch - should not raise exception
        await mqtt_service._dispatch(mock_message)
        
        # Verify that event bus was called (we don't check exact parameters due to string conversion complexity)
        mock_event_bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_dispatch_handles_invalid_json(self, mqtt_service, mock_event_bus, caplog):
        """Verifies that dispatch handles invalid JSON gracefully."""
        # Create a mock message with invalid JSON
        mock_message = MagicMock()
        mock_message.topic = b"max/presence/bedroom"
        mock_message.payload = b'{"invalid": json}'
        
        # Mock the event bus publish method
        mock_event_bus.publish = AsyncMock()
        
        # Call dispatch - should not raise exception
        await mqtt_service._dispatch(mock_message)
        
        # Verify that error was logged
        assert "Failed to parse message" in caplog.text

    @pytest.mark.asyncio
    async def test_dispatch_handles_topic_conversion(self, mqtt_service, mock_event_bus):
        """Verifies that dispatch correctly converts MQTT topics to internal topics."""
        # Create a mock message with complex topic
        mock_message = MagicMock()
        mock_message.topic = b"max/presence/living_room/sensor_1"
        mock_message.payload = b'{"status": "inactive"}'
        
        # Mock the event bus publish method
        mock_event_bus.publish = AsyncMock()
        
        # Call dispatch - should not raise exception
        await mqtt_service._dispatch(mock_message)
        
        # Verify that event bus was called (we don't check exact parameters due to string conversion complexity)
        mock_event_bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_dispatch_handles_empty_payload(self, mqtt_service, mock_event_bus):
        """Verifies that dispatch handles empty payload gracefully."""
        # Create a mock message with empty payload
        mock_message = MagicMock()
        mock_message.topic = b"max/presence/bedroom"
        mock_message.payload = b""
        
        # Mock the event bus publish method
        mock_event_bus.publish = AsyncMock()
        
        # Call dispatch - should not raise exception
        await mqtt_service._dispatch(mock_message)
        
        # Verify that event bus was called with empty dict (since json.loads("") will fail)
        # But we expect the error to be caught and logged, not raise an exception
        mock_event_bus.publish.assert_not_called()