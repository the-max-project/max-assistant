import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, call, patch

import aiomqtt
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
    async def test_start_creates_task(self, mqtt_service):
        """Verifies that start method creates a task to run the service."""
        with patch.object(mqtt_service, "_run", new_callable=AsyncMock) as run:
            await mqtt_service.start()
            task = mqtt_service._task
            await task
            run.assert_awaited_once()
            assert task.done()

    @pytest.mark.asyncio
    async def test_start_is_idempotent_and_restarts_finished_task(self, mqtt_service):
        started = asyncio.Event()
        release = asyncio.Event()

        async def run():
            started.set()
            await release.wait()

        with patch.object(mqtt_service, "_run", side_effect=run) as mock_run:
            await mqtt_service.start()
            first_task = mqtt_service._task
            await started.wait()
            await mqtt_service.start()
            assert mqtt_service._task is first_task
            assert mock_run.call_count == 1
            release.set()
            await first_task
            await mqtt_service.start()
            assert mqtt_service._task is not first_task
            await mqtt_service._task
            assert mock_run.call_count == 2

    @pytest.mark.asyncio
    async def test_stop_cancels_and_waits_for_task(self, mqtt_service, caplog):
        caplog.set_level(logging.INFO)
        started = asyncio.Event()
        teardown = asyncio.Event()

        async def run():
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                teardown.set()

        with patch.object(mqtt_service, "_run", side_effect=run):
            await mqtt_service.start()
            task = mqtt_service._task
            await started.wait()
            await mqtt_service.stop()

        assert task.cancelled()
        assert teardown.is_set()
        assert mqtt_service._task is None
        assert "MQTTService: Stopped." in caplog.text

    @pytest.mark.asyncio
    async def test_stop_without_task(self, mqtt_service):
        await mqtt_service.stop()
        assert mqtt_service._task is None

    @pytest.mark.asyncio
    async def test_run_subscribes_and_dispatches(self, mqtt_service):
        message = MagicMock()
        client = MagicMock()
        client.subscribe = AsyncMock()

        async def messages():
            yield message
            raise asyncio.CancelledError

        client.messages = messages()
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=client)
        context.__aexit__ = AsyncMock(return_value=False)
        with patch("max_assistant.services.mqtt_service.aiomqtt.Client", return_value=context) as factory, \
                patch.object(mqtt_service, "_dispatch", new_callable=AsyncMock) as dispatch:
            await mqtt_service._run()

        factory.assert_called_once_with("test-broker", port=1883)
        assert client.subscribe.await_args_list == [call("max/presence/#"), call("max/state/#")]
        dispatch.assert_awaited_once_with(message)
        context.__aexit__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_retries_mqtt_error(self, mqtt_service, caplog):
        context = MagicMock()
        context.__aenter__ = AsyncMock(side_effect=aiomqtt.MqttError("connection lost"))
        context.__aexit__ = AsyncMock(return_value=False)
        with patch("max_assistant.services.mqtt_service.aiomqtt.Client", return_value=context) as factory, \
                patch("max_assistant.services.mqtt_service.asyncio.sleep", new_callable=AsyncMock) as sleep:
            factory.side_effect = [context, asyncio.CancelledError()]
            await mqtt_service._run()

        sleep.assert_awaited_once_with(5)
        assert factory.call_count == 2
        assert "connection lost" in caplog.text

    @pytest.mark.asyncio
    async def test_dispatch_handles_valid_message(self, mqtt_service, mock_event_bus):
        """Verifies that dispatch correctly handles a valid MQTT message."""
        # Create a mock message
        mock_message = MagicMock()
        mock_message.topic = "max/presence/bedroom"
        mock_message.payload = b'{"status": "active"}'
        
        # Mock the event bus publish method
        mock_event_bus.publish = AsyncMock()
        
        # Call dispatch - should not raise exception
        await mqtt_service._dispatch(mock_message)
        
        mock_event_bus.publish.assert_awaited_once_with("presence.bedroom", {"status": "active"})

    @pytest.mark.asyncio
    async def test_dispatch_handles_invalid_json(self, mqtt_service, mock_event_bus, caplog):
        """Verifies that dispatch handles invalid JSON gracefully."""
        # Create a mock message with invalid JSON
        mock_message = MagicMock()
        mock_message.topic = "max/presence/bedroom"
        mock_message.payload = b'{"invalid": json}'
        
        # Mock the event bus publish method
        mock_event_bus.publish = AsyncMock()
        
        # Call dispatch - should not raise exception
        await mqtt_service._dispatch(mock_message)
        
        # Verify that error was logged
        assert "Failed to parse message" in caplog.text
        mock_event_bus.publish.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("topic, expected", [
        ("max/presence/living_room/sensor_1", "presence.living_room.sensor_1"),
        ("max/state/living_room/light", "state.living_room.light"),
    ])
    async def test_dispatch_handles_topic_conversion(self, mqtt_service, mock_event_bus, topic, expected):
        """Verifies that dispatch correctly converts MQTT topics to internal topics."""
        # Create a mock message with complex topic
        mock_message = MagicMock()
        mock_message.topic = topic
        mock_message.payload = b'{"status": "inactive"}'
        
        # Mock the event bus publish method
        mock_event_bus.publish = AsyncMock()
        
        # Call dispatch - should not raise exception
        await mqtt_service._dispatch(mock_message)
        
        mock_event_bus.publish.assert_awaited_once_with(expected, {"status": "inactive"})

    @pytest.mark.asyncio
    async def test_dispatch_handles_empty_payload(self, mqtt_service, mock_event_bus, caplog):
        """Verifies that dispatch handles empty payload gracefully."""
        # Create a mock message with empty payload
        mock_message = MagicMock()
        mock_message.topic = "max/presence/bedroom"
        mock_message.payload = b""
        
        # Mock the event bus publish method
        mock_event_bus.publish = AsyncMock()
        
        # Call dispatch - should not raise exception
        await mqtt_service._dispatch(mock_message)
        
        assert "Failed to parse message" in caplog.text
        mock_event_bus.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_handles_invalid_utf8(self, mqtt_service, mock_event_bus, caplog):
        message = MagicMock(topic="max/presence/bedroom", payload=b"\xff")
        await mqtt_service._dispatch(message)
        assert "Failed to parse message" in caplog.text
        mock_event_bus.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_handles_publish_failure(self, mqtt_service, mock_event_bus, caplog):
        message = MagicMock(topic="max/state/light", payload=b'{"on": true}')
        mock_event_bus.publish.side_effect = RuntimeError("publish failed")
        await mqtt_service._dispatch(message)
        mock_event_bus.publish.assert_awaited_once_with("state.light", {"on": True})
        assert "publish failed" in caplog.text