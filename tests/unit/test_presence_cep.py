import asyncio
import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from max_assistant.services.presence_cep import PresenceCEPWorker


@pytest.fixture
def mock_config_path(tmp_path):
    """Provides a mock config path for testing."""
    return tmp_path / "test_config.yaml"


@pytest.fixture
def presence_cep_worker():
    """Provides a PresenceCEPWorker instance for testing."""
    with patch('max_assistant.services.presence_cep.resolve_project_path') as mock_resolve:
        # Mock the config path to not exist, so default rules are loaded
        mock_resolve.return_value = MagicMock()
        mock_resolve.return_value.exists.return_value = False
        return PresenceCEPWorker(
            broker_host="test-broker",
            broker_port=1883,
            config_path="test_config.yaml"
        )


class TestPresenceCEPWorker:
    """Tests for the PresenceCEPWorker class."""

    def test_init_sets_default_values(self, presence_cep_worker):
        """Verifies that PresenceCEPWorker initializes with correct default values."""
        # Since the config file doesn't exist, default rules should be loaded
        assert presence_cep_worker.broker_host == "test-broker"
        assert presence_cep_worker.broker_port == 1883
        assert presence_cep_worker._thread is None
        assert isinstance(presence_cep_worker._stop_event, threading.Event)
        # Default rules should be loaded (not empty list)
        assert len(presence_cep_worker._rules) > 0

    def test_start_already_running(self, presence_cep_worker):
        """Verifies that start handles already running thread gracefully."""
        # Mock an existing thread that is alive
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        presence_cep_worker._thread = mock_thread
        
        # Call start - should not create a new thread
        presence_cep_worker.start()
        
        # Thread should remain the same
        assert presence_cep_worker._thread == mock_thread

    @pytest.mark.asyncio
    async def test_process_raw_sensor_handles_valid_message(self, presence_cep_worker):
        """Verifies that process_raw_sensor handles valid sensor messages correctly."""
        # Mock an aiomqtt client
        mock_client = AsyncMock()
        
        # Create a mock message with valid topic and payload
        mock_message = MagicMock()
        mock_message.topic = b"home/bedroom/sensor/door"
        mock_message.payload = b'{"state": "open", "user_id": "user123"}'
        
        # Mock the _evaluate_sequences method to avoid complex async calls
        with patch.object(presence_cep_worker, '_evaluate_sequences') as mock_evaluate:
            await presence_cep_worker._process_raw_sensor(mock_client, mock_message)
            
            # Verify that _evaluate_sequences was called
            assert mock_evaluate.called

    @pytest.mark.asyncio
    async def test_process_raw_sensor_handles_invalid_json(self, presence_cep_worker):
        """Verifies that process_raw_sensor handles invalid JSON gracefully."""
        # Mock an aiomqtt client
        mock_client = AsyncMock()
        
        # Create a mock message with invalid payload
        mock_message = MagicMock()
        mock_message.topic = b"home/bedroom/sensor/door"
        mock_message.payload = b'{"invalid": json}'
        
        # Call process_raw_sensor - should not raise exception
        # We can't easily test logging, but we just want to make sure it doesn't crash
        try:
            await presence_cep_worker._process_raw_sensor(mock_client, mock_message)
        except Exception as e:
            pytest.fail(f"process_raw_sensor raised exception: {e}")

    @pytest.mark.asyncio
    async def test_evaluate_sequences_handles_empty_rules(self, presence_cep_worker):
        """Verifies that evaluate_sequences handles empty rules gracefully."""
        # Set up an empty rules list
        presence_cep_worker._rules = []
        
        # Mock client
        mock_client = AsyncMock()
        
        # Call evaluate_sequences - should not raise exception
        await presence_cep_worker._evaluate_sequences(mock_client, 1000.0)
        
        # Should not crash and should not call publish

    def test_match_sequence_handles_empty_history(self, presence_cep_worker):
        """Verifies that match_sequence handles empty sensor history correctly."""
        # Clear the sensor history
        presence_cep_worker._sensor_history = []
        
        # Create a simple sequence rule
        sequence = [
            {"room": "bedroom", "sensor": "door", "state": "open"}
        ]
        
        # Call match_sequence - should return False
        matched, user_id = presence_cep_worker._match_sequence(sequence, 30.0, 1000.0)
        
        assert matched is False

    def test_match_sequence_handles_matching_sequence(self, presence_cep_worker):
        """Verifies that match_sequence correctly identifies matching sequences."""
        # Add some sensor events to history
        presence_cep_worker._sensor_history = [
            {
                "room": "bedroom",
                "sensor": "door",
                "state": "open",
                "timestamp": 1000.0,
                "user_id": "user123"
            }
        ]
        
        # Create a matching sequence rule
        sequence = [
            {"room": "bedroom", "sensor": "door", "state": "open"}
        ]
        
        # Call match_sequence - should return True
        matched, user_id = presence_cep_worker._match_sequence(sequence, 30.0, 1020.0)
        
        assert matched is True
        assert user_id == "user123"

    @pytest.mark.asyncio
    async def test_listen_and_synthesize_handles_client_disconnect(self, presence_cep_worker):
        """Verifies that listen_and_synthesize handles client disconnects gracefully."""
        # Mock aiomqtt client to raise an exception on subscription
        mock_client = AsyncMock()
        mock_client.__aenter__.side_effect = Exception("Connection failed")
        
        with patch('max_assistant.services.presence_cep.aiomqtt.Client') as mock_client_class:
            mock_client_class.return_value = mock_client
            
            # Mock the stop event to make it stop quickly
            presence_cep_worker._stop_event.set()
            
            # Call listen_and_synthesize - should not raise exception
            try:
                await presence_cep_worker._listen_and_synthesize()
            except Exception as e:
                pytest.fail(f"_listen_and_synthesize raised exception: {e}")

    def test_run_thread_creates_event_loop(self, presence_cep_worker):
        """Verifies that run_thread creates and manages event loop correctly."""
        # Mock the _listen_and_synthesize method to avoid actual async operations
        with patch.object(presence_cep_worker, '_listen_and_synthesize') as mock_listen:
            # Call _run_thread - should not raise exception
            presence_cep_worker._run_thread()
            
            # Verify that listen was called (this is the key verification)
            assert mock_listen.called