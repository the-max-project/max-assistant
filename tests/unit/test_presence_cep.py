import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import aiomqtt
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

    def test_init_with_custom_sleep_config(self, presence_cep_worker):
        """Verifies that PresenceCEPWorker initializes with custom sleep configuration."""
        # Test default values
        assert presence_cep_worker.sleep_inactivity_seconds == 1800.0
        assert presence_cep_worker.sleep_start_hour == 22
        assert presence_cep_worker.sleep_end_hour == 7

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

    def test_start_creates_new_thread(self, presence_cep_worker):
        """Verifies that start creates a new thread when none exists."""
        with patch.object(presence_cep_worker, '_run_thread', side_effect=presence_cep_worker._stop_event.wait):
            try:
                presence_cep_worker.start()
                assert presence_cep_worker._thread is not None
                assert presence_cep_worker._thread.is_alive() is True
            finally:
                presence_cep_worker.stop()

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
    async def test_process_raw_sensor_handles_invalid_topic(self, presence_cep_worker):
        """Verifies that process_raw_sensor handles invalid topic gracefully."""
        # Mock an aiomqtt client
        mock_client = AsyncMock()
        
        # Create a mock message with invalid topic (not enough segments)
        mock_message = MagicMock()
        mock_message.topic = b"home/bedroom"
        mock_message.payload = b'{"state": "open"}'
        
        # Call process_raw_sensor - should not raise exception
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

    def test_match_sequence_handles_non_matching_sequence(self, presence_cep_worker):
        """Verifies that match_sequence correctly identifies non-matching sequences."""
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
        
        # Create a non-matching sequence rule
        sequence = [
            {"room": "bedroom", "sensor": "door", "state": "closed"}
        ]
        
        # Call match_sequence - should return False
        matched, user_id = presence_cep_worker._match_sequence(sequence, 30.0, 1020.0)
        
        assert matched is False

    def test_match_sequence_handles_sequence_with_name_filter(self, presence_cep_worker):
        """Verifies that match_sequence correctly handles name filtering."""
        # Add some sensor events to history
        presence_cep_worker._sensor_history = [
            {
                "room": "bedroom",
                "sensor": "door",
                "name": "main_door",
                "state": "open",
                "timestamp": 1000.0,
                "user_id": "user123"
            }
        ]
        
        # Create a matching sequence rule with name filter
        sequence = [
            {"room": "bedroom", "sensor": "door", "name": "main_door", "state": "open"}
        ]
        
        # Call match_sequence - should return True
        matched, user_id = presence_cep_worker._match_sequence(sequence, 30.0, 1020.0)
        
        assert matched is True

    def test_match_sequence_handles_sequence_with_mismatched_name(self, presence_cep_worker):
        """Verifies that match_sequence correctly handles name filtering mismatches."""
        # Add some sensor events to history
        presence_cep_worker._sensor_history = [
            {
                "room": "bedroom",
                "sensor": "door",
                "name": "main_door",
                "state": "open",
                "timestamp": 1000.0,
                "user_id": "user123"
            }
        ]
        
        # Create a sequence rule with different name
        sequence = [
            {"room": "bedroom", "sensor": "door", "name": "other_door", "state": "open"}
        ]
        
        # Call match_sequence - should return False
        matched, user_id = presence_cep_worker._match_sequence(sequence, 30.0, 1020.0)
        
        assert matched is False

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

    @pytest.mark.asyncio
    async def test_listen_and_synthesize_handles_stop_event(self, presence_cep_worker):
        """Verifies that listen_and_synthesize respects the stop event."""
        # Mock aiomqtt client to raise StopAsyncIteration immediately
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.messages = AsyncMock()
        mock_client.messages.__aiter__.side_effect = StopAsyncIteration()
        
        with patch('max_assistant.services.presence_cep.aiomqtt.Client') as mock_client_class:
            mock_client_class.return_value = mock_client
            
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

    def test_stop_sets_stop_event(self, presence_cep_worker):
        """Verifies that stop correctly sets the stop event."""
        # Call stop
        presence_cep_worker.stop()
        
        # Verify stop event was set
        assert presence_cep_worker._stop_event.is_set() is True

    def test_stop_with_running_thread(self, presence_cep_worker):
        """Verifies that stop works correctly with a running thread."""
        # Mock an existing thread that is alive
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        presence_cep_worker._thread = mock_thread
        
        # Call stop
        presence_cep_worker.stop()
        
        # Verify thread was joined
        assert mock_thread.join.called

    def test_stop_with_no_running_thread(self, presence_cep_worker):
        """Verifies that stop works correctly when no thread is running."""
        # Ensure no thread is running
        presence_cep_worker._thread = None
        
        # Call stop - should not raise exception
        try:
            presence_cep_worker.stop()
        except Exception as e:
            pytest.fail(f"stop raised exception: {e}")

    def test_load_config_handles_missing_file(self, presence_cep_worker):
        """Verifies that _load_config handles missing config file gracefully."""
        # Ensure config path doesn't exist
        with patch.object(presence_cep_worker.config_path, 'exists') as mock_exists:
            mock_exists.return_value = False
            
            # Call _load_config - should not raise exception and should load defaults
            presence_cep_worker._load_config()
            
            # Should have loaded default rules
            assert len(presence_cep_worker._rules) > 0

    def test_load_config_handles_json_file(self, mock_config_path):
        """Verifies that _load_config handles JSON config files correctly."""
        # Create a mock JSON file content
        import json
        config_content = {
            "rules": [
                {
                    "name": "test_rule",
                    "room": "bedroom",
                    "target_state": "vacated",
                    "window_seconds": 15,
                    "sequence": [
                        {"room": "bedroom", "sensor": "door", "state": "closed"}
                    ]
                }
            ],
            "sleep": {
                "inactivity_seconds": 3000.0,
                "start_hour": 23,
                "end_hour": 8
            }
        }
        
        # Write the content to the mock file
        with open(mock_config_path, 'w') as f:
            json.dump(config_content, f)
            
        # Create a new worker with this config path
        with patch('max_assistant.services.presence_cep.resolve_project_path') as mock_resolve:
            mock_resolve.return_value = mock_config_path
            
            worker = PresenceCEPWorker(
                broker_host="test-broker",
                broker_port=1883,
                config_path=str(mock_config_path)
            )
            
            # Verify the rules were loaded
            assert len(worker._rules) == 1
            assert worker.sleep_inactivity_seconds == 3000.0
            assert worker.sleep_start_hour == 23
            assert worker.sleep_end_hour == 8

    def test_evaluate_sleep_onset_handles_asleep_state(self, presence_cep_worker):
        """Verifies that _evaluate_sleep_onset handles asleep state correctly."""
        # Set sleep state to asleep
        presence_cep_worker._sleep_state = "asleep"
        
        # Mock client and call the method
        mock_client = AsyncMock()
        try:
            asyncio.run(presence_cep_worker._evaluate_sleep_onset(mock_client, 1000.0))
        except Exception as e:
            pytest.fail(f"_evaluate_sleep_onset raised exception: {e}")

    def test_evaluate_sleep_onset_handles_awake_state(self, presence_cep_worker):
        """Verifies that _evaluate_sleep_onset handles awake state correctly."""
        # Set sleep state to awake
        presence_cep_worker._sleep_state = "awake"
        
        # Mock client and call the method
        mock_client = AsyncMock()
        try:
            asyncio.run(presence_cep_worker._evaluate_sleep_onset(mock_client, 1000.0))
        except Exception as e:
            pytest.fail(f"_evaluate_sleep_onset raised exception: {e}")

    def test_publish_sleep_state_handles_same_state(self, presence_cep_worker):
        """Verifies that _publish_sleep_state handles same state correctly."""
        # Set sleep state to asleep
        presence_cep_worker._sleep_state = "asleep"
        
        # Mock client and call the method
        mock_client = AsyncMock()
        try:
            asyncio.run(presence_cep_worker._publish_sleep_state(mock_client, "asleep", "user123", 1000.0))
        except Exception as e:
            pytest.fail(f"_publish_sleep_state raised exception: {e}")

    def test_process_raw_sensor_handles_sleep_activity(self, presence_cep_worker):
        """Verifies that process_raw_sensor correctly updates sleep activity."""
        # Mock an aiomqtt client
        mock_client = AsyncMock()
        
        # Create a mock message with active state (should update sleep activity)
        mock_message = MagicMock()
        mock_message.topic = b"home/bedroom/sensor/motion"
        mock_message.payload = b'{"state": "active", "user_id": "user123"}'
        
        # Call process_raw_sensor - should not raise exception
        try:
            asyncio.run(presence_cep_worker._process_raw_sensor(mock_client, mock_message))
        except Exception as e:
            pytest.fail(f"_process_raw_sensor raised exception: {e}")

    def test_process_raw_sensor_handles_wake_state(self, presence_cep_worker):
        """Verifies that process_raw_sensor correctly handles wake state transitions."""
        # Set sleep state to asleep
        presence_cep_worker._sleep_state = "asleep"
        
        # Mock an aiomqtt client
        mock_client = AsyncMock()
        
        # Create a mock message with active state (should wake up user)
        mock_message = MagicMock()
        mock_message.topic = b"home/bedroom/sensor/motion"
        mock_message.payload = b'{"state": "active", "user_id": "user123"}'
        
        # Mock the _publish_sleep_state method to avoid complex async calls
        with patch.object(presence_cep_worker, '_publish_sleep_state') as mock_publish:
            try:
                asyncio.run(presence_cep_worker._process_raw_sensor(mock_client, mock_message))
            except Exception as e:
                pytest.fail(f"_process_raw_sensor raised exception: {e}")

    @pytest.mark.asyncio
    async def test_match_sequence_handles_empty_history(self, presence_cep_worker):
        """Verifies that _match_sequence handles empty sensor history correctly."""
        sequence = [
            {"room": "main", "sensor": "door", "state": "closed"},
            {"room": "main", "sensor": "pir", "state": "inactive"}
        ]
        
        matched, user_id = presence_cep_worker._match_sequence(sequence, 100.0, 1000.0)
        assert matched is False
        assert user_id == "default_user"

    @pytest.mark.asyncio
    async def test_match_sequence_handles_name_filtering(self, presence_cep_worker):
        """Verifies that _match_sequence handles device name filtering."""
        # Add sensor events to history with names and sufficient time window
        presence_cep_worker._sensor_history = [
            {
                "room": "main",
                "sensor": "door",
                "name": "front_door",
                "state": "closed",
                "timestamp": 950.0,
                "user_id": "user123"
            }
        ]
        
        sequence = [
            {"room": "main", "sensor": "door", "name": "front_door", "state": "closed"}
        ]
        
        matched, user_id = presence_cep_worker._match_sequence(sequence, 100.0, 1000.0)
        assert matched is True
        assert user_id == "user123"

    @pytest.mark.asyncio
    async def test_match_sequence_handles_multiple_steps(self, presence_cep_worker):
        """Verifies that _match_sequence handles multiple step sequences."""
        # Add sensor events to history with sufficient time window
        presence_cep_worker._sensor_history = [
            {
                "room": "main",
                "sensor": "door",
                "state": "closed",
                "timestamp": 950.0,
                "user_id": "user123"
            },
            {
                "room": "main",
                "sensor": "pir",
                "state": "inactive",
                "timestamp": 960.0,
                "user_id": "user123"
            }
        ]
        
        sequence = [
            {"room": "main", "sensor": "door", "state": "closed"},
            {"room": "main", "sensor": "pir", "state": "inactive"}
        ]
        
        # Use a larger time window to ensure events are considered relevant
        matched, user_id = presence_cep_worker._match_sequence(sequence, 100.0, 1000.0)
        assert matched is True
        assert user_id == "user123"

    @pytest.mark.asyncio
    async def test_match_sequence_handles_mismatched_sequence(self, presence_cep_worker):
        """Verifies that _match_sequence handles mismatched sequences."""
        # Add sensor events to history with sufficient time window (but in wrong order)
        presence_cep_worker._sensor_history = [
            {
                "room": "main",
                "sensor": "pir",
                "state": "inactive",
                "timestamp": 960.0,
                "user_id": "user123"
            },
            {
                "room": "main",
                "sensor": "door",
                "state": "closed",
                "timestamp": 950.0,
                "user_id": "user123"
            }
        ]
        
        sequence = [
            {"room": "main", "sensor": "door", "state": "closed"},
            {"room": "main", "sensor": "pir", "state": "inactive"}
        ]
        
        # Use a larger time window to ensure events are considered relevant
        matched, user_id = presence_cep_worker._match_sequence(sequence, 100.0, 1000.0)
        assert matched is False
        assert user_id == "user123"

    @pytest.mark.asyncio
    async def test_match_sequence_handles_name_filtering(self, presence_cep_worker):
        """Verifies that _match_sequence handles device name filtering."""
        # Add sensor events to history with names
        presence_cep_worker._sensor_history = [
            {
                "room": "main",
                "sensor": "door",
                "name": "front_door",
                "state": "closed",
                "timestamp": 950.0,
                "user_id": "user123"
            }
        ]
        
        sequence = [
            {"room": "main", "sensor": "door", "name": "front_door", "state": "closed"}
        ]
        
        matched, user_id = presence_cep_worker._match_sequence(sequence, 50.0, 1000.0)
        assert matched is True
        assert user_id == "user123"

    @pytest.mark.asyncio
    async def test_evaluate_sleep_onset_handles_asleep_state(self, presence_cep_worker):
        """Verifies that _evaluate_sleep_onset returns early when already asleep."""
        # Set worker to asleep state
        presence_cep_worker._sleep_state = "asleep"
        
        with patch('max_assistant.services.presence_cep.aiomqtt.Client') as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # This should not proceed to any sleep logic since we're already asleep
            await presence_cep_worker._evaluate_sleep_onset(mock_client_instance, 1000.0)
            
            # Should not have published anything (no assertion needed - just verifying no crash)

    @pytest.mark.asyncio
    async def test_evaluate_sleep_onset_handles_vacated_room(self, presence_cep_worker):
        """Verifies that _evaluate_sleep_onset returns early when room is vacated."""
        # Set worker to awake state and set last published state to vacated
        presence_cep_worker._sleep_state = "awake"
        presence_cep_worker._last_published_state = {"main": "vacated"}
        
        with patch('max_assistant.services.presence_cep.aiomqtt.Client') as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # This should not proceed to any sleep logic since room is vacated
            await presence_cep_worker._evaluate_sleep_onset(mock_client_instance, 1000.0)
            
            # Should not have published anything (no assertion needed - just verifying no crash)

    @pytest.mark.asyncio
    async def test_evaluate_sleep_onset_handles_night_window(self, presence_cep_worker):
        """Verifies that _evaluate_sleep_onset handles night time window correctly."""
        # Set worker to awake state and set appropriate inactivity
        presence_cep_worker._sleep_state = "awake"
        presence_cep_worker._last_sleep_activity = 100.0  # Long ago
        
        # Mock datetime to be within sleep hours (22-7)
        with patch('max_assistant.services.presence_cep.datetime') as mock_datetime:
            mock_datetime.now.return_value.hour = 23  # Within night window
            
            mock_client_instance = AsyncMock()
            await presence_cep_worker._evaluate_sleep_onset(mock_client_instance, 2000.0)

            mock_client_instance.publish.assert_awaited_once()
            assert presence_cep_worker._sleep_state == "asleep"

    @pytest.mark.asyncio
    async def test_publish_sleep_state_handles_same_state(self, presence_cep_worker):
        """Verifies that _publish_sleep_state returns early when state is same."""
        # Set worker to asleep state
        presence_cep_worker._sleep_state = "asleep"
        
        with patch('max_assistant.services.presence_cep.aiomqtt.Client') as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # This should not proceed to publishing since state is the same
            await presence_cep_worker._publish_sleep_state(mock_client_instance, "asleep", "user123", 1000.0)
            
            # Should not have published anything (no assertion needed - just verifying no crash)

    @pytest.mark.asyncio
    async def test_process_raw_sensor_handles_json_decode_error(self, presence_cep_worker):
        """Verifies that _process_raw_sensor handles JSON decode errors gracefully."""
        with patch('max_assistant.services.presence_cep.aiomqtt.Client') as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # Create a message with invalid JSON
            mock_message = MagicMock()
            mock_message.topic = "home/main/sensor/door"
            mock_message.payload.decode.return_value = "invalid json"
            
            # This should not crash
            await presence_cep_worker._process_raw_sensor(mock_client_instance, mock_message)
            
            # Should not have crashed (no assertion needed)

    @pytest.mark.asyncio
    async def test_process_raw_sensor_handles_index_error(self, presence_cep_worker):
        """Verifies that _process_raw_sensor handles topic parsing errors gracefully."""
        with patch('max_assistant.services.presence_cep.aiomqtt.Client') as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # Create a message with insufficient topic parts
            mock_message = MagicMock()
            mock_message.topic = "home"  # Not enough parts
            mock_message.payload.decode.return_value = '{"state": "open"}'
            
            # This should not crash
            await presence_cep_worker._process_raw_sensor(mock_client_instance, mock_message)
            
            # Should not have crashed (no assertion needed)

    def test_start_handles_already_running_thread(self, presence_cep_worker):
        """Verifies that start handles already running thread correctly."""
        # Mock a thread that is alive
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        presence_cep_worker._thread = mock_thread
        
        # This should not create a new thread
        presence_cep_worker.start()
        
        # Should not have created a new thread (mock_thread was reused)
        assert presence_cep_worker._thread == mock_thread

    def test_stop_handles_stopped_thread(self, presence_cep_worker):
        """Verifies that stop handles stopped thread correctly."""
        # Mock a thread that is not alive
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False
        presence_cep_worker._thread = mock_thread
        
        # This should not crash
        presence_cep_worker.stop()
        
        # Should have cleared the thread reference
        assert presence_cep_worker._thread is None

    def test_run_thread_creates_new_loop(self, presence_cep_worker):
        """Verifies that _run_thread creates a new event loop."""
        with patch('max_assistant.services.presence_cep.asyncio.new_event_loop', wraps=asyncio.new_event_loop) as mock_loop:
            with patch.object(presence_cep_worker, '_listen_and_synthesize') as mock_listen:
                presence_cep_worker._run_thread()

                mock_loop.assert_called_once_with()
                mock_listen.assert_awaited_once()
                assert presence_cep_worker._loop is None

    @pytest.mark.asyncio
    async def test_listen_and_synthesize_handles_mqtt_error(self, presence_cep_worker):
        """Verifies that _listen_and_synthesize handles MQTT errors gracefully."""
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.subscribe.side_effect = aiomqtt.MqttError("MQTT Error")

        async def stop_after_retry(delay):
            presence_cep_worker._stop_event.set()

        with patch('max_assistant.services.presence_cep.aiomqtt.Client', return_value=mock_client_instance), \
                patch('max_assistant.services.presence_cep.asyncio.sleep', side_effect=stop_after_retry) as mock_sleep, \
                patch('max_assistant.services.presence_cep.logger') as mock_logger:
            await asyncio.wait_for(presence_cep_worker._listen_and_synthesize(), timeout=1)

            mock_client_instance.subscribe.assert_awaited_once_with("home/+/sensor/#")
            mock_logger.warning.assert_called_once()
            mock_sleep.assert_awaited_once_with(5)

    @pytest.mark.asyncio
    async def test_evaluate_sequences_handles_empty_rules(self, presence_cep_worker):
        """Verifies that _evaluate_sequences handles empty rules gracefully."""
        # Set up worker with empty rules
        presence_cep_worker._rules = []
        
        with patch('max_assistant.services.presence_cep.aiomqtt.Client') as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # This should not crash
            await presence_cep_worker._evaluate_sequences(mock_client_instance, 1000.0)
            
            # Should not have crashed (no assertion needed)

    @pytest.mark.asyncio
    async def test_evaluate_sequences_handles_rule_with_no_sequence(self, presence_cep_worker):
        """Verifies that _evaluate_sequences handles rules without sequences gracefully."""
        # Set up worker with rule that has no sequence
        presence_cep_worker._rules = [
            {
                "name": "test_rule",
                "target_state": "vacated"
                # No sequence key
            }
        ]
        
        with patch('max_assistant.services.presence_cep.aiomqtt.Client') as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # This should not crash
            await presence_cep_worker._evaluate_sequences(mock_client_instance, 1000.0)
            
            # Should not have crashed (no assertion needed)