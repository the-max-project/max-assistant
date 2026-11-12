# Copyright (c) 2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Threaded synthetic sensor worker for presence debouncing and multi-sensor sequence correlation.
Loads sequential trigger rules from a configuration file (YAML/JSON) anchored to the project root.
Supports both 4-part (home/{room}/sensor/{type}) and 5-part (home/{room}/sensor/{type}/{name}) topics.
"""
import asyncio
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional
import aiomqtt

from max_assistant.utils.path_utils import resolve_project_path

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


class PresenceCEPWorker:
    """Threaded synthetic sensor worker for presence debouncing and sequence matching."""

    def __init__(
            self,
            broker_host: str = "127.0.0.1",
            broker_port: int = 1883,
            config_path: str = "cep_config.yaml"
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.config_path = resolve_project_path(config_path)

        self._rules: List[Dict[str, Any]] = []
        self._sensor_history: List[Dict[str, Any]] = []  # Ring buffer of recent state events
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._load_config()

    def _load_config(self):
        """Loads CEP sequence rules from a YAML or JSON file."""
        if not self.config_path.exists():
            logger.warning(
                f"CEP config '{self.config_path}' not found. Falling back to default single-room door/pir heuristic."
            )
            self._rules = [{
                "name": "default_room_exit",
                "room": "main",
                "target_state": "vacated",
                "window_seconds": 15,
                "sequence": [
                    {"room": "main", "sensor": "door", "state": "closed"},
                    {"room": "main", "sensor": "pir", "state": "inactive"}
                ]
            }]
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                if self.config_path.suffix.lower() in (".yaml", ".yml") and yaml:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
                self._rules = data.get("rules", [])
                logger.info(f"PresenceCEPWorker: Loaded {len(self._rules)} sequence rules from '{self.config_path}'.")
        except Exception as e:
            logger.error(f"PresenceCEPWorker: Error loading config file '{self.config_path}': {e}", exc_info=True)

    def start(self):
        """Starts the CEP worker in a dedicated background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        thread = threading.Thread(target=self._run_thread, daemon=True, name="PresenceCEPThread")
        thread.start()
        self._thread = thread
        logger.info("PresenceCEPWorker: Background thread started.")

    def _run_thread(self):
        """Creates an independent asyncio loop for the background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._listen_and_synthesize())
        finally:
            loop.close()

    async def _listen_and_synthesize(self):
        """Main listening loop subscribing to raw hardware telemetry."""
        while not self._stop_event.is_set():
            try:
                async with aiomqtt.Client(self.broker_host, port=self.broker_port) as client:
                    await client.subscribe("home/+/sensor/#")
                    logger.info("PresenceCEPWorker: Subscribed to raw sensors.")
                    async for message in client.messages:
                        await self._process_raw_sensor(client, message)
            except aiomqtt.MqttError as e:
                logger.warning(f"PresenceCEPWorker: Broker disconnected: {e}. Retrying in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"PresenceCEPWorker: Loop error: {e}", exc_info=True)
                await asyncio.sleep(2)

    async def _process_raw_sensor(self, client: aiomqtt.Client, message):
        """Parses incoming MQTT sensor telemetry and evaluates sequence triggers."""
        try:
            # Supports: home/{room}/sensor/{type} OR home/{room}/sensor/{type}/{name}
            parts = str(message.topic).split("/")
            room = parts[1]
            device_type = parts[3]
            payload = json.loads(message.payload.decode())
            state = payload.get("state")

            # Extract optional device name from 5th topic segment or payload
            device_name = parts[4] if len(parts) > 4 else (payload.get("name") or payload.get("device_name"))
        except (json.JSONDecodeError, IndexError, UnicodeDecodeError, AttributeError) as e:
            logger.debug(f"Ignoring unparseable sensor message on topic '{message.topic}': {e}")
            return

        now = time.time()

        self._sensor_history.append({
            "room": room,
            "sensor": device_type,
            "name": device_name,
            "state": state,
            "timestamp": now,
            "user_id": payload.get("user_id", "default_user")
        })

        # Prune events older than 60 seconds
        self._sensor_history = [
            evt for evt in self._sensor_history
            if (now - evt["timestamp"]) <= 60
        ]

        await self._evaluate_sequences(client, now)

    async def _evaluate_sequences(self, client: aiomqtt.Client, current_time: float):
        """Checks if recent sensor events satisfy configured sequence steps in order."""
        for rule in self._rules:
            sequence = rule.get("sequence", [])
            if not sequence:
                continue

            window_seconds = rule.get("window_seconds", 30)
            target_room = rule.get("room", "main")
            target_state = rule.get("target_state", "vacated")

            matched, last_user_id = self._match_sequence(sequence, window_seconds, current_time)

            if matched:
                synthetic_topic = f"max/presence/{target_room}"
                event_payload = json.dumps({
                    "state": target_state,
                    "room": target_room,
                    "user_id": last_user_id,
                    "timestamp": current_time,
                    "trigger_rule": rule.get("name")
                })
                await client.publish(synthetic_topic, event_payload)
                logger.info(
                    f"PresenceCEPWorker: Rule '{rule.get('name')}' triggered -> {synthetic_topic}: {target_state}"
                )

                self._sensor_history.clear()
                break

    def _match_sequence(self, sequence: List[Dict[str, Any]], max_window: float, current_time: float):
        """
        Verifies that every step in the sequence occurred in chronological order
        within max_window seconds from now.
        """
        relevant_events = [
            e for e in self._sensor_history
            if (current_time - e["timestamp"]) <= max_window
        ]

        search_idx = 0
        last_timestamp = 0.0
        last_user_id = "default_user"

        for step in sequence:
            req_room = step.get("room")
            req_sensor = step.get("sensor")
            req_name = step.get("name")
            req_state = step.get("state")

            found = False
            for i in range(search_idx, len(relevant_events)):
                evt = relevant_events[i]
                name_matches = (req_name is None) or (evt.get("name") == req_name)

                if (
                    evt["room"] == req_room and
                    evt["sensor"] == req_sensor and
                    name_matches and
                    str(evt["state"]).lower() == str(req_state).lower() and
                    evt["timestamp"] >= last_timestamp
                ):
                    found = True
                    search_idx = i + 1
                    last_timestamp = evt["timestamp"]
                    last_user_id = evt.get("user_id", "default_user")
                    break

            if not found:
                return False, last_user_id

        return True, last_user_id